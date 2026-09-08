
#http://127.0.0.1:5000 


from __future__ import annotations
import time
import httpx
import openai
import os
import io
import html
import json
import argparse
import unittest
import sys
import textwrap
import tempfile
import uuid
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from collections import Counter
from unittest.mock import patch
from string import Template
from types import SimpleNamespace
from typing import Dict, Any, List, Optional

from flask import Flask, request, render_template_string, make_response, send_file
from dotenv import load_dotenv
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

# NOTE: Modern import for the SDK
try:
    from openai import OpenAI  # pip install openai
except ImportError as e:
    raise SystemExit("OpenAI SDK not installed. Run: pip install openai") from e

load_dotenv(override=True)
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  
MAX_UPLOAD_BYTES = 1 * 1024 * 1024
UPLOAD_ARCHIVE_DIR = Path(os.getenv("UPLOAD_ARCHIVE_DIR", "uploaded_files"))
UPLOAD_REPORTS_DIR = UPLOAD_ARCHIVE_DIR / "reports"
PROJECT_TITLE = "Адаптивная система тестирования на основе ИИ"
DOWNLOAD_BASENAME = "dissertation-project-karandashev-la"
REPORT_TIMEZONE = os.getenv("REPORT_TIMEZONE", "Asia/Qyzylorda")

MCQ_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "mcqs": {
            "type": "array",
            "minItems": 10,
            "maxItems": 10,
            "items": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "options": {
                        "type": "array",
                        "minItems": 4,
                        "maxItems": 4,
                        "items": {"type": "string"},
                    },
                    "correct_index": {"type": "integer", "minimum": 0, "maximum": 3},
                    "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
                    "explanation": {"type": "string"},
                },
                "required": ["prompt", "options", "correct_index"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["mcqs"],
    "additionalProperties": False,
}

# -----------------------
# Prompt helpers
# -----------------------

SYSTEM_MSG_TEMPLATE = (
    "You are an assessment designer. Produce valid JSON that conforms strictly to the provided schema. "
    "Generate exactly {question_count} multiple-choice questions (MCQs), each with exactly {option_count} options. "
    "{correct_guidance}"
)

USER_INSTRUCTION_TEMPLATE = (
    """
Context material:
-----------------
{context}

Task:
-----
- Language: {language}
- Create exactly {question_count} MCQs on the topic above.
- Each MCQ MUST have exactly {option_count} plausible options.
- {correct_requirement}
- Vary difficulty (easy/medium/hard) across the set.
- Prefer clarity, unambiguous wording, and curriculum relevance.
- Return ONLY JSON that matches the schema (no extra commentary).
    """
)

QUESTION_MODES: Dict[str, Dict[str, Any]] = {
    "single": {
        "options": 4,
        "min_correct": 1,
        "max_correct": 1,
        "count": 10,
        "system_guidance": "Each MCQ has exactly one correct answer indicated by the field `correct_index` (0-based).",
        "user_requirement": "Provide the zero-based index of the single correct answer in the field \"correct_index\".",
    },
    "multiple": {
        "options": 7,
        "min_correct": 1,
        "max_correct": 3,
        "count": 10,
        "system_guidance": "Each MCQ may have multiple correct answers. Provide all correct answers in the field `correct_indices` as a list of zero-based indexes (1 to 3 items).",
        "user_requirement": "Between 1 and 3 options must be correct. Provide their zero-based indexes (sorted ascending) in the field \"correct_indices\".",
    },
    "combined": {
        "composite": True,
        "components": [
            {"mode": "single", "count": 10},
            {"mode": "multiple", "count": 10},
        ],
        "system_guidance": "Create a blended set with 10 single-answer MCQs (4 options, 1 correct) and 10 multi-answer MCQs (7 options, up to 3 correct).",
        "user_requirement": "Provide 20 MCQs total: 10 with a single correct answer (4 options) and 10 with multiple correct answers (7 options, 1-3 correct). Return them in a single JSON payload.",
    },
}



def _build_custom_mode_config(settings: Dict[str, int]) -> Dict[str, Any]:
    questions = int(settings.get("questions", 0))
    options = int(settings.get("options", 0))
    correct = int(settings.get("correct", 0))

    if not 1 <= questions <= 20:
        raise ValueError("custom_questions_out_of_range")
    if not 2 <= options <= 8:
        raise ValueError("custom_options_out_of_range")
    if not 1 <= correct <= 3:
        raise ValueError("custom_correct_out_of_range")
    if correct > options:
        raise ValueError("custom_correct_exceeds_options")

    single_answer = correct == 1
    if single_answer:
        system_guidance = "Each MCQ has exactly one correct answer indicated by the field `correct_index` (0-based)."
        user_requirement = 'Provide the zero-based index of the single correct answer in the field "correct_index".'
    else:
        system_guidance = (
            f"Each MCQ has exactly {correct} correct answers. Provide all correct answers in the field `correct_indices` "
            f"as a sorted list of {correct} zero-based indexes."
        )
        user_requirement = (
            f"Exactly {correct} options must be correct. Provide their zero-based indexes (sorted ascending) in the field "
            '"correct_indices".'
        )

    return {
        "options": options,
        "min_correct": correct,
        "max_correct": correct,
        "count": questions,
        "system_guidance": system_guidance,
        "user_requirement": user_requirement,
        "single_answer": single_answer,
        "result_mode": "custom",
        "custom_settings": {
            "questions": questions,
            "options": options,
            "correct": correct,
        },
    }


DEFAULT_MODE = "single"

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")



def _require_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your environment or a .env file."
        )
    api_key = api_key.strip().strip("\"'")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is empty. Check your environment or .env file."
        )
    return api_key


from openai import OpenAI
from openai import APIConnectionError, APIStatusError, RateLimitError

def _chat_completion(api_key, messages, model=DEFAULT_MODEL, temperature=0.2, **extra):
    """
    Обёртка над OpenAI chat.completions.create с таймаутом и понятными ошибками.
    """

    client = OpenAI(api_key=api_key, timeout=60.0)

    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    kwargs.update(extra)

    try:
        return client.chat.completions.create(**kwargs)
    except RateLimitError as e:
        # слишком много запросов / исчерпан квота
        raise RuntimeError("rate_limited") from e
    except APIStatusError as e:
        # ошибки вида 401, 403, 500 и т.п. от API
        # сюда же попадёт 'unsupported_country_region_territory'
        raise RuntimeError(f"openai_status_error_{e.status_code}") from e
    except APIConnectionError as e:
        # проблемы с сетью / DNS / TLS
        raise RuntimeError("openai_connection_error") from e
    except Exception as e:
        # всё остальное
        raise RuntimeError("generation_failed") from e




def _normalise_mcq_payload(data: Any) -> Dict[str, Any]:
    """Ensure the JSON payload exposes an `mcqs` array at the top level."""
    if isinstance(data, dict):
        if "mcqs" in data and isinstance(data["mcqs"], list):
            return data
        for key in ("output", "result", "data", "response"):
            nested = data.get(key)
            if isinstance(nested, dict) and "mcqs" in nested and isinstance(nested["mcqs"], list):
                return nested
        for alias in ("questions", "items", "mcq_list"):
            alias_value = data.get(alias)
            coerced = _coerce_questions_array(alias_value)
            if coerced is not None:
                return {"mcqs": coerced}
    keys = list(data.keys()) if isinstance(data, dict) else []
    raise ValueError("Model did not return the expected JSON structure" + (f"; top-level keys: {keys}" if keys else ""))


def _coerce_questions_array(value: Any) -> Optional[List[Dict[str, Any]]]:
    """Map alternative question structures to the canonical MCQ payload."""
    if not isinstance(value, list) or not value:
        return None

    mcqs: List[Dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            return None

        normalized = dict(item)

        if "prompt" not in normalized:
            question_text = normalized.get("question")
            if isinstance(question_text, str):
                normalized["prompt"] = question_text

        if "options" not in normalized:
            for option_key in ("options", "choices", "answers", "answer_choices"):
                opts = normalized.get(option_key)
                if isinstance(opts, list):
                    normalized["options"] = opts
                    break

        options = normalized.get("options")
        if not isinstance(options, list):
            return None

        if "correct_index" not in normalized:
            if isinstance(normalized.get("answer_index"), int):
                normalized["correct_index"] = normalized["answer_index"]
            elif isinstance(normalized.get("correct_option"), int):
                normalized["correct_index"] = normalized["correct_option"]
            elif isinstance(normalized.get("correct_choice"), int):
                normalized["correct_index"] = normalized["correct_choice"]
            else:
                for answer_key in ("answer", "correct_answer"):
                    answer = normalized.get(answer_key)
                    if isinstance(answer, int):
                        normalized["correct_index"] = answer
                        break
                    if isinstance(answer, str):
                        try:
                            normalized["correct_index"] = options.index(answer)
                            break
                        except ValueError:
                            continue

        if "correct_indices" not in normalized:
            for key in ("correct_indices", "correct_indexes", "answer_indexes", "answer_indices", "correct_options", "correct_choices"):
                value = normalized.get(key)
                if isinstance(value, list) and value:
                    normalized["correct_indices"] = value
                    break
        if "correct_indices" not in normalized and isinstance(normalized.get("correct_index"), int):
            normalized["correct_indices"] = [normalized["correct_index"]]

        if "difficulty" not in normalized:
            level = normalized.get("level") or normalized.get("difficulty_level")
            if isinstance(level, str):
                normalized["difficulty"] = level.lower()

        if "explanation" not in normalized:
            rationale = normalized.get("rationale") or normalized.get("explanation_text")
            if isinstance(rationale, str):
                normalized["explanation"] = rationale

        if "prompt" not in normalized or (
            "correct_index" not in normalized and "correct_indices" not in normalized
        ):
            return None

        mcqs.append(normalized)

    return mcqs




def _read_pdf_document(data: bytes, page_start: Optional[int] = None, page_end: Optional[int] = None) -> str:
    """Extract text from a PDF payload."""
    try:
        from PyPDF2 import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "Reading .pdf files requires PyPDF2. Install via: pip install PyPDF2"
        ) from exc

    reader = PdfReader(io.BytesIO(data))
    pages = list(getattr(reader, "pages", []))
    total_pages = len(pages)
    if page_start is not None and page_start < 1:
        raise ValueError("page_range_invalid")
    if page_end is not None and page_end < 1:
        raise ValueError("page_range_invalid")
    if page_start is not None and page_end is not None and page_start > page_end:
        raise ValueError("page_range_invalid")
    if page_start is not None and page_start > total_pages:
        raise ValueError("page_range_out_of_bounds")

    start_index = (page_start - 1) if page_start is not None else 0
    end_index = page_end if page_end is not None else total_pages
    end_index = min(end_index, total_pages)

    chunks: List[str] = []
    for page in pages[start_index:end_index]:
        try:
            extracted = page.extract_text()
        except Exception:
            extracted = None
        if extracted:
            chunks.append(extracted)
    joined = "\n".join(part.strip() for part in chunks if part and part.strip())
    return joined


def _read_office_document(data: bytes, suffix: str) -> str:
    """Extract text from .doc/.docx payloads."""
    if suffix == ".docx":
        try:
            from docx import Document
        except ImportError as exc:
            raise RuntimeError(
                "Reading .docx files requires python-docx. Install via: pip install python-docx"
            ) from exc

        document = Document(io.BytesIO(data))
        parts: List[str] = []
        for paragraph in document.paragraphs:
            if paragraph.text:
                parts.append(paragraph.text)
        for table in getattr(document, "tables", []):
            for row in table.rows:
                for cell in row.cells:
                    if cell.text:
                        parts.append(cell.text)
        return "\n".join(part.strip() for part in parts if part.strip())

    if suffix == ".doc":
        # Prefer COM automation on Windows when available to avoid external antiword dependency.
        try:
            from win32com.client import Dispatch  # type: ignore
        except Exception:
            Dispatch = None

        if Dispatch:
            tmp = tempfile.NamedTemporaryFile(suffix=".doc", delete=False)
            word = None
            doc = None
            try:
                tmp.write(data)
                tmp.flush()
                tmp.close()

                word = Dispatch("Word.Application")
                word.Visible = False
                word.DisplayAlerts = 0
                doc = word.Documents.Open(tmp.name)
                text = doc.Content.Text
                if text:
                    return text.replace("\r", "\n")
            except Exception:
                # Fall back to textract below
                pass
            finally:
                try:
                    if doc is not None:
                        doc.Close(False)
                except Exception:
                    pass
                try:
                    if word is not None:
                        word.Quit()
                except Exception:
                    pass
                try:
                    os.remove(tmp.name)
                except OSError:
                    pass

        try:
            import textract
        except ImportError as exc:
            raise RuntimeError(
                "Reading .doc files requires textract. On pip>=24 install the patched copy "
                "from ./textract-1.6.4 (pip install ./textract-1.6.4) or temporarily use pip<24 "
                "and pip install textract. Alternatively, install pywin32 with Microsoft Word "
                "so the app can read .doc via COM automation."
            ) from exc

        tmp = tempfile.NamedTemporaryFile(suffix=".doc", delete=False)
        try:
            tmp.write(data)
            tmp.flush()
            tmp.close()
            extracted = textract.process(tmp.name)
        except Exception as exc:
            raise RuntimeError(
                "Failed to read .doc. Install antiword (ensure its binary is on PATH) "
                "or install pywin32 with Microsoft Word so the app can use COM automation."
            ) from exc
        finally:
            try:
                os.remove(tmp.name)
            except OSError:
                pass
        return extracted.decode("utf-8", errors="ignore")

    raise ValueError("Unsupported Office document type")


def _extract_message_content(completion: Any) -> str:
    """Extract the text content from the first chat completion choice."""
    choices = getattr(completion, "choices", None)
    if not choices:
        raise ValueError("Model did not return any choices")

    message = getattr(choices[0], "message", None)
    content = None

    if hasattr(message, "content"):
        content = message.content
    elif isinstance(message, dict):
        content = message.get("content")
    else:
        content = message

    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                if "text" in part:
                    parts.append(str(part["text"]))
                elif "content" in part:
                    parts.append(str(part["content"]))
            elif part:
                parts.append(str(part))
        content = ''.join(parts).strip()
    elif content is not None:
        content = str(content).strip()

    if not content:
        raise ValueError("Model returned empty content")
    return content


def _coerce_answer_candidate(value: Any, options: List[str]) -> Optional[int]:
    """Convert a model-provided answer marker to a zero-based option index."""
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return None
        if candidate.isdigit():
            try:
                return int(candidate)
            except (TypeError, ValueError):
                return None
        try:
            return options.index(candidate)
        except ValueError:
            return None
    return None

def _normalise_single_answer(item: Dict[str, Any], options: List[str]) -> int:
    idx = item.get("correct_index")
    if isinstance(idx, list):
        if len(idx) == 1:
            idx = idx[0]
        else:
            raise AssertionError("Expected exactly one correct answer")
    if idx is None:
        correct_indices = item.get("correct_indices")
        if isinstance(correct_indices, list) and correct_indices:
            if len(correct_indices) == 1:
                idx = correct_indices[0]
            else:
                raise AssertionError("Expected exactly one correct answer")
    if idx is None:
        for key in ("answer", "correct_answer"):
            candidate = item.get(key)
            if candidate is not None:
                idx = candidate
                break
    if idx is None:
        answers = item.get("answers") or item.get("correct_answers")
        if isinstance(answers, list) and answers:
            idx = answers[0]
    coerced = _coerce_answer_candidate(idx, options)
    if coerced is None:
        raise AssertionError("Missing correct_index for single-answer MCQ")
    if not (0 <= coerced < len(options)):
        raise AssertionError("correct_index for single-answer MCQ is out of range")
    return coerced


def _normalise_multiple_answers(item: Dict[str, Any], options: List[str], min_answers: int, max_answers: int) -> List[int]:
    candidates = item.get("correct_indices")
    if candidates is None:
        for key in ("correct_indexes", "answer_indexes", "correct_options", "correct_choices"):
            value = item.get(key)
            if isinstance(value, list):
                candidates = value
                break
    if candidates is None and "correct_index" in item:
        candidates = [item["correct_index"]]
    if candidates is None:
        for key in ("answers", "correct_answers"):
            value = item.get(key)
            if isinstance(value, list) and value:
                candidates = value
                break
    if candidates is None:
        raise AssertionError("Missing correct_indices for multi-answer MCQ")
    if not isinstance(candidates, list):
        candidates = [candidates]

    unique_indices: List[int] = []
    for candidate in candidates:
        coerced = _coerce_answer_candidate(candidate, options)
        if coerced is not None and coerced not in unique_indices:
            unique_indices.append(coerced)

    if not unique_indices:
        unique_indices.append(0)

    for idx in range(len(options)):
        if len(unique_indices) >= min_answers:
            break
        if idx not in unique_indices:
            unique_indices.append(idx)

    if len(unique_indices) > max_answers:
        unique_indices = unique_indices[:max_answers]

    for idx in unique_indices:
        if not (0 <= idx < len(options)):
            raise AssertionError("correct_indices includes out-of-range index")

    return sorted(unique_indices)




def _generate_mode_payload(
    api_key: str,
    text: str,
    language: str,
    mode_key: str,
    *,
    config_override: Optional[Dict[str, Any]] = None,
    additional_context: str = "",
) -> Dict[str, Any]:
    config = dict(config_override or QUESTION_MODES[mode_key])
    if config.get("composite") and config_override is None:
        raise ValueError("Composite mode cannot be generated directly")

    option_count = config["options"]
    question_count = config.get("count", 10)

    context_material = text[:15000]
    if additional_context.strip():
        context_material = (
            f"{context_material}\n\n"
            "Additional user instructions for this generation:\n"
            "-----------------------------------------------\n"
            f"{additional_context.strip()[:3000]}"
        )

    base_instruction = USER_INSTRUCTION_TEMPLATE.format(
        context=context_material,
        language=language,
        option_count=option_count,
        question_count=question_count,
        correct_requirement=config["user_requirement"],
    )

    system_msg = SYSTEM_MSG_TEMPLATE.format(
        option_count=option_count,
        question_count=question_count,
        correct_guidance=config["system_guidance"],
    )

    expected = config.get("count", 10)
    data = None
    last_count = None
    retry_hint = ""

    for attempt in range(3):
        instruction = base_instruction if not retry_hint else f"{base_instruction}\n{retry_hint}"
        completion = _chat_completion(
            api_key=api_key,
            model=MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": instruction},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            timeout=40.0,  # тот же лимит, что и в _chat_completion
        )

        content = _extract_message_content(completion)
        data = _normalise_mcq_payload(json.loads(content))
        actual = len(data.get("mcqs", []))
        if actual == expected:
            break
        last_count = actual
        retry_hint = textwrap.dedent(
            f"""IMPORTANT:
- The previous attempt produced {actual} questions, but exactly {expected} are required.
- Regenerate the entire set from scratch with exactly {expected} questions."""
        )

    else:
        raise RuntimeError(
            f"model_returned_wrong_count:{mode_key}:{expected}:{last_count or 0}"
        )

    processed: List[Dict[str, Any]] = []
    is_single_answer = config.get("single_answer", mode_key == "single")
    for i, raw in enumerate(data["mcqs"], 1):
        item = dict(raw)
        options = item.get("options", [])
        if not isinstance(options, list):
            raise AssertionError(f"MCQ {i} options must be a list")
        if len(options) != option_count:
            raise AssertionError(
                f"MCQ {i} must have exactly {option_count} options for mode {mode_key}"
            )
        if not isinstance(item.get("prompt"), str):
            raise AssertionError(f"MCQ {i} is missing prompt text")
        item["prompt"] = item["prompt"].strip()

        if is_single_answer:
            idx = _normalise_single_answer(item, options)
            item["correct_index"] = idx
            item["correct_indices"] = [idx]
        else:
            indices = _normalise_multiple_answers(
                item,
                options,
                config["min_correct"],
                config["max_correct"],
            )
            item["correct_indices"] = indices
            if len(indices) == 1:
                item["correct_index"] = indices[0]
            else:
                item.pop("correct_index", None)
        item.setdefault("answer_mode", mode_key)
        processed.append(item)

    result_mode = config.get("result_mode", mode_key)
    payload = {
        "mcqs": processed,
        "mode": result_mode,
        "options_per_question": option_count,
        "answer_cardinality": {"min": config["min_correct"], "max": config["max_correct"]},
        "language": language,
        "total_questions": expected,
    }
    if config_override and config.get("custom_settings"):
        payload["custom_settings"] = config["custom_settings"]
    return payload


def generate_mcqs_from_text(
    text: str,
    language: str = "ru",
    mode: str = DEFAULT_MODE,
    custom_settings: Optional[Dict[str, int]] = None,
    additional_context: str = "",
) -> Dict[str, Any]:
    """Call OpenAI and return parsed JSON with MCQs (10, 20, or custom)."""
    api_key = _require_api_key()
    mode_key = (mode or DEFAULT_MODE).lower()

    if mode_key == "custom":
        if not custom_settings:
            raise ValueError("Custom settings are required for custom mode")
        config = _build_custom_mode_config(custom_settings)
        return _generate_mode_payload(
            api_key,
            text,
            language,
            mode_key,
            config_override=config,
            additional_context=additional_context,
        )

    if mode_key not in QUESTION_MODES:
        raise ValueError(f"Unsupported question type: {mode}")

    config = QUESTION_MODES[mode_key]
    if config.get("composite"):
        combined_mcqs: List[Dict[str, Any]] = []
        components_meta: List[Dict[str, Any]] = []
        options_summary: Dict[str, Any] = {}
        cardinality_summary: Dict[str, Any] = {}
        total = 0

        for component in config["components"]:
            sub_mode = component["mode"]
            sub_data = _generate_mode_payload(
                api_key,
                text,
                language,
                sub_mode,
                additional_context=additional_context,
            )
            count = len(sub_data["mcqs"])
            total += count
            components_meta.append({
                "mode": sub_mode,
                "count": count,
                "options": sub_data["options_per_question"],
                "answer_cardinality": sub_data["answer_cardinality"],
            })
            options_summary[sub_mode] = sub_data["options_per_question"]
            cardinality_summary[sub_mode] = sub_data["answer_cardinality"]
            for item in sub_data["mcqs"]:
                combined_mcqs.append(dict(item))

        return {
            "mcqs": combined_mcqs,
            "mode": mode_key,
            "components": components_meta,
            "options_per_question": options_summary,
            "answer_cardinality": cardinality_summary,
            "language": language,
            "total_questions": total,
        }

    return _generate_mode_payload(api_key, text, language, mode_key, additional_context=additional_context)

RU_TEXT = {
    "project_title": PROJECT_TITLE,
    "heading": PROJECT_TITLE,
    "hero_badge": "AI-инструмент для диссертационного проекта",
    "hero_subtitle": "Загрузите конспект, PDF или Word-файл, задайте фокус и получите готовые вопросы с вариантами ответов.",
    "visual_upload_title": "1. Материал",
    "visual_upload_text": "Файл, страницы и язык генерации.",
    "visual_focus_title": "2. Фокус",
    "visual_focus_text": "Акценты, темы и исключения.",
    "visual_result_title": "3. Результат",
    "visual_result_text": "Готовый тест и JSON для проверки.",
    "upload_label": "Загрузите файл с темой/конспектом (.txt/.md/.csv/.json/.doc/.docx/.pdf, до 1 МБ):",
    "lang_label": "Язык генерации:",
    "lang_ru": "Русский",
    "lang_kk": "Қазақша",
    "mode_label": "Тип теста:",
    "mode_single_label": "С одним ответом",
    "mode_multi_label": "С несколькими ответами",
    "mode_combined_label": "Комбинированный (10+10)",
    "mode_custom_label": "Свои настройки",
    "context_heading": "Дополнительный контекст",
    "context_toggle_label": "Использовать дополнительный контекст",
    "context_hint": "Напишите, на что сделать акцент и что не учитывать при составлении вопросов.",
    "context_label": "Ваши инструкции:",
    "context_placeholder": "Например: сделать акцент на терминах и формулах; не учитывать историю создания и биографии.",
    "custom_heading": "Дополнительные параметры",
    "custom_hint": "Чтобы использовать собственные значения, выберите режим \"Свои настройки\" и заполните поля ниже.",
    "custom_questions_label": "Сколько вопросов (до 20):",
    "custom_options_label": "Сколько вариантов ответа (до 8):",
    "custom_correct_label": "Сколько правильных ответов (до 3):",
    "page_range_heading": "Диапазон страниц PDF",
    "page_range_toggle_label": "Ограничить PDF диапазоном страниц",
    "page_range_hint": "Оставьте поля пустыми, чтобы анализировать весь документ. Нумерация страниц начинается с 1.",
    "page_start_label": "С какой страницы:",
    "page_end_label": "По какую страницу:",
    "summary_custom_prefix": "Режим: свои настройки —",
    "summary_custom_questions": "вопросов",
    "summary_custom_options": "вариантов ответа",
    "summary_custom_correct": "правильных ответов на вопрос",
    "submit_label": "Сгенерировать",
    "loading_label": "Генерирую...",
    "copy_json_label": "Копировать JSON",
    "download_json_label": "Скачать JSON",
    "download_docx_label": "Скачать DOCX",
    "download_pdf_label": "Скачать PDF",
    "practice_label": "Режим тренировки",
    "check_answers_label": "Проверить",
    "reset_practice_label": "Сбросить",
    "print_label": "Печать",
    "file_selected_label": "Выбран файл:",
    "questions_tab_label": "Вопросы",
    "json_tab_label": "JSON",
    "error_label": "Ошибка:",
    "done_title": "Готово! Вопросы готовы:",
    "json_title": "JSON результата проекта:",
    "summary_single": "Тип теста: один правильный ответ (4 варианта)",
    "summary_multiple": "Тип теста: несколько правильных ответов (7 вариантов)",
    "summary_combined_prefix": "Тип теста: комбинированный —",
    "summary_fallback": "Тип теста:",
    "questions_word": "вопросов",
    "single_count_phrase": "с одним ответом",
    "multi_count_phrase": "с несколькими ответами",
    "question_label": "Вопрос",
    "answer_format_label": "Формат ответа",
    "answer_format_multi": "несколько правильных вариантов",
    "answer_format_single": "один правильный вариант",
    "multi_answer_label": "Правильные варианты:",
    "single_answer_label": "Правильный вариант",
    "explanation_label": "Пояснение:",
}


CUSTOM_ERROR_MESSAGES = {
    "custom_invalid_numbers": "Введите корректные числа для пользовательского режима.",
    "custom_questions_out_of_range": "Количество вопросов должно быть от 1 до 20.",
    "custom_options_out_of_range": "Количество вариантов должно быть от 2 до 8.",
    "custom_correct_out_of_range": "Количество правильных ответов должно быть от 1 до 3.",
    "custom_correct_exceeds_options": "Правильных ответов не может быть больше, чем вариантов ответов.",
    "page_range_invalid": "Укажите корректный диапазон страниц: начало и конец должны быть положительными числами, начало не больше конца.",
    "page_range_out_of_bounds": "Начальная страница выходит за пределы PDF-файла.",
    "upload_too_large": "Файл слишком большой. Максимальный размер загрузки — 1 МБ.",
    "Reading .pdf files requires PyPDF2. Install via: pip install PyPDF2": "Для обработки PDF установите пакет PyPDF2 (pip install PyPDF2).",
    "generation_failed": "?? ??????? ????????? ????????? ??????. ????????? ????? ??? ?????????? ?????????? ??????????.",
    "generation_failed": "Во время генерации теста произошла непредвиденная ошибка. Попробуйте ещё раз.",
    "openai_connection_error": "Не удалось связаться с сервисом ИИ (проблема с сетью). Попробуйте ещё раз позже.",
    "rate_limited": "Превышен лимит запросов к модели. Подождите немного и попробуйте снова.",
    "openai_status_error_403": "Сервис ИИ вернул ошибку 403 (доступ запрещён). Часто это связано с регионом сервера.",
    "openai_status_error_401": "Ошибка авторизации в OpenAI (проверьте API-ключ).",
}


HTML_TEMPLATE = Template(textwrap.dedent("""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>$project_title</title>
  <style>
    :root {
      --bg: #f4f7fb;
      --surface: #ffffff;
      --surface-soft: #f8fafc;
      --text: #172033;
      --muted: #667085;
      --line: #d9e1ec;
      --primary: #2563eb;
      --primary-dark: #1d4ed8;
      --accent: #0f766e;
      --danger: #dc2626;
      --shadow: 0 18px 45px rgba(22, 34, 51, 0.10);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background:
        linear-gradient(90deg, rgba(248, 251, 255, 0.30) 0%, rgba(248, 251, 255, 0.88) 28%, rgba(248, 251, 255, 0.92) 72%, rgba(248, 251, 255, 0.30) 100%),
        radial-gradient(circle at top left, rgba(37, 99, 235, 0.10), transparent 34rem),
        url("/static/images/quiz-background.png") center top / cover fixed,
        linear-gradient(180deg, #f8fbff 0%, var(--bg) 100%);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      line-height: 1.5;
    }
    .shell { width: min(1120px, calc(100% - 32px)); margin: 0 auto; padding: 28px 0 56px; }
    .hero {
      position: relative;
      min-height: 330px;
      display: flex;
      align-items: flex-end;
      overflow: hidden;
      border-radius: 8px;
      margin-bottom: 18px;
      padding: 34px;
      color: #fff;
      background:
        linear-gradient(90deg, rgba(15, 23, 42, 0.86) 0%, rgba(15, 23, 42, 0.56) 48%, rgba(15, 23, 42, 0.20) 100%),
        url("https://images.unsplash.com/photo-1516321318423-f06f85e504b3?auto=format&fit=crop&w=1600&q=80") center/cover;
      box-shadow: var(--shadow);
    }
    .hero-content { max-width: 650px; }
    .hero h1 { color: #fff; text-wrap: balance; }
    .hero-subtitle { max-width: 580px; margin: 14px 0 0; color: rgba(255,255,255,0.86); font-size: 1.05rem; }
    .hero-badge {
      display: inline-flex;
      align-items: center;
      min-height: 30px;
      margin-bottom: 12px;
      padding: 0.25rem 0.65rem;
      border-radius: 8px;
      background: rgba(255,255,255,0.16);
      border: 1px solid rgba(255,255,255,0.26);
      color: #e0f2fe;
      font-size: 0.8rem;
      font-weight: 800;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      backdrop-filter: blur(8px);
    }
    .visual-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 14px;
      margin-bottom: 18px;
    }
    .visual-card {
      min-height: 148px;
      display: flex;
      align-items: flex-end;
      overflow: hidden;
      position: relative;
      border-radius: 8px;
      padding: 16px;
      color: #fff;
      border: 1px solid rgba(255,255,255,0.35);
      box-shadow: 0 14px 34px rgba(22, 34, 51, 0.12);
      background: #172033 center/cover;
    }
    .visual-card::before {
      content: "";
      position: absolute;
      inset: 0;
      background: linear-gradient(180deg, rgba(15,23,42,0.04) 0%, rgba(15,23,42,0.78) 100%);
    }
    .visual-card > div { position: relative; z-index: 1; }
    .visual-card strong { display: block; font-size: 1rem; margin-bottom: 3px; }
    .visual-card span { display: block; color: rgba(255,255,255,0.82); font-size: 0.9rem; }
    .visual-upload {
      background:
        linear-gradient(135deg, rgba(15, 23, 42, 0.10), rgba(15, 23, 42, 0.10)),
        radial-gradient(circle at 76% 22%, rgba(255,255,255,0.42) 0 8%, transparent 9%),
        linear-gradient(90deg, transparent 0 17%, rgba(255,255,255,0.92) 17% 20%, transparent 20% 100%),
        repeating-linear-gradient(0deg, rgba(255,255,255,0.20) 0 1px, transparent 1px 12px),
        linear-gradient(135deg, #0f766e 0%, #2563eb 100%);
    }
    .visual-focus {
      background:
        radial-gradient(circle at 50% 42%, transparent 0 18%, rgba(255,255,255,0.92) 18% 21%, transparent 21% 100%),
        radial-gradient(circle at 50% 42%, rgba(255,255,255,0.95) 0 5%, transparent 6%),
        linear-gradient(135deg, transparent 0 48%, rgba(255,255,255,0.55) 48% 52%, transparent 52% 100%),
        repeating-linear-gradient(90deg, rgba(255,255,255,0.14) 0 1px, transparent 1px 18px),
        linear-gradient(135deg, #7c3aed 0%, #0f766e 100%);
    }
    .visual-result {
      background:
        linear-gradient(0deg, rgba(255,255,255,0.84) 0 8%, transparent 8% 100%),
        linear-gradient(90deg, transparent 0 13%, rgba(255,255,255,0.84) 13% 24%, transparent 24% 37%, rgba(255,255,255,0.84) 37% 49%, transparent 49% 62%, rgba(255,255,255,0.84) 62% 75%, transparent 75% 100%),
        radial-gradient(circle at 78% 22%, rgba(255,255,255,0.42) 0 8%, transparent 9%),
        linear-gradient(135deg, #dc2626 0%, #2563eb 100%);
    }
    .eyebrow { margin: 0 0 6px; color: var(--accent); font-size: 0.78rem; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase; }
    h1 { margin: 0; font-size: clamp(2rem, 5vw, 3.1rem); line-height: 1.05; letter-spacing: 0; }
    h2, h3 { margin: 0; line-height: 1.2; letter-spacing: 0; }
    h2 { font-size: 1.35rem; }
    h3 { font-size: 1rem; }
    .header-note { max-width: 360px; color: var(--muted); margin: 0; }
    .panel, .card {
      background: rgba(255, 255, 255, 0.92);
      border: 1px solid rgba(217, 225, 236, 0.95);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }
    .panel { padding: 22px; margin-bottom: 22px; }
    .card { padding: 18px; margin-bottom: 16px; }
    .form-section {
      padding: 18px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
      margin-bottom: 16px;
    }
    .section-head { display: flex; justify-content: space-between; gap: 16px; align-items: flex-start; margin-bottom: 14px; }
    .muted { color: var(--muted); font-size: 0.92rem; }
    .section-head .muted, .form-section p { margin: 4px 0 0; }
    .control-grid, .custom-grid, .page-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px;
    }
    .mode-picker { margin-top: 14px; }
    .mode-select-native {
      position: absolute;
      width: 1px;
      height: 1px;
      overflow: hidden;
      clip: rect(0, 0, 0, 0);
      white-space: nowrap;
      border: 0;
      padding: 0;
    }
    .mode-tabs {
      display: flex;
      align-items: flex-end;
      gap: 4px;
      overflow-x: auto;
      padding: 2px 2px 0;
      border-bottom: 1px solid var(--line);
    }
    .mode-tab {
      position: relative;
      flex: 1 0 150px;
      min-height: 58px;
      border: 1px solid var(--line);
      border-bottom: 0;
      border-radius: 8px 8px 0 0;
      padding: 0.72rem 0.85rem;
      background: #eef4ff;
      color: #334155;
      font: inherit;
      font-weight: 850;
      text-align: left;
      cursor: pointer;
      box-shadow: inset 0 -10px 18px rgba(37, 99, 235, 0.05);
      transition: transform 0.15s ease, background 0.15s ease, color 0.15s ease;
    }
    .mode-tab:hover { transform: translateY(-2px); background: #e0ecff; }
    .mode-tab.is-active {
      min-height: 66px;
      background: #fff;
      color: var(--primary-dark);
      transform: translateY(1px);
      box-shadow: 0 -10px 24px rgba(37, 99, 235, 0.12);
    }
    .mode-tab small {
      display: block;
      margin-top: 2px;
      color: var(--muted);
      font-size: 0.76rem;
      font-weight: 750;
    }
    .mode-description {
      display: none;
      padding: 14px 16px;
      border: 1px solid var(--line);
      border-top: 0;
      border-radius: 0 0 8px 8px;
      background: #fff;
      color: var(--muted);
      font-size: 0.93rem;
    }
    .mode-description.is-active { display: block; }
    label { display: flex; flex-direction: column; gap: 7px; font-size: 0.94rem; font-weight: 650; color: #344054; }
    .option-toggle {
      flex-direction: row;
      align-items: center;
      gap: 10px;
      width: fit-content;
      cursor: pointer;
    }
    .option-toggle input {
      width: 18px;
      height: 18px;
      accent-color: var(--primary);
    }
    .optional-fields { margin-top: 14px; }
    .optional-fields[hidden] { display: none; }
    select, input[type=number], textarea {
      width: 100%;
      border: 1px solid #cbd5e1;
      border-radius: 8px;
      background: #fff;
      color: var(--text);
      font: inherit;
      outline: none;
      transition: border-color 0.15s ease, box-shadow 0.15s ease, background 0.15s ease;
    }
    select, input[type=number] { min-height: 44px; padding: 0.55rem 0.7rem; }
    textarea { min-height: 124px; padding: 0.75rem 0.85rem; resize: vertical; }
    select:focus, input[type=number]:focus, textarea:focus {
      border-color: var(--primary);
      box-shadow: 0 0 0 4px rgba(37, 99, 235, 0.14);
    }
    input[type=file] {
      width: 100%;
      margin-top: 8px;
      padding: 14px;
      border: 1px dashed #9fb0c6;
      border-radius: 8px;
      background: var(--surface-soft);
      color: var(--muted);
    }
    input[type=file]::file-selector-button {
      margin-right: 12px;
      border: 0;
      border-radius: 8px;
      background: #e0ecff;
      color: var(--primary-dark);
      padding: 0.55rem 0.8rem;
      font-weight: 750;
      cursor: pointer;
    }
    .file-meta {
      display: none;
      margin-top: 10px;
      padding: 10px 12px;
      border-radius: 8px;
      background: #eef6ff;
      color: #1e3a8a;
      font-size: 0.9rem;
      font-weight: 700;
    }
    .file-meta.is-visible { display: block; }
    .custom-settings { border-style: dashed; background: #fbfcfe; }
    .custom-settings.is-disabled { opacity: 0.55; }
    .actions { display: flex; justify-content: flex-end; gap: 10px; flex-wrap: wrap; padding-top: 4px; }
    .btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 46px;
      padding: 0.72rem 1.2rem;
      border-radius: 8px;
      border: 1px solid var(--primary);
      background: var(--primary);
      color: #fff;
      font-weight: 800;
      cursor: pointer;
      box-shadow: 0 12px 26px rgba(37, 99, 235, 0.22);
      transition: transform 0.15s ease, background 0.15s ease;
    }
    .btn:hover { background: var(--primary-dark); transform: translateY(-1px); }
    .btn:disabled { cursor: wait; opacity: 0.72; transform: none; }
    .btn.secondary {
      min-height: 40px;
      padding: 0.55rem 0.8rem;
      border-color: #cbd5e1;
      background: #fff;
      color: #1f2937;
      box-shadow: none;
      font-weight: 750;
    }
    .btn.secondary:hover { background: #f8fafc; transform: translateY(-1px); }
    .alert { border-color: rgba(220, 38, 38, 0.35); background: #fff7f7; color: #991b1b; box-shadow: none; }
    .result-summary { border-left: 4px solid var(--accent); }
    .result-head {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 16px;
    }
    .result-tools { display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }
    .practice-score {
      display: none;
      margin-top: 12px;
      padding: 12px 14px;
      border-radius: 8px;
      background: #ecfdf5;
      color: #065f46;
      font-weight: 800;
    }
    .practice-score.is-visible { display: block; }
    .tabs {
      display: flex;
      gap: 8px;
      margin: 18px 0 14px;
      padding: 6px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.72);
      width: fit-content;
    }
    .tab-btn {
      min-height: 38px;
      padding: 0.5rem 0.9rem;
      border: 0;
      border-radius: 8px;
      background: transparent;
      color: var(--muted);
      font: inherit;
      font-weight: 800;
      cursor: pointer;
    }
    .tab-btn.is-active {
      background: var(--primary);
      color: #fff;
      box-shadow: 0 8px 18px rgba(37, 99, 235, 0.18);
    }
    .tab-panel[hidden] { display: none; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(290px, 1fr)); gap: 16px; margin: 20px 0; }
    .question-card { box-shadow: none; }
    .question-card.is-correct { border-color: rgba(15, 118, 110, 0.45); background: #f0fdfa; }
    .question-card.is-wrong { border-color: rgba(220, 38, 38, 0.35); background: #fff7f7; }
    .question-index { color: var(--primary); font-weight: 800; font-size: 0.84rem; text-transform: uppercase; }
    .question-title { display: block; margin: 8px 0 12px; font-size: 1.02rem; }
    ol { padding-left: 1.25rem; margin: 0.75rem 0; }
    li { margin: 0.38rem 0; }
    .option-row { display: flex; gap: 8px; align-items: flex-start; }
    .practice-input { display: none; margin-top: 0.22rem; }
    .practice-mode .practice-input { display: inline-block; }
    .practice-mode .answer-line, .practice-mode .explanation-line { display: none; }
    .answer-line { margin-top: 12px; padding-top: 12px; border-top: 1px solid var(--line); }
    pre {
      white-space: pre-wrap;
      word-wrap: break-word;
      background: #0f172a;
      color: #dbeafe;
      padding: 1rem;
      border-radius: 8px;
      overflow: auto;
      max-height: 520px;
    }
    .toast {
      position: fixed;
      right: 18px;
      bottom: 18px;
      z-index: 20;
      display: none;
      max-width: 320px;
      padding: 12px 14px;
      border-radius: 8px;
      background: #111827;
      color: #fff;
      box-shadow: 0 16px 42px rgba(17, 24, 39, 0.24);
      font-weight: 750;
    }
    .toast.is-visible { display: block; }
    @media (max-width: 760px) {
      .shell { width: min(100% - 20px, 1120px); padding-top: 22px; }
      .hero { min-height: 310px; padding: 24px; background-position: center; }
      .visual-grid { grid-template-columns: 1fr; }
      .app-header, .section-head { display: block; }
      .header-note { margin-top: 12px; }
      .mode-tabs { gap: 3px; }
      .mode-tab { flex-basis: 132px; min-height: 54px; padding: 0.62rem 0.7rem; }
      .mode-tab.is-active { min-height: 60px; }
      .panel, .form-section, .card { padding: 14px; }
      .result-head { display: block; }
      .result-tools { justify-content: stretch; margin-top: 12px; }
      .result-tools .btn { flex: 1 1 100%; }
      .tabs { width: 100%; }
      .tab-btn { flex: 1; }
      .actions { display: block; }
      .btn { width: 100%; }
    }
    @media print {
      body { background: #fff; }
      .hero, .visual-grid, form, .result-tools, .card:has(#jsonOut), .toast { display: none !important; }
      .shell { width: 100%; padding: 0; }
      .card, .question-card { box-shadow: none; border-color: #d0d5dd; break-inside: avoid; }
      .grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <header class="hero">
      <div class="hero-content">
        <div class="hero-badge">$hero_badge</div>
        <h1>$heading</h1>
        <p class="hero-subtitle">$hero_subtitle</p>
      </div>
    </header>

    <section class="visual-grid" aria-label="Основные шаги">
      <div class="visual-card visual-upload">
        <div>
          <strong>$visual_upload_title</strong>
          <span>$visual_upload_text</span>
        </div>
      </div>
      <div class="visual-card visual-focus">
        <div>
          <strong>$visual_focus_title</strong>
          <span>$visual_focus_text</span>
        </div>
      </div>
      <div class="visual-card visual-result">
        <div>
          <strong>$visual_result_title</strong>
          <span>$visual_result_text</span>
        </div>
      </div>
    </section>

    <form id="uploadForm" class="panel" enctype="multipart/form-data" method="post" action="/upload">
      <section class="form-section">
        <div class="section-head">
          <div>
            <h3>Материал</h3>
            <p class="muted">$upload_label</p>
          </div>
        </div>
        <input id="sourceFile" type="file" name="file" accept=".txt,.md,.csv,.json,.doc,.docx,.pdf" required>
        <div id="fileMeta" class="file-meta" aria-live="polite"></div>
      </section>

      <section class="form-section">
        <div class="section-head">
          <div>
            <h3>Параметры генерации</h3>
            <p class="muted">Выберите язык и формат будущего теста.</p>
          </div>
        </div>
        <div class="control-grid">
          <label>
            <span>$lang_label</span>
            <select name="language">
              <option value="ru">$lang_ru</option>
              <option value="en">English</option>
              <option value="kk">$lang_kk</option>
            </select>
          </label>
        </div>
        <div class="mode-picker" aria-label="$mode_label">
          <span class="mode-select-native">
            <select name="mode" aria-hidden="true" tabindex="-1">
              <option value="single"{% if (selected_mode or 'single') == 'single' %} selected{% endif %}>$mode_single_label</option>
              <option value="multiple"{% if (selected_mode or 'single') == 'multiple' %} selected{% endif %}>$mode_multi_label</option>
              <option value="combined"{% if (selected_mode or 'single') == 'combined' %} selected{% endif %}>$mode_combined_label</option>
              <option value="custom"{% if (selected_mode or 'single') == 'custom' %} selected{% endif %}>$mode_custom_label</option>
            </select>
          </span>
          <div class="mode-tabs" role="tablist" aria-label="$mode_label">
            <button class="mode-tab{% if (selected_mode or 'single') == 'single' %} is-active{% endif %}" type="button" role="tab" aria-selected="{% if (selected_mode or 'single') == 'single' %}true{% else %}false{% endif %}" aria-controls="modeDescSingle" data-mode-value="single">
              $mode_single_label
              <small>10 вопросов / 4 варианта</small>
            </button>
            <button class="mode-tab{% if (selected_mode or 'single') == 'multiple' %} is-active{% endif %}" type="button" role="tab" aria-selected="{% if (selected_mode or 'single') == 'multiple' %}true{% else %}false{% endif %}" aria-controls="modeDescMultiple" data-mode-value="multiple">
              $mode_multi_label
              <small>до 3 правильных</small>
            </button>
            <button class="mode-tab{% if (selected_mode or 'single') == 'combined' %} is-active{% endif %}" type="button" role="tab" aria-selected="{% if (selected_mode or 'single') == 'combined' %}true{% else %}false{% endif %}" aria-controls="modeDescCombined" data-mode-value="combined">
              $mode_combined_label
              <small>20 вопросов</small>
            </button>
            <button class="mode-tab{% if (selected_mode or 'single') == 'custom' %} is-active{% endif %}" type="button" role="tab" aria-selected="{% if (selected_mode or 'single') == 'custom' %}true{% else %}false{% endif %}" aria-controls="modeDescCustom" data-mode-value="custom">
              $mode_custom_label
              <small>ручной формат</small>
            </button>
          </div>
          <div id="modeDescSingle" class="mode-description{% if (selected_mode or 'single') == 'single' %} is-active{% endif %}" role="tabpanel">Обычный тест: один правильный ответ в каждом вопросе.</div>
          <div id="modeDescMultiple" class="mode-description{% if (selected_mode or 'single') == 'multiple' %} is-active{% endif %}" role="tabpanel">Тест с несколькими правильными вариантами: удобно для тем, где ответ может быть составным.</div>
          <div id="modeDescCombined" class="mode-description{% if (selected_mode or 'single') == 'combined' %} is-active{% endif %}" role="tabpanel">Смешанный набор: сначала вопросы с одним ответом, затем вопросы с несколькими правильными ответами.</div>
          <div id="modeDescCustom" class="mode-description{% if (selected_mode or 'single') == 'custom' %} is-active{% endif %}" role="tabpanel">Свои настройки: количество вопросов, вариантов и правильных ответов задаются ниже.</div>
        </div>
      </section>

      <section class="form-section optional-block">
        <div class="section-head">
          <div>
            <h3>$context_heading</h3>
            <p class="muted">$context_hint</p>
          </div>
        </div>
        <label class="option-toggle" for="enableAdditionalContext">
          <input id="enableAdditionalContext" type="checkbox" name="enable_additional_context" value="1"{% if enable_additional_context %} checked{% endif %}>
          <span>$context_toggle_label</span>
        </label>
        <div id="additionalContextFields" class="optional-fields"{% if not enable_additional_context %} hidden{% endif %}>
          <label for="additionalContext">
            <span>$context_label</span>
            <textarea id="additionalContext" name="additional_context" placeholder="$context_placeholder"{% if not enable_additional_context %} disabled{% endif %}>{{ additional_context }}</textarea>
          </label>
        </div>
      </section>

      <section class="form-section optional-block">
        <div class="section-head">
          <div>
            <h3>$page_range_heading</h3>
            <p class="muted">$page_range_hint</p>
          </div>
        </div>
        <label class="option-toggle" for="enablePageRange">
          <input id="enablePageRange" type="checkbox" name="enable_page_range" value="1"{% if enable_page_range %} checked{% endif %}>
          <span>$page_range_toggle_label</span>
        </label>
        <div id="pageRangeFields" class="page-grid optional-fields"{% if not enable_page_range %} hidden{% endif %}>
          <label>
            <span>$page_start_label</span>
            <input type="number" name="page_start" min="1" value="{{ page_defaults.start }}"{% if not enable_page_range %} disabled{% endif %}>
          </label>
          <label>
            <span>$page_end_label</span>
            <input type="number" name="page_end" min="1" value="{{ page_defaults.end }}"{% if not enable_page_range %} disabled{% endif %}>
          </label>
        </div>
      </section>

      <section id="customControls" class="form-section custom-settings{% if (selected_mode or 'single') != 'custom' %} is-disabled{% endif %}">
        <div class="section-head">
          <div>
            <h3>$custom_heading</h3>
            <p class="muted">$custom_hint</p>
          </div>
        </div>
        <div class="custom-grid">
          <label>
            <span>$custom_questions_label</span>
            <input type="number" name="custom_question_count" min="1" max="20" value="{{ custom_defaults.questions }}" {% if (selected_mode or 'single') != 'custom' %}disabled{% endif %}>
          </label>
          <label>
            <span>$custom_options_label</span>
            <input type="number" name="custom_option_count" min="2" max="8" value="{{ custom_defaults.options }}" {% if (selected_mode or 'single') != 'custom' %}disabled{% endif %}>
          </label>
          <label>
            <span>$custom_correct_label</span>
            <input type="number" name="custom_correct_count" min="1" max="3" value="{{ custom_defaults.correct }}" {% if (selected_mode or 'single') != 'custom' %}disabled{% endif %}>
          </label>
        </div>
      </section>

      <div class="actions">
        <button id="submitBtn" class="btn" type="submit" data-default-label="$submit_label" data-loading-label="$loading_label">$submit_label</button>
      </div>
    </form>
  {% if error %}
    <div class="card alert">
      <strong>$error_label</strong> {{ error }}
    </div>
  {% endif %}
  {% if result %}
    <div class="card result-summary">
      <div class="result-head">
        <div>
          <h2>$done_title</h2>
          <div class="muted">
            {% if result.mode == 'single' %}
              $summary_single
            {% elif result.mode == 'multiple' %}
              $summary_multiple
            {% elif result.mode == 'custom' %}
              $summary_custom_prefix {{ result.custom_settings.questions }} $summary_custom_questions, {{ result.options_per_question }} $summary_custom_options, {{ result.custom_settings.correct }} $summary_custom_correct
            {% elif result.mode == 'combined' %}
              $summary_combined_prefix {{ result.total_questions }} $questions_word ({{ result.components[0].count }} $single_count_phrase, {{ result.components[1].count }} $multi_count_phrase)
            {% else %}
              $summary_fallback {{ result.mode }}
            {% endif %}
          </div>
        </div>
        <div class="result-tools" aria-label="Действия с результатом">
          <button class="btn secondary" type="button" id="copyJsonBtn">$copy_json_label</button>
          <button class="btn secondary" type="button" id="downloadJsonBtn">$download_json_label</button>
          <button class="btn secondary" type="button" id="downloadDocxBtn">$download_docx_label</button>
          <button class="btn secondary" type="button" id="downloadPdfBtn">$download_pdf_label</button>
          <button class="btn secondary" type="button" id="practiceBtn">$practice_label</button>
          <button class="btn secondary" type="button" id="checkPracticeBtn" hidden>$check_answers_label</button>
          <button class="btn secondary" type="button" id="resetPracticeBtn" hidden>$reset_practice_label</button>
          <button class="btn secondary" type="button" id="printBtn">$print_label</button>
        </div>
      </div>
      <div id="practiceScore" class="practice-score"></div>
    </div>
    <div class="tabs" role="tablist" aria-label="Результаты">
      <button class="tab-btn is-active" type="button" role="tab" aria-selected="true" aria-controls="questionsPanel" data-tab-target="questionsPanel">$questions_tab_label</button>
      <button class="tab-btn" type="button" role="tab" aria-selected="false" aria-controls="jsonPanel" data-tab-target="jsonPanel">$json_tab_label</button>
    </div>

    <div id="questionsPanel" class="tab-panel" role="tabpanel">
      <div id="questionsGrid" class="grid">
        {% for q in result.mcqs %}
          {% set q_index = loop.index0 %}
          <div class="card question-card" data-question-index="{{ q_index }}">
            <div class="question-index">$question_label {{ loop.index }}</div>
            {% if result.mode == 'combined' %}
              <div class="muted">$answer_format_label: {% if q.answer_mode == 'multiple' %}$answer_format_multi{% else %}$answer_format_single{% endif %}</div>
            {% endif %}
            <strong class="question-title">{{ q.prompt }}</strong>
            <ol>
              {% for opt in q.options %}
                <li>
                  <label class="option-row">
                    <input
                      class="practice-input"
                      type="{% if q.correct_indices is defined and q.correct_indices|length > 1 %}checkbox{% else %}radio{% endif %}"
                      name="practice_q_{{ q_index }}"
                      value="{{ loop.index0 }}"
                      data-correct="{% if q.correct_indices is defined and loop.index0 in q.correct_indices %}1{% elif q.correct_index is defined and loop.index0 == q.correct_index %}1{% else %}0{% endif %}"
                    >
                    <span>{{ opt }}</span>
                  </label>
                </li>
              {% endfor %}
            </ol>
            {% if q.correct_indices is defined and q.correct_indices %}
              <div class="muted answer-line">
                {% if q.correct_indices|length > 1 %}
                  $multi_answer_label
                {% else %}
                  $single_answer_label:
                {% endif %}
                {% for idx in q.correct_indices %}
                  {{ idx + 1 }}{% if not loop.last %}, {% endif %}
                {% endfor %}
              </div>
            {% elif q.correct_index is defined %}
              <div class="muted answer-line">$single_answer_label: {{ q.correct_index + 1 }}</div>
            {% endif %}
            {% if q.explanation %}
              <div class="muted explanation-line">$explanation_label {{ q.explanation }}</div>
            {% endif %}
          </div>
        {% endfor %}
      </div>
    </div>

    <div id="jsonPanel" class="tab-panel" role="tabpanel" hidden>
      <div class="card">
        <h2>$json_title</h2>
        <pre id="jsonOut">{{ json_pretty|safe }}</pre>
      </div>
    </div>
  {% endif %}
  </main>
  <div id="toast" class="toast" role="status" aria-live="polite"></div>
  <script>
    (function() {
      var modeSelect = document.querySelector("select[name='mode']");
      var modeButtons = Array.prototype.slice.call(document.querySelectorAll("[data-mode-value]"));
      var modeDescriptions = Array.prototype.slice.call(document.querySelectorAll(".mode-description"));
      var customControls = document.getElementById("customControls");
      if (!modeSelect || !customControls) { return; }
      var inputs = Array.prototype.slice.call(customControls.querySelectorAll("input[type='number']"));
      function sync() {
        var activeMode = modeSelect.value || 'single';
        var isCustom = modeSelect.value === 'custom';
        modeButtons.forEach(function(button) {
          var isActive = button.getAttribute("data-mode-value") === activeMode;
          button.classList.toggle("is-active", isActive);
          button.setAttribute("aria-selected", isActive ? "true" : "false");
        });
        modeDescriptions.forEach(function(panel) {
          var mode = panel.id.replace("modeDesc", "").toLowerCase();
          var isActive = mode === activeMode || (mode === "multiple" && activeMode === "multiple");
          panel.classList.toggle("is-active", isActive);
        });
        customControls.classList.toggle('is-disabled', !isCustom);
        inputs.forEach(function(input) {
          input.disabled = !isCustom;
        });
      }
      modeSelect.addEventListener('change', sync);
      modeButtons.forEach(function(button) {
        button.addEventListener("click", function() {
          modeSelect.value = button.getAttribute("data-mode-value") || "single";
          modeSelect.dispatchEvent(new Event("change", { bubbles: true }));
        });
      });
      sync();

      function bindOptionalSection(toggleId, fieldsId) {
        var toggle = document.getElementById(toggleId);
        var fields = document.getElementById(fieldsId);
        if (!toggle || !fields) { return; }
        var sectionInputs = Array.prototype.slice.call(fields.querySelectorAll("input, textarea, select"));
        function updateVisibility() {
          var enabled = toggle.checked;
          fields.hidden = !enabled;
          sectionInputs.forEach(function(input) {
            input.disabled = !enabled;
          });
        }
        toggle.addEventListener("change", updateVisibility);
        updateVisibility();
      }

      bindOptionalSection("enableAdditionalContext", "additionalContextFields");
      bindOptionalSection("enablePageRange", "pageRangeFields");

      var form = document.getElementById("uploadForm");
      var submitBtn = document.getElementById("submitBtn");
      var fileInput = document.getElementById("sourceFile");
      var fileMeta = document.getElementById("fileMeta");
      var toast = document.getElementById("toast");
      var jsonOut = document.getElementById("jsonOut");
      var copyJsonBtn = document.getElementById("copyJsonBtn");
      var downloadJsonBtn = document.getElementById("downloadJsonBtn");
      var downloadDocxBtn = document.getElementById("downloadDocxBtn");
      var downloadPdfBtn = document.getElementById("downloadPdfBtn");
      var practiceBtn = document.getElementById("practiceBtn");
      var checkPracticeBtn = document.getElementById("checkPracticeBtn");
      var resetPracticeBtn = document.getElementById("resetPracticeBtn");
      var questionsGrid = document.getElementById("questionsGrid");
      var practiceScore = document.getElementById("practiceScore");
      var tabButtons = Array.prototype.slice.call(document.querySelectorAll("[data-tab-target]"));
      var printBtn = document.getElementById("printBtn");

      function showToast(message) {
        if (!toast) { return; }
        toast.textContent = message;
        toast.classList.add("is-visible");
        window.clearTimeout(showToast.timer);
        showToast.timer = window.setTimeout(function() {
          toast.classList.remove("is-visible");
        }, 2400);
      }

      if (fileInput && fileMeta) {
        fileInput.addEventListener("change", function() {
          var file = fileInput.files && fileInput.files[0];
          if (!file) {
            fileMeta.classList.remove("is-visible");
            fileMeta.textContent = "";
            return;
          }
          var sizeMb = file.size ? " (" + (file.size / 1024 / 1024).toFixed(2) + " MB)" : "";
          fileMeta.textContent = "$file_selected_label " + file.name + sizeMb;
          fileMeta.classList.add("is-visible");
        });
      }

      if (form && submitBtn) {
        form.addEventListener("submit", function() {
          submitBtn.disabled = true;
          submitBtn.textContent = submitBtn.getAttribute("data-loading-label") || "$loading_label";
        });
      }

      if (copyJsonBtn && jsonOut) {
        copyJsonBtn.addEventListener("click", function() {
          var text = jsonOut.textContent || "";
          if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(text).then(function() {
              showToast("JSON скопирован");
            }).catch(function() {
              showToast("Не удалось скопировать JSON");
            });
          } else {
            var range = document.createRange();
            range.selectNodeContents(jsonOut);
            var selection = window.getSelection();
            selection.removeAllRanges();
            selection.addRange(range);
            document.execCommand("copy");
            selection.removeAllRanges();
            showToast("JSON скопирован");
          }
        });
      }

      if (downloadJsonBtn && jsonOut) {
        downloadJsonBtn.addEventListener("click", function() {
          var blob = new Blob([jsonOut.textContent || ""], { type: "application/json;charset=utf-8" });
          var link = document.createElement("a");
          var stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
          link.href = URL.createObjectURL(blob);
          link.download = "dissertation-project-karandashev-la-result-" + stamp + ".json";
          document.body.appendChild(link);
          link.click();
          link.remove();
          URL.revokeObjectURL(link.href);
          showToast("JSON скачан");
        });
      }

      function downloadStudentFile(format, button) {
        if (!jsonOut) { return; }
        var previousLabel = button ? button.textContent : "";
        if (button) {
          button.disabled = true;
          button.textContent = "Готовлю файл...";
        }
        fetch("/download/" + format, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: jsonOut.textContent || "{}"
        }).then(function(response) {
          if (!response.ok) {
            throw new Error("download_failed");
          }
          var disposition = response.headers.get("Content-Disposition") || "";
          var match = disposition.match(/filename="?([^"]+)"?/);
          var filename = match ? match[1] : "dissertation-project-karandashev-la-test." + format;
          return response.blob().then(function(blob) {
            return { blob: blob, filename: filename };
          });
        }).then(function(file) {
          var link = document.createElement("a");
          link.href = URL.createObjectURL(file.blob);
          link.download = file.filename;
          document.body.appendChild(link);
          link.click();
          link.remove();
          URL.revokeObjectURL(link.href);
          showToast("Файл скачан");
        }).catch(function() {
          showToast("Не удалось подготовить файл");
        }).finally(function() {
          if (button) {
            button.disabled = false;
            button.textContent = previousLabel;
          }
        });
      }

      if (downloadDocxBtn) {
        downloadDocxBtn.addEventListener("click", function() {
          downloadStudentFile("docx", downloadDocxBtn);
        });
      }

      if (downloadPdfBtn) {
        downloadPdfBtn.addEventListener("click", function() {
          downloadStudentFile("pdf", downloadPdfBtn);
        });
      }

      function resetPracticeState() {
        if (!questionsGrid) { return; }
        Array.prototype.slice.call(questionsGrid.querySelectorAll(".practice-input")).forEach(function(input) {
          input.checked = false;
        });
        Array.prototype.slice.call(questionsGrid.querySelectorAll(".question-card")).forEach(function(card) {
          card.classList.remove("is-correct", "is-wrong");
        });
        if (practiceScore) {
          practiceScore.classList.remove("is-visible");
          practiceScore.textContent = "";
        }
      }

      if (practiceBtn && questionsGrid) {
        practiceBtn.addEventListener("click", function() {
          var enabled = !questionsGrid.classList.contains("practice-mode");
          questionsGrid.classList.toggle("practice-mode", enabled);
          practiceBtn.textContent = enabled ? "Показать ответы" : "$practice_label";
          if (checkPracticeBtn) { checkPracticeBtn.hidden = !enabled; }
          if (resetPracticeBtn) { resetPracticeBtn.hidden = !enabled; }
          resetPracticeState();
          showToast(enabled ? "Режим тренировки включён" : "Ответы снова показаны");
        });
      }

      if (checkPracticeBtn && questionsGrid) {
        checkPracticeBtn.addEventListener("click", function() {
          var cards = Array.prototype.slice.call(questionsGrid.querySelectorAll(".question-card"));
          var correctCount = 0;
          cards.forEach(function(card) {
            var inputs = Array.prototype.slice.call(card.querySelectorAll(".practice-input"));
            var isCorrect = inputs.length > 0 && inputs.every(function(input) {
              return input.checked === (input.getAttribute("data-correct") === "1");
            });
            card.classList.toggle("is-correct", isCorrect);
            card.classList.toggle("is-wrong", !isCorrect);
            if (isCorrect) { correctCount += 1; }
          });
          if (practiceScore) {
            practiceScore.textContent = "Результат: " + correctCount + " из " + cards.length;
            practiceScore.classList.add("is-visible");
          }
          showToast("Проверка завершена");
        });
      }

      if (resetPracticeBtn) {
        resetPracticeBtn.addEventListener("click", resetPracticeState);
      }

      if (tabButtons.length) {
        tabButtons.forEach(function(button) {
          button.addEventListener("click", function() {
            var targetId = button.getAttribute("data-tab-target");
            tabButtons.forEach(function(item) {
              var isActive = item === button;
              item.classList.toggle("is-active", isActive);
              item.setAttribute("aria-selected", isActive ? "true" : "false");
            });
            Array.prototype.slice.call(document.querySelectorAll(".tab-panel")).forEach(function(panel) {
              panel.hidden = panel.id !== targetId;
            });
          });
        });
      }

      if (printBtn) {
        printBtn.addEventListener("click", function() {
          window.print();
        });
      }
    })();
  </script>
</body>
</html>
"""))
HTML = HTML_TEMPLATE.substitute(**RU_TEXT)


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES





def _render_html(**context):
    defaults = context.get("custom_defaults") or {}
    normalized = {
        "questions": str(defaults.get("questions") or "10"),
        "options": str(defaults.get("options") or "4"),
        "correct": str(defaults.get("correct") or "1"),
    }
    context["custom_defaults"] = normalized
    page_defaults = context.get("page_defaults") or {}
    context["page_defaults"] = {
        "start": str(page_defaults.get("start") or ""),
        "end": str(page_defaults.get("end") or ""),
    }
    context["additional_context"] = str(context.get("additional_context") or "")
    context["enable_additional_context"] = bool(
        context.get("enable_additional_context") or context["additional_context"].strip()
    )
    context["enable_page_range"] = bool(
        context.get("enable_page_range")
        or context["page_defaults"]["start"].strip()
        or context["page_defaults"]["end"].strip()
    )
    html = render_template_string(HTML, **context)
    response = make_response(html)
    response.headers.setdefault("Content-Type", "text/html; charset=utf-8")
    return response


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.errorhandler(RequestEntityTooLarge)
def handle_upload_too_large(_error):
    return _render_html(error=CUSTOM_ERROR_MESSAGES["upload_too_large"], selected_mode=DEFAULT_MODE), 413


def _parse_optional_page_number(value: Optional[str]) -> Optional[int]:
    if value is None or not str(value).strip():
        return None
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        raise ValueError("page_range_invalid")
    if parsed < 1:
        raise ValueError("page_range_invalid")
    return parsed


def _safe_archive_filename(original_filename: str, request_id: str) -> str:
    safe_name = secure_filename(original_filename or "upload")
    if not safe_name:
        safe_name = "upload"
    return f"{request_id}_{safe_name}"


def _request_report_base(request_id: str, file_storage, raw_size: int) -> Dict[str, Any]:
    forwarded_for = request.headers.get("X-Forwarded-For", "")
    client_ip = (forwarded_for.split(",", 1)[0].strip() if forwarded_for else "") or request.remote_addr or ""
    return {
        "request_id": request_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "request": {
            "method": request.method,
            "path": request.path,
            "host": request.host,
            "client_ip": client_ip,
            "remote_addr": request.remote_addr or "",
            "x_forwarded_for": forwarded_for,
            "origin": request.headers.get("Origin", ""),
            "referer": request.headers.get("Referer", ""),
            "user_agent": request.headers.get("User-Agent", ""),
        },
        "file": {
            "original_name": getattr(file_storage, "filename", "") or "",
            "content_type": getattr(file_storage, "content_type", "") or "",
            "size_bytes": raw_size,
        },
    }


def _write_upload_report(report: Dict[str, Any]) -> Path:
    UPLOAD_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = UPLOAD_REPORTS_DIR / f"{report['request_id']}.json"
    readable_report = _build_readable_upload_report(report)
    report_path.write_text(json.dumps(readable_report, ensure_ascii=False, indent=2), encoding="utf-8")

    index_path = UPLOAD_ARCHIVE_DIR / "upload_reports.jsonl"
    with index_path.open("a", encoding="utf-8") as index_file:
        index_file.write(json.dumps(readable_report, ensure_ascii=False) + "\n")
    return report_path


def _yes_no(value: Any) -> str:
    return "Да" if bool(value) else "Нет"


def _format_report_datetime(value: Any) -> Dict[str, str]:
    raw_value = str(value or "").strip()
    try:
        parsed = datetime.fromisoformat(raw_value.replace("Z", "+00:00"))
    except ValueError:
        return {
            "Дата и время": raw_value,
            "Дата": "",
            "Время": "",
            "Часовой пояс": "",
        }
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    try:
        target_zone = ZoneInfo(REPORT_TIMEZONE)
    except Exception:
        target_zone = datetime.now().astimezone().tzinfo or timezone.utc

    local_dt = parsed.astimezone(target_zone)
    offset = local_dt.strftime("%z")
    formatted_offset = f"UTC{offset[:3]}:{offset[3:]}" if offset else ""
    timezone_name = getattr(target_zone, "key", str(target_zone))
    return {
        "Дата и время": local_dt.strftime("%d.%m.%Y %H:%M:%S"),
        "Дата": local_dt.strftime("%d.%m.%Y"),
        "Время": local_dt.strftime("%H:%M:%S"),
        "Часовой пояс": f"{timezone_name} ({formatted_offset})" if formatted_offset else timezone_name,
    }


def _build_readable_upload_report(report: Dict[str, Any]) -> Dict[str, Any]:
    request_info = report.get("request", {}) if isinstance(report.get("request"), dict) else {}
    file_info = report.get("file", {}) if isinstance(report.get("file"), dict) else {}
    form_info = report.get("form", {}) if isinstance(report.get("form"), dict) else {}
    features = form_info.get("features", {}) if isinstance(form_info.get("features"), dict) else {}
    page_range = form_info.get("page_range", {}) if isinstance(form_info.get("page_range"), dict) else {}
    custom_settings = form_info.get("custom_settings", {}) if isinstance(form_info.get("custom_settings"), dict) else {}
    result_info = report.get("result", {}) if isinstance(report.get("result"), dict) else {}
    display_time = _format_report_datetime(report.get("created_at", ""))

    return {
        "Отчёт о запросе": {
            "Проект": PROJECT_TITLE,
            "ID запроса": report.get("request_id", ""),
            "Дата и время": display_time["Дата и время"],
            "Дата": display_time["Дата"],
            "Время": display_time["Время"],
            "Часовой пояс": display_time["Часовой пояс"],
            "Статус": "Успешно" if result_info.get("status") == "success" else "Ошибка",
        },
        "Файл": {
            "Что это": "Информация о файле, который пользователь загрузил на сайт.",
            "Исходное имя файла": file_info.get("original_name", ""),
            "Тип файла": file_info.get("content_type", ""),
            "Размер в байтах": file_info.get("size_bytes", 0),
            "Размер в МБ": round((int(file_info.get("size_bytes") or 0) / 1024 / 1024), 4),
            "Сохранённое имя": file_info.get("stored_name", ""),
            "Путь к сохранённому файлу": file_info.get("stored_path", ""),
        },
        "Параметры генерации": {
            "Что это": "Настройки, с которыми пользователь отправил файл на обработку.",
            "Язык генерации": form_info.get("language", ""),
            "Тип теста": form_info.get("mode", ""),
            "Дополнительный контекст включён": _yes_no(features.get("additional_context")),
            "Текст дополнительного контекста": form_info.get("additional_context", ""),
            "Диапазон страниц PDF включён": _yes_no(features.get("pdf_page_range")),
            "Начальная страница PDF": page_range.get("start", ""),
            "Конечная страница PDF": page_range.get("end", ""),
            "Свои настройки включены": _yes_no(features.get("custom_settings")),
            "Количество вопросов": custom_settings.get("questions", ""),
            "Количество вариантов ответа": custom_settings.get("options", ""),
            "Количество правильных ответов": custom_settings.get("correct", ""),
        },
        "Откуда был запрос": {
            "Что это": "Сетевые данные и источник перехода, которые передал браузер или прокси.",
            "IP пользователя": request_info.get("client_ip", ""),
            "Remote addr": request_info.get("remote_addr", ""),
            "X-Forwarded-For": request_info.get("x_forwarded_for", ""),
            "Сайт-источник Origin": request_info.get("origin", ""),
            "Страница-источник Referer": request_info.get("referer", ""),
            "User-Agent браузера": request_info.get("user_agent", ""),
            "Метод": request_info.get("method", ""),
            "Путь": request_info.get("path", ""),
            "Host": request_info.get("host", ""),
        },
        "Результат обработки": {
            "Что это": "Итог обработки запроса сервером.",
            "Статус": result_info.get("status", ""),
            "Сгенерировано вопросов": result_info.get("questions_generated", ""),
            "Режим результата": result_info.get("result_mode", ""),
            "Код ошибки": result_info.get("error_key", ""),
            "Сообщение ошибки": result_info.get("error_message", ""),
        },
        "Технические данные": {
            "Что это": "Исходные поля отчёта для отладки и автоматической обработки.",
            "raw": report,
        },
    }


def _archive_upload(file_storage, raw: bytes, request_id: str) -> Dict[str, Any]:
    UPLOAD_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    archived_name = _safe_archive_filename(getattr(file_storage, "filename", "") or "", request_id)
    archived_path = UPLOAD_ARCHIVE_DIR / archived_name
    archived_path.write_bytes(raw)
    return {
        "stored_name": archived_name,
        "stored_path": str(archived_path.resolve()),
    }


def _student_test_title(data: Dict[str, Any]) -> str:
    total = data.get("total_questions") or len(data.get("mcqs", []))
    return f"{PROJECT_TITLE}. Тест для студентов ({total} вопросов)"


def _answer_mode_hint(item: Dict[str, Any], result_mode: str) -> str:
    answer_mode = item.get("answer_mode") or result_mode
    if answer_mode == "multiple":
        return "Выберите один или несколько правильных вариантов."
    return "Выберите один правильный вариант."


def _iter_student_questions(data: Dict[str, Any]):
    result_mode = str(data.get("mode") or "single")
    for index, item in enumerate(data.get("mcqs", []), 1):
        if not isinstance(item, dict):
            continue
        prompt = str(item.get("prompt") or "").strip()
        options = item.get("options") or []
        if not prompt or not isinstance(options, list):
            continue
        yield index, prompt, [str(option) for option in options], _answer_mode_hint(item, result_mode)


def _build_student_docx(data: Dict[str, Any]) -> io.BytesIO:
    try:
        from docx import Document
    except ImportError as exc:
        raise RuntimeError("python-docx is required for DOCX export") from exc

    document = Document()
    document.add_heading(_student_test_title(data), level=1)
    document.add_paragraph("ФИО: ________________________________    Группа: ______________    Дата: __________")
    document.add_paragraph("")

    for index, prompt, options, hint in _iter_student_questions(data):
        document.add_paragraph(f"{index}. {prompt}", style=None)
        document.add_paragraph(hint)
        for option_index, option in enumerate(options):
            marker = chr(ord("A") + option_index)
            document.add_paragraph(f"{marker}. {option}", style=None)
        document.add_paragraph("")

    buffer = io.BytesIO()
    document.save(buffer)
    buffer.seek(0)
    return buffer


def _find_pdf_font() -> Optional[str]:
    candidates = [
        Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "arial.ttf",
        Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "calibri.ttf",
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def _build_student_pdf(data: Dict[str, Any]) -> io.BytesIO:
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_LEFT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    except ImportError as exc:
        raise RuntimeError("reportlab is required for PDF export") from exc

    font_name = "Helvetica"
    font_path = _find_pdf_font()
    if font_path:
        font_name = "StudentExportFont"
        if font_name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(font_name, font_path))

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "StudentTitle",
        parent=styles["Title"],
        fontName=font_name,
        fontSize=17,
        leading=22,
        textColor=colors.HexColor("#172033"),
        alignment=TA_LEFT,
        spaceAfter=12,
    )
    body_style = ParagraphStyle(
        "StudentBody",
        parent=styles["BodyText"],
        fontName=font_name,
        fontSize=10.5,
        leading=15,
        textColor=colors.HexColor("#172033"),
        spaceAfter=5,
    )
    hint_style = ParagraphStyle(
        "StudentHint",
        parent=body_style,
        textColor=colors.HexColor("#667085"),
        fontSize=9,
        leading=13,
    )

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
    )
    story = [
        Paragraph(html.escape(_student_test_title(data)), title_style),
        Paragraph(html.escape("ФИО: ________________________________    Группа: ______________    Дата: __________"), body_style),
        Spacer(1, 6),
    ]
    for index, prompt, options, hint in _iter_student_questions(data):
        story.append(Paragraph(html.escape(f"{index}. {prompt}"), body_style))
        story.append(Paragraph(html.escape(hint), hint_style))
        for option_index, option in enumerate(options):
            marker = chr(ord("A") + option_index)
            story.append(Paragraph(html.escape(f"{marker}. {option}"), body_style))
        story.append(Spacer(1, 7))

    doc.build(story)
    buffer.seek(0)
    return buffer


@app.post("/download/docx")
def download_docx():
    data = request.get_json(silent=True) or {}
    buffer = _build_student_docx(data)
    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"{DOWNLOAD_BASENAME}-test.docx",
        mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )


@app.post("/download/pdf")
def download_pdf():
    data = request.get_json(silent=True) or {}
    buffer = _build_student_pdf(data)
    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"{DOWNLOAD_BASENAME}-test.pdf",
        mimetype="application/pdf",
    )


def read_text_from_bytes(raw: bytes, filename: str, page_start: Optional[int] = None, page_end: Optional[int] = None) -> str:
    """Return textual content extracted from an uploaded file payload."""
    suffix = Path(filename or "").suffix.lower()
    if suffix == ".pdf":
        return _read_pdf_document(raw, page_start=page_start, page_end=page_end)

    if suffix in {".doc", ".docx"}:
        return _read_office_document(raw, suffix)

    if isinstance(raw, bytes):
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw.decode("cp1251", errors="ignore")
    return str(raw)


def read_text_from_file(file_storage, page_start: Optional[int] = None, page_end: Optional[int] = None) -> str:
    """Return textual content extracted from an uploaded file."""
    filename = getattr(file_storage, "filename", "") or ""
    raw = file_storage.stream.read()
    return read_text_from_bytes(raw, filename, page_start=page_start, page_end=page_end)


@app.get("/")
def home():
    return _render_html(selected_mode=DEFAULT_MODE)



@app.post("/upload")
def upload():
    f = request.files.get("file")
    language = request.form.get("language", "ru")
    mode = request.form.get("mode", DEFAULT_MODE)
    enable_additional_context = request.form.get("enable_additional_context") == "1" or bool(request.form.get("additional_context", "").strip())
    enable_page_range = request.form.get("enable_page_range") == "1" or bool(
        request.form.get("page_start", "").strip() or request.form.get("page_end", "").strip()
    )
    additional_context = request.form.get("additional_context", "") if enable_additional_context else ""
    custom_defaults = {
        "questions": request.form.get("custom_question_count", ""),
        "options": request.form.get("custom_option_count", ""),
        "correct": request.form.get("custom_correct_count", ""),
    }
    page_defaults = {
        "start": request.form.get("page_start", "") if enable_page_range else "",
        "end": request.form.get("page_end", "") if enable_page_range else "",
    }
    if not f:
        return _render_html(
            error="Файл не загружен",
            selected_mode=mode,
            custom_defaults=custom_defaults,
            page_defaults=page_defaults,
            additional_context=additional_context,
            enable_additional_context=enable_additional_context,
            enable_page_range=enable_page_range,
        )

    request_id = uuid.uuid4().hex
    raw = f.stream.read()
    report = _request_report_base(request_id, f, len(raw))
    report["form"] = {
        "language": language,
        "mode": mode,
        "features": {
            "additional_context": enable_additional_context,
            "pdf_page_range": enable_page_range,
            "custom_settings": mode == "custom",
        },
        "additional_context": additional_context,
        "page_range": {
            "start": page_defaults["start"],
            "end": page_defaults["end"],
        },
        "custom_settings": custom_defaults,
    }

    try:
        if len(raw) > MAX_UPLOAD_BYTES:
            raise ValueError("upload_too_large")

        archive_info = _archive_upload(f, raw, request_id)
        report["file"].update(archive_info)

        page_start = _parse_optional_page_number(page_defaults["start"])
        page_end = _parse_optional_page_number(page_defaults["end"])
        if page_start is not None and page_end is not None and page_start > page_end:
            raise ValueError("page_range_invalid")

        text_content = read_text_from_bytes(raw, getattr(f, "filename", "") or "", page_start=page_start, page_end=page_end)
        if not text_content.strip():
            raise ValueError("Файл пустой")

        custom_settings = None
        if mode == "custom":
            try:
                custom_settings = {
                    "questions": int(custom_defaults["questions"]),
                    "options": int(custom_defaults["options"]),
                    "correct": int(custom_defaults["correct"]),
                }
            except (TypeError, ValueError):
                raise ValueError("custom_invalid_numbers")
            try:
                _build_custom_mode_config(custom_settings)
            except ValueError as exc:
                raise ValueError(str(exc))
            else:
                custom_defaults = {k: str(custom_settings[k]) for k in ("questions", "options", "correct")}

        data = generate_mcqs_from_text(
            text_content,
            language=language,
            mode=mode,
            custom_settings=custom_settings,
            additional_context=additional_context,
        )
        report["result"] = {
            "status": "success",
            "questions_generated": data.get("total_questions") or len(data.get("mcqs", [])),
            "result_mode": data.get("mode", mode),
        }
        report["report_path"] = str((_write_upload_report(report)).resolve())
        json_pretty = json.dumps(data, ensure_ascii=False, indent=2)
        return _render_html(
            result=data,
            json_pretty=json_pretty,
            selected_mode=mode,
            custom_defaults=custom_defaults,
            page_defaults=page_defaults,
            additional_context=additional_context,
            enable_additional_context=enable_additional_context,
            enable_page_range=enable_page_range,
        )
    except Exception as e:
        error_key = str(e)
        if isinstance(e, RuntimeError) and str(e).startswith("model_returned_wrong_count"):
            error_key = "generation_failed"
        message = CUSTOM_ERROR_MESSAGES.get(error_key, str(e))
        report["result"] = {
            "status": "error",
            "error_key": error_key,
            "error_message": message,
        }
        try:
            report["report_path"] = str((_write_upload_report(report)).resolve())
        except Exception:
            pass
        return _render_html(
            error=message,
            selected_mode=mode,
            custom_defaults=custom_defaults,
            page_defaults=page_defaults,
            additional_context=additional_context,
            enable_additional_context=enable_additional_context,
            enable_page_range=enable_page_range,
        )





# -----------------------
# Unit tests (kept, and fixed per UTF-8 bytes issue)
# Run with:  python mcq_generator_app.py --self-test
# -----------------------

class _Tests(unittest.TestCase):
    def setUp(self):
        # Ensure an API key is present for functions that require it
        os.environ.setdefault("OPENAI_API_KEY", "test-key")
        self.client = app.test_client()

    @staticmethod
    def _fake_mcq_set(
        n_questions: int = 10,
        options: int = 4,
        correct_per_question: int = 1,
        include_correct_index: bool = True,
    ) -> str:
        mcqs = []
        for i in range(n_questions):
            option_values = [f"A{i}{j}" for j in range(options)]
            span = min(correct_per_question, options)
            raw_indices = [(i + offset) % options for offset in range(span)]
            correct_indices = list(dict.fromkeys(raw_indices))
            mcq = {
                "prompt": f"Q{i+1}: Sample?",
                "options": option_values,
                "difficulty": "easy",
                "explanation": "because",
                "correct_indices": correct_indices,
            }
            if include_correct_index and correct_indices:
                mcq["correct_index"] = correct_indices[0]
            mcqs.append(mcq)
        return json.dumps({"mcqs": mcqs}, ensure_ascii=False)

    # FIX: decode bytes to UTF-8 string before assertion
    def test_home_page_ok(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn(PROJECT_TITLE, r.data.decode('utf-8'))

    def test_upload_empty_file(self):
        data = {"file": (io.BytesIO(b""), "empty.txt")}
        r = self.client.post("/upload", data=data, content_type='multipart/form-data')
        self.assertEqual(r.status_code, 200)
        self.assertIn("Файл пустой", r.get_data(as_text=True))

    def test_upload_generates_10x4(self):
        # Patch OpenAI call to avoid network
        with patch.object(sys.modules[__name__], "_chat_completion") as mocked:
            mocked.return_value = SimpleNamespace(
                choices=[SimpleNamespace(message={"content": self._fake_mcq_set(10, 4)})]
            )
            data = {"file": (io.BytesIO(b"Topic: Algebra"), "topic.txt"), "language": "ru"}
            r = self.client.post("/upload", data=data, content_type='multipart/form-data')
            self.assertEqual(r.status_code, 200)
            text = r.get_data(as_text=True)
            self.assertIn("Готово! Вопросы готовы", text)
            self.assertIn("Вопрос 1", text)

    def test_upload_passes_additional_context(self):
        with patch.object(sys.modules[__name__], "_chat_completion") as mocked:
            mocked.return_value = SimpleNamespace(
                choices=[SimpleNamespace(message={"content": self._fake_mcq_set(10, 4)})]
            )
            data = {
                "file": (io.BytesIO(b"Topic: Algebra"), "topic.txt"),
                "language": "ru",
                "additional_context": "Не учитывать вводную часть.",
            }
            r = self.client.post("/upload", data=data, content_type='multipart/form-data')
        self.assertEqual(r.status_code, 200)
        messages = mocked.call_args.kwargs["messages"]
        self.assertIn("Не учитывать вводную часть", messages[1]["content"])

    def test_generate_handles_structured_message_content(self):
        payload = [{"text": self._fake_mcq_set(10, 4)}]
        with patch.object(sys.modules[__name__], "_chat_completion") as mocked:
            mocked.return_value = SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=payload))]
            )
            data = generate_mcqs_from_text("topic")
            self.assertEqual(len(data["mcqs"]), 10)

    def test_generate_includes_additional_context(self):
        with patch.object(sys.modules[__name__], "_chat_completion") as mocked:
            mocked.return_value = SimpleNamespace(
                choices=[SimpleNamespace(message={"content": self._fake_mcq_set(10, 4)})]
            )
            data = generate_mcqs_from_text(
                "topic",
                additional_context="Сделать акцент на формулах, не учитывать биографии.",
            )
        self.assertEqual(len(data["mcqs"]), 10)
        messages = mocked.call_args.kwargs["messages"]
        self.assertIn("Сделать акцент на формулах", messages[1]["content"])
        self.assertIn("Additional user instructions", messages[1]["content"])

    def test_generate_handles_nested_payload(self):
        nested_json = json.dumps({"output": json.loads(self._fake_mcq_set(10, 4))}, ensure_ascii=False)
        with patch.object(sys.modules[__name__], "_chat_completion") as mocked:
            mocked.return_value = SimpleNamespace(
                choices=[SimpleNamespace(message={"content": nested_json})]
            )
            data = generate_mcqs_from_text("topic")
            self.assertEqual(len(data["mcqs"]), 10)


    def test_generate_handles_questions_payload(self):
        base = json.loads(self._fake_mcq_set(10, 4))
        questions = []
        for item in base["mcqs"]:
            questions.append({
                "question": item["prompt"],
                "choices": item["options"],
                "answer": item["options"][item["correct_index"]],
                "level": item.get("difficulty", "easy"),
                "rationale": item.get("explanation", "")
            })
        payload = json.dumps({"questions": questions}, ensure_ascii=False)
        with patch.object(sys.modules[__name__], "_chat_completion") as mocked:
            mocked.return_value = SimpleNamespace(
                choices=[SimpleNamespace(message={"content": payload})]
            )
            data = generate_mcqs_from_text("topic")
            self.assertEqual(len(data["mcqs"]), 10)

    def test_generate_custom_mode_supports_settings(self):
        payload = self._fake_mcq_set(4, options=6, correct_per_question=2, include_correct_index=False)
        with patch.object(sys.modules[__name__], "_chat_completion") as mocked:
            mocked.return_value = SimpleNamespace(
                choices=[SimpleNamespace(message={"content": payload})]
            )
            data = generate_mcqs_from_text(
                "topic",
                mode="custom",
                custom_settings={"questions": 4, "options": 6, "correct": 2},
            )
        self.assertEqual(data["mode"], "custom")
        self.assertEqual(data["total_questions"], 4)
        self.assertEqual(data["options_per_question"], 6)
        self.assertEqual(data["custom_settings"].get("correct"), 2)
        for item in data["mcqs"]:
            self.assertEqual(len(item["options"]), 6)
            self.assertEqual(len(item["correct_indices"]), 2)
            self.assertEqual(item.get("answer_mode"), "custom")

    def test_upload_custom_mode_renders_summary(self):
        payload = self._fake_mcq_set(5, options=6, correct_per_question=2, include_correct_index=False)
        with patch.object(sys.modules[__name__], "_chat_completion") as mocked:
            mocked.return_value = SimpleNamespace(
                choices=[SimpleNamespace(message={"content": payload})]
            )
            data = {
                "file": (io.BytesIO(b"Topic: custom"), "topic.txt"),
                "language": "ru",
                "mode": "custom",
                "custom_question_count": "5",
                "custom_option_count": "6",
                "custom_correct_count": "2",
            }
            r = self.client.post("/upload", data=data, content_type='multipart/form-data')
        self.assertEqual(r.status_code, 200)
        html = r.get_data(as_text=True)
        self.assertIn(RU_TEXT["summary_custom_prefix"], html)
        self.assertIn("5", html)
        self.assertIn("6", html)

    def test_upload_custom_mode_validates_limits(self):
        data = {
            "file": (io.BytesIO(b"Topic: invalid"), "topic.txt"),
            "language": "ru",
            "mode": "custom",
            "custom_question_count": "25",
            "custom_option_count": "9",
            "custom_correct_count": "4",
        }
        r = self.client.post("/upload", data=data, content_type='multipart/form-data')
        self.assertEqual(r.status_code, 200)
        html = r.get_data(as_text=True)
        self.assertIn(CUSTOM_ERROR_MESSAGES["custom_questions_out_of_range"], html)
    def test_generate_multiple_mode_supports_multiple_answers(self):
        payload = self._fake_mcq_set(10, options=7, correct_per_question=2, include_correct_index=False)
        with patch.object(sys.modules[__name__], "_chat_completion") as mocked:
            mocked.return_value = SimpleNamespace(
                choices=[SimpleNamespace(message={"content": payload})]
            )
            data = generate_mcqs_from_text("topic", mode="multiple")
        self.assertEqual(len(data["mcqs"]), 10)
        for item in data["mcqs"]:
            self.assertEqual(len(item["options"]), 7)
            self.assertGreaterEqual(len(item["correct_indices"]), 1)
            self.assertLessEqual(len(item["correct_indices"]), 3)
            self.assertTrue(all(0 <= idx < 7 for idx in item["correct_indices"]))
            if len(item["correct_indices"]) > 1:
                self.assertNotIn("correct_index", item)

    def test_upload_multiple_mode_renders_multi_answers(self):
        payload = self._fake_mcq_set(10, options=7, correct_per_question=3, include_correct_index=False)
        with patch.object(sys.modules[__name__], "_chat_completion") as mocked:
            mocked.return_value = SimpleNamespace(
                choices=[SimpleNamespace(message={"content": payload})]
            )
            data = {
                "file": (io.BytesIO(b"Topic: multi"), "topic.txt"),
                "language": "ru",
                "mode": "multiple",
            }
            r = self.client.post("/upload", data=data, content_type='multipart/form-data')
        self.assertEqual(r.status_code, 200)
        html = r.get_data(as_text=True)
        self.assertIn("7 вариантов", html)
        self.assertIn("Правильные варианты", html)

    def test_generate_combined_mode_returns_20(self):
        payload_single = self._fake_mcq_set(10, options=4, correct_per_question=1)
        payload_multi = self._fake_mcq_set(10, options=7, correct_per_question=2, include_correct_index=False)
        responses = [
            SimpleNamespace(choices=[SimpleNamespace(message={"content": payload_single})]),
            SimpleNamespace(choices=[SimpleNamespace(message={"content": payload_multi})]),
        ]
        with patch.object(sys.modules[__name__], "_chat_completion") as mocked:
            mocked.side_effect = responses
            data = generate_mcqs_from_text("topic", mode="combined")
        self.assertEqual(data["mode"], "combined")
        self.assertEqual(len(data["mcqs"]), 20)
        counts = Counter(q.get("answer_mode") for q in data["mcqs"])
        self.assertEqual(counts.get("single"), 10)
        self.assertEqual(counts.get("multiple"), 10)
        self.assertEqual(data.get("total_questions"), 20)
        components = {comp["mode"]: comp for comp in data.get("components", [])}
        self.assertIn("single", components)
        self.assertIn("multiple", components)
        self.assertEqual(components["single"].get("count"), 10)
        self.assertEqual(components["multiple"].get("count"), 10)

    def test_upload_combined_mode_renders_summary(self):
        payload_single = self._fake_mcq_set(10, options=4, correct_per_question=1)
        payload_multi = self._fake_mcq_set(10, options=7, correct_per_question=3, include_correct_index=False)
        responses = [
            SimpleNamespace(choices=[SimpleNamespace(message={"content": payload_single})]),
            SimpleNamespace(choices=[SimpleNamespace(message={"content": payload_multi})]),
        ]
        with patch.object(sys.modules[__name__], "_chat_completion") as mocked:
            mocked.side_effect = responses
            data = {
                "file": (io.BytesIO(b"Topic: combined"), "topic.txt"),
                "language": "ru",
                "mode": "combined",
            }
            r = self.client.post("/upload", data=data, content_type='multipart/form-data')
        self.assertEqual(r.status_code, 200)
        html = r.get_data(as_text=True)
        combo_label = "комбинированный"
        total_phrase = "20 вопросов"
        format_phrase = "Формат ответа"
        self.assertIn(combo_label, html)
        self.assertIn(total_phrase, html)
        self.assertIn(format_phrase, html)



    def test_upload_pdf_triggers_pdf_loader(self):
        with patch.object(sys.modules[__name__], "_read_pdf_document") as pdf_mock:
            with patch.object(sys.modules[__name__], "_chat_completion") as chat_mock:
                pdf_mock.return_value = "Topic from PDF"
                chat_mock.return_value = SimpleNamespace(
                    choices=[SimpleNamespace(message={"content": self._fake_mcq_set(10, 4)})]
                )
                data = {"file": (io.BytesIO(b"pdf-bytes"), "topic.pdf"), "language": "ru"}
                r = self.client.post("/upload", data=data, content_type='multipart/form-data')
        self.assertEqual(r.status_code, 200)
        pdf_mock.assert_called_once()

    def test_upload_pdf_passes_page_range(self):
        with patch.object(sys.modules[__name__], "_read_pdf_document") as pdf_mock:
            with patch.object(sys.modules[__name__], "_chat_completion") as chat_mock:
                pdf_mock.return_value = "Topic from selected PDF pages"
                chat_mock.return_value = SimpleNamespace(
                    choices=[SimpleNamespace(message={"content": self._fake_mcq_set(10, 4)})]
                )
                data = {
                    "file": (io.BytesIO(b"pdf-bytes"), "topic.pdf"),
                    "language": "ru",
                    "page_start": "2",
                    "page_end": "4",
                }
                r = self.client.post("/upload", data=data, content_type='multipart/form-data')
        self.assertEqual(r.status_code, 200)
        pdf_mock.assert_called_once()
        _, kwargs = pdf_mock.call_args
        self.assertEqual(kwargs.get("page_start"), 2)
        self.assertEqual(kwargs.get("page_end"), 4)

    def test_upload_rejects_invalid_page_range(self):
        with patch.object(sys.modules[__name__], "_read_pdf_document") as pdf_mock:
            with patch.object(sys.modules[__name__], "_chat_completion") as chat_mock:
                data = {
                    "file": (io.BytesIO(b"pdf-bytes"), "topic.pdf"),
                    "language": "ru",
                    "page_start": "5",
                    "page_end": "2",
                }
                r = self.client.post("/upload", data=data, content_type='multipart/form-data')
        self.assertEqual(r.status_code, 200)
        self.assertIn(CUSTOM_ERROR_MESSAGES["page_range_invalid"], r.get_data(as_text=True))
        pdf_mock.assert_not_called()
        chat_mock.assert_not_called()

    def test_upload_docx_triggers_office_loader(self):
        with patch.object(sys.modules[__name__], "_read_office_document") as office_mock:
            with patch.object(sys.modules[__name__], "_chat_completion") as chat_mock:
                office_mock.return_value = "Topic from Word"
                chat_mock.return_value = SimpleNamespace(
                    choices=[SimpleNamespace(message={"content": self._fake_mcq_set(10, 4)})]
                )
                data = {"file": (io.BytesIO(b"docx-bytes"), "topic.docx"), "language": "ru"}
                r = self.client.post("/upload", data=data, content_type='multipart/form-data')
                self.assertEqual(r.status_code, 200)
                office_mock.assert_called_once()
                args, _ = office_mock.call_args
                self.assertEqual(args[1], ".docx")

    def test_upload_doc_triggers_office_loader(self):
        with patch.object(sys.modules[__name__], "_read_office_document") as office_mock:
            with patch.object(sys.modules[__name__], "_chat_completion") as chat_mock:
                office_mock.return_value = "Topic from Word"
                chat_mock.return_value = SimpleNamespace(
                    choices=[SimpleNamespace(message={"content": self._fake_mcq_set(10, 4)})]
                )
                data = {"file": (io.BytesIO(b"doc-bytes"), "topic.doc"), "language": "ru"}
                r = self.client.post("/upload", data=data, content_type='multipart/form-data')
                self.assertEqual(r.status_code, 200)
                office_mock.assert_called_once()
                args, _ = office_mock.call_args
                self.assertEqual(args[1], ".doc")

    def test_validation_enforces_exact_counts(self):
        # 9 questions should raise the generation count error used by the upload handler
        with patch.object(sys.modules[__name__], "_chat_completion") as mocked:
            mocked.return_value = SimpleNamespace(
                choices=[SimpleNamespace(message={"content": self._fake_mcq_set(9, 4)})]
            )
            with self.assertRaises(RuntimeError):
                generate_mcqs_from_text("topic")

        # 5 options should raise assertion
        with patch.object(sys.modules[__name__], "_chat_completion") as mocked:
            mocked.return_value = SimpleNamespace(
                choices=[SimpleNamespace(message={"content": self._fake_mcq_set(10, 5)})]
            )
            with self.assertRaises(AssertionError):
                generate_mcqs_from_text("topic")

        # Multiple-answer mode should enforce 7 options
        with patch.object(sys.modules[__name__], "_chat_completion") as mocked:
            mocked.return_value = SimpleNamespace(
                choices=[SimpleNamespace(message={"content": self._fake_mcq_set(10, options=6, correct_per_question=2, include_correct_index=False)})]
            )
            with self.assertRaises(AssertionError):
                generate_mcqs_from_text("topic", mode="multiple")

        # Multiple-answer mode should enforce max 3 correct options
        with patch.object(sys.modules[__name__], "_chat_completion") as mocked:
            mocked.return_value = SimpleNamespace(
                choices=[SimpleNamespace(message={"content": self._fake_mcq_set(10, options=7, correct_per_question=4, include_correct_index=False)})]
            )
            data = generate_mcqs_from_text("topic", mode="multiple")
            for item in data["mcqs"]:
                self.assertLessEqual(len(item["correct_indices"]), 3)
                self.assertGreaterEqual(len(item["correct_indices"]), 1)

    # EXTRA TESTS
    def test_health_endpoint(self):
        r = self.client.get("/healthz")
        self.assertEqual(r.status_code, 200)
        self.assertIn("ok", r.get_data(as_text=True))

    def test_upload_cp1251_source(self):
        # Simulate Cyrillic source encoded in cp1251
        raw = "Тема: Списки и кортежи".encode('cp1251')
        with patch.object(sys.modules[__name__], "_chat_completion") as mocked:
            mocked.return_value = SimpleNamespace(
                choices=[SimpleNamespace(message={"content": self._fake_mcq_set(10, 4)})]
            )
            data = {"file": (io.BytesIO(raw), "topic.txt"), "language": "ru"}
            r = self.client.post("/upload", data=data, content_type='multipart/form-data')
            self.assertEqual(r.status_code, 200)
            self.assertIn("Готово! Вопросы готовы", r.get_data(as_text=True))

    def test_upload_invalid_json_from_model_shows_error(self):
        with patch.object(sys.modules[__name__], "_chat_completion") as mocked:
            mocked.return_value = SimpleNamespace(
                choices=[SimpleNamespace(message={"content": "not-json"})]
            )
            data = {"file": (io.BytesIO(b"Any topic"), "topic.txt"), "language": "ru"}
            r = self.client.post("/upload", data=data, content_type='multipart/form-data')
            self.assertEqual(r.status_code, 200)
            self.assertIn("Ошибка", r.get_data(as_text=True))

    def test_download_docx_student_version_excludes_answers(self):
        payload = json.loads(self._fake_mcq_set(2, 4))
        payload["total_questions"] = 2
        payload["mode"] = "single"
        r = self.client.post("/download/docx", json=payload)
        self.assertEqual(r.status_code, 200)
        self.assertIn(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            r.headers.get("Content-Type", ""),
        )
        from docx import Document
        document = Document(io.BytesIO(r.data))
        text = "\n".join(paragraph.text for paragraph in document.paragraphs)
        self.assertIn(PROJECT_TITLE, text)
        self.assertIn("Тест для студентов", text)
        self.assertIn("Q1: Sample?", text)
        self.assertIn("A. A00", text)
        self.assertNotIn("Правильный вариант", text)
        self.assertNotIn("because", text)

    def test_download_pdf_student_version(self):
        payload = json.loads(self._fake_mcq_set(2, 4))
        payload["total_questions"] = 2
        payload["mode"] = "single"
        r = self.client.post("/download/pdf", json=payload)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers.get("Content-Type"), "application/pdf")
        self.assertTrue(r.data.startswith(b"%PDF"))


def _run_tests():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(_Tests)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true", help="run unit tests and exit")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    if args.self_test:
        raise SystemExit(_run_tests())
    else:
        app.run(debug=True, port=args.port)

# python mcq_generator_app.py --port 5000
# cloudflared tunnel --url http://127.0.0.1:5000
