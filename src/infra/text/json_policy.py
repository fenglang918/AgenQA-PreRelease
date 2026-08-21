"""Model-aware JSON cleaning/parsing helpers.

Centralizes JSON cleanup so all roles using the same model get consistent
sanitization + optional LLM cleanup behavior.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from typing import Any, Dict, Iterable, Optional

from infra.text.json_llm_cleaner import maybe_clean_json_with_llm
from infra.text.json_sanitize import sanitize_json_text
from infra.text.fenced_blocks import extract_fenced_blocks

logger = logging.getLogger(__name__)


def _extract_first_json_value(text: str) -> tuple[str, Any] | None:
    """Extract the first JSON value (object/array) substring from free-form text.

    Uses json.JSONDecoder.raw_decode so braces inside strings won't break extraction.
    Returns (json_substring, parsed_value) or None.
    """
    if not isinstance(text, str) or not text:
        return None
    decoder = json.JSONDecoder()
    # Scan for possible JSON starts. Most outputs are short; this is fine.
    for m in re.finditer(r"[\{\[]", text):
        start = m.start()
        try:
            val, end = decoder.raw_decode(text[start:])
        except Exception:
            continue
        return text[start : start + end], val
    return None


def _normalize_quotes(s: str) -> str:
    return (
        s.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )


def _remove_trailing_commas(s: str) -> str:
    return re.sub(r",(\s*[}\]])", r"\1", s)


def _iter_json_candidates(text: str) -> list[str]:
    raw = text.strip()
    if not raw:
        return []
    candidates: list[str] = []
    # Prefer line-based fenced extraction so embedded ``` inside JSON strings
    # (e.g., Markdown code blocks in a field) won't truncate the payload.
    blocks = extract_fenced_blocks(raw)
    if blocks:
        for _lang, inner in blocks:
            inner = (inner or "").strip()
            if not inner:
                continue
            extracted = _extract_first_json_value(inner)
            candidates.append(extracted[0] if extracted else inner)
    else:
        # Fallback for non-standard fencing that isn't on its own line.
        for m in re.finditer(r"```(?:json)?\s*(.*?)```", raw, flags=re.IGNORECASE | re.DOTALL):
            inner = (m.group(1) or "").strip()
            if not inner:
                continue
            extracted = _extract_first_json_value(inner)
            candidates.append(extracted[0] if extracted else inner)
    extracted_raw = _extract_first_json_value(raw)
    if extracted_raw:
        candidates.append(extracted_raw[0])
    candidates.append(raw)
    # Deduplicate while preserving order.
    seen: set[str] = set()
    uniq: list[str] = []
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        uniq.append(cand)
    return uniq


def _keys_satisfy_required(obj: Dict[str, Any], required_keys: Optional[Iterable[Any]]) -> bool:
    if not required_keys:
        return True
    keys_lower = {str(k).lower(): k for k in obj.keys()}
    for req in required_keys:
        if isinstance(req, (list, tuple, set)):
            ok = False
            for opt in req:
                if str(opt).lower() in keys_lower:
                    ok = True
                    break
            if not ok:
                return False
        else:
            if str(req).lower() not in keys_lower:
                return False
    return True


def _flatten_required_keys(required_keys: Optional[Iterable[Any]]) -> list[str]:
    if not required_keys:
        return []
    out: list[str] = []
    for req in required_keys:
        if isinstance(req, (list, tuple, set)):
            for opt in req:
                if isinstance(opt, str) and opt.strip():
                    out.append(opt.strip())
        elif isinstance(req, str) and req.strip():
            out.append(req.strip())
    return out


def _try_json_loads(s: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
        # Deterministic normalization: some models wrap the object in a singleton list.
        if isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], dict):
            logger.warning("Normalized singleton JSON list to object.")
            return obj[0]
        return None
    except Exception:
        return None


def _try_literal_eval(s: str) -> Optional[Dict[str, Any]]:
    try:
        patched = re.sub(r"\btrue\b", "True", s, flags=re.IGNORECASE)
        patched = re.sub(r"\bfalse\b", "False", patched, flags=re.IGNORECASE)
        patched = re.sub(r"\bnull\b", "None", patched, flags=re.IGNORECASE)
        obj = ast.literal_eval(patched)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def clean_json_text(
    text: str,
    *,
    generator: Dict[str, Any],
    task_name: str,
    lang: str = "zh",
    required_keys: Optional[Iterable[Any]] = None,
    prompt_body: Optional[str] = None,
    snapshot_dir: Optional[Any] = None,
    allow_python: bool = False,
) -> Optional[str]:
    """Return a valid JSON object string (or None if not recoverable).

    Steps:
    1) Try local repairs: quote normalization, trailing comma removal,
       JSON sanitization for control chars/escapes.
    2) If still invalid, optionally call the LLM cleaner (model-aware).
    """
    if not isinstance(text, str) or not text.strip():
        return None

    candidates = _iter_json_candidates(text)
    for cand in candidates:
        base = _normalize_quotes(cand.strip())
        variants = [base, _remove_trailing_commas(base)]
        for variant in variants:
            obj = _try_json_loads(variant)
            if obj is not None and _keys_satisfy_required(obj, required_keys):
                return json.dumps(obj, ensure_ascii=False)
            obj = _try_json_loads(sanitize_json_text(variant))
            if obj is not None and _keys_satisfy_required(obj, required_keys):
                return json.dumps(obj, ensure_ascii=False)
            if allow_python:
                obj = _try_literal_eval(variant)
                if obj is not None and _keys_satisfy_required(obj, required_keys):
                    return json.dumps(obj, ensure_ascii=False)

    # LLM cleaner fallback (model-aware, only if enabled).
    cleaned = maybe_clean_json_with_llm(
        generator=generator,
        lang=lang or "zh",
        task_name=task_name,
        raw_text=candidates[0] if candidates else text,
        required_keys=_flatten_required_keys(required_keys),
        prompt_body=prompt_body,
        snapshot_dir=snapshot_dir,
    )
    if not cleaned:
        return None
    # Final parse after cleaner.
    obj = _try_json_loads(cleaned) or _try_json_loads(sanitize_json_text(cleaned))
    if obj is None and allow_python:
        obj = _try_literal_eval(cleaned)
    if obj is None:
        return None
    if not _keys_satisfy_required(obj, required_keys):
        return None
    return json.dumps(obj, ensure_ascii=False)


__all__ = ["clean_json_text"]
