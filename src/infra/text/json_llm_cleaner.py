"""Optional LLM-based JSON cleaner (fallback).

Why:
- Some models output almost-JSON that fails parsing due to control characters,
  invalid escapes (LaTeX like \tau), or minor formatting issues.
- A second pass with a reliable model (e.g. Claude) can normalize the output
  without changing semantics.

This module is opt-in by default. For Gemini outputs, it will use qwen3-max
as the default cleaner if SCICLONE_JSON_CLEANER_MODEL is not set.
Disable default with:
  - SCICLONE_JSON_CLEANER_DISABLE_DEFAULT=1

Enable explicitly by setting:
  - SCICLONE_JSON_CLEANER_MODEL=claude_sonnet4_5
Optionally:
  - SCICLONE_JSON_CLEANER_MAX_TOKENS=2048
    (When unset, defaults to the source role's generation.max_tokens)
"""

from __future__ import annotations

import copy
import json
import logging
import os
from typing import Any, Dict, Iterable, Optional

from infra.llm.inference import resolve_inference
from infra.prompt.prompt_builder import build_messages_with_background

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(str(raw).strip())
    except Exception:
        return default


def _resolve_cleaner_max_tokens(generator: Dict[str, Any], default: int = 2048) -> int:
    """Resolve max_tokens for the cleaner.

    Priority:
    1) SCICLONE_JSON_CLEANER_MAX_TOKENS (explicit override)
    2) source generator.generation.max_tokens (sync with role)
    3) fallback default
    """
    raw_env = os.environ.get("SCICLONE_JSON_CLEANER_MAX_TOKENS")
    if isinstance(raw_env, str) and raw_env.strip():
        return _env_int("SCICLONE_JSON_CLEANER_MAX_TOKENS", default)
    gen = (generator or {}).get("generation") if isinstance(generator, dict) else None
    if isinstance(gen, dict):
        raw = gen.get("max_tokens")
        if raw is not None:
            try:
                return int(raw)
            except Exception:
                pass
    return default


def _is_gemini_model(model_name: Optional[str]) -> bool:
    if not model_name:
        return False
    return "gemini" in str(model_name).lower()


def _resolve_cleaner_model(generator: Dict[str, Any]) -> Optional[str]:
    explicit = (os.environ.get("SCICLONE_JSON_CLEANER_MODEL") or "").strip()
    if explicit:
        return explicit
    if os.environ.get("SCICLONE_JSON_CLEANER_DISABLE_DEFAULT", "").strip().lower() in {"1", "true", "yes", "on"}:
        return None
    src_model = generator.get("model_name") if isinstance(generator, dict) else None
    if _is_gemini_model(str(src_model) if src_model else None):
        return (os.environ.get("SCICLONE_JSON_CLEANER_GEMINI_MODEL") or "qwen3-max").strip()
    return None


def maybe_clean_json_with_llm(
    *,
    generator: Dict[str, Any],
    lang: str,
    task_name: str,
    raw_text: str,
    required_keys: Iterable[str],
    prompt_body: Optional[str] = None,
    extra_requirements: Optional[str] = None,
    snapshot_dir: Optional[Any] = None,
) -> Optional[str]:
    """Return cleaned JSON text, or None if disabled/failed.

    This is meant as a *fallback* when local sanitization is insufficient.
    """
    model = _resolve_cleaner_model(generator)
    if not model:
        return None
    if not isinstance(raw_text, str) or not raw_text.strip():
        return None

    cleaner_gen = copy.deepcopy(generator) if isinstance(generator, dict) else {}
    cleaner_gen["model_name"] = model
    # Make the cleaner deterministic and simple.
    cleaner_gen.setdefault("client", {})
    if isinstance(cleaner_gen.get("client"), dict):
        cleaner_gen["client"]["stream"] = False
    cleaner_gen.setdefault("generation", {})
    if isinstance(cleaner_gen.get("generation"), dict):
        cleaner_gen["generation"]["temperature"] = 0.0
        cleaner_gen["generation"]["max_tokens"] = _resolve_cleaner_max_tokens(generator, 2048)

    resolved = resolve_inference(cleaner_gen)
    session = resolved.session
    chat_args = dict(resolved.chat_args)

    keys = [k for k in required_keys if isinstance(k, str) and k.strip()]
    keys_str = ", ".join(keys)
    extra = extra_requirements.strip() if isinstance(extra_requirements, str) and extra_requirements.strip() else ""

    cleaner_prompt = (
        f"You are a strict JSON cleaner for task={task_name}.\n"
        "Input below is intended to be ONE JSON object, but may contain invalid escapes/control characters.\n\n"
        "Output rules (VERY IMPORTANT):\n"
        "- Output ONLY the cleaned JSON object text. No code fences, no commentary.\n"
        "- Preserve the original meaning and wording as much as possible.\n"
        "- Fix JSON validity issues: escape literal newlines/tabs inside strings; fix invalid backslash escapes.\n"
        "- LaTeX: if you see sequences like \\tau or \\frac written as \\t / \\f etc due to bad escaping, "
        "ensure the JSON source uses double backslashes (e.g. \\\\tau, \\\\frac).\n"
        f"- The JSON object MUST contain these keys: {keys_str}.\n"
        "- Do NOT add any other keys.\n"
    )
    if extra:
        cleaner_prompt += f"\nAdditional requirements:\n{extra}\n"
    if isinstance(prompt_body, str) and prompt_body.strip():
        cleaner_prompt += "\nOriginal task prompt (for topic grounding, do not change topic):\n<task_prompt>\n"
        cleaner_prompt += prompt_body
        cleaner_prompt += "\n</task_prompt>\n"

    cleaner_prompt += "\nBroken JSON to clean:\n<broken_json>\n"
    cleaner_prompt += raw_text
    cleaner_prompt += "\n</broken_json>\n"
    cleaner_prompt += "\nIf the input is clearly truncated/incomplete and cannot be repaired without inventing content, output exactly:\nINCOMPLETE\n"

    messages = build_messages_with_background(cleaner_prompt, lang=lang or "zh")
    resp = session.chat(messages, **chat_args)
    text = session.extract_text(resp, default="").strip()

    if snapshot_dir is not None:
        try:
            from pathlib import Path

            snap = Path(snapshot_dir)
            snap.mkdir(parents=True, exist_ok=True)
            (snap / f"raw_response_cleaner.{task_name}.json").write_text(
                json.dumps(resp, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (snap / f"raw_response_cleaner.{task_name}.txt").write_text(text, encoding="utf-8")
        except Exception:
            pass

    if text.strip() == "INCOMPLETE":
        return None
    if not text:
        return None
    return text
