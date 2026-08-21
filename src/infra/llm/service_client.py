"""LLM service client helpers.

Standalone HTTP clients only（不再依赖外部 llm_service 仓库）。
内置 OpenAI 兼容 chat/completions 与 responses API 的简易实现，支持流式聚合。
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, List

from .service_loader import load_llm_service_full_config


logger = logging.getLogger(__name__)

PROXY_ENV_VARS = (
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
)

SESSION_ID_HEADER = "x-idealab-session-id"
SESSION_ID_ENV_VARS = ("IDEALAB_SESSION_ID", "SCICLONE_IDEALAB_SESSION_ID", "SCICLONE_SESSION_ID")


def _first_nonempty_text(payload: Any) -> str:
    """Extract the first non-empty text block from OpenAI-compatible responses."""
    if not isinstance(payload, dict):
        return ""
    allow_reasoning = _env_flag("SCICLONE_ALLOW_REASONING_AS_TEXT", default=False)
    # Chat/Completions style
    try:
        choices = payload.get("choices") or []
        if isinstance(choices, list) and choices:
            choice0 = choices[0] or {}
            msg = choice0.get("message") or {}
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
            reasoning = msg.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning.strip():
                # Some gateways/models (e.g. thinking variants) may place all visible text in reasoning_content
                # while leaving content empty. Treat it as a fallback even if SCICLONE_ALLOW_REASONING_AS_TEXT is off.
                if allow_reasoning or not (isinstance(content, str) and content.strip()):
                    return reasoning.strip()
            text = choice0.get("text")
            if isinstance(text, str) and text.strip():
                return text.strip()
    except Exception:
        pass

    # Responses API style
    try:
        outputs = payload.get("output") or payload.get("outputs") or []
        if isinstance(outputs, list) and outputs:
            # Reasoning models may emit a reasoning item before the assistant
            # message. Inspect every output item instead of assuming the first
            # item owns visible text.
            for output_item in outputs:
                if not isinstance(output_item, dict):
                    continue
                content_items = output_item.get("content") or []
                if isinstance(content_items, list):
                    for item in content_items:
                        if not isinstance(item, dict):
                            continue
                        if item.get("type") in {"text", "output_text"}:
                            val = item.get("text") or item.get("value") or item.get("text_block")
                            if isinstance(val, str) and val.strip():
                                return val.strip()
                        if isinstance(item.get("text"), dict):
                            raw_val = item["text"].get("value")
                            if isinstance(raw_val, str) and raw_val.strip():
                                return raw_val.strip()
    except Exception:
        pass
    return ""


def _extract_error_message(payload: Any) -> str:
    """Best-effort extraction of an error message from gateway/non-OpenAI payloads."""
    if not isinstance(payload, dict):
        return ""

    err = payload.get("error")
    if isinstance(err, str) and err.strip():
        return err.strip()
    if isinstance(err, dict):
        for k in ("message", "error", "detail", "msg"):
            v = err.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        for v in err.values():
            if isinstance(v, str) and v.strip():
                return v.strip()

    for k in ("message", "detail", "text", "error_message"):
        v = payload.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _looks_incomplete_structured_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.startswith("```"):
        return stripped.count("```") < 2
    if stripped.startswith("{"):
        return not stripped.endswith("}")
    if stripped.startswith("["):
        return not stripped.endswith("]")
    return False


def _should_retry_short_output(
    text: str,
    *,
    min_chars: int,
    response_mime_type: Optional[str],
) -> bool:
    if _looks_incomplete_structured_text(text):
        return True
    stripped = text.strip()
    if len(stripped) >= min_chars:
        return False
    mime = (response_mime_type or "").strip().lower()
    if mime.startswith("application/json"):
        return True
    lead = stripped.lstrip()
    if lead.startswith("{") or lead.startswith("[") or lead.startswith("```"):
        return True
    return False


def _normalize_api_base(raw_base: str) -> str:
    base = raw_base.strip().rstrip("/")
    if not base:
        raise ValueError("API base URL must not be empty")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return base


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    val = str(raw).strip().lower()
    return val not in {"", "0", "false", "off", "no"}


def _get_idealab_session_id() -> Optional[str]:
    for name in SESSION_ID_ENV_VARS:
        val = os.environ.get(name)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def _should_attach_session_id(base_url: str | None) -> bool:
    if _env_flag("SCICLONE_SESSION_HEADER_ALWAYS", default=False):
        return True
    if not base_url:
        return False
    return "idealab" in str(base_url).lower()


def _maybe_add_idealab_session_id(headers: Dict[str, str], base_url: str | None) -> None:
    if not isinstance(headers, dict):
        return
    if not _should_attach_session_id(base_url):
        return
    existing = {k.lower() for k in headers.keys() if isinstance(k, str)}
    if SESSION_ID_HEADER in existing:
        return
    session_id = _get_idealab_session_id()
    if session_id:
        headers[SESSION_ID_HEADER] = session_id


def _decode_sse_line(raw_line: Any, response: Any) -> str:
    """Decode one SSE line robustly for both AIMux and Idealab style gateways."""
    if raw_line is None:
        return ""
    if isinstance(raw_line, str):
        return raw_line.strip()

    resp_encoding = getattr(response, "encoding", None)
    candidates: List[str] = ["utf-8"]
    if isinstance(resp_encoding, str) and resp_encoding.strip():
        normalized = resp_encoding.strip().lower()
        if normalized not in candidates:
            candidates.append(normalized)

    for encoding in candidates:
        try:
            return raw_line.decode(encoding).strip()
        except Exception:
            continue

    return raw_line.decode("utf-8", errors="replace").strip()


def _is_gemini_model(model_name: Optional[str]) -> bool:
    if not model_name:
        return False
    return "gemini" in str(model_name).lower()


def _is_claude_model(model_name: Optional[str]) -> bool:
    if not model_name:
        return False
    return "claude" in str(model_name).lower()


def _derive_idealab_vertex_base(openai_v1_base: str) -> Optional[str]:
    """Derive Idealab Vertex-compatible base URL from an OpenAI-compat base URL."""
    base = (openai_v1_base or "").strip().rstrip("/")
    if not base:
        return None
    # Prefer explicit env override.
    env_base = os.environ.get("IDEALAB_VERTEX_BASE_URL") or os.environ.get("IDEALAB_VERTEX_BASE")
    if isinstance(env_base, str) and env_base.strip():
        return env_base.strip().rstrip("/")

    # Common Idealab mapping:
    # https://idealab.alibaba-inc.com/api/openai/v1 -> https://idealab.alibaba-inc.com/api/vertex/v1beta
    if "/api/openai" not in base:
        return None
    if base.endswith("/api/openai/v1"):
        return base[: -len("/api/openai/v1")] + "/api/vertex/v1beta"
    if base.endswith("/api/openai"):
        return base[: -len("/api/openai")] + "/api/vertex/v1beta"
    # Fallback: replace the segment if present.
    return base.replace("/api/openai/v1", "/api/vertex/v1beta").replace("/api/openai", "/api/vertex/v1beta")


def _normalize_anthropic_base(raw_base: str) -> str:
    base = (raw_base or "").strip().rstrip("/")
    if not base:
        raise ValueError("Anthropic base URL must not be empty")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return base


def _derive_anthropic_base(openai_v1_base: str) -> Optional[str]:
    """Derive Anthropic Messages base URL (ending with /v1) from an OpenAI-compat base URL."""
    env_base = (
        os.environ.get("ANTHROPIC_BASE_URL")
        or os.environ.get("IDEALAB_ANTHROPIC_BASE_URL")
        or os.environ.get("IDEALAB_ANTHROPIC_BASE")
    )
    if isinstance(env_base, str) and env_base.strip():
        try:
            return _normalize_anthropic_base(env_base)
        except Exception:
            return env_base.strip().rstrip("/")

    base = (openai_v1_base or "").strip().rstrip("/")
    if not base:
        return None

    base_lc = base.lower()
    # AiArena mapping: /api/openai/v1 -> /api/claude/v1
    if "aiarena.alibaba-inc.com" in base_lc:
        if "/api/openai" in base_lc:
            prefix = base[: base_lc.find("/api/openai")]
            return _normalize_anthropic_base(f"{prefix}/api/claude")
        # Already a claude base
        if "/api/claude" in base_lc:
            return _normalize_anthropic_base(base)

    # Idealab mapping: /api/openai/v1 -> /api/code
    if "idealab.alibaba-inc.com" in base_lc and "/api/openai" in base_lc:
        prefix = base[: base_lc.find("/api/openai")]
        return _normalize_anthropic_base(f"{prefix}/api/code")

    return None


def _resolve_anthropic_token(fallback_api_key: Optional[str]) -> str:
    for name in ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY"):
        val = os.environ.get(name)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return (fallback_api_key or "").strip()


def _openai_messages_to_anthropic(messages: Any) -> tuple[str | None, List[Dict[str, Any]]]:
    """Convert OpenAI chat messages into Anthropic Messages format."""
    if isinstance(messages, str):
        return None, [{"role": "user", "content": [{"type": "text", "text": messages}]}]
    if not isinstance(messages, list):
        return None, [{"role": "user", "content": [{"type": "text", "text": str(messages)}]}]

    system_parts: list[str] = []
    out: List[Dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            out.append({"role": "user", "content": [{"type": "text", "text": str(msg)}]})
            continue
        role = str(msg.get("role") or "user").strip().lower()
        content = msg.get("content")

        def _content_to_anthropic_blocks(val: Any) -> List[Dict[str, Any]]:
            if isinstance(val, str):
                return [{"type": "text", "text": val}]
            if not isinstance(val, list):
                if val is None:
                    return [{"type": "text", "text": ""}]
                return [{"type": "text", "text": str(val)}]

            out_blocks: List[Dict[str, Any]] = []
            for it in val:
                if isinstance(it, str):
                    if it:
                        out_blocks.append({"type": "text", "text": it})
                    continue
                if not isinstance(it, dict):
                    out_blocks.append({"type": "text", "text": str(it)})
                    continue

                item_type = str(it.get("type") or "").strip().lower()
                text = it.get("text")
                if isinstance(text, str):
                    out_blocks.append({"type": "text", "text": text})
                    continue

                if item_type in {"input_file", "file", "document"}:
                    mime = str(
                        it.get("mime_type") or it.get("mime") or it.get("media_type") or "application/octet-stream"
                    ).strip() or "application/octet-stream"
                    data = it.get("data") or it.get("base64")
                    if isinstance(data, str) and data.strip():
                        out_blocks.append(
                            {
                                "type": "document",
                                "source": {
                                    "type": "base64",
                                    "media_type": mime,
                                    "data": data.strip(),
                                },
                            }
                        )
                        continue

                # Fallback: preserve whatever textual residue we can extract.
                raw = it.get("content")
                if isinstance(raw, str) and raw.strip():
                    out_blocks.append({"type": "text", "text": raw})

            return out_blocks or [{"type": "text", "text": ""}]

        blocks = _content_to_anthropic_blocks(content)
        text = "\n".join(
            [str(block.get("text") or "") for block in blocks if isinstance(block, dict) and block.get("type") == "text"]
        ).strip()
        if role == "system":
            if text.strip():
                system_parts.append(text.strip())
            continue

        anth_role = "assistant" if role in {"assistant", "model"} else "user"
        out.append({"role": anth_role, "content": blocks})

    system = "\n\n".join(system_parts).strip() if system_parts else None
    if not out:
        out = [{"role": "user", "content": [{"type": "text", "text": ""}]}]
    return system, out


def _anthropic_messages_call(
    *,
    anthropic_base_v1: str,
    api_key: str,
    model: str,
    messages: Any,
    temperature: Optional[float],
    max_tokens: Optional[int],
    include_thinking: bool,
    expose_thinking: bool,
    thinking_budget_tokens: Optional[int],
    timeout_seconds: int,
    extra_headers: Optional[Dict[str, str]],
) -> Dict[str, Any]:
    """Call Anthropic Messages endpoint and normalize to OpenAI-style response."""
    try:
        import requests  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Anthropic Messages client requires 'requests'. Please install it: pip install requests"
        ) from exc

    base = _normalize_anthropic_base(anthropic_base_v1)
    url = f"{base}/messages"
    headers: Dict[str, str] = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "anthropic-version": "2023-06-01",
    }
    if isinstance(extra_headers, dict):
        for k, v in extra_headers.items():
            if not isinstance(k, str) or not isinstance(v, str):
                continue
            if k.lower() in {"authorization", "content-type"}:
                continue
            headers[k] = v
    _maybe_add_idealab_session_id(headers, base)

    system, msgs = _openai_messages_to_anthropic(messages)
    payload: Dict[str, Any] = {
        "model": model,
        "max_tokens": int(max_tokens) if isinstance(max_tokens, int) else 1024,
        "messages": msgs,
    }
    if system:
        payload["system"] = system
    if include_thinking:
        # Idealab/Claude extended-thinking constraints:
        # - thinking.budget_tokens must be >= 1024
        # - max_tokens must be > thinking.budget_tokens
        # - temperature must be exactly 1
        try:
            budget = int(thinking_budget_tokens) if isinstance(thinking_budget_tokens, int) else 1024
        except Exception:
            budget = 1024
        budget = max(1024, budget)
        payload["thinking"] = {"type": "enabled", "budget_tokens": budget}
        # Ensure max_tokens > budget_tokens (keep a small visible-text margin).
        try:
            cur_max = int(payload.get("max_tokens") or 0)
        except Exception:
            cur_max = 0
        payload["max_tokens"] = max(cur_max, budget + 128)
        payload["temperature"] = 1
    else:
        if isinstance(temperature, (int, float)):
            payload["temperature"] = float(temperature)

    def _post(hdrs: Dict[str, str]) -> tuple[int, Dict[str, Any]]:
        resp = requests.post(url, headers=hdrs, json=payload, timeout=timeout_seconds)
        try:
            data = resp.json()
        except Exception:  # noqa: BLE001
            data = {"status": resp.status_code, "text": resp.text}
        return resp.status_code, data

    status, data = _post(headers)
    # Some endpoints accept x-api-key instead of Bearer.
    if status in {401, 403} and isinstance(data, dict):
        alt_headers = dict(headers)
        alt_headers.pop("Authorization", None)
        alt_headers["x-api-key"] = api_key
        status2, data2 = _post(alt_headers)
        if status2 not in {401, 403}:
            status, data = status2, data2

    if status != 200:
        return {"status": status, "error": data, "_gateway": {"api": "anthropic_messages", "url": url}}

    text_parts: list[str] = []
    thinking_parts: list[str] = []
    try:
        content_items = data.get("content") if isinstance(data, dict) else None
        if isinstance(content_items, list):
            for it in content_items:
                if not isinstance(it, dict):
                    continue
                typ = it.get("type")
                if typ == "text" and isinstance(it.get("text"), str) and it["text"].strip():
                    text_parts.append(it["text"])
                if typ == "thinking":
                    raw = it.get("text") or it.get("thinking")
                    if isinstance(raw, str) and raw.strip():
                        thinking_parts.append(raw)
    except Exception:
        pass
    text = "\n".join(text_parts).strip()
    reasoning_text = "\n".join(thinking_parts).strip() if (include_thinking and expose_thinking) else ""

    usage = {}
    try:
        u = data.get("usage") if isinstance(data, dict) else None
        if isinstance(u, dict):
            pt = u.get("input_tokens")
            ct = u.get("output_tokens")
            if isinstance(pt, int):
                usage["prompt_tokens"] = pt
            if isinstance(ct, int):
                usage["completion_tokens"] = ct
            if usage.get("prompt_tokens") is not None and usage.get("completion_tokens") is not None:
                usage["total_tokens"] = int(usage["prompt_tokens"]) + int(usage["completion_tokens"])
    except Exception:
        pass

    stop_reason = None
    try:
        stop_reason = data.get("stop_reason") if isinstance(data, dict) else None
    except Exception:
        stop_reason = None

    message: Dict[str, Any] = {"role": "assistant", "content": text}
    if reasoning_text:
        message["reasoning_content"] = reasoning_text
    openai_like: Dict[str, Any] = {
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": stop_reason or "stop",
            }
        ],
        "_gateway": {
            "api": "anthropic_messages",
            "model": model,
            "stop_reason": stop_reason,
            "thinking_enabled": bool(include_thinking),
        },
    }
    if usage:
        openai_like["usage"] = usage
    return openai_like


def _openai_messages_to_gemini_contents(messages: Any) -> List[Dict[str, Any]]:
    """Convert OpenAI chat messages into Gemini 'contents' format."""
    if isinstance(messages, str):
        return [{"role": "user", "parts": [{"text": messages}]}]
    if not isinstance(messages, list):
        return [{"role": "user", "parts": [{"text": str(messages)}]}]

    contents: List[Dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            contents.append({"role": "user", "parts": [{"text": str(msg)}]})
            continue
        role = (msg.get("role") or "user").strip().lower()
        content = msg.get("content")
        # Gemini uses roles: user / model. We'll map assistant->model.
        gem_role = "model" if role in {"assistant", "model"} else "user"

        def _content_to_parts(val: Any) -> List[Dict[str, Any]]:
            if isinstance(val, str):
                return [{"text": val}]
            if val is None:
                return [{"text": ""}]
            if not isinstance(val, list):
                return [{"text": str(val)}]

            out_parts: List[Dict[str, Any]] = []
            for item in val:
                if isinstance(item, str):
                    if item:
                        out_parts.append({"text": item})
                    continue
                if not isinstance(item, dict):
                    out_parts.append({"text": str(item)})
                    continue

                typ = str(item.get("type") or "").strip().lower()
                if typ in {"text", "input_text", "output_text"}:
                    text_val = item.get("text")
                    if isinstance(text_val, str) and text_val:
                        out_parts.append({"text": text_val})
                    continue

                if typ in {"input_file", "file", "document"}:
                    mime = item.get("mime_type") or item.get("mime") or item.get("media_type") or "application/octet-stream"
                    data = item.get("data") or item.get("base64") or ""
                    if isinstance(mime, str) and isinstance(data, str) and mime.strip() and data.strip():
                        out_parts.append({"inline_data": {"mime_type": mime.strip(), "data": data.strip()}})
                    continue

                if typ == "image_url":
                    url_obj = item.get("image_url")
                    url = None
                    if isinstance(url_obj, dict):
                        url = url_obj.get("url")
                    if isinstance(url_obj, str):
                        url = url_obj
                    if isinstance(url, str) and url.startswith("data:") and ";base64," in url:
                        try:
                            header, b64 = url.split(";base64,", 1)
                            mime = header[len("data:") :].strip() or "image/png"
                            if b64.strip():
                                out_parts.append({"inline_data": {"mime_type": mime, "data": b64.strip()}})
                        except Exception:
                            pass
                    continue

                # Best-effort fallback: accept plain "text" field if present.
                if isinstance(item.get("text"), str) and str(item.get("text") or "").strip():
                    out_parts.append({"text": str(item.get("text") or "")})

            if not out_parts:
                out_parts = [{"text": ""}]
            return out_parts

        parts = _content_to_parts(content)
        if role == "system":
            # If a system message exists, inline it as a user instruction.
            gem_role = "user"
            sys_prefix = {"text": "SYSTEM:"}
            parts = [sys_prefix] + parts

        contents.append({"role": gem_role, "parts": parts})

    if not contents:
        contents = [{"role": "user", "parts": [{"text": ""}]}]
    return contents


def _messages_look_tagged_protocol(messages: Any) -> bool:
    """Heuristic: detect SciClone tagged protocol prompts (e.g. [Step]...[/Step])."""
    try:
        if not isinstance(messages, list) or not messages:
            return False
        last = messages[-1]
        if not isinstance(last, dict):
            return False
        content = last.get("content")
        if not isinstance(content, str) or not content:
            return False
        # Tagged prompts include these sentinel tags in the output format section.
        return (
            "[Step]" in content
            and "[/Step]" in content
            and "[Question]" in content
            and "[/Question]" in content
            and "[Answer]" in content
            and "[/Answer]" in content
        )
    except Exception:
        return False


def _estimate_messages_chars(messages: Any) -> int:
    """Best-effort estimate of the textual size of messages."""
    if messages is None:
        return 0
    if isinstance(messages, str):
        return len(messages)
    if isinstance(messages, list):
        total = 0
        for m in messages:
            if isinstance(m, str):
                total += len(m)
                continue
            if not isinstance(m, dict):
                total += len(str(m))
                continue
            c = m.get("content")
            if isinstance(c, str):
                total += len(c)
            elif isinstance(c, list):
                for item in c:
                    if isinstance(item, str):
                        total += len(item)
                    elif isinstance(item, dict) and isinstance(item.get("text"), str):
                        total += len(item["text"])
                    else:
                        total += len(str(item))
            else:
                total += len("" if c is None else str(c))
        return total
    return len(str(messages))


def _idealab_vertex_generate_content(
    *,
    vertex_base: str,
    api_key: str,
    model: str,
    messages: Any,
    temperature: Optional[float],
    max_output_tokens: Optional[int],
    timeout_seconds: int,
    extra_headers: Optional[Dict[str, str]],
    include_thoughts: bool,
    expose_thoughts: bool,
    response_mime_type: Optional[str],
) -> Dict[str, Any]:
    """Call Idealab Vertex-compatible Gemini endpoint and normalize to OpenAI-style response."""
    try:
        import requests  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Gemini native (vertex) client requires 'requests'. Please install it: pip install requests"
        ) from exc

    url = f"{vertex_base.rstrip('/')}/models/{model}:generateContent"
    headers: Dict[str, str] = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    if isinstance(extra_headers, dict):
        for k, v in extra_headers.items():
            if not isinstance(k, str) or not isinstance(v, str):
                continue
            # Never allow overriding auth/content-type here.
            if k.lower() in {"authorization", "content-type"}:
                continue
            headers[k] = v
    _maybe_add_idealab_session_id(headers, vertex_base)

    generation_cfg: Dict[str, Any] = {"responseModalities": ["TEXT"]}
    if isinstance(response_mime_type, str) and response_mime_type.strip():
        generation_cfg["responseMimeType"] = response_mime_type.strip()
    if isinstance(temperature, (int, float)):
        generation_cfg["temperature"] = float(temperature)
    if isinstance(max_output_tokens, int):
        generation_cfg["maxOutputTokens"] = max_output_tokens
    mime_lc = str(response_mime_type or "").strip().lower()
    include_thoughts_with_json = _env_flag("SCICLONE_GEMINI_VERTEX_INCLUDE_THOUGHTS_WITH_JSON", default=False)
    use_thoughts = bool(include_thoughts) and (include_thoughts_with_json or not mime_lc.startswith("application/json"))
    if use_thoughts:
        generation_cfg["thinkingConfig"] = {"includeThoughts": True}

    payload: Dict[str, Any] = {
        "contents": _openai_messages_to_gemini_contents(messages),
        "generationConfig": generation_cfg,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout_seconds)
    try:
        data = resp.json()
    except Exception:  # noqa: BLE001
        data = {"status": resp.status_code, "text": resp.text}

    if resp.status_code != 200:
        return {"status": resp.status_code, "error": data, "_gateway": {"api": "idealab_vertex", "url": url}}

    # Extract text from candidates/parts.
    text_parts: list[str] = []
    thought_parts: list[str] = []
    finish_reason: Optional[str] = None
    try:
        candidates = data.get("candidates") or []
        if candidates and isinstance(candidates, list):
            c0 = candidates[0] or {}
            fr = c0.get("finishReason") or c0.get("finish_reason")
            if isinstance(fr, str) and fr.strip():
                finish_reason = fr.strip()
            content = c0.get("content") or {}
            parts = content.get("parts") or []
            if isinstance(parts, list):
                for p in parts:
                    if isinstance(p, dict) and isinstance(p.get("text"), str) and p["text"].strip():
                        if include_thoughts and p.get("thought") is True:
                            thought_parts.append(p["text"])
                        else:
                            text_parts.append(p["text"])
    except Exception:
        pass
    text = "\n".join(text_parts).strip()
    reasoning_text = "\n".join(thought_parts).strip() if expose_thoughts else ""

    # Map usage if present.
    usage = {}
    usage_raw = data.get("usageMetadata") or data.get("usage_metadata") or {}
    if isinstance(usage_raw, dict):
        pt = usage_raw.get("promptTokenCount")
        ct = usage_raw.get("candidatesTokenCount")
        tt = usage_raw.get("totalTokenCount")
        if isinstance(pt, int):
            usage["prompt_tokens"] = pt
        if isinstance(ct, int):
            usage["completion_tokens"] = ct
        if isinstance(tt, int):
            usage["total_tokens"] = tt

    message: Dict[str, Any] = {"role": "assistant", "content": text}
    if reasoning_text:
        message["reasoning_content"] = reasoning_text
    mapped_finish = "stop"
    if isinstance(finish_reason, str):
        fr_norm = finish_reason.strip().upper()
        if fr_norm in {"MAX_TOKENS", "LENGTH"}:
            mapped_finish = "length"
        elif fr_norm in {"SAFETY", "CONTENT_FILTER"}:
            mapped_finish = "content_filter"
    openai_like: Dict[str, Any] = {
        "choices": [{"index": 0, "message": message, "finish_reason": mapped_finish}],
        "_gateway": {"api": "idealab_vertex", "modelVersion": data.get("modelVersion"), "finish_reason_raw": finish_reason},
    }
    # Debug aid: when the gateway returns empty text, keep a compact copy of the raw response.
    try:
        if not text.strip() and not reasoning_text.strip() and _env_flag("SCICLONE_DEBUG_VERTEX_EMPTY_RESPONSE", default=True):
            openai_like["_gateway"]["raw"] = data
    except Exception:
        pass
    if usage:
        openai_like["usage"] = usage
    return openai_like


def _idealab_vertex_stream_generate_content(
    *,
    vertex_base: str,
    api_key: str,
    model: str,
    messages: Any,
    temperature: Optional[float],
    max_output_tokens: Optional[int],
    timeout_seconds: int,
    extra_headers: Optional[Dict[str, str]],
    include_thoughts: bool,
    expose_thoughts: bool,
    response_mime_type: Optional[str],
) -> Dict[str, Any]:
    """Call Idealab Vertex-compatible Gemini streaming endpoint and normalize to OpenAI-style response."""
    try:
        import requests  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Gemini native (vertex) client requires 'requests'. Please install it: pip install requests"
        ) from exc

    url = f"{vertex_base.rstrip('/')}/models/{model}:streamGenerateContent"
    headers: Dict[str, str] = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    if isinstance(extra_headers, dict):
        for k, v in extra_headers.items():
            if not isinstance(k, str) or not isinstance(v, str):
                continue
            if k.lower() in {"authorization", "content-type"}:
                continue
            headers[k] = v
    _maybe_add_idealab_session_id(headers, vertex_base)

    generation_cfg: Dict[str, Any] = {"responseModalities": ["TEXT"]}
    if isinstance(response_mime_type, str) and response_mime_type.strip():
        generation_cfg["responseMimeType"] = response_mime_type.strip()
    if isinstance(temperature, (int, float)):
        generation_cfg["temperature"] = float(temperature)
    if isinstance(max_output_tokens, int):
        generation_cfg["maxOutputTokens"] = max_output_tokens
    mime_lc = str(response_mime_type or "").strip().lower()
    include_thoughts_with_json = _env_flag("SCICLONE_GEMINI_VERTEX_INCLUDE_THOUGHTS_WITH_JSON", default=False)
    use_thoughts = bool(include_thoughts) and (include_thoughts_with_json or not mime_lc.startswith("application/json"))
    if use_thoughts:
        generation_cfg["thinkingConfig"] = {"includeThoughts": True}

    payload: Dict[str, Any] = {
        "contents": _openai_messages_to_gemini_contents(messages),
        "generationConfig": generation_cfg,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=timeout_seconds, stream=True)
    try:
        if resp.status_code != 200:
            try:
                err = resp.json()
            except Exception:  # noqa: BLE001
                err = {"status": resp.status_code, "text": resp.text}
            return {"status": resp.status_code, "error": err, "_gateway": {"api": "idealab_vertex", "url": url}}

        text_parts: list[str] = []
        thought_parts: list[str] = []
        usage: Dict[str, Any] = {}
        model_version: Optional[str] = None
        finish_reason: Optional[str] = None

        for raw_line in resp.iter_lines(decode_unicode=False):
            if not raw_line:
                continue
            line = _decode_sse_line(raw_line, resp)
            if not line:
                continue

            if line.startswith("data:"):
                data_str = line[len("data:") :].strip()
            else:
                data_str = line
            if not data_str or data_str == "[DONE]":
                continue

            try:
                chunk = json.loads(data_str)
            except Exception:
                continue
            if not isinstance(chunk, dict):
                continue

            if isinstance(chunk.get("modelVersion"), str):
                model_version = chunk.get("modelVersion")

            usage_raw = chunk.get("usageMetadata") or chunk.get("usage_metadata") or {}
            if isinstance(usage_raw, dict):
                pt = usage_raw.get("promptTokenCount")
                ct = usage_raw.get("candidatesTokenCount")
                tt = usage_raw.get("totalTokenCount")
                if isinstance(pt, int):
                    usage["prompt_tokens"] = pt
                if isinstance(ct, int):
                    usage["completion_tokens"] = ct
                if isinstance(tt, int):
                    usage["total_tokens"] = tt

            candidates = chunk.get("candidates") or []
            if not isinstance(candidates, list) or not candidates:
                continue
            c0 = candidates[0] or {}
            fr = c0.get("finishReason") or c0.get("finish_reason")
            if isinstance(fr, str) and fr.strip():
                finish_reason = fr.strip()

            content = c0.get("content") or {}
            parts = content.get("parts") or []
            if not isinstance(parts, list):
                continue
            for p in parts:
                if isinstance(p, dict) and isinstance(p.get("text"), str) and p["text"].strip():
                    if include_thoughts and p.get("thought") is True:
                        thought_parts.append(p["text"])
                    else:
                        text_parts.append(p["text"])

        text = "\n".join(text_parts).strip()
        reasoning_text = "\n".join(thought_parts).strip() if expose_thoughts else ""
        message: Dict[str, Any] = {"role": "assistant", "content": text}
        if reasoning_text:
            message["reasoning_content"] = reasoning_text

        mapped_finish = "stop"
        if isinstance(finish_reason, str):
            fr_norm = finish_reason.strip().upper()
            if fr_norm in {"MAX_TOKENS", "LENGTH"}:
                mapped_finish = "length"
            elif fr_norm in {"SAFETY", "CONTENT_FILTER"}:
                mapped_finish = "content_filter"

        out: Dict[str, Any] = {
            "choices": [{"index": 0, "message": message, "finish_reason": mapped_finish}],
            "_gateway": {"api": "idealab_vertex_stream", "modelVersion": model_version, "finish_reason_raw": finish_reason},
        }
        if usage:
            out["usage"] = usage
        return out
    finally:
        try:
            resp.close()
        except Exception:
            pass


def _expand_env(val: Any) -> Any:
    """Expand environment variables in a string value if present.

    Supports patterns like "$VAR" and "${VAR}".
    """
    if isinstance(val, str):
        try:
            return os.path.expandvars(val)
        except Exception:
            return val
    return val


def _clear_proxy_env_for_inference(service_id: str) -> None:
    """Optionally clear proxy environment variables for inference backends.

    Some environments require proxy variables (e.g. macOS system proxy / Clash) to
    reach remote endpoints; clearing them will break connectivity and may surface
    as SSL `OSError: [Errno 22] Invalid argument`.

    Set `SCICLONE_CLEAR_PROXY=1` to force clearing.
    """
    flag = os.environ.get("SCICLONE_CLEAR_PROXY", "").strip().lower()
    if not flag or flag in {"0", "false", "off", "no"}:
        return
    cleared = [name for name in PROXY_ENV_VARS if os.environ.pop(name, None) is not None]
    if cleared:
        service_type = "local" if service_id.startswith("local:") else "remote"
        logger.info(
            "Cleared proxy variables for %s inference backend (%s): %s",
            service_type,
            service_id,
            ", ".join(cleared)
        )


def _derive_service_id(api_base: str, provided_id: Optional[str], model_name: Optional[str]) -> str:
    if provided_id:
        return provided_id

    normalized = api_base.lower()
    prefix = "local" if ("127.0.0.1" in normalized or "localhost" in normalized) else "remote"

    suffix_source = (model_name or "configured").strip() or "configured"
    safe_suffix = suffix_source.replace(" ", "-").replace(":", "-")
    return f"{prefix}:{safe_suffix}"


def _infer_api_channel_from_base(base_url: str | None) -> str:
    base = str(base_url or "").strip().lower()
    if not base:
        return "custom"
    if "aimux.alibaba-inc.com" in base:
        return "aimux"
    if "idealab.alibaba-inc.com" in base:
        return "idealab"
    if "aiarena.alibaba-inc.com" in base:
        return "aiarena"
    if "dashscope.aliyuncs.com" in base or "bailian" in base:
        return "dashscope"
    if "35.220.164.252" in base or "123.129.219.111" in base:
        return "sjtu"
    return "custom"


def _normalize_api_channel(raw_channel: Any, *, base_url: str | None) -> str:
    if isinstance(raw_channel, str) and raw_channel.strip():
        return raw_channel.strip().lower()
    return _infer_api_channel_from_base(base_url)


@dataclass
class GatewayTarget:
    api_channel: str
    api_style: str
    api_base: str


def _gateway_rule_matches(
    rule: Dict[str, Any],
    *,
    api_channel: str,
    model_name: str,
    api_style: str,
) -> bool:
    match = rule.get("match")
    if not isinstance(match, dict):
        return False

    chan = match.get("api_channel")
    if isinstance(chan, str) and chan.strip():
        if api_channel != chan.strip().lower():
            return False
    elif isinstance(chan, list):
        allowed = {str(x).strip().lower() for x in chan if str(x).strip()}
        if allowed and api_channel not in allowed:
            return False

    style = match.get("api_style")
    if isinstance(style, str) and style.strip():
        if api_style != style.strip().lower():
            return False

    prefix = match.get("model_prefix")
    if isinstance(prefix, str) and prefix.strip():
        if not model_name.lower().startswith(prefix.strip().lower()):
            return False

    regex = match.get("model_regex")
    if isinstance(regex, str) and regex.strip():
        try:
            if re.search(regex, model_name, flags=re.IGNORECASE) is None:
                return False
        except re.error:
            return False

    return True


def _resolve_gateway_target(
    *,
    base_url: str,
    api_channel: str,
    api_style: str,
    model_name: str,
    gateway_routing: Optional[List[Dict[str, Any]]],
) -> GatewayTarget:
    normalized_channel = _normalize_api_channel(api_channel, base_url=base_url)
    normalized_style = str(api_style or "chat").strip().lower() or "chat"
    normalized_base = _normalize_api_base(base_url)
    model = str(model_name or "").strip()

    if isinstance(gateway_routing, list):
        for rule in gateway_routing:
            if not isinstance(rule, dict):
                continue
            if not _gateway_rule_matches(
                rule,
                api_channel=normalized_channel,
                model_name=model,
                api_style=normalized_style,
            ):
                continue
            target = rule.get("target")
            if not isinstance(target, dict):
                continue
            target_channel = _normalize_api_channel(target.get("api_channel"), base_url=base_url)
            target_style = str(target.get("api_style") or normalized_style).strip().lower() or normalized_style
            target_base = target.get("api_base")
            if isinstance(target_base, str) and target_base.strip():
                resolved_base = _normalize_api_base(target_base)
            elif target_style == "gemini_vertex":
                resolved_base = _derive_idealab_vertex_base(base_url) or normalized_base
            elif target_style == "anthropic_messages":
                resolved_base = _derive_anthropic_base(base_url) or normalized_base
            else:
                resolved_base = normalized_base
            return GatewayTarget(
                api_channel=target_channel,
                api_style=target_style,
                api_base=resolved_base,
            )

    if normalized_channel == "idealab" and _is_gemini_model(model):
        vertex_base = _derive_idealab_vertex_base(base_url)
        if vertex_base:
            return GatewayTarget(
                api_channel="idealab_vertex",
                api_style="gemini_vertex",
                api_base=vertex_base,
            )

    if normalized_channel in {"idealab", "aiarena"} and _is_claude_model(model):
        anthropic_base = _derive_anthropic_base(base_url)
        if anthropic_base:
            return GatewayTarget(
                api_channel=f"{normalized_channel}_messages",
                api_style="anthropic_messages",
                api_base=anthropic_base,
            )

    return GatewayTarget(
        api_channel=normalized_channel,
        api_style=normalized_style,
        api_base=normalized_base,
    )


@dataclass
class LLMServiceSession:
    """Lightweight wrapper around in-repo HTTP clients."""

    service_id: str
    model_name: str
    base_url: str
    api_key: Optional[str]
    default_headers: Dict[str, str]
    timeout: Optional[int]
    stream: bool
    extra_headers: Optional[Dict[str, str]]
    alt_base_urls: Optional[List[str]] = None
    api_channel: str = "custom"
    api_style: str = "chat"  # "chat" (default) or "responses"
    gateway_routing: Optional[List[Dict[str, Any]]] = None

    def __post_init__(self) -> None:
        """Initialize underlying client (always in-repo HTTP clients)."""

        normalized_base = _normalize_api_base(self.base_url)
        if self.api_style == "responses":
            self._client = _ResponsesApiClient(
                base_url=normalized_base,
                api_key=self.api_key,
                timeout=self.timeout,
                extra_headers=self.extra_headers,
                default_headers=self.default_headers,
            )
        else:
            self._client = _SimpleApiClient(
                base_url=normalized_base,
                api_key=self.api_key,
                timeout=self.timeout,
                stream=self.stream,
                extra_headers=self.extra_headers,
                default_headers=self.default_headers,
            )
        self._is_simple_client = True
        try:
            self.base_url = self._client.get_active_base_url()
        except Exception:
            self.base_url = normalized_base

        # 清理代理设置，避免经由 Clash 等代理访问内网/本地服务。
        _clear_proxy_env_for_inference(self.service_id)

    @property
    def client(self):
        return self._client

    def chat(self, messages: Any, **kwargs: Any) -> Dict[str, Any]:
        model = kwargs.get("model") or self.model_name
        gateway_target = _resolve_gateway_target(
            base_url=self.base_url,
            api_channel=self.api_channel,
            api_style=self.api_style,
            model_name=str(model or ""),
            gateway_routing=self.gateway_routing,
        )
        enable_vertex_fallback = _env_flag("SCICLONE_GEMINI_VERTEX_FALLBACK", default=True)
        is_gemini = _is_gemini_model(str(model) if model is not None else None)
        prefer_vertex = (
            _env_flag("SCICLONE_GEMINI_PREFER_VERTEX", default=True) if is_gemini else False
        )
        enable_claude_messages = _env_flag("SCICLONE_CLAUDE_MESSAGES_GATEWAY", default=True)
        is_claude = _is_claude_model(str(model) if model is not None else None)
        prefer_claude_messages = (
            _env_flag("SCICLONE_CLAUDE_PREFER_MESSAGES", default=True) if is_claude else False
        )

        def _make_direct_client(target: GatewayTarget):
            if target.api_style == "responses":
                return _ResponsesApiClient(
                    base_url=target.api_base,
                    api_key=self.api_key,
                    timeout=self.timeout,
                    extra_headers=self.extra_headers,
                    default_headers=self.default_headers,
                )
            return _SimpleApiClient(
                base_url=target.api_base,
                api_key=self.api_key,
                timeout=self.timeout,
                stream=self.stream,
                extra_headers=self.extra_headers,
                default_headers=self.default_headers,
            )

        direct_target = GatewayTarget(
            api_channel=_normalize_api_channel(self.api_channel, base_url=self.base_url),
            api_style=self.api_style,
            api_base=_normalize_api_base(self.base_url),
        )

        def _direct_chat(target: GatewayTarget, *, fallback_to_session: bool = False) -> Dict[str, Any]:
            use_session_client = (
                fallback_to_session
                or (
                    target.api_style == self.api_style
                    and _normalize_api_base(target.api_base) == _normalize_api_base(self.base_url)
                )
            )
            client = self._client if use_session_client else _make_direct_client(target)
            return client.chat(messages, **kwargs)

        def _try_vertex() -> Optional[Dict[str, Any]]:
            if not enable_vertex_fallback:
                return None
            if gateway_target.api_style != "gemini_vertex":
                return None
            if not isinstance(self.api_key, str) or not self.api_key.strip():
                return None
            vertex_base = gateway_target.api_base
            if not vertex_base:
                return None
            # Map OpenAI args -> Gemini generationConfig.
            temperature = kwargs.get("temperature")
            include_thoughts = _env_flag("SCICLONE_GEMINI_VERTEX_INCLUDE_THOUGHTS", default=False)
            expose_thoughts = _env_flag("SCICLONE_GEMINI_VERTEX_EXPOSE_THOUGHTS", default=False)
            response_mime_type = os.environ.get("SCICLONE_GEMINI_VERTEX_RESPONSE_MIME_TYPE")
            # If the prompt expects tagged protocol output, prefer plain text even if JSON mime type is set.
            try:
                if (
                    isinstance(response_mime_type, str)
                    and response_mime_type.strip().lower().startswith("application/json")
                    and _messages_look_tagged_protocol(messages)
                    and not _env_flag("SCICLONE_GEMINI_VERTEX_FORCE_JSON_MIME_TYPE", default=False)
                ):
                    response_mime_type = "text/plain"
            except Exception:
                pass
            max_output_tokens = kwargs.get("max_output_tokens")
            if max_output_tokens is None:
                # Best-effort mapping: OpenAI max_tokens -> Gemini maxOutputTokens.
                max_tokens = kwargs.get("max_tokens")
                if isinstance(max_tokens, int):
                    max_output_tokens = max_tokens
            # Gemini/Vertex often spends token budget on hidden thoughts; too-small maxOutputTokens can yield no text.
            # For large prompts, we use a larger minimum to ensure visible TEXT output.
            msg_chars = _estimate_messages_chars(messages)
            try:
                min_out_small = int(os.getenv("SCICLONE_GEMINI_VERTEX_MIN_OUTPUT_TOKENS", "200") or "200")
            except Exception:
                min_out_small = 200
            try:
                min_out_large = int(
                    os.getenv("SCICLONE_GEMINI_VERTEX_MIN_OUTPUT_TOKENS_LARGE_PROMPT", "4096") or "4096"
                )
            except Exception:
                min_out_large = 4096
            try:
                large_prompt_chars = int(os.getenv("SCICLONE_GEMINI_VERTEX_LARGE_PROMPT_CHARS", "2000") or "2000")
            except Exception:
                large_prompt_chars = 2000
            min_out = min_out_large if msg_chars >= large_prompt_chars else min_out_small

            if isinstance(max_output_tokens, int):
                max_output_tokens = max(min_out, max_output_tokens)
            else:
                max_output_tokens = min_out
            timeout_seconds = int(self.timeout) if isinstance(self.timeout, int) else 300
            try:
                vertex_attempts = max(1, int(os.getenv("SCICLONE_GEMINI_VERTEX_RETRY_TIMES", "2") or "2"))
            except Exception:
                vertex_attempts = 2
            try:
                vertex_delay = float(os.getenv("SCICLONE_GEMINI_VERTEX_RETRY_DELAY_SECONDS", "1.0") or "1.0")
            except Exception:
                vertex_delay = 1.0
            try:
                max_out_cap = int(os.getenv("SCICLONE_GEMINI_VERTEX_MAX_OUTPUT_TOKENS", "12288") or "12288")
            except Exception:
                max_out_cap = 12288
            short_retry_enabled = _env_flag("SCICLONE_GEMINI_VERTEX_SHORT_OUTPUT_RETRY", default=True)
            try:
                short_min_chars = int(os.getenv("SCICLONE_GEMINI_VERTEX_MIN_OUTPUT_CHARS", "120") or "120")
            except Exception:
                short_min_chars = 120
            force_text_on_short = _env_flag("SCICLONE_GEMINI_VERTEX_FORCE_TEXT_MIME_ON_SHORT", default=True)
            use_vertex_stream = _env_flag("SCICLONE_GEMINI_VERTEX_STREAM", default=False)
            try:
                current_max_out = max_output_tokens if isinstance(max_output_tokens, int) else min_out
                last_resp: Optional[Dict[str, Any]] = None
                delay = vertex_delay
                effective_mime = response_mime_type
                for attempt in range(1, vertex_attempts + 1):
                    # Try streaming endpoint first (if enabled) to avoid early-stop on long responses.
                    if use_vertex_stream:
                        last_resp = _idealab_vertex_stream_generate_content(
                            vertex_base=vertex_base,
                            api_key=self.api_key,
                            model=str(model),
                            messages=messages,
                            temperature=temperature if isinstance(temperature, (int, float)) else None,
                            max_output_tokens=int(current_max_out),
                            timeout_seconds=timeout_seconds,
                            extra_headers=self.extra_headers,
                            include_thoughts=include_thoughts,
                            expose_thoughts=expose_thoughts,
                            response_mime_type=effective_mime,
                        )
                        text = _first_nonempty_text(last_resp)
                        if text and short_retry_enabled and _should_retry_short_output(
                            text,
                            min_chars=short_min_chars,
                            response_mime_type=effective_mime,
                        ):
                            logger.warning(
                                "Gemini vertex(stream) response looks truncated/too short; retrying (len=%s, mime=%s, attempt=%s/%s)",
                                len(text.strip()),
                                (effective_mime or "").strip() or "default",
                                attempt,
                                vertex_attempts,
                            )
                            if (
                                force_text_on_short
                                and isinstance(effective_mime, str)
                                and effective_mime.strip().lower().startswith("application/json")
                            ):
                                effective_mime = "text/plain"
                            text = ""
                        if text:
                            return last_resp

                    # Non-streaming fallback (generateContent).
                    last_resp = _idealab_vertex_generate_content(
                        vertex_base=vertex_base,
                        api_key=self.api_key,
                        model=str(model),
                        messages=messages,
                        temperature=temperature if isinstance(temperature, (int, float)) else None,
                        max_output_tokens=int(current_max_out),
                        timeout_seconds=timeout_seconds,
                        extra_headers=self.extra_headers,
                        include_thoughts=include_thoughts,
                        expose_thoughts=expose_thoughts,
                        response_mime_type=effective_mime,
                    )
                    text = _first_nonempty_text(last_resp)
                    if text:
                        if short_retry_enabled and _should_retry_short_output(
                            text,
                            min_chars=short_min_chars,
                            response_mime_type=effective_mime,
                        ):
                            logger.warning(
                                "Gemini vertex response looks truncated/too short; retrying (len=%s, mime=%s, attempt=%s/%s)",
                                len(text.strip()),
                                (effective_mime or "").strip() or "default",
                                attempt,
                                vertex_attempts,
                            )
                            if (
                                force_text_on_short
                                and isinstance(effective_mime, str)
                                and effective_mime.strip().lower().startswith("application/json")
                            ):
                                effective_mime = "text/plain"
                            text = ""
                        else:
                            return last_resp
                    if attempt >= vertex_attempts:
                        break
                    current_max_out = min(max_out_cap, int(current_max_out) * 2)
                    try:
                        time.sleep(delay)
                    except Exception:
                        pass
                    delay *= 2.0
                return last_resp
            except Exception as exc:  # noqa: BLE001
                logger.warning("Gemini vertex fallback failed (model=%s, vertex_base=%s): %s", model, vertex_base, exc)
                return None

        def _try_anthropic() -> Optional[Dict[str, Any]]:
            if not enable_claude_messages:
                return None
            if gateway_target.api_style != "anthropic_messages":
                return None
            api_key = _resolve_anthropic_token(self.api_key)
            if not api_key:
                return None
            base = gateway_target.api_base
            if not base:
                return None
            temperature = kwargs.get("temperature")
            max_tokens = kwargs.get("max_tokens")
            if max_tokens is None:
                mot = kwargs.get("max_output_tokens")
                if isinstance(mot, int):
                    max_tokens = mot
            thinking_cfg = kwargs.get("thinking")
            include_thinking = False
            thinking_budget_tokens: Optional[int] = None
            if isinstance(thinking_cfg, dict):
                thinking_type = str(thinking_cfg.get("type") or "").strip().lower()
                include_thinking = thinking_type == "enabled"
                raw_budget_cfg = thinking_cfg.get("budget_tokens")
                if raw_budget_cfg is not None:
                    try:
                        thinking_budget_tokens = int(raw_budget_cfg)
                    except Exception:
                        thinking_budget_tokens = None
            if not include_thinking:
                include_thinking = _env_flag("SCICLONE_CLAUDE_MESSAGES_INCLUDE_THINKING", default=False)
            expose_thinking = _env_flag("SCICLONE_CLAUDE_MESSAGES_EXPOSE_THINKING", default=False)
            if thinking_budget_tokens is None:
                raw_budget = os.environ.get("SCICLONE_CLAUDE_MESSAGES_THINKING_BUDGET_TOKENS")
                if isinstance(raw_budget, str) and raw_budget.strip():
                    try:
                        thinking_budget_tokens = int(raw_budget.strip())
                    except Exception:
                        thinking_budget_tokens = None
            timeout_seconds = int(self.timeout) if isinstance(self.timeout, int) else 300
            try:
                return _anthropic_messages_call(
                    anthropic_base_v1=base,
                    api_key=api_key,
                    model=str(model),
                    messages=messages,
                    temperature=temperature if isinstance(temperature, (int, float)) else None,
                    max_tokens=max_tokens if isinstance(max_tokens, int) else None,
                    include_thinking=include_thinking,
                    expose_thinking=expose_thinking,
                    thinking_budget_tokens=thinking_budget_tokens,
                    timeout_seconds=timeout_seconds,
                    extra_headers=self.extra_headers,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Claude messages call failed (model=%s, base=%s): %s", model, base, exc)
                return None

        if gateway_target.api_style == "gemini_vertex" and is_gemini and prefer_vertex:
            vertex_resp = _try_vertex()
            if isinstance(vertex_resp, dict) and _first_nonempty_text(vertex_resp):
                return vertex_resp
            try:
                resp = _direct_chat(direct_target, fallback_to_session=True)
            except Exception as exc:  # noqa: BLE001
                if isinstance(vertex_resp, dict):
                    if _first_nonempty_text(vertex_resp):
                        return vertex_resp
                    status = vertex_resp.get("status") if isinstance(vertex_resp, dict) else None
                    if isinstance(status, int) and status >= 400:
                        err_payload = vertex_resp.get("error") if isinstance(vertex_resp, dict) else None
                        err = _extract_error_message(err_payload) or _extract_error_message(vertex_resp)
                        raise RuntimeError(
                            f"Gemini vertex fallback also failed (status={status}, base={self.base_url}, model={model}, error={err})"
                        ) from exc
                raise
            if not _first_nonempty_text(resp):
                vertex_resp2 = _try_vertex()
                if isinstance(vertex_resp2, dict) and _first_nonempty_text(vertex_resp2):
                    return vertex_resp2
            return resp

        if gateway_target.api_style == "anthropic_messages" and is_claude and prefer_claude_messages:
            claude_resp = _try_anthropic()
            if isinstance(claude_resp, dict) and _first_nonempty_text(claude_resp):
                return claude_resp
            try:
                resp = _direct_chat(direct_target, fallback_to_session=True)
            except Exception as exc:  # noqa: BLE001
                if isinstance(claude_resp, dict):
                    if _first_nonempty_text(claude_resp):
                        return claude_resp
                    status = claude_resp.get("status") if isinstance(claude_resp, dict) else None
                    if isinstance(status, int) and status >= 400:
                        err_payload = claude_resp.get("error") if isinstance(claude_resp, dict) else None
                        err = _extract_error_message(err_payload) or _extract_error_message(claude_resp)
                        url = None
                        try:
                            gw = claude_resp.get("_gateway") if isinstance(claude_resp, dict) else None
                            if isinstance(gw, dict):
                                url = gw.get("url")
                        except Exception:
                            url = None
                        raise RuntimeError(
                            f"Claude messages gateway also failed (status={status}, url={url or 'unknown'}, model={model}, error={err})"
                        ) from exc
                raise
            if not _first_nonempty_text(resp):
                claude_resp2 = _try_anthropic()
                if isinstance(claude_resp2, dict) and _first_nonempty_text(claude_resp2):
                    return claude_resp2
            return resp

        try:
            if gateway_target.api_style in {"chat", "responses"}:
                resp = _direct_chat(gateway_target)
            else:
                resp = _direct_chat(direct_target, fallback_to_session=True)
        except Exception as exc:  # noqa: BLE001
            # OpenAI-compat first; optionally fall back to Gemini Vertex.
            msg = str(exc).lower()
            if gateway_target.api_style == "gemini_vertex" and is_gemini and (
                "empty content" in msg or "max_tokens" in msg or "connection" in msg
            ):
                vertex_resp = _try_vertex()
                if isinstance(vertex_resp, dict) and _first_nonempty_text(vertex_resp):
                    return vertex_resp
            if gateway_target.api_style == "anthropic_messages" and is_claude and ("empty content" in msg or "connection" in msg or "protocol" in msg):
                claude_resp = _try_anthropic()
                if isinstance(claude_resp, dict) and _first_nonempty_text(claude_resp):
                    return claude_resp
            raise

        if gateway_target.api_style == "gemini_vertex" and is_gemini and not _first_nonempty_text(resp):
            vertex_resp = _try_vertex()
            if isinstance(vertex_resp, dict) and _first_nonempty_text(vertex_resp):
                return vertex_resp
        if gateway_target.api_style == "anthropic_messages" and is_claude and not _first_nonempty_text(resp):
            claude_resp = _try_anthropic()
            if isinstance(claude_resp, dict) and _first_nonempty_text(claude_resp):
                return claude_resp
        return resp

    def extract_text(self, response: Dict[str, Any], default: str = "") -> str:
        if not isinstance(response, dict):
            return default
        return _first_nonempty_text(response) or default

    def install_env(self) -> None:
        """Expose backend information through OPENAI_* environment variables."""
        try:
            self._client.install_env()
            try:
                self.base_url = self._client.get_active_base_url()
            except Exception:
                pass
            return
        except Exception:
            pass
        # Minimal env
        os.environ["OPENAI_API_KEY"] = self.api_key or ""
        os.environ["OPENAI_BASE_URL"] = _normalize_api_base(self.base_url)


def create_llm_service_session(config: Dict[str, Any]) -> LLMServiceSession:
    """Construct an `LLMServiceSession` from a generator config block."""

    if not config:
        raise ValueError("Missing generator configuration for inference service")

    # Allow concise configs with only `service_id` by resolving through llm_service services.json.
    # This keeps YAML short and matches the repository convention: set LLM_SERVICES_JSON and use service_id.
    try:
        service_id = config.get("service_id")
        has_api_base = bool(config.get("api_base") or config.get("base_url"))
        has_model = bool(config.get("model_name"))
        if isinstance(service_id, str) and service_id.strip() and (not has_api_base or not has_model):
            service_cfg_path = Path(
                os.getenv(
                    "LLM_SERVICES_JSON",
                    "config/services.json",
                )
            )
            base = load_llm_service_full_config(
                service_cfg_path,
                service_id.strip(),
                explicit_model=config.get("service_model"),
                fallback_model=config.get("model_name"),
            )
            merged: Dict[str, Any] = dict(base)
            for key, val in (config or {}).items():
                if key in {"client", "generation"} and isinstance(merged.get(key), dict) and isinstance(val, dict):
                    merged[key] = {**merged[key], **val}
                else:
                    merged[key] = val
            config = merged
    except Exception:
        # Keep original error behavior below if the config is still incomplete.
        pass

    raw_service_type = config.get("service_type", "private_endpoint")
    service_type = raw_service_type or "private_endpoint"
    if service_type not in {"private_endpoint"}:
        raise ValueError(
            f"Unsupported service_type={service_type!r}; SciClone now expects llm_service private endpoints"
        )

    api_base = _expand_env(config.get("api_base") or config.get("base_url"))
    if not api_base:
        raise ValueError("Private endpoint configuration requires `api_base`")

    model_name = _expand_env(config.get("model_name"))
    if not model_name:
        raise ValueError("Private endpoint configuration requires `model_name`")

    api_key = _expand_env(config.get("api_key"))
    default_headers = config.get("default_headers") or {}
    if not isinstance(default_headers, dict):
        raise ValueError("`default_headers` must be a mapping if provided")

    client_config = config.get("client") or {}
    timeout = client_config.get("timeout")
    # 环境变量覆盖超时（AGENT_CLIENT_TIMEOUT 优先级最高）
    env_timeout = os.environ.get("AGENT_CLIENT_TIMEOUT")
    if env_timeout:
        try:
            timeout = int(env_timeout)
        except ValueError:
            pass
    stream = bool(client_config.get("stream", False))
    # 允许通过环境变量强制开启流式（用于排查连接重置等偶发网络问题）
    env_stream = os.environ.get("SCICLONE_FORCE_STREAM")
    if isinstance(env_stream, str) and env_stream.strip():
        val = env_stream.strip().lower()
        stream = val not in {"0", "false", "off"}
    extra_headers = client_config.get("extra_headers")
    if extra_headers is not None and not isinstance(extra_headers, dict):
        raise ValueError("`client.extra_headers` must be a mapping if provided")
    api_channel = _normalize_api_channel(config.get("api_channel"), base_url=str(api_base))
    gateway_routing = config.get("gateway_routing")
    if gateway_routing is not None and not isinstance(gateway_routing, list):
        raise ValueError("`gateway_routing` must be a list if provided")

    service_id = _derive_service_id(str(api_base), config.get("service_id"), model_name)
    api_style_raw = config.get("api_style") or ("responses" if config.get("use_responses_api") else "chat")
    api_style = str(api_style_raw).strip().lower()
    if api_style not in {"chat", "responses"}:
        api_style = "chat"

    # 允许配置透传 alt_base_urls（来自 services.json）
    alt_base_urls = config.get("alt_base_urls")

    session = LLMServiceSession(
        service_id=service_id,
        model_name=model_name,
        base_url=str(api_base),
        alt_base_urls=alt_base_urls if isinstance(alt_base_urls, list) else None,
        api_key=api_key,
        default_headers=default_headers,
        timeout=timeout,
        stream=stream,
        extra_headers=extra_headers,
        api_channel=api_channel,
        api_style=api_style,
        gateway_routing=gateway_routing if isinstance(gateway_routing, list) else None,
    )

    session.install_env()
    timeout_display = f"{timeout}s" if timeout else "default(300s)"
    logger.info(
        "Backend: %s model=%s timeout=%s",
        session.service_id,
        session.model_name,
        timeout_display,
    )

    return session


class _SimpleApiClient:
    """Minimal OpenAI-compatible HTTP client.

    Supports `chat()` with JSON responses and optional streaming aggregation.
    """

    def __init__(self, *, base_url: str, api_key: str | None, timeout: int | None, stream: bool, extra_headers: Dict[str, str] | None, default_headers: Dict[str, str] | None) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key or ""
        self._timeout = timeout if isinstance(timeout, int) else 300
        self._stream = bool(stream)
        self._headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        if isinstance(default_headers, dict):
            self._headers.update(default_headers)
        if isinstance(extra_headers, dict):
            self._headers.update(extra_headers)
        _maybe_add_idealab_session_id(self._headers, self._base_url)
        _maybe_add_idealab_session_id(self._headers, self._base_url)

    def get_active_base_url(self) -> str:
        return self._base_url

    def install_env(self) -> None:
        os.environ["OPENAI_API_KEY"] = self._api_key
        os.environ["OPENAI_BASE_URL"] = self._base_url

    def chat(self, messages: Any, **kwargs: Any) -> Dict[str, Any]:
        try:
            import requests  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "HTTP fallback client requires 'requests'. Please install it: pip install requests"
            ) from exc

        # 轻量级重试机制：主要应对偶发的网络中断（如 ConnectionReset/超时），
        # 次数与间隔可通过环境变量覆盖，默认 3 次、首次间隔 1s、指数退避。
        try:
            max_attempts = max(1, int(os.getenv("SCICLONE_HTTP_RETRY_TIMES", "3") or "3"))
        except Exception:
            max_attempts = 3
        try:
            base_delay = float(os.getenv("SCICLONE_HTTP_RETRY_DELAY_SECONDS", "1.0") or "1.0")
        except Exception:
            base_delay = 1.0

        url = f"{self._base_url}/chat/completions"
        model = kwargs.get("model")
        if not model:
            raise ValueError("chat() requires 'model' in kwargs")

        payload_base: Dict[str, Any] = {"model": model, "messages": messages}
        model_lower = str(model).lower()
        is_gemini = "gemini" in model_lower
        # Pass through generation overrides
        for k, v in kwargs.items():
            if k != "model" and v is not None:
                payload_base[k] = v
        if is_gemini and "max_tokens" in payload_base and "max_output_tokens" not in payload_base:
            # Gemini OpenAI-compat expects max_output_tokens; mirror max_tokens to avoid truncation.
            payload_base["max_output_tokens"] = payload_base["max_tokens"]

        use_stream_default = bool(payload_base.get("stream")) or self._stream

        last_exc: Exception | None = None
        retry_on_empty = os.getenv("SCICLONE_HTTP_RETRY_ON_EMPTY", "1").strip().lower() not in {"0", "false"}
        fallback_to_nonstream_on_empty = (
            os.getenv("SCICLONE_HTTP_STREAM_FALLBACK_TO_NONSTREAM_ON_EMPTY", "1").strip().lower()
            not in {"0", "false"}
        )
        allow_empty_response = os.getenv("SCICLONE_HTTP_ALLOW_EMPTY_RESPONSE", "0").strip().lower() not in {"0", "false"}
        delay = base_delay
        should_disable_stream = False
        transient_statuses = {408, 409, 425, 429, 500, 502, 503, 504}

        def _is_rate_limit_error(err: Any) -> bool:
            if not isinstance(err, str):
                return False
            if not err.strip():
                return False
            low = err.lower()
            if "rate limit" in low or "too many requests" in low or "throttl" in low:
                return True
            # Common CN gateway messages
            if "限流" in err or "请求过多" in err or "频率" in err:
                return True
            return False

        def _is_transient_http_error(status: Optional[int], err: Any) -> bool:
            if isinstance(status, int) and status in transient_statuses:
                return True
            # Some OpenAI-compat gateways embed rate-limit as status=400 with a provider message.
            if isinstance(status, int) and status == 400 and _is_rate_limit_error(err):
                return True
            return False

        def _sleep_for_retry(resp: Any, fallback_delay: float) -> float:
            """Use Retry-After header if present; otherwise fallback_delay."""
            try:
                headers = getattr(resp, "headers", None)
                if not headers:
                    return fallback_delay
                raw = headers.get("Retry-After")
                if raw is None:
                    return fallback_delay
                s = str(raw).strip()
                if not s:
                    return fallback_delay
                # Retry-After can be seconds or HTTP-date; only handle numeric seconds.
                seconds = float(s)
                if seconds <= 0:
                    return fallback_delay
                return max(fallback_delay, seconds)
            except Exception:
                return fallback_delay

        def _embedded_status(payload: Any) -> Optional[int]:
            if not isinstance(payload, dict):
                return None
            raw = payload.get("status")
            if isinstance(raw, int):
                # Many callers inject HTTP status_code (often 200) into payload["status"].
                # Only treat it as an embedded error status when it's >= 400.
                if raw >= 400:
                    return raw
            if isinstance(raw, str):
                s = raw.strip()
                if s.isdigit():
                    try:
                        v = int(s)
                        if v >= 400:
                            return v
                    except Exception:
                        return None

            # Some gateways return HTTP 200 but encode an error in finish_reason/content.
            # Example: finish_reason="error_finish" and content is a JSON error object with code=429.
            try:
                choices = payload.get("choices")
                if not isinstance(choices, list) or not choices:
                    return None
                first = choices[0]
                if not isinstance(first, dict):
                    return None
                finish = first.get("finish_reason")
                if not (isinstance(finish, str) and "error" in finish.lower()):
                    return None
                msg = first.get("message") if isinstance(first.get("message"), dict) else {}
                content = msg.get("content") if isinstance(msg, dict) else None
                if not isinstance(content, str) or not content.strip():
                    return 500
                content_s = content.strip()
                if not content_s.startswith("{"):
                    return 500
                content_obj = json.loads(content_s)
                if not isinstance(content_obj, dict):
                    return 500
                err = content_obj.get("error")
                if isinstance(err, dict):
                    code = err.get("code")
                    if isinstance(code, int):
                        return code
                    if isinstance(code, str) and code.strip().isdigit():
                        return int(code.strip())
                    if str(err.get("status") or "").upper() == "RESOURCE_EXHAUSTED":
                        return 429
                return 500
            except Exception:
                return None
            return None

        def _embedded_error_message(payload: Any) -> str:
            """Extract an error message from embedded error payloads (including error_finish)."""
            if not isinstance(payload, dict):
                return ""
            # Prefer explicit error fields.
            msg = _extract_error_message(payload) or _extract_error_message(payload.get("error") if isinstance(payload, dict) else None)
            if msg:
                return msg
            try:
                choices = payload.get("choices")
                if not isinstance(choices, list) or not choices:
                    return ""
                first = choices[0]
                if not isinstance(first, dict):
                    return ""
                finish = first.get("finish_reason")
                if not (isinstance(finish, str) and "error" in finish.lower()):
                    return ""
                message = first.get("message")
                if not isinstance(message, dict):
                    return ""
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    # If it's JSON, try to extract a clean message.
                    s = content.strip()
                    if s.startswith("{"):
                        try:
                            obj = json.loads(s)
                            return _extract_error_message(obj) or _extract_error_message(obj.get("error")) or s
                        except Exception:
                            return s
                    return s
            except Exception:
                return ""
            return ""

        def _raise_if_embedded_error(payload: Any) -> None:
            status = _embedded_status(payload)
            if status is None or status < 400:
                return
            err = _embedded_error_message(payload)
            msg = f"HTTP chat failed (status={status}, base={self._base_url}, model={model}, error={err})"
            raise RuntimeError(msg)

        for attempt in range(1, max_attempts + 1):
            use_stream = use_stream_default and (not should_disable_stream)
            payload = dict(payload_base)
            if use_stream:
                payload["stream"] = True
            else:
                payload.pop("stream", None)
            try:
                resp = requests.post(
                    url,
                    headers=self._headers,
                    json=payload,
                    timeout=self._timeout,
                    stream=use_stream,
                )
                if not use_stream:
                    try:
                        data = resp.json()
                    except Exception:  # noqa: BLE001
                        data = {"status": resp.status_code, "text": resp.text}
                    if isinstance(data, dict) and "status" not in data:
                        data = dict(data)
                        data["status"] = resp.status_code
                    # Some gateways return HTTP 200 but embed the real error status in JSON.
                    try:
                        _raise_if_embedded_error(data)
                    except RuntimeError as exc:
                        status = _embedded_status(data)
                        err = _embedded_error_message(data)
                        if _is_transient_http_error(status, err) and attempt < max_attempts:
                            sleep_s = _sleep_for_retry(resp, delay)
                            logger.warning("%s; retrying in %.1fs", str(exc), sleep_s)
                            try:
                                time.sleep(sleep_s)
                            except Exception:
                                pass
                            delay = max(delay, sleep_s) * 2.0
                            continue
                        raise
                    if resp.status_code != 200:
                        err = _embedded_error_message(data) or (resp.text.strip() if isinstance(resp.text, str) else "")
                        msg = f"HTTP chat failed (status={resp.status_code}, base={self._base_url}, model={model}, error={err})"
                        if _is_transient_http_error(resp.status_code, err) and attempt < max_attempts:
                            sleep_s = _sleep_for_retry(resp, delay)
                            logger.warning("%s; retrying in %.1fs", msg, sleep_s)
                            try:
                                time.sleep(sleep_s)
                            except Exception:
                                pass
                            delay = max(delay, sleep_s) * 2.0
                            continue
                        raise RuntimeError(msg)
                    if retry_on_empty and not _first_nonempty_text(data):
                        if attempt >= max_attempts:
                            msg = f"HTTP chat got empty content after {attempt} attempts (base={self._base_url}, model={model})"
                            if allow_empty_response:
                                logger.warning("%s; returning last response.", msg)
                                return data
                            raise RuntimeError(msg)
                        logger.warning(
                            "HTTP chat got empty content (attempt %d/%d, base=%s, model=%s); retrying in %.1fs",
                            attempt,
                            max_attempts,
                            self._base_url,
                            model,
                            delay,
                        )
                        try:
                            time.sleep(delay)
                        except Exception:
                            pass
                        delay *= 2.0
                        continue
                    return data

                # 流式模式：按 OpenAI 兼容 SSE 协议增量读取，再组装为一次性响应。
                if resp.status_code != 200:
                    try:
                        data = resp.json()
                    except Exception:  # noqa: BLE001
                        data = {"status": resp.status_code, "text": resp.text}
                    if isinstance(data, dict) and "status" not in data:
                        data = dict(data)
                        data["status"] = resp.status_code
                    err = _embedded_error_message(data) or (resp.text.strip() if isinstance(resp.text, str) else "")
                    msg = f"HTTP chat failed (status={resp.status_code}, base={self._base_url}, model={model}, error={err})"
                    if _is_transient_http_error(resp.status_code, err) and attempt < max_attempts:
                        sleep_s = _sleep_for_retry(resp, delay)
                        logger.warning("%s; retrying in %.1fs", msg, sleep_s)
                        try:
                            time.sleep(sleep_s)
                        except Exception:
                            pass
                        delay = max(delay, sleep_s) * 2.0
                        continue
                    raise RuntimeError(msg)

                data = self._consume_streaming_response(resp)
                # Some gateways return HTTP 200 but embed the real error status in the first chunk JSON.
                try:
                    _raise_if_embedded_error(data)
                except RuntimeError as exc:
                    status = _embedded_status(data)
                    err = _embedded_error_message(data)
                    if _is_transient_http_error(status, err) and attempt < max_attempts:
                        sleep_s = _sleep_for_retry(resp, delay)
                        logger.warning("%s; retrying in %.1fs", str(exc), sleep_s)
                        try:
                            time.sleep(sleep_s)
                        except Exception:
                            pass
                        delay = max(delay, sleep_s) * 2.0
                        continue
                    raise
                if retry_on_empty and not _first_nonempty_text(data):
                    if fallback_to_nonstream_on_empty and not should_disable_stream:
                        should_disable_stream = True
                        logger.warning(
                            "Streaming chat got empty content (base=%s, model=%s); falling back to non-streaming retries.",
                            self._base_url,
                            model,
                        )
                    if attempt >= max_attempts:
                        msg = f"Streaming chat got empty content after {attempt} attempts (base={self._base_url}, model={model})"
                        if allow_empty_response:
                            logger.warning("%s; returning last response.", msg)
                            return data
                        raise RuntimeError(msg)
                    logger.warning(
                        "Streaming chat got empty content (attempt %d/%d, base=%s, model=%s); retrying in %.1fs",
                        attempt,
                        max_attempts,
                        self._base_url,
                        model,
                        delay,
                    )
                    try:
                        time.sleep(delay)
                    except Exception:
                        pass
                    delay *= 2.0
                    continue
                return data
            except requests.exceptions.RequestException as exc:  # type: ignore[attr-defined]
                last_exc = exc
                if attempt >= max_attempts:
                    logger.warning(
                        "HTTP chat request failed after %d attempts (SimpleApiClient, base=%s, model=%s): %s",
                        max_attempts,
                        self._base_url,
                        model,
                        exc,
                    )
                    raise
                logger.warning(
                    "HTTP chat request failed (attempt %d/%d, base=%s, model=%s): %s; retrying in %.1fs",
                    attempt,
                    max_attempts,
                    self._base_url,
                    model,
                    exc,
                    delay,
                )
                try:
                    time.sleep(delay)
                except Exception:
                    pass
                delay *= 2.0

        if last_exc:
            raise last_exc
        raise RuntimeError("HTTP chat request failed without response or exception")

    def _consume_streaming_response(self, resp: Any) -> Dict[str, Any]:
        """Consume an OpenAI-compatible streaming response and reassemble it."""
        full_text_parts: list[str] = []
        full_reasoning_parts: list[str] = []
        first_chunk: Dict[str, Any] | None = None
        finish_reason: Optional[str] = None

        try:
            for raw_line in resp.iter_lines(decode_unicode=False):
                if not raw_line:
                    continue
                line = _decode_sse_line(raw_line, resp)
                if not line:
                    continue

                if line.startswith("data:"):
                    data_str = line[len("data:") :].strip()
                else:
                    data_str = line

                if not data_str:
                    continue
                if data_str == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)
                except Exception:
                    # 无法解析的行直接跳过，避免中断整次调用。
                    continue

                if not isinstance(data, dict):
                    continue
                if first_chunk is None:
                    first_chunk = data

                choices = data.get("choices") or []
                if not choices:
                    continue
                choice0 = choices[0] or {}
                delta = choice0.get("delta") or {}
                delta_text = delta.get("content") or ""
                delta_reasoning = delta.get("reasoning_content") or ""
                if not (delta_text or delta_reasoning):
                    msg = choice0.get("message") or {}
                    delta_text = msg.get("content") or ""
                    delta_reasoning = msg.get("reasoning_content") or ""

                if delta_text:
                    full_text_parts.append(str(delta_text))
                if delta_reasoning:
                    full_reasoning_parts.append(str(delta_reasoning))

                fr = choice0.get("finish_reason")
                if isinstance(fr, str) and fr:
                    finish_reason = fr
        finally:
            try:
                resp.close()
            except Exception:
                pass

        full_text = "".join(full_text_parts)
        full_reasoning = "".join(full_reasoning_parts)
        base: Dict[str, Any] = first_chunk or {}

        message: Dict[str, Any] = {"role": "assistant", "content": full_text}
        if full_reasoning.strip():
            message["reasoning_content"] = full_reasoning

        return {
            "id": base.get("id"),
            "object": base.get("object") or "chat.completion",
            "created": base.get("created"),
            "model": base.get("model"),
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason or "stop",
                }
            ],
            "usage": base.get("usage") or {},
        }


class _ResponsesApiClient:
    """Minimal OpenAI Responses API client for models like gpt-5.1-2025-11-13."""

    def __init__(self, *, base_url: str, api_key: str | None, timeout: int | None, extra_headers: Dict[str, str] | None, default_headers: Dict[str, str] | None, stream: bool = False) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key or ""
        self._timeout = timeout if isinstance(timeout, int) else 300
        self._stream = bool(stream)
        self._headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        if isinstance(default_headers, dict):
            self._headers.update(default_headers)
        if isinstance(extra_headers, dict):
            self._headers.update(extra_headers)

    def get_active_base_url(self) -> str:
        return self._base_url

    def install_env(self) -> None:
        os.environ["OPENAI_API_KEY"] = self._api_key
        os.environ["OPENAI_BASE_URL"] = self._base_url

    def _messages_to_text(self, messages: Any) -> str:
        """Flatten chat-style messages into a single text block for Responses `input`.

        AiArena 的示例脚本（test_openai_response_api）也是直接传入纯文本，这里尽量对齐：
        将 system/user/assistant 的 content 串联为一段文本。
        """
        if isinstance(messages, str):
            return messages
        if not isinstance(messages, list):
            return str(messages)
        parts: list[str] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for it in content:
                    if isinstance(it, str):
                        parts.append(it)
                    elif isinstance(it, dict):
                        txt = it.get("text") or it.get("content")
                        if isinstance(txt, str):
                            parts.append(txt)
        return "\n\n".join(p for p in parts if p.strip())

    def chat(self, messages: Any, **kwargs: Any) -> Dict[str, Any]:
        try:
            import requests  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "HTTP fallback client requires 'requests'. Please install it: pip install requests"
            ) from exc

        try:
            max_attempts = max(1, int(os.getenv("SCICLONE_HTTP_RETRY_TIMES", "3") or "3"))
        except Exception:
            max_attempts = 3
        try:
            base_delay = float(os.getenv("SCICLONE_HTTP_RETRY_DELAY_SECONDS", "1.0") or "1.0")
        except Exception:
            base_delay = 1.0

        url = f"{self._base_url}/responses"
        model = kwargs.get("model")
        if not model:
            raise ValueError("chat() requires 'model' in kwargs")

        input_text = self._messages_to_text(messages)
        payload: Dict[str, Any] = {"model": model, "input": input_text}
        for k, v in kwargs.items():
            if k != "model" and v is not None:
                payload[k] = v
        # Chat Completions uses max_tokens while the Responses API uses
        # max_output_tokens. Accept the former because the rest of the AgenQA
        # runtime historically exposes it as a provider-neutral setting.
        if "max_tokens" in payload and "max_output_tokens" not in payload:
            payload["max_output_tokens"] = payload.pop("max_tokens")
        last_exc: Exception | None = None
        retry_on_empty = os.getenv("SCICLONE_HTTP_RETRY_ON_EMPTY", "1").strip().lower() not in {"0", "false"}
        delay = base_delay
        for attempt in range(1, max_attempts + 1):
            try:
                resp = requests.post(
                    url,
                    headers=self._headers,
                    json=payload,
                    timeout=self._timeout,
                    stream=self._stream,
                )
                # 解析响应
                try:
                    data = resp.json()
                except Exception:  # noqa: BLE001
                    data = {"status": resp.status_code, "text": resp.text}

                embedded_status = data.get("status") if isinstance(data, dict) else None
                status = embedded_status if isinstance(embedded_status, int) and embedded_status >= 400 else resp.status_code
                if isinstance(status, int) and status >= 400:
                    err = _extract_error_message(data)
                    if not err and isinstance(data, dict):
                        err = _extract_error_message(data.get("error"))
                    if not err:
                        err = str(getattr(resp, "text", "") or "").strip()
                    raise RuntimeError(
                        f"HTTP responses failed (status={status}, base={self._base_url}, model={model}, error={err})"
                    )

                if retry_on_empty and not _first_nonempty_text(data):
                    if attempt >= max_attempts:
                        logger.warning(
                            "Responses API got empty content after %d attempts (base=%s, model=%s); returning last response.",
                            attempt,
                            self._base_url,
                            model,
                        )
                        return data
                    logger.warning(
                        "Responses API got empty content (attempt %d/%d, base=%s, model=%s); retrying in %.1fs",
                        attempt,
                        max_attempts,
                        self._base_url,
                        model,
                        delay,
                    )
                    try:
                        time.sleep(delay)
                    except Exception:
                        pass
                    delay *= 2.0
                    continue

                return data
            except requests.exceptions.RequestException as exc:  # type: ignore[attr-defined]
                last_exc = exc
                if attempt >= max_attempts:
                    logger.warning(
                        "HTTP responses request failed after %d attempts (ResponsesApiClient, base=%s, model=%s): %s",
                        max_attempts,
                        self._base_url,
                        model,
                        exc,
                    )
                    raise
                logger.warning(
                    "HTTP responses request failed (attempt %d/%d, base=%s, model=%s): %s; retrying in %.1fs",
                    attempt,
                    max_attempts,
                    self._base_url,
                    model,
                    exc,
                    delay,
                )
                try:
                    time.sleep(delay)
                except Exception:
                    pass
                delay *= 2.0
        if last_exc:
            raise last_exc
        raise RuntimeError("HTTP responses request failed without response or exception")
