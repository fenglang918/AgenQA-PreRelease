#!/usr/bin/env python3
"""
测试 OpenAI 兼容 API 是否可用（Idealab / AIMux / DMXAPI / AiArena / DashScope / SJTU Relay 等）。

定位：
- 本目录 `infra/llm/probes/*` 是围绕 `infra/llm/` 的“人工探测/连通性验证工具”，不参与 SciClone pipeline 运行时推理调用；
- pipeline 真正发请求的入口是 `infra/llm/inference.py` 与 `infra/llm/service_client.py`。

本脚本提供统一入口：
- /models 探测
- /chat/completions 探测（含 stream）
- /responses 探测（适配 Responses-only 模型）

安全说明：
- 不要把明文 Key 写入仓库；仅通过环境变量或 CLI 参数传入。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import requests


DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_IDEALAB_BASE_URL = os.getenv("IDEALAB_BASE_URL", DEFAULT_OPENAI_BASE_URL)
DEFAULT_AIMUX_BASE_URL = os.getenv("AIMUX_BASE_URL", DEFAULT_OPENAI_BASE_URL)
DEFAULT_DMXAPI_BASE_URL = "https://www.dmxapi.cn/v1"
DEFAULT_AIARENA_BASE_URL = os.getenv("AIARENA_BASE_URL", DEFAULT_OPENAI_BASE_URL)
DEFAULT_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_SJTU_BASE_URL = os.getenv("SJTU_BASE_URL", DEFAULT_OPENAI_BASE_URL)


def _normalize_api_key(api_key: Optional[str]) -> Optional[str]:
    if not isinstance(api_key, str):
        return None
    val = api_key.strip()
    return val or None


def _missing_api_key_error() -> Dict[str, Any]:
    return {
        "success": False,
        "error": (
            "缺少 API Key（请设置 --api-key，或环境变量 "
            "IDEALAB_API_KEY/AIMUX_API_KEY/DMXAPI_API_KEY/AIARENA_API_KEY/DASHSCOPE_API_KEY/BAILIAN_API_KEY/OPENAI_API_KEY/AGENT_API_KEY）"
        ),
    }


def _parse_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "y", "on"):
        return True
    if normalized in ("0", "false", "no", "n", "off"):
        return False
    return None


def _apply_optional_flags(payload: Dict[str, Any]) -> None:
    """Apply optional gateway flags, keeping existing behavior."""
    enable_thinking = _parse_bool(os.getenv("IDEALAB_ENABLE_THINKING"))
    if enable_thinking is not None:
        payload["enable_thinking"] = enable_thinking


def _resolve_test_models_env(source: Optional[str]) -> Optional[str]:
    """Resolve provider-specific model list env var for `test-models`.

    Compatibility policy:
    - Prefer `<SOURCE>_MODELS` when `--source` is explicitly given.
    - Fall back to historical `IDEALAB_MODELS`.
    - Allow generic `OPENAI_COMPAT_MODELS` as a provider-agnostic override.
    """
    src = str(source or "").strip().lower()
    provider_env_map = {
        "idealab": "IDEALAB_MODELS",
        "aimux": "AIMUX_MODELS",
        "dmxapi": "DMXAPI_MODELS",
        "aiarena": "AIARENA_MODELS",
        "dashscope": "DASHSCOPE_MODELS",
        "bailian": "BAILIAN_MODELS",
        "sjtu": "SJTU_MODELS",
        "custom": "OPENAI_COMPAT_MODELS",
    }
    candidate_names: List[str] = []
    if src:
        provider_name = provider_env_map.get(src)
        if provider_name:
            candidate_names.append(provider_name)
    candidate_names.extend(["OPENAI_COMPAT_MODELS", "IDEALAB_MODELS"])
    seen: set[str] = set()
    for name in candidate_names:
        if name in seen:
            continue
        seen.add(name)
        val = os.getenv(name)
        if isinstance(val, str) and val.strip():
            return val
    return None


def resolve_base_url_and_api_key(
    source: Optional[str],
    explicit_base: Optional[str],
    explicit_key: Optional[str],
) -> Tuple[str, Optional[str]]:
    """
    解析 base_url 与 api_key。

    规则：
    - 默认优先 Idealab / AIMux / DMXAPI，其次 AiArena / DashScope / Bailian / OpenAI；
    - 指定 --source 时按对应来源优先；
    - 显式参数（--base-url / --api-key）最高优先级。
    """
    base_url = (
        os.getenv("IDEALAB_BASE_URL")
        or os.getenv("AIMUX_BASE_URL")
        or os.getenv("DMXAPI_BASE_URL")
        or os.getenv("AIARENA_BASE_URL")
        or os.getenv("DASHSCOPE_BASE_URL")
        or os.getenv("BAILIAN_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or DEFAULT_OPENAI_BASE_URL
    )
    api_key: Optional[str] = (
        os.getenv("IDEALAB_API_KEY")
        or os.getenv("AIMUX_API_KEY")
        or os.getenv("DMXAPI_API_KEY")
        or os.getenv("AIARENA_API_KEY")
        or os.getenv("DASHSCOPE_API_KEY")
        or os.getenv("BAILIAN_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or None
    )

    if source:
        src = source.strip().lower()
        if src == "idealab":
            if not explicit_base:
                base_url = os.getenv("IDEALAB_BASE_URL") or DEFAULT_IDEALAB_BASE_URL
            if not explicit_key:
                api_key = os.getenv("IDEALAB_API_KEY") or api_key
        elif src == "aimux":
            if not explicit_base:
                base_url = os.getenv("AIMUX_BASE_URL") or DEFAULT_AIMUX_BASE_URL
            if not explicit_key:
                api_key = os.getenv("AIMUX_API_KEY") or api_key
        elif src == "dmxapi":
            if not explicit_base:
                base_url = os.getenv("DMXAPI_BASE_URL") or DEFAULT_DMXAPI_BASE_URL
            if not explicit_key:
                api_key = os.getenv("DMXAPI_API_KEY") or api_key
        elif src == "aiarena":
            if not explicit_base:
                base_url = os.getenv("AIARENA_BASE_URL") or DEFAULT_AIARENA_BASE_URL
            if not explicit_key:
                api_key = os.getenv("AIARENA_API_KEY") or api_key
        elif src in ("dashscope", "bailian"):
            if not explicit_base:
                base_url = (
                    os.getenv("DASHSCOPE_BASE_URL")
                    or os.getenv("BAILIAN_BASE_URL")
                    or DEFAULT_DASHSCOPE_BASE_URL
                )
            if not explicit_key:
                api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("BAILIAN_API_KEY") or api_key
        elif src == "sjtu":
            if not explicit_base:
                base_url = os.getenv("AGENT_BASE_URL") or DEFAULT_SJTU_BASE_URL
            if not explicit_key:
                api_key = os.getenv("AGENT_API_KEY") or api_key
        elif src == "custom":
            if not explicit_base:
                base_url = os.getenv("OPENAI_BASE_URL") or base_url
            if not explicit_key:
                api_key = os.getenv("OPENAI_API_KEY") or api_key

    if explicit_base:
        base_url = explicit_base
    if explicit_key:
        api_key = explicit_key

    return base_url, _normalize_api_key(api_key)


def extract_message_text(resp_json: Any) -> str:
    """Extract the most relevant visible text from OpenAI-compatible chat responses."""
    if not isinstance(resp_json, dict):
        return ""
    try:
        choices = resp_json.get("choices") or []
        if not isinstance(choices, list) or not choices:
            return ""
        choice0 = choices[0] or {}
        if not isinstance(choice0, dict):
            return ""
        msg = choice0.get("message") or {}
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
            reasoning = msg.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning.strip():
                return reasoning.strip()
        text = choice0.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()
    except Exception:
        return ""
    return ""


def _decode_sse_line(raw_line: Any, response: Any) -> str:
    """Decode one SSE line robustly, preferring UTF-8 for Chinese content."""
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

    # Final fallback: keep the stream readable instead of crashing.
    return raw_line.decode("utf-8", errors="replace").strip()


def consume_sse_stream(response: Any) -> Tuple[str, str]:
    """
    Consume OpenAI-compatible SSE stream for `/chat/completions`.

    Returns:
      (content_text, reasoning_text)

    Notes:
    - Some thinking-style models stream visible text in `delta.reasoning_content` while leaving `delta.content` empty.
    """
    content_parts: List[str] = []
    reasoning_parts: List[str] = []

    for raw_line in response.iter_lines(decode_unicode=False):
        if not raw_line:
            continue
        line = _decode_sse_line(raw_line, response)
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
            payload = json.loads(data_str)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue

        choices = payload.get("choices") or []
        if not isinstance(choices, list) or not choices:
            continue
        choice0 = choices[0] or {}
        if not isinstance(choice0, dict):
            continue

        delta = choice0.get("delta") or {}
        if not isinstance(delta, dict):
            delta = {}

        delta_content = delta.get("content") or ""
        delta_reasoning = delta.get("reasoning_content") or ""

        if not (delta_content or delta_reasoning):
            msg = choice0.get("message") or {}
            if isinstance(msg, dict):
                delta_content = msg.get("content") or ""
                delta_reasoning = msg.get("reasoning_content") or ""

        if delta_content:
            content_parts.append(str(delta_content))
        if delta_reasoning:
            reasoning_parts.append(str(delta_reasoning))

    return "".join(content_parts), "".join(reasoning_parts)


def _request_get_json(url: str, *, headers: Dict[str, str], timeout_s: int) -> Tuple[int, Any]:
    resp = requests.get(url, headers=headers, timeout=timeout_s)
    try:
        return resp.status_code, resp.json()
    except Exception:
        return resp.status_code, {"raw_text": resp.text}


def _request_post_json(
    url: str,
    *,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout_s: int,
) -> Tuple[int, Any]:
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    try:
        return resp.status_code, resp.json()
    except Exception:
        return resp.status_code, {"raw_text": resp.text}


def cmd_models(base_url: str, api_key: Optional[str]) -> Dict[str, Any]:
    api_key = _normalize_api_key(api_key)
    if not api_key:
        return _missing_api_key_error()

    url = f"{base_url.rstrip('/')}/models"
    status, data = _request_get_json(url, headers={"Authorization": f"Bearer {api_key}"}, timeout_s=30)
    return {"success": status == 200, "status_code": status, "url": url, "response": data}


def cmd_chat(base_url: str, api_key: Optional[str], *, model: str, timeout_s: int = 60) -> Dict[str, Any]:
    api_key = _normalize_api_key(api_key)
    if not api_key:
        return _missing_api_key_error()

    url = f"{base_url.rstrip('/')}/chat/completions"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": "say hello"}],
        "max_tokens": 16,
    }
    _apply_optional_flags(payload)

    status, data = _request_post_json(url, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, payload=payload, timeout_s=timeout_s)
    message = extract_message_text(data)
    return {"success": status == 200, "status_code": status, "url": url, "model": model, "message": message, "response": data}


def cmd_test_models(base_url: str, api_key: Optional[str], *, model_names: List[str]) -> Dict[str, Any]:
    api_key = _normalize_api_key(api_key)
    if not api_key:
        return _missing_api_key_error()

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    results: Dict[str, Any] = {}
    available: List[str] = []
    unavailable: List[str] = []

    for model in model_names:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10,
        }
        _apply_optional_flags(payload)

        status, data = _request_post_json(url, headers=headers, payload=payload, timeout_s=30)
        ok = status == 200
        message = extract_message_text(data)
        results[model] = {"success": ok, "status_code": status, "message": message, "response": data}
        (available if ok else unavailable).append(model)

    # 注意：批量测试的“成功”定义为脚本执行成功（而非所有模型都可用）。
    return {
        "success": True,
        "url": url,
        "results": results,
        "available": available,
        "unavailable": unavailable,
        "all_available": len(unavailable) == 0,
    }


def cmd_stream(base_url: str, api_key: Optional[str], *, model: str, timeout_s: int = 300) -> Dict[str, Any]:
    api_key = _normalize_api_key(api_key)
    if not api_key:
        return _missing_api_key_error()

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": (
                    "请用中文进行较长时间的思考与推理，系统性地分步骤分析一个复杂问题，"
                    "输出尽量长、结构化的回答，用于测试流式输出与长连接稳定性。"
                ),
            }
        ],
        "max_tokens": 1024,
        "stream": True,
    }
    _apply_optional_flags(payload)

    content_text = ""
    reasoning_text = ""
    printed_any_content = False

    try:
        with requests.post(url, headers=headers, json=payload, timeout=timeout_s, stream=True) as resp:
            if resp.status_code != 200:
                try:
                    data = resp.json()
                except Exception:
                    data = {"raw_text": resp.text}
                return {"success": False, "status_code": resp.status_code, "url": url, "model": model, "response": data}

            # 实时打印：优先 content；若一直没有 content，则打印 reasoning_content（thinking 模型常见）。
            for raw_line in resp.iter_lines(decode_unicode=False):
                if not raw_line:
                    continue
                line = _decode_sse_line(raw_line, resp)
                if not line:
                    continue
                if not line.startswith("data:"):
                    continue
                data_str = line[len("data:") :].strip()
                if data_str == "[DONE]":
                    break
                try:
                    payload_chunk = json.loads(data_str)
                except Exception:
                    continue
                if not isinstance(payload_chunk, dict):
                    continue
                choices = payload_chunk.get("choices") or []
                if not isinstance(choices, list) or not choices:
                    continue
                choice0 = choices[0] or {}
                if not isinstance(choice0, dict):
                    continue
                delta = choice0.get("delta") or {}
                if not isinstance(delta, dict):
                    delta = {}
                delta_content = delta.get("content") or ""
                delta_reasoning = delta.get("reasoning_content") or ""
                if not (delta_content or delta_reasoning):
                    msg = choice0.get("message") or {}
                    if isinstance(msg, dict):
                        delta_content = msg.get("content") or ""
                        delta_reasoning = msg.get("reasoning_content") or ""

                if delta_content:
                    printed_any_content = True
                    content_text += str(delta_content)
                    print(delta_content, end="", flush=True)
                    continue

                if delta_reasoning and (not printed_any_content):
                    reasoning_text += str(delta_reasoning)
                    print(delta_reasoning, end="", flush=True)
                elif delta_reasoning:
                    reasoning_text += str(delta_reasoning)

    except requests.exceptions.Timeout:
        return {"success": False, "error": "流式请求超时", "url": url, "model": model}
    except requests.exceptions.ConnectionError as e:
        return {"success": False, "error": f"连接错误: {e}", "url": url, "model": model}
    except Exception as e:
        return {"success": False, "error": str(e), "url": url, "model": model}

    # 打印换行（便于终端显示）
    print()

    return {
        "success": True,
        "url": url,
        "model": model,
        "content_text": content_text,
        "reasoning_text": reasoning_text,
        "printed": "content" if printed_any_content else "reasoning_content",
        "full_text": content_text.strip() or reasoning_text.strip(),
    }


def build_input_text(paper_path: Optional[str], extra_instructions: Optional[str]) -> str:
    base_instruction = (
        extra_instructions
        or "你是一个科研助教。阅读下面的学术论文片段，基于内容用中文出一道高质量的单选题，并给出标准答案和简要解析。"
    )
    if not paper_path:
        return base_instruction + "\n\n（当前未提供论文内容，仅用于 Responses API 连通性测试。）"

    try:
        with open(paper_path, "r", encoding="utf-8") as f:
            paper_text = f.read()
    except UnicodeDecodeError:
        with open(paper_path, "r", encoding="utf-8", errors="replace") as f:
            paper_text = f.read()
    except FileNotFoundError:
        return base_instruction + f"\n\n（警告：未找到论文文件 {paper_path}，仅返回指令本身。）"

    max_chars = 20000
    if len(paper_text) > max_chars:
        paper_text = paper_text[:max_chars]

    return base_instruction + "\n\n--- 论文内容开始 ---\n" + paper_text + "\n--- 论文内容结束 ---"


def extract_responses_output_text(resp_json: Any) -> str:
    if not isinstance(resp_json, dict):
        return ""
    outputs = resp_json.get("output") or resp_json.get("outputs") or []
    if not isinstance(outputs, list) or not outputs:
        return ""
    first = outputs[0]
    if not isinstance(first, dict):
        return ""

    # Standard: {type:"message", content:[{type:"output_text", text:"..."}]}
    content_items = first.get("content") or []
    if not isinstance(content_items, list):
        return ""
    for item in content_items:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "output_text":
            txt = item.get("text")
            if isinstance(txt, str) and txt.strip():
                return txt.strip()
        if item_type == "text":
            text_block = item.get("text")
            if isinstance(text_block, dict):
                val = text_block.get("value")
                if isinstance(val, str) and val.strip():
                    return val.strip()
    return ""


def cmd_responses(
    base_url: str,
    api_key: Optional[str],
    *,
    model: str,
    file_path: Optional[str],
    instructions: Optional[str],
) -> Dict[str, Any]:
    api_key = _normalize_api_key(api_key)
    if not api_key:
        return _missing_api_key_error()

    url = f"{base_url.rstrip('/')}/responses"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    input_text = build_input_text(file_path, instructions)
    payload: Dict[str, Any] = {"model": model, "input": input_text}
    _apply_optional_flags(payload)

    status, data = _request_post_json(url, headers=headers, payload=payload, timeout_s=60)
    output_text = extract_responses_output_text(data)
    return {
        "success": status == 200,
        "status_code": status,
        "url": url,
        "model": model,
        "output_preview": output_text[:200],
        "response": data,
    }


def _print_json(obj: Any) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OpenAI-compat API probe tool (Idealab/AIMux/DMXAPI/AiArena/DashScope/SJTU)")
    parser.add_argument("--source", "-s", default=None, help="idealab|aimux|dmxapi|aiarena|dashscope|bailian|sjtu|custom")
    parser.add_argument("--base-url", default=None, help="Override base URL (e.g. https://.../v1)")
    parser.add_argument("--api-key", default=None, help="Override API key")

    sub = parser.add_subparsers(dest="command")

    sub.add_parser("models", help="List /models")

    p_test = sub.add_parser("test-models", help="Test multiple models via /chat/completions")
    p_test.add_argument(
        "--models",
        default=None,
        help="Comma-separated model ids (default: env <SOURCE>_MODELS / OPENAI_COMPAT_MODELS / IDEALAB_MODELS or built-in list)",
    )

    p_chat = sub.add_parser("chat", help="Test one model via /chat/completions")
    p_chat.add_argument("model", help="Model id")

    p_stream = sub.add_parser("stream", help="Stream test via /chat/completions")
    p_stream.add_argument("model", nargs="?", default="qwen3-max", help="Model id (default: qwen3-max)")

    p_resp = sub.add_parser("responses", help="Test /responses (Responses API)")
    p_resp.add_argument("model", help="Model id")
    p_resp.add_argument("--file", dest="file_path", default=None, help="Input file path (optional)")
    p_resp.add_argument("--instructions", default=None, help="Extra instructions (optional)")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    base_url, api_key = resolve_base_url_and_api_key(args.source, args.base_url, args.api_key)

    # Default action: do a minimal chat test (kept for convenience)
    cmd = args.command or "smoke"

    if cmd == "models":
        result = cmd_models(base_url, api_key)
        _print_json(result)
        return 0 if result.get("success") else 1

    if cmd == "test-models":
        env_models = _resolve_test_models_env(args.source)
        if args.models:
            model_names = [m.strip() for m in args.models.split(",") if m.strip()]
        elif env_models:
            model_names = [m.strip() for m in env_models.split(",") if m.strip()]
        else:
            model_names = [
                "gpt-5.2-1211-global",
                "gpt-5-0807-global",
                "gpt-5-mini-0807-global",
                "gemini-2.5-pro-06-17",
                "claude_sonnet4_5",
                "qwen3-max",
            ]
        result = cmd_test_models(base_url, api_key, model_names=model_names)
        _print_json(result)
        return 0 if result.get("success") else 1

    if cmd == "chat":
        result = cmd_chat(base_url, api_key, model=args.model)
        _print_json(result)
        return 0 if result.get("success") else 1

    if cmd == "stream":
        result = cmd_stream(base_url, api_key, model=args.model)
        _print_json(result)
        return 0 if result.get("success") else 1

    if cmd == "responses":
        result = cmd_responses(
            base_url,
            api_key,
            model=args.model,
            file_path=args.file_path,
            instructions=args.instructions,
        )
        _print_json(result)
        return 0 if result.get("success") else 1

    if cmd == "smoke":
        # Keep previous behavior: do a minimal chat test (default model qwen3-max).
        result = cmd_chat(base_url, api_key, model="qwen3-max")
        _print_json(result)
        return 0 if result.get("success") else 1

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
