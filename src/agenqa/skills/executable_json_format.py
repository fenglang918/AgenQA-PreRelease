"""ExecutableJsonFormat runner: repair a role output into strict JSON.

This is a lightweight "Format" stage for executable track structured roles.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any, Dict, List, Optional

from infra.llm.inference import resolve_inference
from infra.prompt.prompt_builder import build_messages_with_background
from infra.prompt.prompt_tracker import log_using_prompt, snapshot_prompt_used, snapshot_rendered_prompt
from infra.llm.service_client import LLMServiceSession
from agenqa.skills.base import BaseSkillRunner
from infra.text.json_sanitize import sanitize_json_text

from agenqa.prompts.executable_json_format import EXECUTABLE_JSON_FORMAT_V1, EXECUTABLE_JSON_FORMAT_V1_EN

logger = logging.getLogger(__name__)


@dataclass
class ExecutableJsonFormatInput:
    task_name: str
    required_keys: List[str]
    raw_output: str
    original_prompt: str = ""
    parse_error: str = ""


@dataclass
class ExecutableJsonFormatConfig:
    generator: Dict[str, Any]
    prompt_path: Any
    prompt_text: Optional[str] = None
    lang: Optional[str] = None
    max_retries: int = 1


class ExecutableJsonFormatRunner:
    def __init__(self, config: ExecutableJsonFormatConfig) -> None:
        self.config = config
        resolved = resolve_inference(config.generator)
        self.session: LLMServiceSession = resolved.session
        self._chat_args: Dict[str, Any] = dict(resolved.chat_args)

        lang_norm = (config.lang or "").strip().lower()
        use_en = lang_norm in {"en", "english"}
        base_text = config.prompt_text if getattr(config, "prompt_text", None) else None
        if base_text is None:
            base_text = EXECUTABLE_JSON_FORMAT_V1_EN if use_en else EXECUTABLE_JSON_FORMAT_V1
        self.prompt_text = base_text
        self.prompt_template = Template(base_text)
        log_using_prompt(logger, Path(str(config.prompt_path)))

    def _build_prompt(self, fmt_in: ExecutableJsonFormatInput, *, parse_error: str = "") -> str:
        payload = {
            "task_name": fmt_in.task_name or "",
            "required_keys_json": json.dumps(list(fmt_in.required_keys or []), ensure_ascii=False),
            "parse_error": parse_error or fmt_in.parse_error or "",
            "original_prompt": fmt_in.original_prompt or "",
            "raw_output": fmt_in.raw_output or "",
        }
        body = self.prompt_template.safe_substitute(payload)
        lang = (self.config.lang or "zh").lower()
        header = f"Role: ExecutableJsonFormat | Language: {lang}" if lang in {"en", "english"} else f"【ExecutableJsonFormat 角色】语言: {lang}"
        return "\n".join([header, "", body])

    @staticmethod
    def _parse_json_object(text: str, *, required_keys: List[str]) -> str:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("ExecutableJsonFormat output is empty")
        s = text.strip()
        if s == "INCOMPLETE":
            raise ValueError("ExecutableJsonFormat returned INCOMPLETE")

        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            obj = json.loads(sanitize_json_text(s))
        if not isinstance(obj, dict):
            raise ValueError("ExecutableJsonFormat output is not a JSON object")

        keys = set(obj.keys())
        required = [k for k in (required_keys or []) if isinstance(k, str) and k.strip()]
        missing = [k for k in required if k not in keys]
        if missing:
            raise ValueError(f"ExecutableJsonFormat missing keys: {missing}")
        extra = [k for k in keys if k not in set(required)]
        if extra:
            for k in extra:
                obj.pop(k, None)
        return json.dumps(obj, ensure_ascii=False)

    def run_one(
        self,
        fmt_in: ExecutableJsonFormatInput,
        *,
        snapshot_dir: Optional[Any] = None,
        unified_prompt_dir: Optional[Path] = None,
        name_prefix: str = "prompt_used.executable_json_format.",
    ) -> str:
        prompt_body = self._build_prompt(fmt_in)
        messages = build_messages_with_background(prompt_body, lang=self.config.lang or "zh")
        response = self.session.chat(messages, **self._chat_args)
        BaseSkillRunner._check_finish_reason(response, "ExecutableJsonFormat")
        text = (self.session.extract_text(response, default="") or "").strip()

        snap_path: Optional[Path] = None
        if snapshot_dir is not None:
            try:
                snap_path = Path(snapshot_dir)
                snap_path.mkdir(parents=True, exist_ok=True)
                (snap_path / "input_view.json").write_text(
                    json.dumps(
                        {
                            "task_name": fmt_in.task_name,
                            "required_keys": list(fmt_in.required_keys or []),
                            "parse_error": fmt_in.parse_error,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                snapshot_prompt_used(
                    Path(str(self.config.prompt_path)),
                    snap_path,
                    content=self.prompt_text,
                    name_prefix="prompt_used.",
                    logger=logger,
                )
                snapshot_rendered_prompt(prompt_body, snap_path, filename="prompt_rendered.txt", logger=logger)
                request_meta = {
                    "model": self.session.model_name,
                    "service_id": self.session.service_id,
                    "chat_args": self._chat_args,
                }
                (snap_path / "request_meta.json").write_text(
                    json.dumps(request_meta, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                try:
                    (snap_path / "raw_response.json").write_text(
                        json.dumps(response, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                except Exception:
                    pass
                (snap_path / "raw_text.txt").write_text(text or "", encoding="utf-8")
            except Exception as exc:  # noqa: BLE001
                logger.warning("ExecutableJsonFormat snapshot write failed: %s", str(exc))
                snap_path = None

        last_exc: Optional[Exception] = None
        attempts = 1 + max(0, int(self.config.max_retries))
        for attempt in range(attempts):
            parse_error = fmt_in.parse_error
            if last_exc is not None:
                parse_error = f"{parse_error}\n\n[format_parse_error]\n{last_exc}".strip()
            try:
                if attempt == 0:
                    return self._parse_json_object(text, required_keys=list(fmt_in.required_keys or []))
                retry_prompt = self._build_prompt(fmt_in, parse_error=parse_error)
                retry_messages = build_messages_with_background(retry_prompt, lang=self.config.lang or "zh")
                retry_resp = self.session.chat(retry_messages, **self._chat_args)
                BaseSkillRunner._check_finish_reason(retry_resp, "ExecutableJsonFormat(retry)")
                text = (self.session.extract_text(retry_resp, default="") or "").strip()
                if snap_path is not None:
                    try:
                        (snap_path / f"raw_response.retry_{attempt}.json").write_text(
                            json.dumps(retry_resp, ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                        (snap_path / f"raw_text.retry_{attempt}.txt").write_text(text or "", encoding="utf-8")
                    except Exception:
                        pass
                return self._parse_json_object(text, required_keys=list(fmt_in.required_keys or []))
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                continue
        raise ValueError(f"ExecutableJsonFormat failed after retries: {last_exc}")


__all__ = [
    "ExecutableJsonFormatInput",
    "ExecutableJsonFormatConfig",
    "ExecutableJsonFormatRunner",
]
