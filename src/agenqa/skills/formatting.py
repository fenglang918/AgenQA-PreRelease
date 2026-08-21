"""Format 角色：在 Draft 草稿基础上整理正式题目并自检。

当前主要服务于 Extend/Revise/QA‑Init 的角色拆分实验：
在同一物理场景下分离“构思草稿”（Draft）与“格式化 + 自检”（Format）。
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any, Dict, List, Optional

from infra.llm.inference import resolve_inference
from infra.prompt.prompt_builder import build_messages_with_background
from infra.prompt.prompt_tracker import log_using_prompt, snapshot_prompt_used
from infra.llm.service_client import LLMServiceSession
from agenqa.domain.format_schema import (
    FIELD_STEP,
    FIELD_QUESTION,
    FIELD_WORLD_CONTRACT,
    FIELD_SOLUTION,
    FIELD_ANSWER,
    FIELD_VALIDATION_ERRORS,
    FIELD_VALIDATION_PASSED,
    format_output_to_dict,
)
from agenqa.skills.base import BaseSkillRunner
from infra.text.fenced_blocks import extract_preferred_fenced_block
from infra.text.json_sanitize import sanitize_json_text
from infra.text.json_policy import clean_json_text

logger = logging.getLogger(__name__)

_TAG_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass
class FormatInput:
    draft_json: str
    prev_step: int
    step: int
    question_type: Optional[str] = None


@dataclass
class FormatOutput:
    step: int
    question: str
    world_contract: str
    solution: str
    answer: str
    validation_passed: bool
    validation_errors: List[str]


@dataclass
class FormatConfig:
    generator: Dict[str, Any]
    prompt_path: Any  # Path-like
    prompt_text: Optional[str] = None
    lang: Optional[str] = None
    mode: Optional[str] = None
    # 输出协议：json（默认）或 tagged（带字段标记的纯文本）
    protocol: Optional[str] = None


class FormatRunner:
    """给定 FormatInput，调用 LLM 生成 FormatOutput。"""

    def __init__(self, config: FormatConfig) -> None:
        self.config = config
        proto = (getattr(config, "protocol", None) or "json").strip().lower()
        self._protocol: str = proto if proto in ("json", "tagged") else "json"
        resolved = resolve_inference(config.generator)
        self.session: LLMServiceSession = resolved.session
        self._chat_args: Dict[str, Any] = dict(resolved.chat_args)

        base_text = (
            config.prompt_text
            if getattr(config, "prompt_text", None)
            else config.prompt_path.read_text(encoding="utf-8")
        )
        self.prompt_text: str = base_text
        self.prompt_template = Template(base_text)
        log_using_prompt(logger, config.prompt_path)

        try:
            logger.info(
                "FormatRunner configured: model=%s service_id=%s chat_args=%s",
                self.session.model_name,
                self.session.service_id,
                json.dumps(self._chat_args, ensure_ascii=False),
            )
        except Exception:
            pass

    def _build_prompt(self, fmt_in: FormatInput) -> str:
        lang = (self.config.lang or "zh").lower()
        mode_label = (self.config.mode or "Extend").strip() or "Extend"
        payload: Dict[str, Any] = {
            "prev_step": fmt_in.prev_step,
            "step": fmt_in.step,
            "draft_json": fmt_in.draft_json,
            "question_type": getattr(fmt_in, "question_type", None) or "",
        }
        body = self.prompt_template.safe_substitute(payload)
        if lang in {"en", "english"}:
            header = f"Role: Format | Mode: {mode_label} | Language: {lang}"
        else:
            header = f"【Format 角色 · {mode_label}】语言: {lang}"
        return "\n".join(
            [
                header,
                "",
                body,
            ]
        )

    def _parse_output(self, text: str) -> FormatOutput:
        if not text:
            raise ValueError("Format output is empty")
        candidate = text.strip()
        think_end = "</think>"
        if think_end in candidate:
            candidate = candidate.split(think_end, 1)[1].strip()

        fenced = extract_preferred_fenced_block(candidate, preferred_langs=("json",))
        if fenced:
            candidate = fenced

        def _parse_json(payload: str) -> Dict[str, Any]:
            try:
                data = json.loads(payload)
            except json.JSONDecodeError as e:  # noqa: BLE001
                try:
                    data = json.loads(sanitize_json_text(payload))
                except Exception as e2:  # noqa: BLE001
                    logger.error("Format JSON parse failed after sanitization: %s", str(e2))
                    logger.error("Original JSONDecodeError: %s", str(e))
                    logger.error("Payload preview: %s", payload[:300])
                    raise ValueError(f"Invalid Format JSON output: {e2}") from e2
            except Exception as e:  # noqa: BLE001
                logger.error("Format JSON parse failed: %s", str(e))
                logger.error("Payload preview: %s", payload[:300])
                raise ValueError(f"Invalid Format JSON output: {e}") from e
            if not isinstance(data, dict):
                logger.error("Format JSON output is not an object: type=%s", type(data).__name__)
                logger.error("Payload preview: %s", payload[:300])
                raise ValueError("Invalid Format JSON output: not an object")
            return data

        data: Optional[Dict[str, Any]] = None
        json_exc: Exception | None = None
        # Default: prefer JSON; in tagged mode, still try JSON if the output looks like JSON.
        looks_like_json = candidate.lstrip().startswith("{")
        if self._protocol != "tagged" or looks_like_json:
            try:
                data = _parse_json(candidate)
            except Exception as e:  # noqa: BLE001
                data = None
                json_exc = e

        if data is None and self._protocol in ("tagged", "json"):
            tagged = self._parse_tagged(text)
            if tagged is not None:
                return tagged

        if data is None:
            raise ValueError(f"Format output parse failed: {json_exc}") from json_exc

        def _as_str(val: Any) -> str:
            if val is None:
                return ""
            if isinstance(val, (str, int, float, bool)):
                return str(val)
            try:
                return json.dumps(val, ensure_ascii=False)
            except Exception as e:  # noqa: BLE001
                logger.warning("Format: failed to JSON-encode value, using str(): %s", str(e))
                return str(val)

        def _as_list(val: Any) -> List[str]:
            items: List[str] = []
            if val is None:
                return items
            if isinstance(val, list):
                for v in val:
                    s = _as_str(v).strip()
                    if s:
                        items.append(s)
                return items
            s = _as_str(val).strip()
            if s:
                items.append(s)
            return items

        def _as_obj_list(val: Any) -> List[Dict[str, Any]]:
            items: List[Dict[str, Any]] = []
            if val is None:
                return items
            if isinstance(val, list):
                for v in val:
                    if isinstance(v, dict):
                        items.append(v)
                        continue
                    if isinstance(v, str) and v.strip():
                        try:
                            parsed = json.loads(v)
                            if isinstance(parsed, dict):
                                items.append(parsed)
                        except Exception:
                            continue
                return items
            if isinstance(val, dict):
                return [val]
            return items

        try:
            step_val = int(data.get(FIELD_STEP))
        except Exception as e:  # noqa: BLE001
            logger.error("Format JSON missing/invalid step: %s", str(e))
            raise ValueError("Invalid Format output: invalid step") from e

        q = _as_str(data.get(FIELD_QUESTION)).strip()
        wc = _as_str(data.get(FIELD_WORLD_CONTRACT)).strip()
        s = _as_str(data.get(FIELD_SOLUTION)).strip()
        a = _as_str(data.get(FIELD_ANSWER)).strip()
        if not (q and s and a):
            logger.error("Format JSON missing required fields: question/solution/answer")
            raise ValueError("Invalid Format output: missing required fields")

        v_pass = bool(data.get(FIELD_VALIDATION_PASSED))
        v_err = _as_list(data.get(FIELD_VALIDATION_ERRORS))
        if len(v_err) == 1 and v_err[0].strip() in {"[]", "[ ]", "None", "none", "null", "NULL"}:
            v_err = []

        return FormatOutput(
            step=step_val,
            question=q,
            world_contract=wc,
            solution=s,
            answer=a,
            validation_passed=v_pass,
            validation_errors=v_err,
        )

    def _parse_tagged(self, text: str) -> Optional[FormatOutput]:
        """解析带字段标记的纯文本格式。

        约定格式示例：
        [Step] ... [/Step]
        [Question] ... [/Question]
        ...
        """
        if not text:
            return None

        lines = text.splitlines()
        current: Optional[str] = None
        buffers: Dict[str, List[str]] = {}

        fields = {
            FIELD_STEP,
            FIELD_QUESTION,
            FIELD_WORLD_CONTRACT,
            FIELD_SOLUTION,
            FIELD_ANSWER,
            FIELD_VALIDATION_PASSED,
            FIELD_VALIDATION_ERRORS,
        }

        for raw in lines:
            line = raw.strip()
            if not line:
                if current:
                    buffers.setdefault(current, []).append("")
                continue

            if line.startswith("[/") and line.endswith("]") and len(line) > 3:
                name = line[2:-1].strip()
                if _TAG_NAME_RE.match(name) and name in fields:
                    current = None
                    continue

            if line.startswith("[") and line.endswith("]") and len(line) > 2:
                name = line[1:-1].strip()
                if _TAG_NAME_RE.match(name) and name in fields:
                    current = name
                    buffers.setdefault(current, [])
                    continue

            if current:
                buffers.setdefault(current, []).append(raw)

        def _join(name: str) -> str:
            return "\n".join(buffers.get(name, [])).strip()

        def _as_list(name: str) -> List[str]:
            raw_items = buffers.get(name, [])
            items: List[str] = []
            for ln in raw_items:
                s = ln.strip()
                if not s:
                    continue
                if s.startswith("- "):
                    s = s[2:].strip()
                items.append(s)
            return items

        step_raw = _join(FIELD_STEP)
        try:
            step_val = int(step_raw)
        except Exception:
            # Tolerate mild deviations like "This is step 1 ..." by extracting the first integer.
            m = re.search(r"-?\d+", step_raw)
            if not m:
                return None
            try:
                step_val = int(m.group(0))
            except Exception:
                return None

        q = _join(FIELD_QUESTION)
        wc = _join(FIELD_WORLD_CONTRACT)
        s = _join(FIELD_SOLUTION)
        a = _join(FIELD_ANSWER)
        if not (q and s and a):
            return None

        v_pass_raw = _join(FIELD_VALIDATION_PASSED).strip().lower()
        # Accept "true/false" or "validation_passed=true" style variants.
        if "true" in v_pass_raw and "false" not in v_pass_raw:
            v_pass = True
        elif "false" in v_pass_raw and "true" not in v_pass_raw:
            v_pass = False
        else:
            v_pass = v_pass_raw in ("true", "1", "yes", "y")
        v_err = _as_list(FIELD_VALIDATION_ERRORS)
        if len(v_err) == 1 and v_err[0].strip() in {"[]", "[ ]", "None", "none", "null", "NULL"}:
            v_err = []

        return FormatOutput(
            step=step_val,
            question=q,
            world_contract=wc,
            solution=s,
            answer=a,
            validation_passed=v_pass,
            validation_errors=v_err,
        )

    def run_one(
        self,
        fmt_in: FormatInput,
        *,
        snapshot_dir: Optional[Any] = None,
        unified_prompt_dir: Optional[Path] = None,
        name_prefix: str = "prompt_used.format.",
    ) -> Optional[FormatOutput]:
        """对单条 FormatInput 调用一次 LLM，返回 FormatOutput。"""
        prompt_body = self._build_prompt(fmt_in)
        messages = build_messages_with_background(prompt_body, lang=self.config.lang or "zh")
        response = self.session.chat(messages, **self._chat_args)
        BaseSkillRunner._check_finish_reason(response, "Format")
        text = self.session.extract_text(response, default="")

        snap_path: Optional[Path] = None
        if snapshot_dir is not None:
            try:
                snap_path = Path(snapshot_dir)
                snap_path.mkdir(parents=True, exist_ok=True)
                (snap_path / "prompt.txt").write_text(prompt_body, encoding="utf-8")
                try:
                    (snap_path / "response.json").write_text(
                        json.dumps(response, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                except Exception:
                    (snap_path / "response.txt").write_text(str(response), encoding="utf-8")
                (snap_path / "response_text.txt").write_text(text or "", encoding="utf-8")
            except Exception:
                snap_path = None
        if unified_prompt_dir is not None:
            try:
                snapshot_prompt_used(
                    self.config.prompt_path,
                    Path(unified_prompt_dir),
                    content=self.prompt_text,
                    name_prefix=name_prefix,
                    logger=logger,
                )
            except Exception:
                pass

        retry_on_parse_fail = (
            os.getenv("SCICLONE_FORMAT_RETRY_ON_PARSE_FAIL", "1").strip().lower()
            not in {"0", "false", "off", "no"}
        )
        out: FormatOutput
        try:
            out = self._parse_output(text)
        except Exception as first_exc:  # noqa: BLE001
            if not retry_on_parse_fail:
                raise
            cleaned_out: Optional[FormatOutput] = None
            try:
                cleaned_text = clean_json_text(
                    text,
                    generator=self.config.generator,
                    task_name="format",
                    lang=self.config.lang or "zh",
                    required_keys=[
                        FIELD_STEP,
                        FIELD_QUESTION,
                        FIELD_WORLD_CONTRACT,
                        FIELD_SOLUTION,
                        FIELD_ANSWER,
                        FIELD_VALIDATION_PASSED,
                        FIELD_VALIDATION_ERRORS,
                    ],
                    prompt_body=prompt_body,
                    snapshot_dir=snap_path,
                )
                if cleaned_text:
                    cleaned_out = self._parse_output(cleaned_text)
                    if snap_path is not None:
                        try:
                            (snap_path / "response_cleaned_text.txt").write_text(cleaned_text, encoding="utf-8")
                        except Exception:
                            pass
            except Exception:
                cleaned_out = None

            if cleaned_out is not None:
                out = cleaned_out
            else:
                logger.warning("Format parse failed; retrying once with repair prompt: %s", first_exc)
                if self._protocol == "tagged":
                    protocol_req = (
                        "Re-output using the tagged protocol (no extra text). "
                        "All fields must be present and non-empty (except validation_errors may be empty):\n"
                        f"[{FIELD_STEP}]{fmt_in.step}[/{FIELD_STEP}]\n"
                        f"[{FIELD_QUESTION}]...[/Question]\n"
                        f"[{FIELD_WORLD_CONTRACT}]... (may be empty if no extra contract block is needed)[/{FIELD_WORLD_CONTRACT}]\n"
                        f"[{FIELD_SOLUTION}]...[/Solution]\n"
                        f"[{FIELD_ANSWER}]...[/Answer]\n"
                        f"[{FIELD_VALIDATION_PASSED}]true/false[/{FIELD_VALIDATION_PASSED}]\n"
                        f"[{FIELD_VALIDATION_ERRORS}]... (0+ lines, optional)[/{FIELD_VALIDATION_ERRORS}]\n"
                        "\nIf you choose JSON instead, output ONE JSON object with the same fields."
                    )
                else:
                    protocol_req = (
                        "Re-output ONE complete JSON object only (no ``` fences, no extra text) with keys: "
                        f"{FIELD_STEP}, {FIELD_QUESTION}, {FIELD_WORLD_CONTRACT}, {FIELD_SOLUTION}, {FIELD_ANSWER}, "
                        f"{FIELD_VALIDATION_PASSED}, {FIELD_VALIDATION_ERRORS}."
                    )
                repair_prompt = (
                    "Your previous Format output did not follow the required protocol or was empty.\n\n"
                    f"{protocol_req}\n\n"
                    "IMPORTANT:\n"
                    f"- The Step must be exactly {fmt_in.step}.\n"
                    "- Do NOT invent a new question. Only reformat the given Draft JSON.\n"
                    "- Ensure Solution is non-empty.\n\n"
                    "Here is the original Format prompt you must follow (verbatim):\n"
                    "<format_prompt>\n"
                    f"{prompt_body}\n"
                    "</format_prompt>\n\n"
                    "Previous output (for reference, may be empty/truncated):\n"
                    "<previous_output>\n"
                    f"{text}\n"
                    "</previous_output>\n"
                )
                repair_messages = build_messages_with_background(repair_prompt, lang=self.config.lang or "zh")
                repair_args = dict(self._chat_args)
                try:
                    repair_args["temperature"] = 0.2
                except Exception:
                    pass
                repair_response = self.session.chat(repair_messages, **repair_args)
                BaseSkillRunner._check_finish_reason(repair_response, "Format(repair)")
                repair_text = self.session.extract_text(repair_response, default="")
                if snap_path is not None:
                    try:
                        (snap_path / "response_repair.json").write_text(
                            json.dumps(repair_response, ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                    except Exception:
                        (snap_path / "response_repair.txt").write_text(str(repair_response), encoding="utf-8")
                    (snap_path / "response_repair_text.txt").write_text(repair_text or "", encoding="utf-8")
                out = self._parse_output(repair_text)
        if snap_path is not None:
            try:
                parsed_payload = format_output_to_dict(out)
                (snap_path / "parsed.json").write_text(
                    json.dumps(parsed_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception:
                pass
        return out
