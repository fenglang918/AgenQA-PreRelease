"""Diagnose 角色：针对单道题做诊断分析（Revise 前置模块）。

仅负责识别问题与给出修复建议，不直接重写题目。
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any, Dict, List, Optional

from infra.llm.inference import resolve_inference
from infra.prompt.prompt_builder import build_messages_with_background
from infra.prompt.prompt_tracker import log_using_prompt, snapshot_prompt_used
from infra.llm.service_client import LLMServiceSession
from agenqa.skills.base import BaseSkillRunner
from infra.text.fenced_blocks import extract_preferred_fenced_block
from infra.text.json_sanitize import sanitize_json_text
from infra.text.json_policy import clean_json_text
from agenqa.domain.diagnose_schema import (
    FIELD_ISSUES,
    FIELD_FIX_SUGGESTIONS,
    FIELD_DIAGNOSIS,
    diagnose_output_to_dict,
)

logger = logging.getLogger(__name__)

_TAG_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass
class DiagnoseInput:
    known_0: str
    question: str
    answer: str
    solver_feedback: Optional[str]
    director_notes: Optional[str]
    solver_answers: Optional[str]
    background: Optional[str] = None  # 累积的背景条目（可选）
    solver_reasoning: Optional[str] = None  # solver 推导过程摘要（可选）


@dataclass
class DiagnoseOutput:
    issues: List[str]
    fix_suggestions: List[str]
    diagnosis: str


@dataclass
class DiagnoseConfig:
    generator: Dict[str, Any]
    prompt_path: Any  # Path-like
    prompt_text: Optional[str] = None
    lang: Optional[str] = None
    # 输出协议：json（默认）或 tagged（带字段标记的纯文本）
    protocol: Optional[str] = None


class DiagnoseRunner:
    """给定 DiagnoseInput，调用 LLM 生成 DiagnoseOutput。"""

    def __init__(self, config: DiagnoseConfig) -> None:
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
                "DiagnoseRunner configured: model=%s service_id=%s chat_args=%s",
                self.session.model_name,
                self.session.service_id,
                json.dumps(self._chat_args, ensure_ascii=False),
            )
        except Exception:
            pass

    def _build_prompt(self, diag_in: DiagnoseInput) -> str:
        lang = (self.config.lang or "zh").lower()
        use_en = lang in {"en", "english"}
        # If premise summary is empty, show a placeholder.
        if use_en:
            background_text = diag_in.background if diag_in.background else "(no premise summary yet)"
            reasoning_text = diag_in.solver_reasoning if diag_in.solver_reasoning else "(no solver reasoning summary yet)"
            solver_answers_text = diag_in.solver_answers if diag_in.solver_answers else "(no solver answer records yet)"
            header = f"Role: Diagnose | Mode: Revise | Language: {lang}"
        else:
            background_text = diag_in.background if diag_in.background else "（当前无前提摘要）"
            reasoning_text = diag_in.solver_reasoning if diag_in.solver_reasoning else "（当前无 solver 推导过程摘要）"
            solver_answers_text = diag_in.solver_answers if diag_in.solver_answers else "（当前无 solver 答案记录）"
            header = f"【Diagnose 角色 · Revise】语言: {lang}"
        payload: Dict[str, Any] = {
            "known_0": diag_in.known_0,
            "question": diag_in.question,
            "answer": diag_in.answer,
            "solver_feedback": diag_in.solver_feedback or "",
            "director_notes": diag_in.director_notes or "",
            "solver_answers": solver_answers_text,
            "background": background_text,
            "solver_reasoning": reasoning_text,
        }
        body = self.prompt_template.safe_substitute(payload)
        return "\n".join(
            [
                header,
                "",
                body,
            ]
        )

    def _parse_output(self, text: str) -> DiagnoseOutput:
        if not text:
            raise ValueError("Diagnose output is empty")
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
                    logger.error("Diagnose JSON parse failed after sanitization: %s", str(e2))
                    logger.error("Original JSONDecodeError: %s", str(e))
                    logger.error("Payload preview: %s", payload[:300])
                    raise ValueError(f"Invalid Diagnose JSON output: {e2}") from e2
            except Exception as e:  # noqa: BLE001
                logger.error("Diagnose JSON parse failed: %s", str(e))
                logger.error("Payload preview: %s", payload[:300])
                raise ValueError(f"Invalid Diagnose JSON output: {e}") from e
            if not isinstance(data, dict):
                logger.error("Diagnose JSON output is not an object: type=%s", type(data).__name__)
                raise ValueError("Invalid Diagnose JSON output: not an object")
            return data

        data: Optional[Dict[str, Any]] = None
        json_exc: Exception | None = None
        # 默认优先 JSON；在 tagged 模式下若 JSON 失败则尝试带标记格式
        if self._protocol != "tagged":
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
            raise ValueError(f"Diagnose output parse failed: {json_exc}") from json_exc

        def _as_str(val: Any) -> str:
            if val is None:
                return ""
            if isinstance(val, (str, int, float, bool)):
                return str(val)
            try:
                return json.dumps(val, ensure_ascii=False)
            except Exception as e:  # noqa: BLE001
                logger.warning("Diagnose: failed to JSON-encode value, using str(): %s", str(e))
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

        issues = _as_list(data.get(FIELD_ISSUES))
        fix_suggestions = _as_list(data.get(FIELD_FIX_SUGGESTIONS))
        diagnosis = _as_str(data.get(FIELD_DIAGNOSIS)).strip()
        if not diagnosis:
            logger.error("Diagnose JSON missing required field: diagnosis")
            raise ValueError("Invalid Diagnose output: missing diagnosis")

        return DiagnoseOutput(
            issues=issues,
            fix_suggestions=fix_suggestions,
            diagnosis=diagnosis,
        )

    def _parse_tagged(self, text: str) -> Optional[DiagnoseOutput]:
        """解析带字段标记的纯文本格式。

        约定格式示例：
        [issues] ... [/issues]
        [fix_suggestions] ... [/fix_suggestions]
        [diagnosis] ... [/diagnosis]
        """
        if not text:
            return None

        lines = text.splitlines()
        current: Optional[str] = None
        buffers: Dict[str, List[str]] = {}

        fields = {
            FIELD_ISSUES,
            FIELD_FIX_SUGGESTIONS,
            FIELD_DIAGNOSIS,
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

        issues = _as_list(FIELD_ISSUES)
        fix_suggestions = _as_list(FIELD_FIX_SUGGESTIONS)
        diagnosis = _join(FIELD_DIAGNOSIS)
        if not diagnosis:
            return None

        return DiagnoseOutput(
            issues=issues,
            fix_suggestions=fix_suggestions,
            diagnosis=diagnosis,
        )

    def run_one(
        self,
        diag_in: DiagnoseInput,
        *,
        snapshot_dir: Optional[Any] = None,
        unified_prompt_dir: Optional[Path] = None,
        name_prefix: str = "prompt_used.diagnose_revise.diagnose.",
    ) -> Optional[DiagnoseOutput]:
        prompt_body = self._build_prompt(diag_in)
        messages = build_messages_with_background(prompt_body, lang=self.config.lang or "zh")
        response = self.session.chat(messages, **self._chat_args)
        BaseSkillRunner._check_finish_reason(response, "Diagnose")
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

        try:
            out = self._parse_output(text)
        except Exception as first_exc:  # noqa: BLE001
            cleaned_text = clean_json_text(
                text,
                generator=self.config.generator,
                task_name="diagnose",
                lang=self.config.lang or "zh",
                required_keys=[FIELD_ISSUES, FIELD_FIX_SUGGESTIONS, FIELD_DIAGNOSIS],
                prompt_body=prompt_body,
                snapshot_dir=snap_path,
            )
            if not cleaned_text:
                raise
            try:
                out = self._parse_output(cleaned_text)
            except Exception:
                raise first_exc
        if snap_path is not None:
            try:
                parsed_payload = diagnose_output_to_dict(out)
                (snap_path / "parsed.json").write_text(
                    json.dumps(parsed_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception:
                pass
        return out
