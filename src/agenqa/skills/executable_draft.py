"""ExecutableDraft runner: generate the next executable sub-step and its golden code."""

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
from infra.prompt.prompt_tracker import log_using_prompt, snapshot_prompt_used, snapshot_rendered_prompt
from infra.llm.service_client import LLMServiceSession
from infra.text.json_policy import clean_json_text
from agenqa.domain.executable_schema import (
    ExecutableDraftOutput,
    ExecutableSubStep,
    FIELD_STEP_NUMBER,
    FIELD_SUB_STEP,
    FIELD_GOLDEN_STEP_CODE,
    FIELD_DEPENDENCIES,
    FIELD_REQUIRED_FACT_IDS,
    FIELD_PRIMARY_REQUIRED_FACT_ID,
    FIELD_REUSE_PLAN,
)
from agenqa.skills.base import BaseSkillRunner
from infra.text.fenced_blocks import extract_preferred_fenced_block
from infra.text.json_sanitize import sanitize_json_text
from agenqa.skills.executable_json_format import ExecutableJsonFormatConfig, ExecutableJsonFormatInput, ExecutableJsonFormatRunner

from agenqa.prompts.executable_draft_step import EXECUTABLE_DRAFT_STEP_V1, EXECUTABLE_DRAFT_STEP_V1_EN

logger = logging.getLogger(__name__)

_DEF_OR_CLASS_LINE_RE = re.compile(r"^\s*(def|class)\s+([A-Za-z_][A-Za-z0-9_]*)\s*(\(|:)", re.ASCII)


def _infer_function_header_from_code(code: str) -> str:
    if not isinstance(code, str) or not code.strip():
        return ""
    lines = code.splitlines()
    start = None
    parens = 0
    for idx, line in enumerate(lines):
        if _DEF_OR_CLASS_LINE_RE.match(line):
            start = idx
            break
    if start is None:
        return ""
    buf: List[str] = []
    for line in lines[start:]:
        raw = line.rstrip()
        buf.append(raw)
        parens += raw.count("(") - raw.count(")")
        # Stop after the signature terminator.
        if parens <= 0 and raw.endswith(":"):
            break
    return "\n".join(buf).strip()


@dataclass
class ExecutableDraftInput:
    step: int
    director_notes: str
    task_sketch: str
    background: str
    prev_sub_steps: List[ExecutableSubStep]
    dependencies_whitelist: str
    expected_primary_fact_id: str = ""


@dataclass
class ExecutableDraftConfig:
    generator: Dict[str, Any]
    prompt_path: Any
    format_generator: Optional[Dict[str, Any]] = None
    prompt_text: Optional[str] = None
    lang: Optional[str] = None


class ExecutableDraftRunner:
    def __init__(self, config: ExecutableDraftConfig) -> None:
        self.config = config
        resolved = resolve_inference(config.generator)
        self.session: LLMServiceSession = resolved.session
        self._chat_args: Dict[str, Any] = dict(resolved.chat_args)

        lang_norm = (config.lang or "").strip().lower()
        use_en = lang_norm in {"en", "english"}
        base_text = config.prompt_text if getattr(config, "prompt_text", None) else None
        if base_text is None:
            base_text = EXECUTABLE_DRAFT_STEP_V1_EN if use_en else EXECUTABLE_DRAFT_STEP_V1
        self.prompt_text = base_text
        self.prompt_template = Template(base_text)
        log_using_prompt(logger, Path(str(config.prompt_path)))

    def _build_prompt(self, draft_in: ExecutableDraftInput) -> str:
        prev_steps_payload = [s.to_dict() for s in draft_in.prev_sub_steps]
        payload = {
            "step": int(draft_in.step),
            "director_notes": draft_in.director_notes or "",
            "task_sketch": draft_in.task_sketch or "",
            "background": draft_in.background or "",
            "prev_sub_steps_json": json.dumps(prev_steps_payload, ensure_ascii=False),
            "dependencies_whitelist": draft_in.dependencies_whitelist or "",
            "expected_primary_fact_id": draft_in.expected_primary_fact_id or "",
        }
        body = self.prompt_template.safe_substitute(payload)
        lang = (self.config.lang or "zh").lower()
        header = f"Role: ExecutableDraftStep | Language: {lang}" if lang in {"en", "english"} else f"【ExecutableDraftStep 角色】语言: {lang}"
        return "\n".join([header, "", body])

    def _parse_output(self, text: str) -> ExecutableDraftOutput:
        if not text or not str(text).strip():
            raise ValueError("ExecutableDraftStep output is empty")
        candidate = str(text).strip()
        think_end = "</think>"
        if think_end in candidate:
            candidate = candidate.split(think_end, 1)[1].strip()

        fenced = extract_preferred_fenced_block(candidate, preferred_langs=("json",))
        if fenced:
            candidate = fenced

        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            data = json.loads(sanitize_json_text(candidate))
        if not isinstance(data, dict):
            raise ValueError("ExecutableDraftStep JSON output is not an object")

        step_number = str(data.get(FIELD_STEP_NUMBER) or "")
        sub_step_raw = data.get(FIELD_SUB_STEP) or {}
        if not isinstance(sub_step_raw, dict):
            raise ValueError("ExecutableDraftStep sub_step is missing or invalid")
        sub_step = ExecutableSubStep.from_dict(sub_step_raw)
        if step_number:
            sub_step.step_number = step_number
        # Backward/robust mapping: some models may output an alternative schema.
        if (not sub_step.step_description.strip()) and isinstance(sub_step_raw.get("description"), str):
            sub_step.step_description = str(sub_step_raw.get("description") or "")

        golden_step_code = str(data.get(FIELD_GOLDEN_STEP_CODE) or "")
        if not golden_step_code.strip():
            raise ValueError("ExecutableDraftStep golden_step_code is empty")
        dependencies = str(data.get(FIELD_DEPENDENCIES) or "")

        required_fact_ids: list[str] = []
        raw_required = data.get(FIELD_REQUIRED_FACT_IDS)
        if isinstance(raw_required, list):
            required_fact_ids = [str(x) for x in raw_required if str(x).strip()]
        primary_required_fact_id = str(data.get(FIELD_PRIMARY_REQUIRED_FACT_ID) or "").strip() or None
        reuse_plan: list[str] = []
        raw_plan = data.get(FIELD_REUSE_PLAN)
        if isinstance(raw_plan, list):
            reuse_plan = [str(x) for x in raw_plan if str(x).strip()]
        if not sub_step.function_header.strip():
            inferred = _infer_function_header_from_code(golden_step_code)
            if inferred:
                sub_step.function_header = inferred

        return ExecutableDraftOutput(
            step_number=step_number or sub_step.step_number,
            sub_step=sub_step,
            golden_step_code=golden_step_code,
            dependencies=dependencies,
            required_fact_ids=required_fact_ids,
            primary_required_fact_id=primary_required_fact_id,
            reuse_plan=reuse_plan,
        )

    def run_one(
        self,
        draft_in: ExecutableDraftInput,
        *,
        snapshot_dir: Optional[Any] = None,
        unified_prompt_dir: Optional[Path] = None,
        name_prefix: str = "prompt_used.executable_draft_step.",
    ) -> ExecutableDraftOutput:
        prompt_body = self._build_prompt(draft_in)
        messages = build_messages_with_background(prompt_body, lang=self.config.lang or "zh")
        response = self.session.chat(messages, **self._chat_args)
        BaseSkillRunner._check_finish_reason(response, "ExecutableDraftStep")
        text = self.session.extract_text(response, default="")

        snap_path: Optional[Path] = None
        if snapshot_dir is not None:
            try:
                snap_path = Path(snapshot_dir)
                snap_path.mkdir(parents=True, exist_ok=True)
                input_payload = {
                    "step": draft_in.step,
                    "director_notes": draft_in.director_notes,
                    "task_sketch": draft_in.task_sketch,
                    "background": draft_in.background,
                    "prev_sub_steps": [s.to_dict() for s in draft_in.prev_sub_steps],
                    "dependencies_whitelist": draft_in.dependencies_whitelist,
                }
                (snap_path / "input_view.json").write_text(
                    json.dumps(input_payload, ensure_ascii=False, indent=2),
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
                logger.warning("ExecutableDraftStep snapshot write failed: %s", str(exc))
                snap_path = None

        required_keys = [
            FIELD_STEP_NUMBER,
            FIELD_SUB_STEP,
            FIELD_GOLDEN_STEP_CODE,
            FIELD_DEPENDENCIES,
            FIELD_REQUIRED_FACT_IDS,
            FIELD_PRIMARY_REQUIRED_FACT_ID,
            FIELD_REUSE_PLAN,
        ]
        cleaned = clean_json_text(
            text or "",
            generator=self.config.generator,
            task_name="ExecutableDraftStep",
            lang=(self.config.lang or "zh"),
            required_keys=required_keys,
            prompt_body=prompt_body,
            snapshot_dir=(snap_path / "json_policy") if snap_path is not None else None,
            allow_python=False,
        )
        out: ExecutableDraftOutput
        try:
            out = self._parse_output(cleaned if cleaned else text)
        except Exception as first_exc:  # noqa: BLE001
            fmt_gen = self.config.format_generator or self.config.generator
            fmt_runner = ExecutableJsonFormatRunner(
                ExecutableJsonFormatConfig(
                    generator=fmt_gen if isinstance(fmt_gen, dict) else {},
                    prompt_path=Path("src/agenqa/prompts/executable_json_format.prompt"),
                    lang=self.config.lang,
                    max_retries=1,
                )
            )
            fmt_text = fmt_runner.run_one(
                ExecutableJsonFormatInput(
                    task_name="ExecutableDraftStep",
                    required_keys=required_keys,
                    raw_output=str(text or ""),
                    original_prompt=prompt_body,
                    parse_error=str(first_exc),
                ),
                snapshot_dir=(snap_path.parent / "executable_json_format.executable_draft_step") if snap_path is not None else None,
                unified_prompt_dir=unified_prompt_dir,
            )
            out = self._parse_output(fmt_text)

        if snapshot_dir is not None:
            try:
                if snap_path is None:
                    snap_path = Path(snapshot_dir)
                    snap_path.mkdir(parents=True, exist_ok=True)
                (snap_path / "output.json").write_text(
                    json.dumps(
                        {
                            FIELD_STEP_NUMBER: out.step_number,
                            FIELD_SUB_STEP: out.sub_step.to_dict(),
                            FIELD_GOLDEN_STEP_CODE: out.golden_step_code,
                            FIELD_DEPENDENCIES: out.dependencies,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("ExecutableDraftStep snapshot write failed: %s", str(exc))

        if unified_prompt_dir is not None:
            try:
                snapshot_prompt_used(
                    Path(str(self.config.prompt_path)),
                    Path(unified_prompt_dir),
                    content=self.prompt_text,
                    name_prefix=name_prefix,
                    logger=logger,
                )
            except Exception:
                pass
        return out


__all__ = [
    "ExecutableDraftInput",
    "ExecutableDraftConfig",
    "ExecutableDraftRunner",
]
