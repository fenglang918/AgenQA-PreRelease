"""ExecutableReviseStep runner: revise the current executable step based on Diagnose output."""

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
from agenqa.domain.executable_schema import (
    ExecutableDraftOutput,
    ExecutableSubStep,
    FIELD_DEPENDENCIES,
    FIELD_GOLDEN_STEP_CODE,
    FIELD_STEP_NUMBER,
    FIELD_SUB_STEP,
)
from agenqa.skills.base import BaseSkillRunner
from infra.text.fenced_blocks import extract_preferred_fenced_block
from infra.text.json_sanitize import sanitize_json_text

from agenqa.prompts.executable_revise_step import EXECUTABLE_REVISE_STEP_V1, EXECUTABLE_REVISE_STEP_V1_EN

logger = logging.getLogger(__name__)


@dataclass
class ExecutableReviseStepInput:
    step: int
    director_notes: str
    task_sketch: str
    background: str
    prev_sub_steps: List[ExecutableSubStep]
    current_sub_step: ExecutableSubStep
    current_golden_step_code: str
    diagnose_json: str
    dependencies_whitelist: str


@dataclass
class ExecutableReviseStepConfig:
    generator: Dict[str, Any]
    prompt_path: Any
    prompt_text: Optional[str] = None
    lang: Optional[str] = None


class ExecutableReviseStepRunner:
    def __init__(self, config: ExecutableReviseStepConfig) -> None:
        self.config = config
        resolved = resolve_inference(config.generator)
        self.session: LLMServiceSession = resolved.session
        self._chat_args: Dict[str, Any] = dict(resolved.chat_args)

        lang_norm = (config.lang or "").strip().lower()
        use_en = lang_norm in {"en", "english"}
        base_text = config.prompt_text if getattr(config, "prompt_text", None) else None
        if base_text is None:
            base_text = EXECUTABLE_REVISE_STEP_V1_EN if use_en else EXECUTABLE_REVISE_STEP_V1
        self.prompt_text = base_text
        self.prompt_template = Template(base_text)
        log_using_prompt(logger, Path(str(config.prompt_path)))

    def _build_prompt(self, revise_in: ExecutableReviseStepInput) -> str:
        prev_steps_payload = [s.to_dict() for s in revise_in.prev_sub_steps]
        payload = {
            "step": int(revise_in.step),
            "director_notes": revise_in.director_notes or "",
            "task_sketch": revise_in.task_sketch or "",
            "background": revise_in.background or "",
            "prev_sub_steps_json": json.dumps(prev_steps_payload, ensure_ascii=False),
            "current_sub_step_json": json.dumps(revise_in.current_sub_step.to_dict(), ensure_ascii=False),
            "current_golden_step_code": revise_in.current_golden_step_code or "",
            "diagnose_json": revise_in.diagnose_json or "",
            "dependencies_whitelist": revise_in.dependencies_whitelist or "",
        }
        body = self.prompt_template.safe_substitute(payload)
        lang = (self.config.lang or "zh").lower()
        header = (
            f"Role: ExecutableReviseStep | Language: {lang}"
            if lang in {"en", "english"}
            else f"【ExecutableReviseStep 角色】语言: {lang}"
        )
        return "\n".join([header, "", body])

    def _parse_output(self, text: str) -> ExecutableDraftOutput:
        if not text or not str(text).strip():
            raise ValueError("ExecutableReviseStep output is empty")
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
            raise ValueError("ExecutableReviseStep JSON output is not an object")

        step_number = str(data.get(FIELD_STEP_NUMBER) or "")
        sub_step_raw = data.get(FIELD_SUB_STEP) or {}
        if not isinstance(sub_step_raw, dict):
            raise ValueError("ExecutableReviseStep sub_step is missing or invalid")
        sub_step = ExecutableSubStep.from_dict(sub_step_raw)
        if step_number:
            sub_step.step_number = step_number

        golden_step_code = str(data.get(FIELD_GOLDEN_STEP_CODE) or "")
        if not golden_step_code.strip():
            raise ValueError("ExecutableReviseStep golden_step_code is empty")
        dependencies = str(data.get(FIELD_DEPENDENCIES) or "")

        return ExecutableDraftOutput(
            step_number=step_number or sub_step.step_number,
            sub_step=sub_step,
            golden_step_code=golden_step_code,
            dependencies=dependencies,
        )

    def run_one(
        self,
        revise_in: ExecutableReviseStepInput,
        *,
        snapshot_dir: Optional[Any] = None,
        unified_prompt_dir: Optional[Path] = None,
        name_prefix: str = "prompt_used.executable_revise_step.",
    ) -> ExecutableDraftOutput:
        prompt_body = self._build_prompt(revise_in)
        messages = build_messages_with_background(prompt_body, lang=self.config.lang or "zh")
        response = self.session.chat(messages, **self._chat_args)
        BaseSkillRunner._check_finish_reason(response, "ExecutableReviseStep")
        text = self.session.extract_text(response, default="")
        out = self._parse_output(text)

        if snapshot_dir is not None:
            try:
                snap_path = Path(snapshot_dir)
                snap_path.mkdir(parents=True, exist_ok=True)
                (snap_path / "input_view.json").write_text(
                    json.dumps(
                        {
                            "step": revise_in.step,
                            "director_notes": revise_in.director_notes,
                            "task_sketch": revise_in.task_sketch,
                            "background": revise_in.background,
                            "prev_sub_steps": [s.to_dict() for s in revise_in.prev_sub_steps],
                            "current_sub_step": revise_in.current_sub_step.to_dict(),
                            "current_golden_step_code": revise_in.current_golden_step_code,
                            "diagnose_json": revise_in.diagnose_json,
                            "dependencies_whitelist": revise_in.dependencies_whitelist,
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
                logger.warning("ExecutableReviseStep snapshot write failed: %s", str(exc))

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
    "ExecutableReviseStepInput",
    "ExecutableReviseStepConfig",
    "ExecutableReviseStepRunner",
]
