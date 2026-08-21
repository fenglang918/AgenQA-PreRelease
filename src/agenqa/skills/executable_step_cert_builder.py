"""ExecutableStepCertBuilder role: produce premise/fact/cert for executable track.

This aligns executable track with the semantic pipeline's KnownTree v2 mechanism by
emitting StepCert-style outputs: premise_delta / fact_delta / step_cert / key_fact_id.
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
from infra.text.json_policy import clean_json_text
from agenqa.domain.executable_schema import ExecutableSubStep
from agenqa.domain.step_cert_schema import (
    FIELD_FACT_DELTA,
    FIELD_KEY_FACT_ID,
    FIELD_PREMISE_DELTA,
    FIELD_STEP_CERT,
)
from agenqa.skills.base import BaseSkillRunner
from infra.text.fenced_blocks import extract_preferred_fenced_block
from infra.text.json_sanitize import sanitize_json_text
from agenqa.skills.executable_json_format import ExecutableJsonFormatConfig, ExecutableJsonFormatInput, ExecutableJsonFormatRunner

from agenqa.prompts.executable_step_cert_builder import (
    EXECUTABLE_STEP_CERT_BUILDER_V1,
    EXECUTABLE_STEP_CERT_BUILDER_V1_EN,
)

logger = logging.getLogger(__name__)


@dataclass
class ExecutableStepCertBuilderInput:
    step: int
    director_notes: str
    task_sketch: str
    background: str
    prev_sub_steps: List[ExecutableSubStep]
    tail_sub_step: ExecutableSubStep
    golden_step_code: str
    memory_json: str
    expected_primary_fact_id: str = ""
    observed_required_fact_ids: List[str] | None = None
    observed_primary_required_fact_id: str = ""


@dataclass
class ExecutableStepCertBuilderOutput:
    premise_delta: List[Dict[str, Any]]
    fact_delta: List[Dict[str, Any]]
    step_cert: Dict[str, Any]
    key_fact_id: str


@dataclass
class ExecutableStepCertBuilderConfig:
    generator: Dict[str, Any]
    prompt_path: Any
    format_generator: Optional[Dict[str, Any]] = None
    prompt_text: Optional[str] = None
    lang: Optional[str] = None


class ExecutableStepCertBuilderRunner:
    def __init__(self, config: ExecutableStepCertBuilderConfig) -> None:
        self.config = config
        resolved = resolve_inference(config.generator)
        self.session: LLMServiceSession = resolved.session
        self._chat_args: Dict[str, Any] = dict(resolved.chat_args)

        lang_norm = (config.lang or "").strip().lower()
        use_en = lang_norm in {"en", "english"}
        base_text = config.prompt_text if getattr(config, "prompt_text", None) else None
        if base_text is None:
            base_text = EXECUTABLE_STEP_CERT_BUILDER_V1_EN if use_en else EXECUTABLE_STEP_CERT_BUILDER_V1
        self.prompt_text = base_text
        self.prompt_template = Template(base_text)
        log_using_prompt(logger, Path(str(config.prompt_path)))

    def _build_prompt(self, cert_in: ExecutableStepCertBuilderInput) -> str:
        prev_steps_payload = [s.to_dict() for s in (cert_in.prev_sub_steps or [])]
        payload: Dict[str, Any] = {
            "step": int(cert_in.step),
            "director_notes": cert_in.director_notes or "",
            "task_sketch": cert_in.task_sketch or "",
            "background": cert_in.background or "",
            "prev_sub_steps_json": json.dumps(prev_steps_payload, ensure_ascii=False),
            "tail_sub_step_json": json.dumps(cert_in.tail_sub_step.to_dict(), ensure_ascii=False),
            "golden_step_code": cert_in.golden_step_code or "",
            "memory_json": cert_in.memory_json or "",
            "expected_primary_fact_id": cert_in.expected_primary_fact_id or "",
            "observed_required_fact_ids_json": json.dumps(list(cert_in.observed_required_fact_ids or []), ensure_ascii=False),
            "observed_primary_required_fact_id": cert_in.observed_primary_required_fact_id or "",
        }
        body = self.prompt_template.safe_substitute(payload)
        lang = (self.config.lang or "zh").lower()
        header = (
            f"Role: ExecutableStepCertBuilder | Language: {lang}"
            if lang in {"en", "english"}
            else f"【ExecutableStepCertBuilder 角色】语言: {lang}"
        )
        return "\n".join([header, "", body])

    def _parse_output(self, text: str) -> ExecutableStepCertBuilderOutput:
        if not text or not str(text).strip():
            raise ValueError("ExecutableStepCertBuilder output is empty")
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
            raise ValueError("ExecutableStepCertBuilder JSON output is not an object")

        premise_delta = data.get(FIELD_PREMISE_DELTA) or []
        fact_delta = data.get(FIELD_FACT_DELTA) or []
        step_cert = data.get(FIELD_STEP_CERT) or {}
        key_fact_id = data.get(FIELD_KEY_FACT_ID)

        if not isinstance(premise_delta, list):
            raise ValueError("premise_delta must be a list")
        if not isinstance(fact_delta, list):
            raise ValueError("fact_delta must be a list")
        if not isinstance(step_cert, dict):
            raise ValueError("step_cert must be an object")
        if not isinstance(key_fact_id, str) or not key_fact_id.strip():
            raise ValueError("key_fact_id must be a non-empty string")

        for entry in premise_delta:
            if not isinstance(entry, dict):
                raise ValueError("premise_delta entries must be objects")
            if not isinstance(entry.get("id"), str) or not str(entry.get("id") or "").strip():
                raise ValueError("premise_delta entries must include non-empty id")
        for entry in fact_delta:
            if not isinstance(entry, dict):
                raise ValueError("fact_delta entries must be objects")
            if not isinstance(entry.get("id"), str) or not str(entry.get("id") or "").strip():
                raise ValueError("fact_delta entries must include non-empty id")

        return ExecutableStepCertBuilderOutput(
            premise_delta=premise_delta,
            fact_delta=fact_delta,
            step_cert=step_cert,
            key_fact_id=str(key_fact_id).strip(),
        )

    def run_one(
        self,
        cert_in: ExecutableStepCertBuilderInput,
        *,
        snapshot_dir: Optional[Any] = None,
        unified_prompt_dir: Optional[Path] = None,
        name_prefix: str = "prompt_used.executable_step_cert_builder.",
    ) -> ExecutableStepCertBuilderOutput:
        prompt_body = self._build_prompt(cert_in)
        messages = build_messages_with_background(prompt_body, lang=self.config.lang or "zh")
        response = self.session.chat(messages, **self._chat_args)
        BaseSkillRunner._check_finish_reason(response, "ExecutableStepCertBuilder")
        text = self.session.extract_text(response, default="")
        text = text or ""

        snap_path: Optional[Path] = None
        if snapshot_dir is not None:
            try:
                snap_path = Path(snapshot_dir)
                snap_path.mkdir(parents=True, exist_ok=True)
                input_payload = {
                    "step": cert_in.step,
                    "director_notes": cert_in.director_notes,
                    "task_sketch": cert_in.task_sketch,
                    "background": cert_in.background,
                    "prev_sub_steps": [s.to_dict() for s in (cert_in.prev_sub_steps or [])],
                    "tail_sub_step": cert_in.tail_sub_step.to_dict(),
                    "memory_json": cert_in.memory_json,
                    "expected_primary_fact_id": cert_in.expected_primary_fact_id,
                    "observed_required_fact_ids": list(cert_in.observed_required_fact_ids or []),
                    "observed_primary_required_fact_id": cert_in.observed_primary_required_fact_id,
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
                logger.warning("ExecutableStepCertBuilder snapshot write failed: %s", str(exc))
                snap_path = None

        required_keys = [FIELD_PREMISE_DELTA, FIELD_FACT_DELTA, FIELD_STEP_CERT, FIELD_KEY_FACT_ID]
        cleaned = clean_json_text(
            text,
            generator=self.config.generator,
            task_name="ExecutableStepCertBuilder",
            lang=(self.config.lang or "zh"),
            required_keys=required_keys,
            prompt_body=prompt_body,
            snapshot_dir=(snap_path / "json_policy") if snap_path is not None else None,
            allow_python=False,
        )
        out: ExecutableStepCertBuilderOutput
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
                    task_name="ExecutableStepCertBuilder",
                    required_keys=required_keys,
                    raw_output=str(text or ""),
                    original_prompt=prompt_body,
                    parse_error=str(first_exc),
                ),
                snapshot_dir=(snap_path.parent / "executable_json_format.executable_step_cert_builder") if snap_path is not None else None,
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
                            FIELD_PREMISE_DELTA: out.premise_delta,
                            FIELD_FACT_DELTA: out.fact_delta,
                            FIELD_STEP_CERT: out.step_cert,
                            FIELD_KEY_FACT_ID: out.key_fact_id,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("ExecutableStepCertBuilder snapshot write failed: %s", str(exc))

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
    "ExecutableStepCertBuilderConfig",
    "ExecutableStepCertBuilderInput",
    "ExecutableStepCertBuilderOutput",
    "ExecutableStepCertBuilderRunner",
]
