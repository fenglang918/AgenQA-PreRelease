"""ExecutableExtract runner: extract a executable task skeleton from paper background."""

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
    ExecutableExtractOutput,
    ExecutableSubStep,
    FIELD_EXECUTABLE_SUITABLE,
    FIELD_NOTES,
    FIELD_TASK_SKETCH,
    FIELD_INITIAL_SUB_STEPS,
    FIELD_ESTIMATED_DIFFICULTY,
)
from agenqa.skills.base import BaseSkillRunner
from infra.text.fenced_blocks import extract_preferred_fenced_block
from infra.text.json_sanitize import sanitize_json_text

from agenqa.prompts.executable_extract import EXECUTABLE_EXTRACT_V1, EXECUTABLE_EXTRACT_V1_EN

logger = logging.getLogger(__name__)


@dataclass
class ExecutableExtractInput:
    director_notes: str
    paper_background: str
    problem_description: str
    dependencies_whitelist: str


@dataclass
class ExecutableExtractConfig:
    generator: Dict[str, Any]
    prompt_path: Any
    prompt_text: Optional[str] = None
    lang: Optional[str] = None


class ExecutableExtractRunner:
    def __init__(self, config: ExecutableExtractConfig) -> None:
        self.config = config
        resolved = resolve_inference(config.generator)
        self.session: LLMServiceSession = resolved.session
        self._chat_args: Dict[str, Any] = dict(resolved.chat_args)

        lang_norm = (config.lang or "").strip().lower()
        use_en = lang_norm in {"en", "english"}
        base_text = config.prompt_text if getattr(config, "prompt_text", None) else None
        if base_text is None:
            base_text = EXECUTABLE_EXTRACT_V1_EN if use_en else EXECUTABLE_EXTRACT_V1
        self.prompt_text = base_text
        self.prompt_template = Template(base_text)
        log_using_prompt(logger, Path(str(config.prompt_path)))

    def _build_prompt(self, extract_in: ExecutableExtractInput) -> str:
        payload = {
            "director_notes": extract_in.director_notes or "",
            "paper_background": extract_in.paper_background or "",
            "problem_description": extract_in.problem_description or "",
            "dependencies_whitelist": extract_in.dependencies_whitelist or "",
        }
        body = self.prompt_template.safe_substitute(payload)
        lang = (self.config.lang or "zh").lower()
        header = f"Role: ExecutableExtract | Language: {lang}" if lang in {"en", "english"} else f"【ExecutableExtract 角色】语言: {lang}"
        return "\n".join([header, "", body])

    def _parse_output(self, text: str) -> ExecutableExtractOutput:
        if not text or not str(text).strip():
            raise ValueError("ExecutableExtract output is empty")
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
            raise ValueError("ExecutableExtract JSON output is not an object")

        suitable_raw = data.get(FIELD_EXECUTABLE_SUITABLE)
        executable_suitable = bool(suitable_raw) if suitable_raw is not None else False
        notes = str(data.get(FIELD_NOTES) or "")
        task_sketch = str(data.get(FIELD_TASK_SKETCH) or "")
        est = data.get(FIELD_ESTIMATED_DIFFICULTY)
        estimated = str(est) if isinstance(est, str) and est.strip() else None

        steps_raw = data.get(FIELD_INITIAL_SUB_STEPS) or []
        steps: List[ExecutableSubStep] = []
        if isinstance(steps_raw, list):
            for item in steps_raw:
                if isinstance(item, dict):
                    steps.append(ExecutableSubStep.from_dict(item))
        elif isinstance(steps_raw, dict):
            steps.append(ExecutableSubStep.from_dict(steps_raw))

        return ExecutableExtractOutput(
            executable_suitable=executable_suitable,
            notes=notes,
            task_sketch=task_sketch,
            initial_sub_steps=steps,
            estimated_difficulty=estimated,
        )

    def run_one(
        self,
        extract_in: ExecutableExtractInput,
        *,
        snapshot_dir: Optional[Any] = None,
        unified_prompt_dir: Optional[Path] = None,
        name_prefix: str = "prompt_used.executable_extract.",
    ) -> ExecutableExtractOutput:
        prompt_body = self._build_prompt(extract_in)
        messages = build_messages_with_background(prompt_body, lang=self.config.lang or "zh")
        response = self.session.chat(messages, **self._chat_args)
        BaseSkillRunner._check_finish_reason(response, "ExecutableExtract")
        text = self.session.extract_text(response, default="")
        out = self._parse_output(text)

        if snapshot_dir is not None:
            try:
                snap_path = Path(snapshot_dir)
                snap_path.mkdir(parents=True, exist_ok=True)
                input_payload = {
                    "director_notes": extract_in.director_notes,
                    "paper_background": extract_in.paper_background,
                    "problem_description": extract_in.problem_description,
                    "dependencies_whitelist": extract_in.dependencies_whitelist,
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
                (snap_path / "output.json").write_text(
                    json.dumps(
                        {
                            FIELD_EXECUTABLE_SUITABLE: out.executable_suitable,
                            FIELD_NOTES: out.notes,
                            FIELD_TASK_SKETCH: out.task_sketch,
                            FIELD_INITIAL_SUB_STEPS: [s.to_dict() for s in out.initial_sub_steps],
                            FIELD_ESTIMATED_DIFFICULTY: out.estimated_difficulty,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("ExecutableExtract snapshot write failed: %s", str(exc))

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
    "ExecutableExtractInput",
    "ExecutableExtractConfig",
    "ExecutableExtractRunner",
]
