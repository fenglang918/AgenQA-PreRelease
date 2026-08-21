"""PathFold role: generate folded path questions (scaffolded + direct)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any, Dict, Optional

from infra.llm.inference import resolve_inference
from infra.prompt.prompt_builder import build_messages_with_background
from infra.prompt.prompt_tracker import log_using_prompt, snapshot_prompt_used, snapshot_rendered_prompt
from infra.llm.service_client import LLMServiceSession
from agenqa.domain.path_fold_schema import (
    FIELD_FOLD_NOTES,
    FIELD_QUESTION_DIRECT,
    FIELD_QUESTION_SCAFFOLDED,
)
from agenqa.skills.base import BaseSkillRunner
from infra.text.fenced_blocks import extract_preferred_fenced_block
from infra.text.json_sanitize import sanitize_json_text

logger = logging.getLogger(__name__)


@dataclass
class PathFoldInput:
    step: int
    question_type: str
    premise_bank_json: str
    history_json: str


@dataclass
class PathFoldOutput:
    question_scaffolded: str
    question_direct: str
    fold_notes: str


@dataclass
class PathFoldConfig:
    generator: Dict[str, Any]
    prompt_path: Any
    prompt_text: Optional[str] = None
    lang: Optional[str] = None


class PathFoldRunner:
    def __init__(self, config: PathFoldConfig) -> None:
        self.config = config
        resolved = resolve_inference(config.generator)
        self.session: LLMServiceSession = resolved.session
        self._chat_args: Dict[str, Any] = dict(resolved.chat_args)

        base_text = (
            config.prompt_text if getattr(config, "prompt_text", None) else config.prompt_path.read_text(encoding="utf-8")
        )
        self.prompt_text: str = base_text
        self.prompt_template = Template(base_text)
        log_using_prompt(logger, Path(str(config.prompt_path)))

    def _build_prompt(self, fold_in: PathFoldInput) -> str:
        lang = (self.config.lang or "zh").lower()
        payload: Dict[str, Any] = {
            "step": int(fold_in.step),
            "question_type": fold_in.question_type,
            "premise_bank_json": fold_in.premise_bank_json,
            "history_json": fold_in.history_json,
        }
        body = self.prompt_template.safe_substitute(payload)
        header = f"Role: PathFold | Language: {lang}" if lang in {"en", "english"} else f"【PathFold 角色】语言: {lang}"
        return "\n".join([header, "", body])

    def _parse_output(self, text: str) -> PathFoldOutput:
        if not text or not str(text).strip():
            raise ValueError("PathFold output is empty")
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
            raise ValueError("PathFold JSON output is not an object")

        def _as_str(val: Any) -> str:
            if val is None:
                return ""
            if isinstance(val, (str, int, float, bool)):
                return str(val)
            try:
                return json.dumps(val, ensure_ascii=False)
            except Exception:
                return str(val)

        q_scaf = _as_str(data.get(FIELD_QUESTION_SCAFFOLDED)).strip()
        q_dir = _as_str(data.get(FIELD_QUESTION_DIRECT)).strip()
        notes = _as_str(data.get(FIELD_FOLD_NOTES)).strip()
        if not q_scaf:
            raise ValueError("PathFold missing question_scaffolded")
        if not q_dir:
            raise ValueError("PathFold missing question_direct")
        return PathFoldOutput(question_scaffolded=q_scaf, question_direct=q_dir, fold_notes=notes)

    def run_one(
        self,
        fold_in: PathFoldInput,
        *,
        snapshot_dir: Optional[Any] = None,
        unified_prompt_dir: Optional[Path] = None,
        name_prefix: str = "prompt_used.path_fold.",
    ) -> PathFoldOutput:
        prompt_body = self._build_prompt(fold_in)
        messages = build_messages_with_background(prompt_body, lang=self.config.lang or "zh")
        response = self.session.chat(messages, **self._chat_args)
        BaseSkillRunner._check_finish_reason(response, "PathFold")
        text = self.session.extract_text(response, default="")
        out = self._parse_output(text)

        if snapshot_dir is not None:
            try:
                snap_path = Path(snapshot_dir)
                snap_path.mkdir(parents=True, exist_ok=True)
                input_payload = {
                    "step": fold_in.step,
                    "question_type": fold_in.question_type,
                    "premise_bank_json": fold_in.premise_bank_json,
                    "history_json": fold_in.history_json,
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
                request_meta = {"model": self.session.model_name, "service_id": self.session.service_id, "chat_args": self._chat_args}
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
                            FIELD_QUESTION_SCAFFOLDED: out.question_scaffolded,
                            FIELD_QUESTION_DIRECT: out.question_direct,
                            FIELD_FOLD_NOTES: out.fold_notes,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("PathFold snapshot write failed: %s", str(exc))

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
    "PathFoldInput",
    "PathFoldOutput",
    "PathFoldConfig",
    "PathFoldRunner",
]
