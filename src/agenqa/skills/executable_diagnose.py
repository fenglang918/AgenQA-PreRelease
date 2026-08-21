"""ExecutableDiagnose runner: diagnose failures in the latest executable step."""

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
from agenqa.domain.diagnose_schema import FIELD_DIAGNOSIS, FIELD_FIX_SUGGESTIONS, FIELD_ISSUES, diagnose_output_to_dict
from agenqa.skills.base import BaseSkillRunner
from infra.text.fenced_blocks import extract_preferred_fenced_block
from infra.text.json_sanitize import sanitize_json_text

from agenqa.prompts.executable_diagnose import EXECUTABLE_DIAGNOSE_V1, EXECUTABLE_DIAGNOSE_V1_EN

logger = logging.getLogger(__name__)


@dataclass
class ExecutableDiagnoseInput:
    step: int
    director_notes: str
    executable_tail_json: str
    eval_error: str


@dataclass
class ExecutableDiagnoseConfig:
    generator: Dict[str, Any]
    prompt_path: Any
    prompt_text: Optional[str] = None
    lang: Optional[str] = None


@dataclass
class ExecutableDiagnoseOutput:
    issues: List[str]
    fix_suggestions: List[str]
    diagnosis: str


class ExecutableDiagnoseRunner:
    def __init__(self, config: ExecutableDiagnoseConfig) -> None:
        self.config = config
        resolved = resolve_inference(config.generator)
        self.session: LLMServiceSession = resolved.session
        self._chat_args: Dict[str, Any] = dict(resolved.chat_args)

        lang_norm = (config.lang or "").strip().lower()
        use_en = lang_norm in {"en", "english"}
        base_text = config.prompt_text if getattr(config, "prompt_text", None) else None
        if base_text is None:
            base_text = EXECUTABLE_DIAGNOSE_V1_EN if use_en else EXECUTABLE_DIAGNOSE_V1
        self.prompt_text = base_text
        self.prompt_template = Template(base_text)
        log_using_prompt(logger, Path(str(config.prompt_path)))

    def _build_prompt(self, diag_in: ExecutableDiagnoseInput) -> str:
        payload = {
            "step": int(diag_in.step),
            "director_notes": diag_in.director_notes or "",
            "executable_tail_json": diag_in.executable_tail_json or "",
            "eval_error": diag_in.eval_error or "",
        }
        body = self.prompt_template.safe_substitute(payload)
        lang = (self.config.lang or "zh").lower()
        header = f"Role: ExecutableDiagnose | Language: {lang}" if lang in {"en", "english"} else f"【ExecutableDiagnose 角色】语言: {lang}"
        return "\n".join([header, "", body])

    def _parse_output(self, text: str) -> ExecutableDiagnoseOutput:
        if not text or not str(text).strip():
            raise ValueError("ExecutableDiagnose output is empty")
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
            raise ValueError("ExecutableDiagnose JSON output is not an object")

        def _as_list(val: Any) -> List[str]:
            if val is None:
                return []
            if isinstance(val, list):
                out: List[str] = []
                for item in val:
                    s = str(item).strip()
                    if s:
                        out.append(s)
                return out
            s = str(val).strip()
            return [s] if s else []

        issues = _as_list(data.get(FIELD_ISSUES))
        fix_suggestions = _as_list(data.get(FIELD_FIX_SUGGESTIONS))
        diagnosis = str(data.get(FIELD_DIAGNOSIS) or "").strip()
        return ExecutableDiagnoseOutput(issues=issues, fix_suggestions=fix_suggestions, diagnosis=diagnosis)

    def run_one(
        self,
        diag_in: ExecutableDiagnoseInput,
        *,
        snapshot_dir: Optional[Any] = None,
        unified_prompt_dir: Optional[Path] = None,
        name_prefix: str = "prompt_used.executable_diagnose.",
    ) -> ExecutableDiagnoseOutput:
        prompt_body = self._build_prompt(diag_in)
        messages = build_messages_with_background(prompt_body, lang=self.config.lang or "zh")
        response = self.session.chat(messages, **self._chat_args)
        BaseSkillRunner._check_finish_reason(response, "ExecutableDiagnose")
        text = self.session.extract_text(response, default="")
        out = self._parse_output(text)

        if snapshot_dir is not None:
            try:
                snap_path = Path(snapshot_dir)
                snap_path.mkdir(parents=True, exist_ok=True)
                (snap_path / "input_view.json").write_text(
                    json.dumps(
                        {
                            "step": diag_in.step,
                            "director_notes": diag_in.director_notes,
                            "executable_tail_json": diag_in.executable_tail_json,
                            "eval_error": diag_in.eval_error,
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
                    json.dumps(diagnose_output_to_dict(out), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("ExecutableDiagnose snapshot write failed: %s", str(exc))

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
    "ExecutableDiagnoseInput",
    "ExecutableDiagnoseConfig",
    "ExecutableDiagnoseOutput",
    "ExecutableDiagnoseRunner",
]
