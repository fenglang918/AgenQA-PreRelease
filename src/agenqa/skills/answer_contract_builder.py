"""AnswerContractBuilder 角色：从格式化题目中提取契约约束。"""

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
from agenqa.domain.contracts.answer_contract_builder_schema import (
    FIELD_ANSWER_SEMANTICS,
    FIELD_ANSWER_STYLE,
    FIELD_SUPPORT_WITNESS,
)
from agenqa.skills.base import BaseSkillRunner
from infra.text.fenced_blocks import extract_preferred_fenced_block
from infra.text.json_sanitize import sanitize_json_text
from infra.text.json_policy import clean_json_text

logger = logging.getLogger(__name__)


@dataclass
class AnswerContractBuilderInput:
    step: int
    question: str
    answer: str
    question_type: Optional[str] = None


@dataclass
class AnswerContractBuilderOutput:
    answer_style: Dict[str, Any]
    answer_semantics: Dict[str, Any]
    support_witness: list[Dict[str, Any]]


@dataclass
class AnswerContractBuilderConfig:
    generator: Dict[str, Any]
    prompt_path: Any
    prompt_text: Optional[str] = None
    lang: Optional[str] = None


class AnswerContractBuilderRunner:
    def __init__(self, config: AnswerContractBuilderConfig) -> None:
        self.config = config
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

    def _build_prompt(self, step_in: AnswerContractBuilderInput) -> str:
        lang = (self.config.lang or "zh").lower()
        payload: Dict[str, Any] = {
            "step": step_in.step,
            "question": step_in.question,
            "answer": step_in.answer,
            "question_type": step_in.question_type or "",
        }
        body = self.prompt_template.safe_substitute(payload)
        if lang in {"en", "english"}:
            header = f"Role: AnswerContractBuilder | Language: {lang}"
        else:
            header = f"【AnswerContractBuilder 角色】语言: {lang}"
        return "\n".join([header, "", body])

    def _parse_output(self, text: str) -> AnswerContractBuilderOutput:
        if not text:
            raise ValueError("AnswerContractBuilder output is empty")
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
                    raise ValueError(f"Invalid AnswerContractBuilder JSON output: {e2}") from e2
            except Exception as e:  # noqa: BLE001
                raise ValueError(f"Invalid AnswerContractBuilder JSON output: {e}") from e
            if not isinstance(data, dict):
                raise ValueError("AnswerContractBuilder JSON output is not an object")
            return data

        data = _parse_json(candidate)

        answer_style = data.get(FIELD_ANSWER_STYLE)
        if answer_style is None:
            answer_style = {}
        if not isinstance(answer_style, dict):
            raise ValueError("answer_style must be an object")

        answer_semantics = data.get(FIELD_ANSWER_SEMANTICS)
        if answer_semantics is None:
            answer_semantics = {}
        if not isinstance(answer_semantics, dict):
            raise ValueError("answer_semantics must be an object")

        support_witness = data.get(FIELD_SUPPORT_WITNESS)
        if support_witness is None:
            support_witness = []
        if not isinstance(support_witness, list):
            raise ValueError("support_witness must be a list")

        return AnswerContractBuilderOutput(
            answer_style=answer_style,
            answer_semantics=answer_semantics,
            support_witness=[x for x in support_witness if isinstance(x, dict)],
        )

    def run_one(
        self,
        step_in: AnswerContractBuilderInput,
        *,
        snapshot_dir: Optional[Any] = None,
        unified_prompt_dir: Optional[Path] = None,
        name_prefix: str = "prompt_used.answer_contract_builder.",
    ) -> AnswerContractBuilderOutput:
        prompt_body = self._build_prompt(step_in)
        messages = build_messages_with_background(prompt_body, lang=self.config.lang or "zh")
        response = self.session.chat(messages, **self._chat_args)
        BaseSkillRunner._check_finish_reason(response, "AnswerContractBuilder")
        text = self.session.extract_text(response, default="")

        snap_path: Optional[Path] = None
        if snapshot_dir is not None:
            try:
                snap_path = Path(snapshot_dir)
                snap_path.mkdir(parents=True, exist_ok=True)
                input_payload = {
                    "step": step_in.step,
                    "question": step_in.question,
                    "answer": step_in.answer,
                    "question_type": step_in.question_type,
                }
                (snap_path / "input_view.json").write_text(
                    json.dumps(input_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                snapshot_prompt_used(
                    self.config.prompt_path,
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
                    (snap_path / "raw_response.txt").write_text(str(response), encoding="utf-8")
                (snap_path / "raw_response_text.txt").write_text(text or "", encoding="utf-8")
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
                task_name="answer_contract_builder",
                lang=self.config.lang or "zh",
                required_keys=[FIELD_ANSWER_STYLE, FIELD_ANSWER_SEMANTICS, FIELD_SUPPORT_WITNESS],
                prompt_body=prompt_body,
                snapshot_dir=snap_path,
            )
            if not cleaned_text:
                raise
            if snap_path is not None:
                try:
                    (snap_path / "raw_response_cleaned_text.txt").write_text(cleaned_text, encoding="utf-8")
                except Exception:
                    pass
            try:
                out = self._parse_output(cleaned_text)
            except Exception:
                raise first_exc

        if snap_path is not None:
            try:
                parsed_payload = {
                    FIELD_ANSWER_STYLE: out.answer_style,
                    FIELD_ANSWER_SEMANTICS: out.answer_semantics,
                    FIELD_SUPPORT_WITNESS: out.support_witness,
                }
                (snap_path / "parsed_output.json").write_text(
                    json.dumps(parsed_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception:
                pass
        return out


__all__ = ["AnswerContractBuilderInput", "AnswerContractBuilderOutput", "AnswerContractBuilderConfig", "AnswerContractBuilderRunner"]
