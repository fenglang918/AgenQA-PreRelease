"""StepCertBuilder 角色：从最终 QA 中提取 premise/fact/step_cert。"""

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
from agenqa.domain.step_cert_schema import (
    FIELD_PREMISE_DELTA,
    FIELD_FACT_DELTA,
    FIELD_STEP_CERT,
    FIELD_KEY_FACT_ID,
)
from agenqa.skills.base import BaseSkillRunner
from infra.text.fenced_blocks import extract_preferred_fenced_block
from infra.text.json_sanitize import sanitize_json_text
from infra.text.json_policy import clean_json_text
from agenqa.validators.reuse_consistency import extract_boxed_letter, parse_mcq_options, is_mcq_question

logger = logging.getLogger(__name__)


@dataclass
class StepCertInput:
    step: int
    question: str
    solution: str
    answer: str
    question_type: Optional[str] = None
    memory_json: Optional[str] = None


@dataclass
class StepCertOutput:
    premise_delta: List[Dict[str, Any]]
    fact_delta: List[Dict[str, Any]]
    step_cert: Dict[str, Any]
    key_fact_id: str


@dataclass
class StepCertConfig:
    generator: Dict[str, Any]
    prompt_path: Any
    prompt_text: Optional[str] = None
    lang: Optional[str] = None
    protocol: Optional[str] = None


class StepCertBuilderRunner:
    def __init__(self, config: StepCertConfig) -> None:
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

    def _build_prompt(self, step_in: StepCertInput) -> str:
        lang = (self.config.lang or "zh").lower()
        payload: Dict[str, Any] = {
            "step": step_in.step,
            "question": step_in.question,
            "solution": step_in.solution,
            "answer": step_in.answer,
            "question_type": step_in.question_type or "",
            "memory_json": step_in.memory_json or "",
        }
        body = self.prompt_template.safe_substitute(payload)
        if lang in {"en", "english"}:
            header = f"Role: StepCertBuilder | Language: {lang}"
        else:
            header = f"【StepCertBuilder 角色】语言: {lang}"
        return "\n".join([header, "", body])

    def _parse_output(self, text: str) -> StepCertOutput:
        if not text:
            raise ValueError("StepCertBuilder output is empty")
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
                    raise ValueError(f"Invalid StepCertBuilder JSON output: {e2}") from e2
            except Exception as e:  # noqa: BLE001
                raise ValueError(f"Invalid StepCertBuilder JSON output: {e}") from e
            if not isinstance(data, dict):
                raise ValueError("StepCertBuilder JSON output is not an object")
            return data

        data = _parse_json(candidate)

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
            if not isinstance(entry.get("id"), str):
                raise ValueError("premise_delta entries must include id")
        for entry in fact_delta:
            if not isinstance(entry, dict):
                raise ValueError("fact_delta entries must be objects")
            if not isinstance(entry.get("id"), str):
                raise ValueError("fact_delta entries must include id")

        return StepCertOutput(
            premise_delta=premise_delta,
            fact_delta=fact_delta,
            step_cert=step_cert,
            key_fact_id=str(key_fact_id).strip(),
        )

    @staticmethod
    def _autofill_mcq_key_fact_fields(question: str, answer: str, fact_delta: List[Dict[str, Any]], key_fact_id: str) -> None:
        if not is_mcq_question(question):
            return
        letter = extract_boxed_letter(answer or "")
        if not letter:
            return
        options = parse_mcq_options(question)
        expected_text = (options.get(letter) or "").strip()
        if not expected_text:
            return

        target: Optional[Dict[str, Any]] = None
        for fact in fact_delta:
            if isinstance(fact, dict) and fact.get("id") == key_fact_id:
                target = fact
                break
        if not target:
            return

        updated = False
        mcq_choice = str(target.get("mcq_choice") or "").strip().upper()
        if not mcq_choice:
            target["mcq_choice"] = letter
            updated = True

        mcq_choice_text = str(target.get("mcq_choice_text") or "").strip()
        if not mcq_choice_text:
            target["mcq_choice_text"] = expected_text
            mcq_choice_text = expected_text
            updated = True

        statement = str(target.get("statement") or target.get("text") or "").strip()
        if not statement:
            target["statement"] = f"{letter}. {mcq_choice_text}"
            updated = True
        else:
            new_statement = statement
            if letter not in new_statement:
                new_statement = f"{letter}. {new_statement}"
            if mcq_choice_text and mcq_choice_text not in new_statement:
                new_statement = f"{new_statement} {mcq_choice_text}"
            if new_statement != statement:
                target["statement"] = new_statement
                updated = True

        if updated:
            logger.warning(
                "StepCertBuilder MCQ key_fact missing fields; autofilled from boxed answer/options (key_fact_id=%s, letter=%s)",
                key_fact_id,
                letter,
            )

    @staticmethod
    def _validate_mcq_key_fact(question: str, answer: str, fact_delta: List[Dict[str, Any]], key_fact_id: str) -> None:
        if not is_mcq_question(question):
            return
        letter = extract_boxed_letter(answer or "")
        if not letter:
            return
        options = parse_mcq_options(question)
        expected_text = options.get(letter)
        if not expected_text:
            raise ValueError("MCQ options missing for key fact validation")
        target = None
        for fact in fact_delta:
            if isinstance(fact, dict) and fact.get("id") == key_fact_id:
                target = fact
                break
        if not target:
            raise ValueError("key_fact_id not found in fact_delta for MCQ validation")
        mcq_choice = str(target.get("mcq_choice") or "").strip().upper()
        mcq_choice_text = str(target.get("mcq_choice_text") or "").strip()
        statement = str(target.get("statement") or target.get("text") or "").strip()
        if mcq_choice != letter:
            raise ValueError("MCQ key_fact mcq_choice must match boxed answer letter")
        if not mcq_choice_text:
            raise ValueError("MCQ key_fact must include mcq_choice_text")
        expected_text_norm = expected_text.strip()
        if expected_text_norm not in mcq_choice_text and mcq_choice_text not in expected_text_norm:
            logger.warning(
                "StepCertBuilder MCQ key_fact mcq_choice_text mismatch; overwriting with option text (key_fact_id=%s, letter=%s)",
                key_fact_id,
                letter,
            )
            target["mcq_choice_text"] = expected_text_norm
            mcq_choice_text = expected_text_norm
        if not statement:
            target["statement"] = f"{letter}. {mcq_choice_text}"
        else:
            new_statement = statement
            if letter not in new_statement:
                new_statement = f"{letter}. {new_statement}"
            if mcq_choice_text and mcq_choice_text not in new_statement:
                new_statement = f"{new_statement} {mcq_choice_text}"
            if new_statement != statement:
                target["statement"] = new_statement

    @staticmethod
    def _normalize_id_fields(out: StepCertOutput) -> None:
        """Normalize IDs by removing whitespace to avoid downstream mismatches."""
        def _norm(val: str) -> str:
            return re.sub(r"\s+", "", val.strip())

        def _norm_list(items: Any) -> list[str]:
            if not isinstance(items, list):
                return []
            out_items: list[str] = []
            for item in items:
                if not isinstance(item, str):
                    continue
                cleaned = _norm(item)
                if cleaned:
                    out_items.append(cleaned)
            return out_items

        for entry in out.premise_delta or []:
            if isinstance(entry, dict) and isinstance(entry.get("id"), str):
                entry["id"] = _norm(entry["id"])
        for entry in out.fact_delta or []:
            if isinstance(entry, dict) and isinstance(entry.get("id"), str):
                entry["id"] = _norm(entry["id"])

        if isinstance(out.key_fact_id, str):
            out.key_fact_id = _norm(out.key_fact_id)

        cert = out.step_cert if isinstance(out.step_cert, dict) else {}
        for key in ("uses_premise_ids", "uses_fact_ids", "produces_fact_ids"):
            if key in cert:
                cert[key] = _norm_list(cert.get(key))
        if "key_fact_id" in cert and isinstance(cert.get("key_fact_id"), str):
            cert["key_fact_id"] = _norm(cert["key_fact_id"])

    @staticmethod
    def _fix_step_cert_reference_lists(out: StepCertOutput, step_in: StepCertInput) -> None:
        """Fix common LLM mistakes in step_cert reference lists.

        The KnownTree schema differentiates:
        - uses_premise_ids: references premise ids only
        - uses_fact_ids: references fact ids only

        In practice, the model sometimes puts a valid fact id into uses_premise_ids (or vice versa).
        Since step_in.memory_json includes the authoritative id sets (< step), we can auto-correct
        these cross-listed ids before writing memory, while still letting truly-unknown ids fail later.
        """

        def _dedupe(items: Any) -> list[str]:
            if not isinstance(items, list):
                return []
            seen: set[str] = set()
            out_items: list[str] = []
            for it in items:
                if not isinstance(it, str):
                    continue
                s = it.strip()
                if not s or s in seen:
                    continue
                seen.add(s)
                out_items.append(s)
            return out_items

        memory_json = step_in.memory_json or ""
        try:
            mem = json.loads(memory_json) if memory_json.strip() else {}
        except Exception:
            mem = {}

        premise_ids: set[str] = set()
        fact_ids: set[str] = set()
        if isinstance(mem, dict):
            for p in mem.get("premise_bank") or []:
                if isinstance(p, dict) and isinstance(p.get("id"), str) and p["id"].strip():
                    premise_ids.add(p["id"].strip())
            for f in mem.get("fact_bank") or []:
                if isinstance(f, dict) and isinstance(f.get("id"), str) and f["id"].strip():
                    fact_ids.add(f["id"].strip())

        # Also include same-step new ids from deltas (the model may reference them in uses_*_ids).
        for p in out.premise_delta or []:
            if isinstance(p, dict) and isinstance(p.get("id"), str) and p["id"].strip():
                premise_ids.add(p["id"].strip())
        for f in out.fact_delta or []:
            if isinstance(f, dict) and isinstance(f.get("id"), str) and f["id"].strip():
                fact_ids.add(f["id"].strip())

        cert = out.step_cert if isinstance(out.step_cert, dict) else {}
        uses_premise_ids = _dedupe(cert.get("uses_premise_ids"))
        uses_fact_ids = _dedupe(cert.get("uses_fact_ids"))

        move_to_fact = [pid for pid in uses_premise_ids if pid not in premise_ids and pid in fact_ids]
        if move_to_fact:
            logger.warning(
                "StepCertBuilder: auto-fixing id lists: moving fact IDs from uses_premise_ids to uses_fact_ids (step=%s): %s",
                step_in.step,
                sorted(set(move_to_fact)),
            )
            uses_premise_ids = [pid for pid in uses_premise_ids if pid not in set(move_to_fact)]
            uses_fact_ids = uses_fact_ids + move_to_fact

        move_to_premise = [fid for fid in uses_fact_ids if fid not in fact_ids and fid in premise_ids]
        if move_to_premise:
            logger.warning(
                "StepCertBuilder: auto-fixing id lists: moving premise IDs from uses_fact_ids to uses_premise_ids (step=%s): %s",
                step_in.step,
                sorted(set(move_to_premise)),
            )
            uses_fact_ids = [fid for fid in uses_fact_ids if fid not in set(move_to_premise)]
            uses_premise_ids = uses_premise_ids + move_to_premise

        # Final dedupe (preserve order).
        uses_premise_ids = _dedupe(uses_premise_ids)
        uses_fact_ids = _dedupe(uses_fact_ids)
        cert["uses_premise_ids"] = uses_premise_ids
        cert["uses_fact_ids"] = uses_fact_ids
        out.step_cert = cert

    def run_one(
        self,
        step_in: StepCertInput,
        *,
        snapshot_dir: Optional[Any] = None,
        unified_prompt_dir: Optional[Path] = None,
        name_prefix: str = "prompt_used.step_cert_builder.",
    ) -> StepCertOutput:
        prompt_body = self._build_prompt(step_in)
        messages = build_messages_with_background(prompt_body, lang=self.config.lang or "zh")
        response = self.session.chat(messages, **self._chat_args)
        BaseSkillRunner._check_finish_reason(response, "StepCertBuilder")
        text = self.session.extract_text(response, default="")

        snap_path: Optional[Path] = None
        if snapshot_dir is not None:
            try:
                snap_path = Path(snapshot_dir)
                snap_path.mkdir(parents=True, exist_ok=True)
                input_payload = {
                    "step": step_in.step,
                    "question": step_in.question,
                    "solution": step_in.solution,
                    "answer": step_in.answer,
                    "question_type": step_in.question_type,
                    "memory_json": step_in.memory_json,
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
                task_name="step_cert_builder",
                lang=self.config.lang or "zh",
                required_keys=[FIELD_PREMISE_DELTA, FIELD_FACT_DELTA, FIELD_STEP_CERT, FIELD_KEY_FACT_ID],
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
        self._normalize_id_fields(out)
        self._fix_step_cert_reference_lists(out, step_in)
        self._autofill_mcq_key_fact_fields(step_in.question, step_in.answer, out.fact_delta, out.key_fact_id)
        self._validate_mcq_key_fact(step_in.question, step_in.answer, out.fact_delta, out.key_fact_id)

        if snap_path is not None:
            try:
                parsed_payload = {
                    FIELD_PREMISE_DELTA: out.premise_delta,
                    FIELD_FACT_DELTA: out.fact_delta,
                    FIELD_STEP_CERT: out.step_cert,
                    FIELD_KEY_FACT_ID: out.key_fact_id,
                }
                (snap_path / "parsed_output.json").write_text(
                    json.dumps(parsed_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception:
                pass
        return out


__all__ = ["StepCertInput", "StepCertOutput", "StepCertConfig", "StepCertBuilderRunner"]
