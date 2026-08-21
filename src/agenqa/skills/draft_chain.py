"""DraftChain 角色：基于 Memory 视图生成递进题目的显式草稿。"""

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
from infra.prompt.prompt_tracker import log_using_prompt, snapshot_prompt_used, snapshot_rendered_prompt
from infra.llm.service_client import LLMServiceSession
from agenqa.domain.draft_chain_schema import (
    FIELD_DRAFT_QUESTION_EXPLICIT,
    FIELD_DRAFT_SOLUTION_OUTLINE,
    FIELD_DRAFT_ANSWER,
    FIELD_SUBTASKS,
    FIELD_FINAL_SUBTASK_ID,
    FIELD_DEPENDENCIES,
    FIELD_REQUIRED_FACT_IDS,
    FIELD_PRIMARY_REQUIRED_FACT_ID,
    FIELD_REUSE_PLAN,
    FIELD_WORLD_CONTRACT,
)
from agenqa.domain.numeric_oracle_schema import (
    FIELD_ABS_TOL,
    FIELD_NOTES,
    FIELD_ORACLE_CODE,
    FIELD_REL_TOL,
    FIELD_SIG_FIGS,
    FIELD_UNIT,
)
from agenqa.skills.base import BaseSkillRunner
from infra.text.fenced_blocks import extract_preferred_fenced_block
from infra.text.json_sanitize import sanitize_json_text
from infra.text.json_policy import clean_json_text

logger = logging.getLogger(__name__)

_TAG_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass
class DraftChainInput:
    chain_view_json: str
    prev_step: int
    step: int
    director_notes: Optional[str] = None
    question_type: Optional[str] = None
    expected_primary_fact_id: Optional[str] = None


@dataclass
class DraftChainOutput:
    subtasks: List[Dict[str, Any]]
    final_subtask_id: str
    dependencies: Dict[str, List[str]]
    draft_question_explicit: str
    draft_solution_outline: str
    draft_answer: str
    required_fact_ids: List[str]
    primary_required_fact_id: str
    reuse_plan: List[str]
    # Numeric-only optional tool artifacts (draft-time oracle code)
    abs_tol: Optional[float] = None
    rel_tol: Optional[float] = None
    sig_figs: Optional[int] = None
    unit: str = ""
    oracle_code: str = ""
    notes: str = ""
    # Type1 semantics governance (optional; mainly for revise_mode=world_contract).
    world_contract: Optional[Dict[str, Any]] = None


@dataclass
class DraftChainConfig:
    generator: Dict[str, Any]
    prompt_path: Any  # Path-like
    prompt_text: Optional[str] = None
    lang: Optional[str] = None
    # 输出协议：json（默认）或 tagged
    protocol: Optional[str] = None


class DraftChainRunner:
    def __init__(self, config: DraftChainConfig) -> None:
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

    def _build_prompt(self, draft_in: DraftChainInput) -> str:
        lang = (self.config.lang or "zh").lower()
        payload: Dict[str, Any] = {
            "prev_step": draft_in.prev_step,
            "step": draft_in.step,
            "chain_view_json": draft_in.chain_view_json,
            "director_notes": draft_in.director_notes or "",
            "question_type": draft_in.question_type or "",
            "expected_primary_fact_id": draft_in.expected_primary_fact_id or "",
        }
        body = self.prompt_template.safe_substitute(payload)
        if lang in {"en", "english"}:
            header = f"Role: DraftChain | Language: {lang}"
        else:
            header = f"【DraftChain 角色】语言: {lang}"
        return "\n".join([header, "", body])

    def _parse_output(self, text: str) -> DraftChainOutput:
        if not text:
            raise ValueError("DraftChain output is empty")
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
                    raise ValueError(f"Invalid DraftChain JSON output: {e2}") from e2
            except Exception as e:  # noqa: BLE001
                raise ValueError(f"Invalid DraftChain JSON output: {e}") from e
            if not isinstance(data, dict):
                raise ValueError("DraftChain JSON output is not an object")
            return data

        data: Optional[Dict[str, Any]] = None
        json_exc: Exception | None = None
        try:
            data = _parse_json(candidate)
        except Exception as e:  # noqa: BLE001
            data = None
            json_exc = e

        if data is None:
            raise ValueError(f"DraftChain output parse failed: {json_exc}") from json_exc

        def _as_str(val: Any) -> str:
            if val is None:
                return ""
            if isinstance(val, (str, int, float, bool)):
                return str(val)
            try:
                return json.dumps(val, ensure_ascii=False)
            except Exception:
                return str(val)

        def _as_list(val: Any) -> List[str]:
            if val is None:
                return []
            if isinstance(val, list):
                out = []
                for item in val:
                    s = _as_str(item).strip()
                    if s:
                        out.append(s)
                return out
            if isinstance(val, str):
                text = val.strip()
                if not text:
                    return []
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, list):
                        return [_as_str(it).strip() for it in parsed if _as_str(it).strip()]
                except Exception:
                    pass
                lines = [s.strip("- ") for s in text.splitlines() if s.strip()]
                return lines if lines else [text]
            return [_as_str(val).strip()]

        def _as_subtasks(val: Any) -> List[Dict[str, Any]]:
            if val is None:
                return []
            if isinstance(val, list):
                out: List[Dict[str, Any]] = []
                for item in val:
                    if isinstance(item, dict):
                        out.append(dict(item))
                    elif isinstance(item, str) and item.strip():
                        out.append({"id": "", "description": item.strip(), "result": ""})
                return out
            if isinstance(val, dict):
                return [dict(val)]
            if isinstance(val, str):
                text = val.strip()
                if not text:
                    return []
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, list):
                        return _as_subtasks(parsed)
                    if isinstance(parsed, dict):
                        return _as_subtasks([parsed])
                except Exception:
                    return [{"id": "", "description": text, "result": ""}]
            return []

        def _as_dependencies(val: Any) -> Dict[str, List[str]]:
            if val is None:
                return {}
            if isinstance(val, dict):
                out: Dict[str, List[str]] = {}
                for k, v in val.items():
                    key = _as_str(k).strip()
                    if not key:
                        continue
                    out[key] = _as_list(v)
                return out
            if isinstance(val, str):
                text = val.strip()
                if not text:
                    return {}
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, dict):
                        return _as_dependencies(parsed)
                except Exception:
                    return {}
            return {}

        def _as_float(val: Any) -> Optional[float]:
            if val is None:
                return None
            if isinstance(val, (int, float)):
                out = float(val)
                return out if out > 0 else None
            if isinstance(val, str):
                s = val.strip()
                if not s:
                    return None
                try:
                    out = float(s)
                    return out if out > 0 else None
                except Exception:
                    return None
            return None

        def _as_int(val: Any) -> Optional[int]:
            if val is None:
                return None
            if isinstance(val, int):
                return val if val > 0 else None
            if isinstance(val, float):
                try:
                    iv = int(val)
                    return iv if iv > 0 else None
                except Exception:
                    return None
            if isinstance(val, str):
                s = val.strip()
                if not s:
                    return None
                try:
                    iv = int(float(s))
                    return iv if iv > 0 else None
                except Exception:
                    return None
            return None

        def _as_obj_or_null(val: Any) -> Optional[Dict[str, Any]]:
            if val is None:
                return None
            if isinstance(val, dict):
                return dict(val)
            if isinstance(val, str):
                s = val.strip()
                if not s:
                    return None
                try:
                    parsed = json.loads(s)
                    return dict(parsed) if isinstance(parsed, dict) else None
                except Exception:
                    return None
            return None

        def _validate_tolerance(
            abs_tol: Optional[float], rel_tol: Optional[float], sig_figs: Optional[int]
        ) -> tuple[Optional[float], Optional[float], Optional[int]]:
            # Prefer sig_figs if provided; otherwise require at least some abs/rel tolerance.
            if sig_figs is not None:
                return None, None, sig_figs
            if abs_tol is None and rel_tol is None:
                return 1e-6, 1e-4, None
            return abs_tol, rel_tol, None

        subtasks = _as_subtasks(data.get(FIELD_SUBTASKS))
        final_subtask_id = _as_str(data.get(FIELD_FINAL_SUBTASK_ID)).strip()
        dependencies = _as_dependencies(data.get(FIELD_DEPENDENCIES))
        dq = _as_str(data.get(FIELD_DRAFT_QUESTION_EXPLICIT))
        ds = _as_str(data.get(FIELD_DRAFT_SOLUTION_OUTLINE))
        da = _as_str(data.get(FIELD_DRAFT_ANSWER))
        required_fact_ids = _as_list(data.get(FIELD_REQUIRED_FACT_IDS))
        primary_required_fact_id = _as_str(data.get(FIELD_PRIMARY_REQUIRED_FACT_ID)).strip()
        reuse_plan = _as_list(data.get(FIELD_REUSE_PLAN))

        oracle_code = _as_str(data.get(FIELD_ORACLE_CODE)).strip()
        abs_tol = _as_float(data.get(FIELD_ABS_TOL))
        rel_tol = _as_float(data.get(FIELD_REL_TOL))
        sig_figs = _as_int(data.get(FIELD_SIG_FIGS))
        unit = _as_str(data.get(FIELD_UNIT)).strip()
        notes = _as_str(data.get(FIELD_NOTES)).strip()
        abs_tol, rel_tol, sig_figs = _validate_tolerance(abs_tol, rel_tol, sig_figs) if oracle_code else (abs_tol, rel_tol, sig_figs)
        world_contract = _as_obj_or_null(data.get(FIELD_WORLD_CONTRACT))

        return DraftChainOutput(
            subtasks=subtasks,
            final_subtask_id=final_subtask_id,
            dependencies=dependencies,
            draft_question_explicit=dq,
            draft_solution_outline=ds,
            draft_answer=da,
            required_fact_ids=required_fact_ids,
            primary_required_fact_id=primary_required_fact_id,
            reuse_plan=reuse_plan,
            abs_tol=abs_tol,
            rel_tol=rel_tol,
            sig_figs=sig_figs,
            unit=unit,
            oracle_code=oracle_code,
            notes=notes,
            world_contract=world_contract,
        )

    def run_one(
        self,
        draft_in: DraftChainInput,
        *,
        snapshot_dir: Optional[Any] = None,
        unified_prompt_dir: Optional[Path] = None,
        name_prefix: str = "prompt_used.draft_chain.",
    ) -> DraftChainOutput:
        prompt_body = self._build_prompt(draft_in)
        messages = build_messages_with_background(prompt_body, lang=self.config.lang or "zh")
        response = self.session.chat(messages, **self._chat_args)
        BaseSkillRunner._check_finish_reason(response, "DraftChain")
        text = self.session.extract_text(response, default="")

        snap_path: Optional[Path] = None
        if snapshot_dir is not None:
            try:
                snap_path = Path(snapshot_dir)
                snap_path.mkdir(parents=True, exist_ok=True)
                input_payload = {
                    "prev_step": draft_in.prev_step,
                    "step": draft_in.step,
                    "chain_view_json": draft_in.chain_view_json,
                    "director_notes": draft_in.director_notes,
                    "question_type": draft_in.question_type,
                    "expected_primary_fact_id": draft_in.expected_primary_fact_id,
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

        retry_on_parse_fail = os.getenv("SCICLONE_DRAFTCHAIN_RETRY_ON_PARSE_FAIL", "1").strip().lower() not in {
            "0",
            "false",
            "off",
            "no",
        }
        out: DraftChainOutput
        try:
            out = self._parse_output(text)
        except Exception as first_exc:  # noqa: BLE001
            if not retry_on_parse_fail:
                raise
            # Optional: try a dedicated JSON cleaner model (e.g. Claude) before doing a full regeneration repair.
            cleaned_out: Optional[DraftChainOutput] = None
            try:
                cleaned_text = clean_json_text(
                    text,
                    generator=self.config.generator,
                    task_name="draft_chain",
                    lang=self.config.lang or "zh",
                    required_keys=[
                        FIELD_SUBTASKS,
                        FIELD_FINAL_SUBTASK_ID,
                        FIELD_DEPENDENCIES,
                        FIELD_DRAFT_QUESTION_EXPLICIT,
                        FIELD_DRAFT_SOLUTION_OUTLINE,
                        FIELD_DRAFT_ANSWER,
                        FIELD_REQUIRED_FACT_IDS,
                        FIELD_PRIMARY_REQUIRED_FACT_ID,
                        FIELD_REUSE_PLAN,
                    ],
                    prompt_body=prompt_body,
                    snapshot_dir=snap_path,
                )
                if cleaned_text:
                    cleaned_out = self._parse_output(cleaned_text)
                    if snap_path is not None:
                        try:
                            (snap_path / "raw_response_cleaned_text.txt").write_text(cleaned_text, encoding="utf-8")
                        except Exception:
                            pass
            except Exception:
                cleaned_out = None

            if cleaned_out is not None:
                out = cleaned_out
            else:
                logger.warning("DraftChain parse failed; retrying once with repair prompt: %s", first_exc)
                expected_primary = (draft_in.expected_primary_fact_id or "").strip()
                step_rule = ""
                if int(draft_in.step or 0) >= 2:
                    step_rule = (
                        "Step>=2 requirements:\n"
                        f"- primary_required_fact_id must be exactly \"{expected_primary}\".\n"
                        f"- required_fact_ids must include \"{expected_primary}\".\n"
                        f"- reuse_plan must be a JSON array and include a line that explicitly reuses \"{expected_primary}\".\n"
                        "- subtasks must have exactly 2 items with ids sub_prev and sub_step; final_subtask_id=sub_step; "
                        "dependencies=sub_step -> sub_prev.\n"
                    )
                else:
                    step_rule = (
                        "Step==1 requirements:\n"
                        "- required_fact_ids must be an empty array [].\n"
                        "- primary_required_fact_id must be an empty string \"\".\n"
                        "- reuse_plan must be a JSON array (can be empty).\n"
                        "- subtasks should be 1-2 coarse items.\n"
                    )
                repair_prompt = (
                    "You produced an invalid or truncated JSON for DraftChain.\n\n"
                    "Re-output ONE complete JSON object only (no ``` fences, no extra text).\n"
                    "Requirements:\n"
                    "- Must be valid JSON (double quotes, no trailing commas).\n"
                    "- Must include keys: "
                    f"{FIELD_SUBTASKS}, {FIELD_FINAL_SUBTASK_ID}, {FIELD_DEPENDENCIES}, "
                    f"{FIELD_DRAFT_QUESTION_EXPLICIT}, {FIELD_DRAFT_SOLUTION_OUTLINE}, {FIELD_DRAFT_ANSWER}, "
                    f"{FIELD_REQUIRED_FACT_IDS}, {FIELD_PRIMARY_REQUIRED_FACT_ID}, {FIELD_REUSE_PLAN}.\n"
                    "- Keep draft_question_explicit concise (avoid very long LaTeX blocks; keep options short).\n"
                    f"{step_rule}\n"
                    "Follow the original task prompt below (do NOT change the topic):\n"
                    "<draft_chain_prompt>\n"
                    f"{prompt_body}\n"
                    "</draft_chain_prompt>\n\n"
                    "Your previous output (for reference, may be truncated):\n"
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
                BaseSkillRunner._check_finish_reason(repair_response, "DraftChain(repair)")
                repair_text = self.session.extract_text(repair_response, default="")
                if snap_path is not None:
                    try:
                        (snap_path / "raw_response_repair.json").write_text(
                            json.dumps(repair_response, ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                    except Exception:
                        (snap_path / "raw_response_repair.txt").write_text(str(repair_response), encoding="utf-8")
                    (snap_path / "raw_response_repair_text.txt").write_text(repair_text or "", encoding="utf-8")
                out = self._parse_output(repair_text)
        if snap_path is not None:
            try:
                parsed_payload = {
                    FIELD_SUBTASKS: list(out.subtasks or []),
                    FIELD_FINAL_SUBTASK_ID: out.final_subtask_id,
                    FIELD_DEPENDENCIES: dict(out.dependencies or {}),
                    FIELD_DRAFT_QUESTION_EXPLICIT: out.draft_question_explicit,
                    FIELD_DRAFT_SOLUTION_OUTLINE: out.draft_solution_outline,
                    FIELD_DRAFT_ANSWER: out.draft_answer,
                    FIELD_REQUIRED_FACT_IDS: out.required_fact_ids,
                    FIELD_PRIMARY_REQUIRED_FACT_ID: out.primary_required_fact_id,
                    FIELD_REUSE_PLAN: out.reuse_plan,
                }
                (snap_path / "parsed_output.json").write_text(
                    json.dumps(parsed_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception:
                pass
        return out


__all__ = ["DraftChainInput", "DraftChainOutput", "DraftChainConfig", "DraftChainRunner"]
