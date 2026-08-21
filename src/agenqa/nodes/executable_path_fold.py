from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from agenqa.prompts.executable_path_fold import EXECUTABLE_PATH_FOLD_V1, EXECUTABLE_PATH_FOLD_V1_EN
from agenqa.domain.executable_schema import ExecutableRecord, ExecutableSubStep, ExecutableTestCase
from agenqa.domain.folded_question_schema import dumps_folded_question
from agenqa.domain.path_fold_schema import (
    FIELD_FOLD_NOTES,
    FIELD_QUESTION_DIRECT,
    FIELD_QUESTION_SCAFFOLDED,
)
from agenqa.graph.state import AgentState
from agenqa.nodes.executable_e2e_oracle import build_e2e_sub_step, resolve_e2e_spec_from_memory
from agenqa.nodes.utils import idealab_session_id_for_step_node, with_idealab_session_id

logger = logging.getLogger(__name__)


def _executable_path_view(sub_steps: list[ExecutableSubStep], *, variant: str) -> list[Dict[str, Any]]:
    if not sub_steps:
        return []
    head = sub_steps[0]
    tail = sub_steps[-1]
    if variant == "direct":
        return [head.to_dict(), tail.to_dict()] if len(sub_steps) > 1 else [head.to_dict()]
    if variant == "scaffolded":
        out: list[Dict[str, Any]] = [head.to_dict()]
        for mid in sub_steps[1:-1]:
            out.append(
                {
                    "step_number": mid.step_number,
                    "step_description": "",
                    "function_header": mid.function_header,
                    "return_line": mid.return_line,
                    "step_background": None,
                }
            )
        if len(sub_steps) > 1:
            out.append(tail.to_dict())
        return out
    return [s.to_dict() for s in sub_steps]


def _build_path_payload(record: ExecutableRecord, *, variant: str) -> Dict[str, Any]:
    tail_idx = record.qa_idx
    payload: Dict[str, Any] = {
        "record_type": record.record_type,
        "paper_id": record.paper_id,
        "problem_id": record.problem_id,
        "qa_idx": record.qa_idx,
        "step": record.qa_idx,
        "background": record.background,
        "required_dependencies": record.required_dependencies,
        "sub_steps": _executable_path_view(record.sub_steps, variant=variant),
        "test_cases": {
            step: [t.to_dict() for t in tests]
            for step, tests in (record.test_cases or {}).items()
            if isinstance(step, str)
        },
        "estimated_difficulty": record.estimated_difficulty,
        "subject": record.subject,
        "source": record.source,
        "qa_idx_head": 0,
        "qa_idx_tail": tail_idx,
        "step_head": 0,
        "step_tail": tail_idx,
        "variant": variant,
        "path_view": True,
    }
    return payload


def _fallback_fold(
    agent_conf: Dict[str, Any],
    state: AgentState,
    record: ExecutableRecord,
    *,
    enable_e2e_oracle: bool,
) -> Dict[str, Any]:
    if not record.sub_steps:
        direct = _build_path_payload(record, variant="direct")
        scaffolded = _build_path_payload(record, variant="scaffolded")
        notes = "deterministic_fold: empty_sub_steps; no ground-truth code included."
        return {
            FIELD_QUESTION_DIRECT: direct,
            FIELD_QUESTION_SCAFFOLDED: scaffolded,
            FIELD_FOLD_NOTES: notes,
        }

    if enable_e2e_oracle:
        try:
            try:
                step_idx = int(record.qa_idx or getattr(state, "step", 0) or 0)
            except Exception:
                step_idx = 0
            spec = resolve_e2e_spec_from_memory(getattr(state, "memory", None), step=step_idx)
            if spec is None:
                raise ValueError("step_cert.e2e_spec missing")
            raw_tests = (record.test_cases or {}).get("e2e") or []
            tests: list[Dict[str, Any]] = []
            for t in raw_tests:
                if isinstance(t, ExecutableTestCase):
                    tests.append(t.to_dict())
                elif isinstance(t, dict):
                    tests.append(dict(t))
            if not tests:
                raise ValueError("record.test_cases['e2e'] empty")

            e2e_step = build_e2e_sub_step(record=record, spec=spec).to_dict()

            direct_payload = _build_path_payload(record, variant="direct")
            direct_payload["sub_steps"] = [e2e_step]
            direct_payload["test_cases"] = {"e2e": tests}

            scaffolded_payload = _build_path_payload(record, variant="scaffolded")
            interfaces = [
                {
                    "step_number": s.step_number,
                    "step_description": "",
                    "function_header": s.function_header,
                    "return_line": s.return_line,
                    "step_background": None,
                }
                for s in (record.sub_steps or [])
                if s.step_number
            ]
            scaffolded_payload["sub_steps"] = [e2e_step] + interfaces
            scaffolded_payload["test_cases"] = {"e2e": tests}

            notes = (
                "deterministic_fold: e2e_differential_oracle (solve from step_cert.e2e_spec + static expected); "
                "scaffolded includes interface hints only; no ground-truth code included."
            )
            return {
                FIELD_QUESTION_DIRECT: direct_payload,
                FIELD_QUESTION_SCAFFOLDED: scaffolded_payload,
                FIELD_FOLD_NOTES: notes,
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "executable_path_fold deterministic e2e fold failed; fall back to tail-only fold. error=%s", str(exc)
            )

    tail = record.sub_steps[-1]
    tail_id = str(tail.step_number or "").strip()
    tests_tail = (record.test_cases or {}).get(tail_id) or []
    tests_map = {tail_id: [t.to_dict() for t in tests_tail]} if tail_id else {}

    # Deterministic fold is always "e2e-first" for the tail step so that
    # path evaluation can treat sub_steps[0] as the target signature.
    direct_payload = _build_path_payload(record, variant="direct")
    direct_payload["sub_steps"] = [tail.to_dict()]
    direct_payload["test_cases"] = tests_map

    scaffolded_payload = _build_path_payload(record, variant="scaffolded")
    interfaces = _executable_path_view(record.sub_steps, variant="scaffolded")
    if interfaces and tail_id and str(interfaces[-1].get("step_number") or "") == tail_id:
        interfaces = interfaces[:-1]
    scaffolded_payload["sub_steps"] = [tail.to_dict()] + interfaces
    scaffolded_payload["test_cases"] = tests_map

    notes = "deterministic_fold: direct=e2e(tail) only; scaffolded=e2e(tail)+interfaces(prev); no ground-truth code included."
    return {
        FIELD_QUESTION_DIRECT: direct_payload,
        FIELD_QUESTION_SCAFFOLDED: scaffolded_payload,
        FIELD_FOLD_NOTES: notes,
    }


def _build_fold_inputs(record: ExecutableRecord) -> tuple[str, str]:
    head_step = record.sub_steps[0] if record.sub_steps else None
    tail_step = record.sub_steps[-1] if record.sub_steps else None
    tail_id = tail_step.step_number if tail_step else ""
    tests_tail: list[Dict[str, Any]] = []
    raw_tests = (record.test_cases or {}).get(tail_id) or []
    for t in raw_tests:
        if isinstance(t, ExecutableTestCase):
            tests_tail.append(t.to_dict())
        elif isinstance(t, dict):
            tests_tail.append(dict(t))

    premise_obj = {
        "problem_id": record.problem_id,
        "background": record.background,
        "required_dependencies": record.required_dependencies,
        "head_step": head_step.to_dict() if head_step else {},
    }
    history_obj = {
        "problem_id": record.problem_id,
        "tail_step_number": tail_id,
        "sub_steps": [s.to_dict() for s in record.sub_steps],
        "test_cases_tail": tests_tail,
    }
    return (
        json.dumps(premise_obj, ensure_ascii=False),
        json.dumps(history_obj, ensure_ascii=False),
    )


def apply_executable_path_fold(
    agent_conf: Dict[str, Any],
    state: AgentState,
    record: ExecutableRecord,
    *,
    step_dir: Optional[Path] = None,
    op_name: str = "extend",
) -> Dict[str, str]:
    """Populate `record.path_question_*` for executable track and return fold dict.

    Always applies a deterministic fallback fold. If enabled via config, it will
    additionally call an LLM-based ExecutablePathFold role and overwrite the fields.
    """
    ops_conf = agent_conf.get("operators") or {}
    op_conf = (ops_conf.get(op_name) or {}) if isinstance(ops_conf, dict) else {}
    enable_e2e_oracle = bool(op_conf.get("enable_e2e_oracle", False) or op_conf.get("e2e_oracle_enabled", False))

    fallback = _fallback_fold(agent_conf, state, record, enable_e2e_oracle=enable_e2e_oracle)
    direct_payload = fallback.get(FIELD_QUESTION_DIRECT)
    scaffolded_payload = fallback.get(FIELD_QUESTION_SCAFFOLDED)
    if not isinstance(direct_payload, dict):
        raise ValueError("executable_path_fold deterministic fold payload is not a dict (direct)")
    if not isinstance(scaffolded_payload, dict):
        raise ValueError("executable_path_fold deterministic fold payload is not a dict (scaffolded)")
    record.path_question_direct = dumps_folded_question(track="executable", variant="direct", payload=direct_payload)
    record.path_question_scaffolded = dumps_folded_question(track="executable", variant="scaffolded", payload=scaffolded_payload)
    record.path_fold_notes = str(fallback.get(FIELD_FOLD_NOTES) or "")

    fallback_wrapped = {
        FIELD_QUESTION_DIRECT: record.path_question_direct,
        FIELD_QUESTION_SCAFFOLDED: record.path_question_scaffolded,
        FIELD_FOLD_NOTES: record.path_fold_notes,
    }

    agent_block = (agent_conf.get("agent") or {}) if isinstance(agent_conf.get("agent"), dict) else {}
    agent_lang = str(agent_block.get("lang") or "").lower() or None
    use_en = str(agent_lang or "").lower().strip() in {"en", "english"}

    enabled = bool(op_conf.get("enable_path_fold", False) or op_conf.get("path_fold_enabled", False))
    if not enabled:
        return fallback_wrapped

    fold_generator = op_conf.get("path_fold_generator") or op_conf.get("struct_generator") or op_conf.get("generator")
    if not fold_generator:
        director_conf = agent_conf.get("director") or {}
        fold_generator = director_conf.get("generator") or {
            "service_type": "private_endpoint",
            "service_id": director_conf.get("service_id"),
        }
    if not isinstance(fold_generator, dict) or not fold_generator:
        logger.warning("ExecutablePathFold enabled but generator is missing; keep deterministic fold.")
        return fallback_wrapped

    try:
        step_idx = int(record.qa_idx or getattr(state, "step", 0) or 0)
    except Exception:
        step_idx = 0
    fold_generator = with_idealab_session_id(fold_generator, idealab_session_id_for_step_node(state, op_name, step_idx))

    prompt_text = EXECUTABLE_PATH_FOLD_V1_EN if use_en else EXECUTABLE_PATH_FOLD_V1
    prompt_path = Path(op_conf.get("path_fold_prompt_path") or "src/agenqa/prompts/executable_path_fold.prompt")

    try:
        try:
            from agenqa.skills.path_fold import PathFoldConfig, PathFoldInput, PathFoldRunner
        except Exception as exc:  # noqa: BLE001
            logger.warning("ExecutablePathFold enabled but optional deps are missing; keep deterministic fold: %s", str(exc))
            return fallback_wrapped

        fold_runner = PathFoldRunner(
            PathFoldConfig(
                generator=fold_generator,
                prompt_path=prompt_path,
                prompt_text=prompt_text,
                lang=agent_lang,
            )
        )
        premise_bank_json, history_json = _build_fold_inputs(record)
        fold_in = PathFoldInput(
            step=step_idx,
            question_type="executable",
            premise_bank_json=premise_bank_json,
            history_json=history_json,
        )
        snapshot_dir = (step_dir / "subruns_raw" / "executable_path_fold") if isinstance(step_dir, Path) else None
        fold_out = fold_runner.run_one(
            fold_in,
            snapshot_dir=snapshot_dir,
            unified_prompt_dir=Path(state.artifacts_dir) / "00_Prompts_Snapshot",
            name_prefix="prompt_used.executable_path_fold.",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("ExecutablePathFold failed (step=%s): %s", str(step_idx), str(exc))
        return fallback_wrapped

    # LLM fold outputs for executable are JSON objects; wrap them into typed folded_question.
    direct_obj: Dict[str, Any] | None = None
    scaffolded_obj: Dict[str, Any] | None = None
    try:
        if isinstance(fold_out.question_direct, str) and fold_out.question_direct.strip():
            raw = json.loads(fold_out.question_direct)
            direct_obj = raw if isinstance(raw, dict) else None
        if isinstance(fold_out.question_scaffolded, str) and fold_out.question_scaffolded.strip():
            raw = json.loads(fold_out.question_scaffolded)
            scaffolded_obj = raw if isinstance(raw, dict) else None
    except Exception:
        direct_obj, scaffolded_obj = None, None

    if isinstance(direct_obj, dict):
        record.path_question_direct = dumps_folded_question(track="executable", variant="direct", payload=direct_obj)
    if isinstance(scaffolded_obj, dict):
        record.path_question_scaffolded = dumps_folded_question(track="executable", variant="scaffolded", payload=scaffolded_obj)
    record.path_fold_notes = fold_out.fold_notes
    return {
        FIELD_QUESTION_DIRECT: record.path_question_direct,
        FIELD_QUESTION_SCAFFOLDED: record.path_question_scaffolded,
        FIELD_FOLD_NOTES: record.path_fold_notes,
    }


__all__ = [
    "apply_executable_path_fold",
]
