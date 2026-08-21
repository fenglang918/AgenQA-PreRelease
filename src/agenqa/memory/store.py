"""Store helpers: write state + snapshots to disk (no AgentState IO)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, TYPE_CHECKING
import json
import logging

from agenqa.domain.known_tree import KnownTree
from agenqa.domain.contracts.solver_contract_text import compose_solver_question
from agenqa.domain.executable_schema import ExecutableRecord, ExecutableSubStep
from agenqa.domain.folded_question_schema import extract_executable_payload, extract_semantic_question_text, loads_folded_question

if TYPE_CHECKING:
    from agenqa.graph.state import AgentState

logger = logging.getLogger(__name__)


def _latest_kqa_payload(state: AgentState, *, compact_known: bool) -> Dict[str, Any]:
    latest = state.history[-1] if state.history else None
    if not latest:
        return {}
    try:
        step_idx = int(getattr(latest, "qa_idx", getattr(latest, "step", 0)) or 0)
    except Exception:
        step_idx = 0
    memory = KnownTree.normalize_memory(getattr(state, "memory", None))
    known_view = KnownTree.build_edge_solver_view(memory, step_idx)
    if compact_known:
        known_view = KnownTree.compact_kqa_known_view(known_view)
    known_text = KnownTree.to_json(known_view)
    payload: Dict[str, Any] = {
        "paper_id": latest.paper_id,
        "qa_idx": latest.qa_idx,
        "step": latest.qa_idx,  # backward compat
        "known": known_text,
        "question": latest.question,
        "world_contract_text": getattr(latest, "world_contract_text", None) or "",
        "question_for_solver": compose_solver_question(
            getattr(latest, "question", "") or "",
            getattr(latest, "world_contract_text", None),
        ),
        "answer": latest.answer,
        "chain": latest.chain,
    }
    if latest.subject:
        payload["subject"] = latest.subject
    return payload


def _write_jsonl_payload(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_pretty_json_sidecar(jsonl_path: Path, payload: Dict[str, Any]) -> None:
    """
    Write a human-readable JSON snapshot next to the canonical JSONL file.

    - Keeps the `.jsonl` as the stable machine-consumed artifact (1 row).
    - Writes a `.json` sidecar with `indent=2` and (when possible) a parsed `known_tree`.
    """

    if jsonl_path.suffix != ".jsonl":
        return

    pretty_path = jsonl_path.with_suffix(".json")
    pretty_payload: Dict[str, Any] = dict(payload)
    try:
        known = pretty_payload.get("known")
        if isinstance(known, str) and known.strip():
            pretty_payload["known_tree"] = json.loads(known)
    except Exception:
        pass

    pretty_path.write_text(json.dumps(pretty_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_pretty_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_folded_question_sidecar(step_dir: Path, *, variant: str, raw: Any) -> None:
    """Write a human-readable folded_question sidecar derived from the canonical string."""
    if not (isinstance(raw, str) and raw.strip()):
        return
    try:
        fq = loads_folded_question(raw)
        out = step_dir / f"path_folded_question_{variant}.json"
        out.write_text(json.dumps(fq.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        try:
            (step_dir / f"path_folded_question_{variant}.parse_error.txt").write_text(str(exc), encoding="utf-8")
        except Exception:
            pass


def _executable_question_base_payload(latest: ExecutableRecord) -> Dict[str, Any]:
    return {
        "record_type": latest.record_type,
        "paper_id": latest.paper_id,
        "problem_id": latest.problem_id,
        "qa_idx": latest.qa_idx,
        "step": latest.qa_idx,
        "background": latest.background,
        "required_dependencies": latest.required_dependencies,
        "estimated_difficulty": latest.estimated_difficulty,
        "subject": latest.subject,
        "source": latest.source,
    }


def _executable_question_payload(
    latest: ExecutableRecord,
    *,
    view: str,
    variant: str | None = None,
    sub_steps: list[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    payload = _executable_question_base_payload(latest)
    payload["view"] = view
    if variant is not None:
        payload["variant"] = variant
        payload["qa_idx_head"] = 0
        payload["qa_idx_tail"] = latest.qa_idx
        payload["step_head"] = 0
        payload["step_tail"] = latest.qa_idx
    payload["sub_steps"] = sub_steps if sub_steps is not None else [s.to_dict() for s in latest.sub_steps]
    # Keep the on-disk question file minimal: do NOT include test_cases, inputs_artifacts, or golden code.
    return payload


def save_state(state: AgentState, path: Path | None = None) -> Path:
    target = path or (Path(state.artifacts_dir) / "state.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(state.to_json(), encoding="utf-8")
    return target


def dump_latest_kqa_jsonl(state: AgentState, path: Path, *, compact_known: bool = True) -> Path:
    payload = _latest_kqa_payload(state, compact_known=compact_known)
    _write_jsonl_payload(path, payload)
    try:
        _write_pretty_json_sidecar(path, payload)
    except Exception:
        logger.warning("写入 pretty json 失败: %s", str(path))
    return path


def dump_edge_kqa_for_step(
    state: AgentState,
    step_dir: Path,
    filename: str = "edge_kqa.jsonl",
    *,
    compact_known: bool = True,
) -> Path:
    step_dir.mkdir(parents=True, exist_ok=True)
    payload = _latest_kqa_payload(state, compact_known=compact_known)
    path = step_dir / filename
    _write_jsonl_payload(path, payload)
    try:
        _write_pretty_json_sidecar(path, payload)
    except Exception:
        logger.warning("写入 pretty json 失败: %s", str(path))
    return path


def dump_path_kqa_for_step(
    state: AgentState,
    step_dir: Path,
    filename: str = "path_kqa.jsonl",
    *,
    fold_variant: str = "auto",
    compact_known: bool = True,
) -> Path:
    step_dir.mkdir(parents=True, exist_ok=True)
    path = step_dir / filename
    if not state.history:
        path.write_text("", encoding="utf-8")
        try:
            _write_pretty_json_sidecar(path, {})
        except Exception:
            logger.warning("写入 pretty json 失败: %s", str(path))
        return path

    tail = state.history[-1]
    try:
        tail_idx = int(tail.qa_idx) if tail.qa_idx is not None else state.qa_idx
    except Exception:
        tail_idx = state.qa_idx

    memory = KnownTree.normalize_memory(getattr(state, "memory", None))
    known_view = KnownTree.build_path_solver_view(memory, tail_idx)
    if compact_known:
        known_view = KnownTree.compact_kqa_known_view(known_view)
    known_text = KnownTree.to_json(known_view)

    # Path: use the folded question generated by PathFold.
    folded_direct = getattr(tail, "path_question_direct", None)
    folded_scaffolded = getattr(tail, "path_question_scaffolded", None)
    v = str(fold_variant or "auto").strip().lower()
    if v not in {"direct", "scaffolded"}:
        # Fallback: respect state preference if present; otherwise default to direct.
        try:
            pref = str(getattr(state, "path_kqa_variant", "") or "").strip().lower()
        except Exception:
            pref = ""
        v = pref if pref in {"direct", "scaffolded"} else "direct"

    folded_primary = folded_direct if v == "direct" else folded_scaffolded
    folded_alt = folded_scaffolded if v == "direct" else folded_direct

    if not (isinstance(folded_primary, str) and folded_primary.strip()):
        # No folded question available → skip path solve by writing an empty file.
        path.write_text("", encoding="utf-8")
        try:
            _write_pretty_json_sidecar(path, {})
        except Exception:
            logger.warning("写入 pretty json 失败: %s", str(path))
        return path

    try:
        question_primary = extract_semantic_question_text(folded_primary, variant=v)  # type: ignore[arg-type]
    except Exception:
        # Invalid folded payload → skip path solve (fold is optional).
        path.write_text("", encoding="utf-8")
        try:
            _write_pretty_json_sidecar(path, {})
        except Exception:
            logger.warning("写入 pretty json 失败: %s", str(path))
        return path

    payload = {
        "paper_id": tail.paper_id or state.paper_id,
        "qa_idx_head": 0,
        "qa_idx_tail": tail_idx,
        "step_head": 0,
        "step_tail": tail_idx,
        "known": known_text,
        "question": question_primary,
        "world_contract_text": getattr(tail, "world_contract_text", None) or "",
        "question_for_solver": compose_solver_question(
            question_primary,
            getattr(tail, "world_contract_text", None),
        ),
        "answer": tail.answer,
        "subject": tail.subject or state.subject,
        "source": f"path_fold_{v}",
        "variant": v,
        "chain": f"k0,q{tail_idx},a{tail_idx}",
    }
    _write_jsonl_payload(path, payload)
    try:
        _write_pretty_json_sidecar(path, payload)
    except Exception:
        logger.warning("写入 pretty json 失败: %s", str(path))
    try:
        _write_folded_question_sidecar(step_dir, variant=v, raw=folded_primary)
    except Exception:
        pass

    # Also write the other variant (optional) for inspection / ablation.
    if isinstance(folded_alt, str) and folded_alt.strip():
        try:
            try:
                alt_variant = "scaffolded" if v == "direct" else "direct"
                question_alt = extract_semantic_question_text(folded_alt, variant=alt_variant)  # type: ignore[arg-type]
            except Exception:
                question_alt = ""
            if not question_alt.strip():
                return path
            alt_path = step_dir / ("path_scaffolded_kqa.jsonl" if v == "direct" else "path_direct_kqa.jsonl")
            alt_payload = dict(payload)
            alt_payload["question"] = question_alt
            alt_payload["source"] = f"path_fold_{alt_variant}"
            alt_payload["variant"] = alt_variant
            _write_jsonl_payload(alt_path, alt_payload)
            try:
                _write_pretty_json_sidecar(alt_path, alt_payload)
            except Exception:
                logger.warning("写入 pretty json 失败: %s", str(alt_path))
            try:
                _write_folded_question_sidecar(step_dir, variant=alt_variant, raw=folded_alt)
            except Exception:
                pass
        except Exception:
            pass
    return path


def dump_director_decision_for_step(
    state: AgentState,
    step_dir: Path,
    qa_idx: int,
    filename: str = "director_decision.json",
) -> Path:
    step_dir.mkdir(parents=True, exist_ok=True)
    path = step_dir / filename
    if not state.last_decision:
        return path
    payload: Dict[str, Any] = {
        "qa_idx": int(qa_idx),
        "step": int(qa_idx),
        "operation": state.last_decision.operation,
        "reason": state.last_decision.reason,
        "params": state.last_decision.params or {},
    }
    try:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        logger.warning("写入 director_decision 失败: %s", str(path))
    return path


__all__ = [
    "save_state",
    "dump_latest_kqa_jsonl",
    "dump_edge_kqa_for_step",
    "dump_path_kqa_for_step",
    "dump_edge_executable_for_step",
    "dump_path_executable_for_step",
    "dump_director_decision_for_step",
]


def _latest_executable_payload(state: AgentState) -> Dict[str, Any]:
    latest = None
    try:
        latest = state.latest_executable_record()
    except Exception:
        latest = None
    if not isinstance(latest, ExecutableRecord):
        return {}
    payload = latest.to_dict()
    return payload


def dump_edge_executable_for_step(
    state: AgentState,
    step_dir: Path,
    filename: str = "edge_executable.jsonl",
) -> Path:
    step_dir.mkdir(parents=True, exist_ok=True)
    payload = _latest_executable_payload(state)
    path = step_dir / filename
    _write_jsonl_payload(path, payload)
    try:
        _write_pretty_json_sidecar(path, payload)
    except Exception:
        logger.warning("写入 pretty json 失败: %s", str(path))
    # Also write a minimal "pure question" snapshot for human inspection / sharing.
    try:
        latest = state.latest_executable_record()
        if isinstance(latest, ExecutableRecord):
            question_payload = _executable_question_payload(latest, view="edge")
            _write_pretty_json(step_dir / "edge_executable_question.json", question_payload)
    except Exception:
        pass
    return path


def _executable_path_view(sub_steps: list[ExecutableSubStep], *, variant: str) -> list[Dict[str, Any]]:
    if not sub_steps:
        return []
    head = sub_steps[0]
    tail = sub_steps[-1]
    if variant == "direct":
        return [head.to_dict(), tail.to_dict()] if len(sub_steps) > 1 else [head.to_dict()]
    if variant == "scaffolded":
        out: list[Dict[str, Any]] = [head.to_dict()]
        # Middle steps: keep only interface scaffold.
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
    # Unknown variant: keep as-is
    return [s.to_dict() for s in sub_steps]


def dump_path_executable_for_step(
    state: AgentState,
    step_dir: Path,
    filename: str = "path_executable.jsonl",
) -> Path:
    step_dir.mkdir(parents=True, exist_ok=True)
    path = step_dir / filename
    latest = None
    try:
        latest = state.latest_executable_record()
    except Exception:
        latest = None
    if latest is None:
        path.write_text("", encoding="utf-8")
        try:
            _write_pretty_json_sidecar(path, {})
        except Exception:
            logger.warning("写入 pretty json 失败: %s", str(path))
        return path

    if not isinstance(latest, ExecutableRecord):
        path.write_text("", encoding="utf-8")
        try:
            _write_pretty_json_sidecar(path, {})
        except Exception:
            logger.warning("写入 pretty json 失败: %s", str(path))
        return path

    tail_idx = latest.qa_idx
    payload: Dict[str, Any] = latest.to_dict()
    payload.update(
        {
            "qa_idx_head": 0,
            "qa_idx_tail": tail_idx,
            "step_head": 0,
            "step_tail": tail_idx,
            "variant": "direct",
            "source": (latest.source or "executable") + "_path_direct",
        }
    )

    def _folded_sub_steps(variant: str) -> list[Dict[str, Any]] | None:
        raw = getattr(latest, "path_question_direct", None) if variant == "direct" else getattr(latest, "path_question_scaffolded", None)
        if not (isinstance(raw, str) and raw.strip()):
            return None
        try:
            payload = extract_executable_payload(raw, variant=variant)  # type: ignore[arg-type]
        except Exception:
            return None
        sub = payload.get("sub_steps") if isinstance(payload, dict) else None
        return sub if isinstance(sub, list) else None

    payload["sub_steps"] = _folded_sub_steps("direct") or _executable_path_view(latest.sub_steps, variant="direct")
    _write_jsonl_payload(path, payload)
    try:
        _write_pretty_json_sidecar(path, payload)
    except Exception:
        logger.warning("写入 pretty json 失败: %s", str(path))
    try:
        _write_folded_question_sidecar(step_dir, variant="direct", raw=getattr(latest, "path_question_direct", None))
    except Exception:
        pass
    # Minimal "pure question" snapshot (direct path view).
    try:
        question_payload = _executable_question_payload(
            latest,
            view="path",
            variant="direct",
            sub_steps=payload.get("sub_steps") if isinstance(payload.get("sub_steps"), list) else None,
        )
        _write_pretty_json(step_dir / "path_executable_question.json", question_payload)
    except Exception:
        pass

    # Also write scaffolded variant (optional) for inspection/ablation.
    try:
        scaffolded_path = step_dir / "path_scaffolded_executable.jsonl"
        scaffolded_payload = dict(payload)
        scaffolded_payload["variant"] = "scaffolded"
        scaffolded_payload["source"] = (latest.source or "executable") + "_path_scaffolded"
        scaffolded_payload["sub_steps"] = _folded_sub_steps("scaffolded") or _executable_path_view(latest.sub_steps, variant="scaffolded")
        _write_jsonl_payload(scaffolded_path, scaffolded_payload)
        try:
            _write_pretty_json_sidecar(scaffolded_path, scaffolded_payload)
        except Exception:
            logger.warning("写入 pretty json 失败: %s", str(scaffolded_path))
        try:
            _write_folded_question_sidecar(step_dir, variant="scaffolded", raw=getattr(latest, "path_question_scaffolded", None))
        except Exception:
            pass
        # Minimal "pure question" snapshot (scaffolded path view).
        try:
            question_payload = _executable_question_payload(
                latest,
                view="path",
                variant="scaffolded",
                sub_steps=scaffolded_payload.get("sub_steps") if isinstance(scaffolded_payload.get("sub_steps"), list) else None,
            )
            _write_pretty_json(step_dir / "path_scaffolded_executable_question.json", question_payload)
        except Exception:
            pass
    except Exception:
        pass

    return path
