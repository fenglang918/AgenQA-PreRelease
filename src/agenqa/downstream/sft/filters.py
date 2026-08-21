"""Filtering logic for AgenQA downstream SFT export."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from agenqa.graph.state import SolverResult

from .collector import StepSnapshot


@dataclass(frozen=True)
class Phase1FilterConfig:
    require_validation_passed: bool = True
    require_non_empty_plan: bool = True
    require_non_empty_solution: bool = True
    require_non_empty_answer: bool = True
    max_answer_contract_error_count: int = 0
    require_direct_path_variant: bool = True
    min_path_step: int = 2
    require_edge_medium_or_strong: bool = True
    require_path_strong_all_correct: bool = True
    require_explicit_well_posed_when_available: bool = True


@dataclass(frozen=True)
class FilterDecision:
    accepted: bool
    reasons: list[str]
    metadata: dict[str, Any]


def _bool_all(values: list[bool | None]) -> Optional[bool]:
    explicit = [v for v in values if v is not None]
    if not explicit:
        return None
    return all(v is True for v in explicit)


def summarize_solver_view(snapshot: StepSnapshot, target: str) -> dict[str, Any]:
    medium = snapshot.state.get_latest_solver(snapshot.artifacts.step, target, "medium")
    strong_rows = snapshot.state.get_latest_solver_results(snapshot.artifacts.step, target, "strong")
    strong_results = [result for _, result in strong_rows]
    return {
        "medium_correct": getattr(medium, "correct", None) if isinstance(medium, SolverResult) else None,
        "medium_question_well_posed": (
            getattr(medium, "question_well_posed", None) if isinstance(medium, SolverResult) else None
        ),
        "strong_count": len(strong_results),
        "strong_all_correct": bool(strong_results) and all(res.correct is True for res in strong_results),
        "strong_any_incorrect": any(res.correct is False for res in strong_results),
        "strong_all_well_posed": _bool_all([res.question_well_posed for res in strong_results]),
    }


def decide_edge(snapshot: StepSnapshot, config: Phase1FilterConfig) -> FilterDecision:
    reasons: list[str] = []
    fmt = snapshot.format_output
    draft = snapshot.draft_chain
    ac = snapshot.answer_contract_report or {}
    solver = summarize_solver_view(snapshot, "edge")

    if config.require_validation_passed and fmt.get("validation_passed") is not True:
        reasons.append("validation_failed")
    if config.require_non_empty_plan and not str(draft.get("draft_solution_outline") or "").strip():
        reasons.append("missing_plan")
    if config.require_non_empty_solution and not str(fmt.get("Solution") or "").strip():
        reasons.append("missing_solution")
    if config.require_non_empty_answer and not str(fmt.get("Answer") or "").strip():
        reasons.append("missing_answer")
    if int(ac.get("error_count", 0) or 0) > config.max_answer_contract_error_count:
        reasons.append("answer_contract_error")
    if config.require_edge_medium_or_strong:
        if not (solver["medium_correct"] is True or solver["strong_all_correct"] is True):
            reasons.append("edge_solver_not_stable")
    if config.require_explicit_well_posed_when_available:
        explicit = solver.get("strong_all_well_posed")
        if explicit is False:
            reasons.append("edge_not_well_posed")

    return FilterDecision(
        accepted=not reasons,
        reasons=reasons,
        metadata={"solver": solver},
    )


def decide_path_direct(snapshot: StepSnapshot, config: Phase1FilterConfig) -> FilterDecision:
    reasons: list[str] = []
    fmt = snapshot.format_output
    draft = snapshot.draft_chain
    ac = snapshot.answer_contract_report or {}
    path_kqa = snapshot.path_kqa or {}
    solver = summarize_solver_view(snapshot, "path")

    if snapshot.path_kqa is None:
        reasons.append("missing_path_kqa")
    if config.require_direct_path_variant:
        variant = str(path_kqa.get("variant") or "").strip().lower()
        source = str(path_kqa.get("source") or "").strip().lower()
        if variant not in {"", "direct"} and source != "path_fold_direct":
            reasons.append("not_direct_path_variant")
    if int(snapshot.artifacts.step) < config.min_path_step:
        reasons.append("path_not_multistep")
    if config.require_validation_passed and fmt.get("validation_passed") is not True:
        reasons.append("validation_failed")
    if config.require_non_empty_plan and not str(draft.get("draft_solution_outline") or "").strip():
        reasons.append("missing_plan")
    if config.require_non_empty_solution and not str(fmt.get("Solution") or "").strip():
        reasons.append("missing_solution")
    if config.require_non_empty_answer and not str(fmt.get("Answer") or "").strip():
        reasons.append("missing_answer")
    if int(ac.get("error_count", 0) or 0) > config.max_answer_contract_error_count:
        reasons.append("answer_contract_error")
    if config.require_path_strong_all_correct and solver["strong_all_correct"] is not True:
        reasons.append("path_solver_not_stable")
    if config.require_explicit_well_posed_when_available:
        explicit = solver.get("strong_all_well_posed")
        if explicit is False:
            reasons.append("path_not_well_posed")

    return FilterDecision(
        accepted=not reasons,
        reasons=reasons,
        metadata={"solver": solver},
    )
