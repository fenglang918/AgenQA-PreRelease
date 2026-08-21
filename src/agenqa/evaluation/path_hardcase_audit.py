from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


_ROUND_RE = re.compile(r"^round_(\d+)$")
_STEP_STAGE_RE = re.compile(r"^step_(\d+)_(.+)$")
_STRONG_FILE_RE = re.compile(r"^solve_path_strong_(\d+)\.jsonl$")
_OUTPUT_SPEC_RE = re.compile(r"L4\.answer_output_spec_\d+")
_QUESTION_TYPE_RE = re.compile(r'"question_type"\s*:\s*"([^"]+)"')
_TERMINAL_BATCH_STATUSES = {"success", "failed", "cancelled", "error"}
_SUSPECT_FALSE_NEGATIVE_LABELS = {
    "suspect_world_contract",
    "suspect_answer_contract",
    "suspect_judge_false_negative",
}


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(_read_text(path))


def _read_jsonl(path: Path) -> Iterable[tuple[int, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            raw = line.strip()
            if not raw:
                continue
            try:
                yield lineno, json.loads(raw)
            except Exception as exc:  # pragma: no cover - defensive
                yield lineno, {"__error__": f"{type(exc).__name__}: {exc}", "__raw__": raw[:2000]}


def _read_first_jsonl_row(path: Path) -> Any | None:
    for _, row in _read_jsonl(path):
        return row
    return None


def _safe_get(obj: Any, *keys: str, default: Any = None) -> Any:
    cur = obj
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    text = text.replace("\r", "")
    text = re.sub(r"\s+", "", text)
    if text.startswith("$$") and text.endswith("$$") and len(text) >= 4:
        text = text[2:-2]
    return text


def _read_optional_json(path: Path) -> tuple[Any | None, str | None]:
    if not path.exists():
        return None, None
    try:
        return _read_json(path), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _infer_round_and_stage(run_dir: Path, candidate_dir: Path) -> tuple[int | None, str]:
    rel = candidate_dir.relative_to(run_dir)
    parts = rel.parts
    round_id = None
    stage = "unknown"
    if parts:
        match = _ROUND_RE.fullmatch(parts[0])
        if match:
            round_id = int(match.group(1))
    if len(parts) >= 2:
        stage = parts[1]
    return round_id, stage


def _infer_stage_step(stage: str) -> int | None:
    match = _STEP_STAGE_RE.fullmatch(stage)
    if not match:
        return None
    return int(match.group(1))


def _load_path_candidate(candidate_dir: Path) -> tuple[Any | None, Path | None, str | None]:
    json_path = candidate_dir / "path_kqa.json"
    if json_path.exists():
        try:
            return _read_json(json_path), json_path, None
        except Exception as exc:
            return None, json_path, f"{type(exc).__name__}: {exc}"
    jsonl_path = candidate_dir / "path_kqa.jsonl"
    if jsonl_path.exists():
        try:
            return _read_first_jsonl_row(jsonl_path), jsonl_path, None
        except Exception as exc:
            return None, jsonl_path, f"{type(exc).__name__}: {exc}"
    return None, None, None


def _load_state(run_dir: Path) -> Any | None:
    state_path = run_dir / "state.json"
    if not state_path.exists():
        return None
    try:
        return _read_json(state_path)
    except Exception:
        return None


def _normalize_run_status(batch_result: dict[str, Any] | None, state: Any | None) -> tuple[str, str]:
    if isinstance(batch_result, dict):
        status = str(batch_result.get("status") or "").strip() or "unknown"
        error = str(batch_result.get("error") or "").strip()
        return status, error
    if isinstance(state, dict) and state.get("stop_reason"):
        return "completed", ""
    return "running", ""


def _discover_run_dirs(
    *,
    batch_dir: Path | None,
    runs_root: Path | None,
    completed_only: bool,
    include_running: bool,
) -> tuple[list[Path], dict[str, Any]]:
    context: dict[str, Any] = {
        "batch_id": None,
        "batch_dir": str(batch_dir.resolve()) if batch_dir else None,
        "results_by_run_id": {},
    }
    run_dirs: list[Path] = []

    if batch_dir is not None:
        manifest, _ = _read_optional_json(batch_dir / "batch_manifest.json")
        if isinstance(manifest, dict):
            context["batch_id"] = manifest.get("batch_id") or batch_dir.name
        else:
            context["batch_id"] = batch_dir.name

        results_path = batch_dir / "batch_results.jsonl"
        results_by_run_id: dict[str, Any] = {}
        if results_path.exists():
            for _, row in _read_jsonl(results_path):
                if not isinstance(row, dict):
                    continue
                run_id = row.get("run_id")
                if isinstance(run_id, str) and run_id:
                    results_by_run_id[run_id] = row
        context["results_by_run_id"] = results_by_run_id

        for path in sorted((batch_dir / "runs").glob("run_*")):
            if not path.is_dir():
                continue
            state = _load_state(path)
            run_id = None
            if isinstance(state, dict):
                run_id = state.get("run_id")
            if not isinstance(run_id, str) or not run_id:
                run_id = path.name.removeprefix("run_")
            batch_result = results_by_run_id.get(run_id)
            is_terminal = isinstance(batch_result, dict) and str(batch_result.get("status") or "") in _TERMINAL_BATCH_STATUSES
            if completed_only and not is_terminal:
                continue
            if not completed_only and not include_running and not is_terminal:
                continue
            run_dirs.append(path.resolve())
        return run_dirs, context

    if runs_root is None:
        raise ValueError("batch_dir and runs_root cannot both be None")

    for path in sorted(runs_root.rglob("run_*")):
        if not path.is_dir():
            continue
        if not ((path / "state.json").exists() or (path / "run_config.json").exists()):
            continue
        state = _load_state(path)
        is_terminal = bool(isinstance(state, dict) and state.get("stop_reason"))
        if completed_only and not is_terminal:
            continue
        if not completed_only and not include_running and not is_terminal:
            continue
        run_dirs.append(path.resolve())
    return run_dirs, context


def _candidate_dirs_for_run(run_dir: Path) -> list[Path]:
    seen: set[Path] = set()
    for path in run_dir.rglob("path_kqa.json"):
        parent = path.parent.resolve()
        rel_parts = parent.relative_to(run_dir).parts
        if "subruns_raw" in rel_parts or "solve" in rel_parts:
            continue
        seen.add(parent)
    for path in run_dir.rglob("path_kqa.jsonl"):
        parent = path.parent.resolve()
        rel_parts = parent.relative_to(run_dir).parts
        if "subruns_raw" in rel_parts or "solve" in rel_parts:
            continue
        seen.add(parent)
    dirs = list(seen)
    dirs.sort(key=lambda p: str(p.relative_to(run_dir)))
    return dirs


def load_run_state_for_audit(run_dir: Path) -> Any | None:
    return _load_state(run_dir)


def candidate_dirs_for_run(run_dir: Path) -> list[Path]:
    return _candidate_dirs_for_run(run_dir)


def audit_candidate_dir(
    *,
    run_dir: Path,
    candidate_dir: Path,
    state: Any | None,
    batch_id: str | None,
    batch_result: dict[str, Any] | None,
) -> dict[str, Any]:
    return _audit_candidate(
        run_dir=run_dir,
        candidate_dir=candidate_dir,
        state=state,
        batch_id=batch_id,
        batch_result=batch_result,
    )


def _find_next_director_summary(run_dir: Path, round_id: int | None, step: int | None) -> tuple[dict[str, Any] | None, Path | None]:
    if round_id is None or step is None:
        return None, None
    next_path = run_dir / f"round_{round_id + 1}" / "director" / "director_decision.json"
    if not next_path.exists():
        return None, None
    obj, err = _read_optional_json(next_path)
    if err or not isinstance(obj, dict):
        return None, None
    if _coerce_int(obj.get("step")) != step:
        return None, None
    summary = _safe_get(obj, "params", "solver_context", "metrics", "path", "strong_summary", default=None)
    if not isinstance(summary, dict):
        return None, None
    return summary, next_path


def _summarize_solver_index(state: Any, step: int | None, round_id: int | None) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    if not isinstance(state, dict) or step is None or round_id is None:
        return None, []
    solver_index = state.get("solver_index")
    if not isinstance(solver_index, dict):
        return None, []
    step_rec = solver_index.get(str(step))
    if not isinstance(step_rec, dict):
        return None, []
    round_rec = step_rec.get(str(round_id))
    if not isinstance(round_rec, dict):
        return None, []
    path_view = round_rec.get("path")
    if not isinstance(path_view, dict):
        return None, []
    entries: list[dict[str, Any]] = []
    for key, payload in path_view.items():
        if key != "strong" and not str(key).startswith("strong_"):
            continue
        if isinstance(payload, dict):
            entries.append({"tier": key, **payload})
    if not entries:
        return None, []
    correct_count = sum(1 for item in entries if item.get("correct") is True)
    incorrect_count = sum(1 for item in entries if item.get("correct") is False)
    unknown_count = len(entries) - correct_count - incorrect_count
    summary = {
        "total": len(entries),
        "explicit_total": correct_count + incorrect_count,
        "correct_count": correct_count,
        "incorrect_count": incorrect_count,
        "unknown_count": unknown_count,
        "all_correct": incorrect_count == 0 and unknown_count == 0 and len(entries) > 0,
        "any_incorrect": incorrect_count > 0,
    }
    return summary, entries


def _solve_file_sort_key(path: Path) -> tuple[int, str]:
    match = _STRONG_FILE_RE.fullmatch(path.name)
    if match and match.group(1) is not None:
        return int(match.group(1)), path.name
    return 10_000, path.name


def _load_main_path_strong_rows(candidate_dir: Path) -> list[dict[str, Any]]:
    solve_dir = candidate_dir / "solve"
    if not solve_dir.exists():
        return []
    files = sorted(
        [path for path in solve_dir.glob("solve_path_strong_*.jsonl") if _STRONG_FILE_RE.fullmatch(path.name)],
        key=_solve_file_sort_key,
    )
    rows: list[dict[str, Any]] = []
    for path in files:
        row = _read_first_jsonl_row(path)
        if row is None:
            rows.append({"file": path, "row": None})
        else:
            rows.append({"file": path, "row": row})
    return rows


def _summarize_solve_rows(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    correct_count = 0
    incorrect_count = 0
    unknown_count = 0
    for item in rows:
        row = item.get("row")
        if not isinstance(row, dict):
            unknown_count += 1
            continue
        correct = row.get("correct")
        if correct is True:
            correct_count += 1
        elif correct is False:
            incorrect_count += 1
        else:
            unknown_count += 1
    total = len(rows)
    return {
        "total": total,
        "explicit_total": correct_count + incorrect_count,
        "correct_count": correct_count,
        "incorrect_count": incorrect_count,
        "unknown_count": unknown_count,
        "all_correct": incorrect_count == 0 and unknown_count == 0 and total > 0,
        "any_incorrect": incorrect_count > 0,
    }


def _scan_candidate_pathfold(candidate_dir: Path, kqa: Any | None) -> list[str]:
    problems: list[str] = []
    direct_question = None
    scaffolded_question = None

    direct_obj, _ = _read_optional_json(candidate_dir / "path_folded_question_direct.json")
    if isinstance(direct_obj, dict):
        direct_question = _safe_get(direct_obj, "payload", "question_text")
    scaffolded_obj, _ = _read_optional_json(candidate_dir / "path_folded_question_scaffolded.json")
    if isinstance(scaffolded_obj, dict):
        scaffolded_question = _safe_get(scaffolded_obj, "payload", "question_text")
    scaffolded_kqa, _ = _read_optional_json(candidate_dir / "path_scaffolded_kqa.json")

    if isinstance(kqa, dict):
        if direct_question is not None and kqa.get("question") != direct_question:
            problems.append("pathfold_direct_question_mismatch")
        if not (kqa.get("answer") or "").strip():
            problems.append("path_answer_empty")
        variant = kqa.get("variant")
        if variant not in (None, "direct", "auto"):
            problems.append("path_variant_unexpected")
    if isinstance(scaffolded_kqa, dict):
        if scaffolded_question is not None and scaffolded_kqa.get("question") != scaffolded_question:
            problems.append("pathfold_scaffolded_question_mismatch")
        if not (scaffolded_kqa.get("answer") or "").strip():
            problems.append("path_scaffolded_answer_empty")
        variant = scaffolded_kqa.get("variant")
        if variant not in (None, "scaffolded", "auto"):
            problems.append("path_scaffolded_variant_unexpected")
    if isinstance(kqa, dict) and isinstance(scaffolded_kqa, dict):
        if (kqa.get("answer") or "").strip() != (scaffolded_kqa.get("answer") or "").strip():
            problems.append("pathfold_direct_scaffolded_answers_differ")
    return problems


def _collect_judge_suspect_codes(rows: list[dict[str, Any]]) -> list[str]:
    codes: list[str] = []
    for item in rows:
        row = item.get("row")
        if not isinstance(row, dict) or row.get("correct") is not False:
            continue
        solve = _clean_text(row.get("solve"))
        answer = _clean_text(row.get("answer"))
        if solve and answer and solve == answer:
            codes.append("solve_equals_answer_but_correct_false")
        expression_judge = row.get("expression_judge")
        if isinstance(expression_judge, dict):
            for payload in expression_judge.values():
                if not isinstance(payload, dict):
                    continue
                if payload.get("equivalent") == "yes":
                    codes.append("expression_judge_equivalent_yes_but_correct_false")
                    break
    return sorted(set(codes))


def _load_answer_contract_report(candidate_dir: Path) -> tuple[Any | None, Path | None]:
    path = candidate_dir / "answer_contract_report.json"
    if not path.exists():
        return None, None
    obj, _ = _read_optional_json(path)
    return obj, path


def _load_ambiguity_report(candidate_dir: Path) -> tuple[Any | None, Path | None]:
    path = candidate_dir / "solve" / "ambiguity_report.json"
    if not path.exists():
        return None, None
    obj, _ = _read_optional_json(path)
    return obj, path


def _load_director_decision(candidate_dir: Path) -> tuple[Any | None, Path | None]:
    path = candidate_dir / "director_decision.json"
    if not path.exists():
        return None, None
    obj, _ = _read_optional_json(path)
    return obj, path


def _extract_question_type(kqa: Any | None, answer_contract_report: Any | None, director_decision: Any | None) -> str | None:
    if isinstance(answer_contract_report, dict):
        qtype = answer_contract_report.get("question_type")
        if isinstance(qtype, str) and qtype:
            return qtype
    if isinstance(director_decision, dict):
        qtype = _safe_get(director_decision, "params", "question_type")
        if isinstance(qtype, str) and qtype:
            return qtype
    if isinstance(kqa, dict):
        for key in ("question_type",):
            qtype = kqa.get(key)
            if isinstance(qtype, str) and qtype:
                return qtype
        for text_key in ("question_for_solver", "world_contract_text"):
            text = kqa.get(text_key)
            if isinstance(text, str):
                match = _QUESTION_TYPE_RE.search(text)
                if match:
                    return match.group(1)
    return None


def _answer_contract_codes(
    *,
    hard_case_observed: bool,
    answer_contract_report: Any | None,
    kqa: Any | None,
    question_type: str | None,
) -> list[str]:
    if not hard_case_observed:
        return []
    codes: list[str] = []
    if isinstance(answer_contract_report, dict):
        if _coerce_int(answer_contract_report.get("error_count")):
            codes.append("answer_contract_report_error")
        if _coerce_int(answer_contract_report.get("warn_count")):
            codes.append("answer_contract_report_warn")
        if not answer_contract_report.get("answer_contract_ids"):
            codes.append("answer_contract_ids_missing")
    else:
        if question_type in {"Derivation", "Numeric"}:
            codes.append("answer_contract_report_missing")

    if isinstance(kqa, dict):
        question_for_solver = kqa.get("question_for_solver")
        world_contract_text = kqa.get("world_contract_text")
        has_world_output_spec = isinstance(world_contract_text, str) and bool(_OUTPUT_SPEC_RE.search(world_contract_text))
        has_visible_output_spec = isinstance(question_for_solver, str) and bool(_OUTPUT_SPEC_RE.search(question_for_solver))
        if has_world_output_spec and not has_visible_output_spec:
            codes.append("solver_visible_output_spec_missing")
    return sorted(set(codes))


def _world_contract_codes(
    *,
    hard_case_observed: bool,
    ambiguity_report: Any | None,
    pathfold_problems: list[str],
    solver_entries: list[dict[str, Any]],
) -> list[str]:
    if not hard_case_observed:
        return []
    codes: list[str] = []
    if isinstance(ambiguity_report, dict):
        if ambiguity_report.get("type1_suspected") is True:
            codes.append("ambiguity_type1_suspected")
        votes = ambiguity_report.get("wellposed_votes")
        if isinstance(votes, dict):
            if _coerce_int(votes.get("false")):
                codes.append("ambiguity_wellposed_false_vote")
            if _coerce_int(votes.get("null")):
                codes.append("ambiguity_wellposed_null_vote")
    for item in solver_entries:
        wellposed = item.get("question_well_posed")
        if wellposed is False:
            codes.append("solver_question_not_wellposed")
            break
    for problem in pathfold_problems:
        if problem in {"path_answer_empty", "path_scaffolded_answer_empty"}:
            continue
        codes.append(problem)
    return sorted(set(codes))


def _incomplete_codes(
    *,
    kqa: Any | None,
    kqa_error: str | None,
    strong_summary: dict[str, Any] | None,
    solve_rows: list[dict[str, Any]],
    pathfold_problems: list[str],
) -> list[str]:
    codes: list[str] = []
    if kqa is None:
        codes.append("path_kqa_missing")
    if kqa_error:
        codes.append("path_kqa_parse_failed")
    if "path_answer_empty" in pathfold_problems:
        codes.append("path_answer_empty")
    if not solve_rows and strong_summary is None:
        codes.append("path_strong_results_missing")
    summary_total = _coerce_int(_safe_get(strong_summary, "total"))
    if summary_total and solve_rows and len(solve_rows) < summary_total:
        codes.append("path_strong_partial")
    for item in solve_rows:
        row = item.get("row")
        if not isinstance(row, dict):
            codes.append("path_strong_row_missing")
            continue
        finish_reason = str(row.get("finish_reason") or "").lower()
        gateway_finish_reason = str(row.get("gateway_finish_reason") or "").lower()
        correct = row.get("correct")
        if (finish_reason == "length" or gateway_finish_reason == "length") and correct is not True:
            codes.append("solver_output_truncated")
        if (finish_reason == "timeout" or gateway_finish_reason == "timeout") and correct is not True:
            codes.append("solver_output_timeout")
    return sorted(set(codes))


def _review_priority(primary_label: str) -> str:
    if primary_label in _SUSPECT_FALSE_NEGATIVE_LABELS:
        return "high"
    if primary_label == "insufficient_evidence":
        return "medium"
    return "none"


def _build_candidate_id(batch_id: str | None, run_id: str, paper_id: str | None, round_id: int | None, stage: str) -> str:
    batch_part = batch_id or "no_batch"
    paper_part = paper_id or "unknown_paper"
    round_part = f"round_{round_id}" if round_id is not None else "round_unknown"
    return f"{batch_part}::{run_id}::{paper_part}::{round_part}::{stage}"


def _majority_threshold(total: int) -> int:
    if total <= 0:
        return 0
    return math.ceil(total / 2)


def _audit_candidate(
    *,
    run_dir: Path,
    candidate_dir: Path,
    state: Any | None,
    batch_id: str | None,
    batch_result: dict[str, Any] | None,
) -> dict[str, Any]:
    source_paths: list[str] = []
    kqa, kqa_path, kqa_error = _load_path_candidate(candidate_dir)
    if kqa_path is not None:
        source_paths.append(str(kqa_path.resolve()))

    round_id, stage = _infer_round_and_stage(run_dir, candidate_dir)
    step = None
    if isinstance(kqa, dict):
        step = _coerce_int(kqa.get("step_tail"))
        if step is None:
            step = _coerce_int(kqa.get("step"))
        if step is None:
            step = _coerce_int(kqa.get("qa_idx_tail"))
    if step is None:
        step = _infer_stage_step(stage)

    run_id = run_dir.name.removeprefix("run_")
    if isinstance(state, dict) and isinstance(state.get("run_id"), str):
        run_id = state["run_id"]
    paper_id = kqa.get("paper_id") if isinstance(kqa, dict) else None
    if not isinstance(paper_id, str) or not paper_id:
        paper_id = None

    answer_contract_report, answer_contract_path = _load_answer_contract_report(candidate_dir)
    if answer_contract_path is not None:
        source_paths.append(str(answer_contract_path.resolve()))
    ambiguity_report, ambiguity_path = _load_ambiguity_report(candidate_dir)
    if ambiguity_path is not None:
        source_paths.append(str(ambiguity_path.resolve()))
    director_decision, director_decision_path = _load_director_decision(candidate_dir)
    if director_decision_path is not None:
        source_paths.append(str(director_decision_path.resolve()))

    question_type = _extract_question_type(kqa, answer_contract_report, director_decision)
    pathfold_problems = _scan_candidate_pathfold(candidate_dir, kqa)

    next_summary, next_director_path = _find_next_director_summary(run_dir, round_id, step)
    if next_director_path is not None:
        source_paths.append(str(next_director_path.resolve()))
    state_summary, solver_entries = _summarize_solver_index(state, step, round_id)
    solve_rows = _load_main_path_strong_rows(candidate_dir)
    for item in solve_rows:
        source_paths.append(str(Path(item["file"]).resolve()))
    solve_summary = _summarize_solve_rows(solve_rows)

    summary_source = None
    strong_summary = None
    if isinstance(next_summary, dict):
        strong_summary = next_summary
        summary_source = "next_director_metrics"
    elif isinstance(state_summary, dict):
        strong_summary = state_summary
        summary_source = "state_solver_index"
    elif isinstance(solve_summary, dict):
        strong_summary = solve_summary
        summary_source = "solve_files"

    incomplete_codes = _incomplete_codes(
        kqa=kqa,
        kqa_error=kqa_error,
        strong_summary=strong_summary,
        solve_rows=solve_rows,
        pathfold_problems=pathfold_problems,
    )
    path_strong_total = _coerce_int(_safe_get(strong_summary, "total")) or 0
    path_strong_correct = _coerce_int(_safe_get(strong_summary, "correct_count")) or 0
    path_strong_incorrect = _coerce_int(_safe_get(strong_summary, "incorrect_count")) or 0
    path_strong_unknown = _coerce_int(_safe_get(strong_summary, "unknown_count")) or 0
    explicit_total = _coerce_int(_safe_get(strong_summary, "explicit_total")) or 0
    evaluable = not incomplete_codes and explicit_total > 0
    hard_case_observed = bool(evaluable and _safe_get(strong_summary, "any_incorrect") is True)
    hard_case_ge_2 = bool(evaluable and path_strong_incorrect >= 2)
    hard_case_ge_3 = bool(evaluable and path_strong_incorrect >= 3)
    hard_case_majority = bool(evaluable and path_strong_incorrect >= _majority_threshold(path_strong_total))

    world_codes = _world_contract_codes(
        hard_case_observed=hard_case_observed,
        ambiguity_report=ambiguity_report,
        pathfold_problems=pathfold_problems,
        solver_entries=solver_entries,
    )
    answer_codes = _answer_contract_codes(
        hard_case_observed=hard_case_observed,
        answer_contract_report=answer_contract_report,
        kqa=kqa,
        question_type=question_type,
    )
    judge_codes = []
    if hard_case_observed and not world_codes and not answer_codes:
        judge_codes = _collect_judge_suspect_codes(solve_rows)

    if incomplete_codes:
        primary_label = "insufficient_evidence"
    elif not hard_case_observed:
        primary_label = "not_hard"
    elif world_codes:
        primary_label = "suspect_world_contract"
    elif answer_codes:
        primary_label = "suspect_answer_contract"
    elif judge_codes:
        primary_label = "suspect_judge_false_negative"
    else:
        primary_label = "likely_true_hard"

    evidence_codes = sorted(set(incomplete_codes + world_codes + answer_codes + judge_codes))
    run_status, run_error = _normalize_run_status(batch_result, state)
    candidate_id = _build_candidate_id(batch_id, run_id, paper_id, round_id, stage)

    return {
        "candidate_id": candidate_id,
        "batch_id": batch_id,
        "paper_id": paper_id,
        "run_id": run_id,
        "round": round_id,
        "stage": stage,
        "step": step,
        "question_type": question_type,
        "path_strong_total": path_strong_total,
        "path_strong_correct": path_strong_correct,
        "path_strong_incorrect": path_strong_incorrect,
        "path_strong_unknown": path_strong_unknown,
        "hard_case_observed": hard_case_observed,
        "hard_case_ge_2": hard_case_ge_2,
        "hard_case_ge_3": hard_case_ge_3,
        "hard_case_majority": hard_case_majority,
        "primary_label": primary_label,
        "evidence_codes": evidence_codes,
        "review_priority": _review_priority(primary_label),
        "source_paths": sorted(set(source_paths)),
        "evaluable": evaluable,
        "run_status": run_status,
        "run_error": run_error,
        "summary_source": summary_source,
        "path_variant": kqa.get("variant") if isinstance(kqa, dict) else None,
        "chain": kqa.get("chain") if isinstance(kqa, dict) else None,
        "subject": kqa.get("subject") if isinstance(kqa, dict) else None,
        "source": kqa.get("source") if isinstance(kqa, dict) else None,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _pct(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _aggregate_bucket(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "total": len(rows),
        "evaluable": sum(1 for row in rows if row.get("evaluable") is True),
        "hard_case_observed": sum(1 for row in rows if row.get("hard_case_observed") is True),
        "hard_case_ge_2": sum(1 for row in rows if row.get("hard_case_ge_2") is True),
        "hard_case_ge_3": sum(1 for row in rows if row.get("hard_case_ge_3") is True),
        "hard_case_majority": sum(1 for row in rows if row.get("hard_case_majority") is True),
        "suspect_false_negative": sum(1 for row in rows if row.get("primary_label") in _SUSPECT_FALSE_NEGATIVE_LABELS),
        "likely_true_hard": sum(1 for row in rows if row.get("primary_label") == "likely_true_hard"),
        "insufficient_evidence": sum(1 for row in rows if row.get("primary_label") == "insufficient_evidence"),
    }


def _build_group_summary(candidates: list[dict[str, Any]], key_name: str) -> dict[str, dict[str, int]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        value = candidate.get(key_name)
        bucket = "unknown" if value in (None, "") else str(value)
        grouped[bucket].append(candidate)
    out: dict[str, dict[str, int]] = {}
    for bucket in sorted(grouped):
        out[bucket] = _aggregate_bucket(grouped[bucket])
    return out


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_None_\n"
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header_line, sep_line, *body_lines]) + "\n"


def _format_rate(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1%}"


def _render_summary_md(summary: dict[str, Any]) -> str:
    counts = summary["counts"]
    metrics = summary["headline_metrics"]
    label_counts = summary["primary_label_counts"]
    lines = [
        "# Path Hard Case Audit",
        "",
        "## Summary",
        f"- Eval ID: `{summary['eval_id']}`",
        f"- Generated At (UTC): `{summary['generated_at']}`",
        f"- Input Mode: `{summary['input']['mode']}`",
        f"- Total candidates: `{counts['total_candidates']}`",
        f"- Evaluable path candidates: `{counts['evaluable_candidates']}`",
        f"- Incomplete candidates: `{counts['incomplete_candidates']}`",
        f"- Observed hard cases: `{counts['hard_case_observed']}`",
        f"- observed_hard_case_rate: `{_format_rate(metrics['observed_hard_case_rate'])}`",
        f"- moderate_hard_case_rate_ge_2: `{_format_rate(metrics['moderate_hard_case_rate_ge_2'])}`",
        f"- strong_hard_case_rate_ge_3: `{_format_rate(metrics['strong_hard_case_rate_ge_3'])}`",
        f"- consensus_hard_case_rate_majority: `{_format_rate(metrics['consensus_hard_case_rate_majority'])}`",
        f"- suspect_false_negative_rate_within_hard: `{_format_rate(metrics['suspect_false_negative_rate_within_hard'])}`",
        f"- likely_true_hard_rate_within_hard: `{_format_rate(metrics['likely_true_hard_rate_within_hard'])}`",
        "",
        "## Primary Labels",
        _markdown_table(
            ["label", "count"],
            [[label, count] for label, count in sorted(label_counts.items())],
        ),
        "## By Question Type",
        _markdown_table(
            [
                "question_type",
                "total",
                "evaluable",
                "hard_ge_1",
                "hard_ge_2",
                "hard_ge_3",
                "hard_majority",
                "suspect_false_negative",
                "likely_true_hard",
                "insufficient",
            ],
            [
                [
                    bucket,
                    stats["total"],
                    stats["evaluable"],
                    stats["hard_case_observed"],
                    stats["hard_case_ge_2"],
                    stats["hard_case_ge_3"],
                    stats["hard_case_majority"],
                    stats["suspect_false_negative"],
                    stats["likely_true_hard"],
                    stats["insufficient_evidence"],
                ]
                for bucket, stats in summary["by_question_type"].items()
            ],
        ),
        "## By Round",
        _markdown_table(
            [
                "round",
                "total",
                "evaluable",
                "hard_ge_1",
                "hard_ge_2",
                "hard_ge_3",
                "hard_majority",
                "suspect_false_negative",
                "likely_true_hard",
                "insufficient",
            ],
            [
                [
                    bucket,
                    stats["total"],
                    stats["evaluable"],
                    stats["hard_case_observed"],
                    stats["hard_case_ge_2"],
                    stats["hard_case_ge_3"],
                    stats["hard_case_majority"],
                    stats["suspect_false_negative"],
                    stats["likely_true_hard"],
                    stats["insufficient_evidence"],
                ]
                for bucket, stats in summary["by_round"].items()
            ],
        ),
        "## By Incorrect Count",
        _markdown_table(
            [
                "incorrect_count",
                "total",
                "evaluable",
                "hard_ge_1",
                "hard_ge_2",
                "hard_ge_3",
                "hard_majority",
                "suspect_false_negative",
                "likely_true_hard",
                "insufficient",
            ],
            [
                [
                    bucket,
                    stats["total"],
                    stats["evaluable"],
                    stats["hard_case_observed"],
                    stats["hard_case_ge_2"],
                    stats["hard_case_ge_3"],
                    stats["hard_case_majority"],
                    stats["suspect_false_negative"],
                    stats["likely_true_hard"],
                    stats["insufficient_evidence"],
                ]
                for bucket, stats in summary["by_incorrect_count"].items()
            ],
        ),
    ]
    return "\n".join(lines).rstrip() + "\n"


def run_path_hardcase_audit(
    *,
    batch_dir: Path | None = None,
    runs_root: Path | None = None,
    output_dir: Path,
    include_running: bool = True,
    completed_only: bool = False,
    max_candidates: int | None = None,
) -> dict[str, Any]:
    if batch_dir is None and runs_root is None:
        raise ValueError("Must provide either batch_dir or runs_root.")
    if batch_dir is not None and runs_root is not None:
        raise ValueError("Provide only one of batch_dir or runs_root.")

    discovered_runs, context = _discover_run_dirs(
        batch_dir=batch_dir.resolve() if batch_dir else None,
        runs_root=runs_root.resolve() if runs_root else None,
        completed_only=completed_only,
        include_running=include_running,
    )

    candidates: list[dict[str, Any]] = []
    batch_id = context.get("batch_id")
    results_by_run_id = context.get("results_by_run_id") or {}

    for run_dir in discovered_runs:
        state = _load_state(run_dir)
        run_id = run_dir.name.removeprefix("run_")
        if isinstance(state, dict) and isinstance(state.get("run_id"), str):
            run_id = state["run_id"]
        batch_result = results_by_run_id.get(run_id)
        for candidate_dir in _candidate_dirs_for_run(run_dir):
            candidates.append(
                _audit_candidate(
                    run_dir=run_dir,
                    candidate_dir=candidate_dir,
                    state=state,
                    batch_id=batch_id,
                    batch_result=batch_result,
                )
            )
            if max_candidates is not None and len(candidates) >= max_candidates:
                break
        if max_candidates is not None and len(candidates) >= max_candidates:
            break

    hard_cases = [row for row in candidates if row.get("hard_case_observed") is True]
    review_queue = [
        row
        for row in candidates
        if row.get("primary_label") in _SUSPECT_FALSE_NEGATIVE_LABELS
        or row.get("primary_label") == "insufficient_evidence"
    ]
    review_queue.sort(key=lambda row: (row.get("review_priority") != "high", row.get("candidate_id") or ""))

    total_candidates = len(candidates)
    evaluable_candidates = sum(1 for row in candidates if row.get("evaluable") is True)
    incomplete_candidates = sum(1 for row in candidates if row.get("primary_label") == "insufficient_evidence")
    hard_case_count = len(hard_cases)
    moderate_hard_case_count = sum(1 for row in candidates if row.get("hard_case_ge_2") is True)
    strong_hard_case_count = sum(1 for row in candidates if row.get("hard_case_ge_3") is True)
    consensus_hard_case_count = sum(1 for row in candidates if row.get("hard_case_majority") is True)
    suspect_false_negative_count = sum(
        1 for row in candidates if row.get("primary_label") in _SUSPECT_FALSE_NEGATIVE_LABELS
    )
    likely_true_hard_count = sum(1 for row in candidates if row.get("primary_label") == "likely_true_hard")

    summary = {
        "eval_id": output_dir.resolve().name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input": {
            "mode": "batch_dir" if batch_dir is not None else "runs_root",
            "batch_dir": str(batch_dir.resolve()) if batch_dir else None,
            "runs_root": str(runs_root.resolve()) if runs_root else None,
            "include_running": include_running,
            "completed_only": completed_only,
            "max_candidates": max_candidates,
            "discovered_runs": len(discovered_runs),
            "batch_id": batch_id,
        },
        "counts": {
            "total_candidates": total_candidates,
            "evaluable_candidates": evaluable_candidates,
            "incomplete_candidates": incomplete_candidates,
            "hard_case_observed": hard_case_count,
            "hard_case_ge_2": moderate_hard_case_count,
            "hard_case_ge_3": strong_hard_case_count,
            "hard_case_majority": consensus_hard_case_count,
            "review_queue": len(review_queue),
        },
        "headline_metrics": {
            "observed_hard_case_rate": _pct(hard_case_count, evaluable_candidates),
            "moderate_hard_case_rate_ge_2": _pct(moderate_hard_case_count, evaluable_candidates),
            "strong_hard_case_rate_ge_3": _pct(strong_hard_case_count, evaluable_candidates),
            "consensus_hard_case_rate_majority": _pct(consensus_hard_case_count, evaluable_candidates),
            "suspect_false_negative_rate_within_hard": _pct(suspect_false_negative_count, hard_case_count),
            "likely_true_hard_rate_within_hard": _pct(likely_true_hard_count, hard_case_count),
        },
        "primary_label_counts": dict(Counter(row.get("primary_label") or "unknown" for row in candidates)),
        "by_question_type": _build_group_summary(candidates, "question_type"),
        "by_round": _build_group_summary(candidates, "round"),
        "by_incorrect_count": _build_group_summary(candidates, "path_strong_incorrect"),
    }

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "candidates.jsonl", candidates)
    _write_jsonl(output_dir / "hard_cases.jsonl", hard_cases)
    _write_jsonl(output_dir / "review_queue.jsonl", review_queue)
    _write_json(output_dir / "summary.json", summary)
    (output_dir / "summary.md").write_text(_render_summary_md(summary), encoding="utf-8")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit upstream Path hard cases and suspect false negatives.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--batch-dir", type=Path, help="Batch directory containing batch_manifest.json and runs/.")
    group.add_argument("--runs-root", type=Path, help="Root directory that contains one or more run_* directories.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for audit artifacts.")
    parser.add_argument(
        "--include-running",
        action="store_true",
        help="Explicitly include nonterminal runs. This is the default unless --completed-only is set.",
    )
    parser.add_argument("--completed-only", action="store_true", help="Only include terminal runs.")
    parser.add_argument("--max-candidates", type=int, default=None, help="Optional cap for discovered candidates.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    include_running = True
    if args.completed_only:
        include_running = False
    elif args.include_running:
        include_running = True
    run_path_hardcase_audit(
        batch_dir=args.batch_dir,
        runs_root=args.runs_root,
        output_dir=args.output_dir,
        include_running=include_running,
        completed_only=args.completed_only,
        max_candidates=args.max_candidates,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
