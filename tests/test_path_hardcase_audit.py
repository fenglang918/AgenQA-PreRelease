from __future__ import annotations

import json
from pathlib import Path

from agenqa.evaluation.path_hardcase_audit import run_path_hardcase_audit


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_clean_answer_contract_report() -> dict[str, object]:
    return {
        "step": 2,
        "question_type": "Derivation",
        "where": "extend_format",
        "answer_contract_ids": ["ac_step2_derivation_exact_v1"],
        "issue_types_error": [],
        "issue_types_warn": [],
        "error_count": 0,
        "warn_count": 0,
    }


def make_path_kqa(*, step: int, answer: str, include_visible_spec: bool = True) -> dict[str, object]:
    question_for_solver = "Question body"
    if include_visible_spec:
        question_for_solver += '\n- L4.answer_output_spec_1: {"question_type":"Derivation"}'
    return {
        "paper_id": "paper_x",
        "step_tail": step,
        "variant": "direct",
        "question": "Question body",
        "question_for_solver": question_for_solver,
        "world_contract_text": '<contract></contract>\n- L4.answer_output_spec_1: {"question_type":"Derivation"}',
        "answer": answer,
        "subject": "Test Subject",
        "source": "path_fold_direct",
    }


def make_solve_row(
    *,
    answer: str,
    solve: str,
    correct: bool | None,
    finish_reason: str = "stop",
    gateway_finish_reason: str | None = None,
    expression_equivalent: str | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "answer": answer,
        "solve": solve,
        "correct": correct,
        "finish_reason": finish_reason,
        "gateway_finish_reason": gateway_finish_reason,
    }
    if expression_equivalent is not None:
        row["expression_judge"] = {"strong": {"equivalent": expression_equivalent}}
    return row


def make_candidate(
    run_dir: Path,
    *,
    round_id: int,
    step: int,
    answer_contract_report: dict[str, object] | None,
    ambiguity_report: dict[str, object] | None,
    next_director_summary: dict[str, object] | None,
    solve_rows: list[dict[str, object]],
    include_visible_spec: bool = True,
    answer: str = "\\boxed{ref}",
) -> None:
    candidate_dir = run_dir / f"round_{round_id}" / "extend"
    write_json(candidate_dir / "path_kqa.json", make_path_kqa(step=step, answer=answer, include_visible_spec=include_visible_spec))
    if answer_contract_report is not None:
        write_json(candidate_dir / "answer_contract_report.json", answer_contract_report)
    if ambiguity_report is not None:
        write_json(candidate_dir / "solve" / "ambiguity_report.json", ambiguity_report)
    for idx, row in enumerate(solve_rows):
        write_jsonl(candidate_dir / "solve" / f"solve_path_strong_{idx}.jsonl", [row])
    if next_director_summary is not None:
        write_json(
            run_dir / f"round_{round_id + 1}" / "director" / "director_decision.json",
            {
                "step": step,
                "params": {
                    "solver_context": {
                        "metrics": {
                            "path": {
                                "strong_summary": next_director_summary,
                            }
                        }
                    }
                },
            },
        )


def make_run(run_dir: Path, *, run_id: str, stop_reason: str | None) -> None:
    state: dict[str, object] = {"run_id": run_id}
    if stop_reason is not None:
        state["stop_reason"] = stop_reason
    write_json(run_dir / "state.json", state)


def test_run_path_hardcase_audit_labels_and_summary(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    output_dir = tmp_path / "audit"

    world_run = runs_root / "run_world"
    make_run(world_run, run_id="world", stop_reason="reach_max_rounds(6)")
    make_candidate(
        world_run,
        round_id=2,
        step=2,
        answer_contract_report=make_clean_answer_contract_report(),
        ambiguity_report={
            "type1_suspected": True,
            "wellposed_votes": {"true": 1, "false": 1, "null": 0},
        },
        next_director_summary={
            "total": 3,
            "explicit_total": 3,
            "correct_count": 1,
            "incorrect_count": 2,
            "unknown_count": 0,
            "all_correct": False,
            "any_incorrect": True,
        },
        solve_rows=[
            make_solve_row(answer="\\boxed{ref}", solve="\\boxed{wrong1}", correct=False, expression_equivalent="no"),
            make_solve_row(answer="\\boxed{ref}", solve="\\boxed{wrong2}", correct=False, expression_equivalent="no"),
            make_solve_row(answer="\\boxed{ref}", solve="\\boxed{ref}", correct=True, expression_equivalent="yes"),
        ],
    )

    answer_run = runs_root / "run_answer"
    make_run(answer_run, run_id="answer", stop_reason="reach_max_rounds(6)")
    make_candidate(
        answer_run,
        round_id=2,
        step=2,
        answer_contract_report={
            **make_clean_answer_contract_report(),
            "issue_types_error": ["answer_contract_missing_output_spec"],
            "error_count": 1,
        },
        ambiguity_report=None,
        next_director_summary=None,
        include_visible_spec=False,
        solve_rows=[
            make_solve_row(answer="\\boxed{ref}", solve="\\boxed{wrong1}", correct=False, expression_equivalent="no"),
            make_solve_row(answer="\\boxed{ref}", solve="\\boxed{wrong2}", correct=False, expression_equivalent="no"),
            make_solve_row(answer="\\boxed{ref}", solve="\\boxed{ref}", correct=True, expression_equivalent="yes"),
        ],
    )

    judge_run = runs_root / "run_judge"
    make_run(judge_run, run_id="judge", stop_reason="reach_max_rounds(6)")
    make_candidate(
        judge_run,
        round_id=2,
        step=2,
        answer_contract_report=make_clean_answer_contract_report(),
        ambiguity_report=None,
        next_director_summary=None,
        solve_rows=[
            make_solve_row(answer="\\boxed{ref}", solve="\\boxed{ref}", correct=False, expression_equivalent="no"),
            make_solve_row(answer="\\boxed{ref}", solve="\\boxed{ref}", correct=True, expression_equivalent="yes"),
            make_solve_row(answer="\\boxed{ref}", solve="\\boxed{ref}", correct=True, expression_equivalent="yes"),
        ],
    )

    hard_run = runs_root / "run_hard"
    make_run(hard_run, run_id="hard", stop_reason="reach_max_rounds(6)")
    make_candidate(
        hard_run,
        round_id=2,
        step=2,
        answer_contract_report=make_clean_answer_contract_report(),
        ambiguity_report={"type1_suspected": False, "wellposed_votes": {"true": 3, "false": 0, "null": 0}},
        next_director_summary=None,
        solve_rows=[
            make_solve_row(answer="\\boxed{ref}", solve="\\boxed{wrong}", correct=False, expression_equivalent="no"),
            make_solve_row(answer="\\boxed{ref}", solve="\\boxed{ref}", correct=True, expression_equivalent="yes"),
            make_solve_row(answer="\\boxed{ref}", solve="\\boxed{ref}", correct=True, expression_equivalent="yes"),
        ],
    )

    incomplete_run = runs_root / "run_incomplete"
    make_run(incomplete_run, run_id="incomplete", stop_reason=None)
    make_candidate(
        incomplete_run,
        round_id=2,
        step=2,
        answer_contract_report=make_clean_answer_contract_report(),
        ambiguity_report=None,
        next_director_summary=None,
        solve_rows=[
            make_solve_row(
                answer="\\boxed{ref}",
                solve="\\boxed{partial}",
                correct=False,
                finish_reason="length",
                gateway_finish_reason="length",
                expression_equivalent="no",
            )
        ],
    )

    summary = run_path_hardcase_audit(runs_root=runs_root, output_dir=output_dir)
    assert summary["counts"]["total_candidates"] == 5
    assert summary["counts"]["evaluable_candidates"] == 4
    assert summary["counts"]["incomplete_candidates"] == 1
    assert summary["counts"]["hard_case_observed"] == 4
    assert summary["headline_metrics"]["observed_hard_case_rate"] == 1.0
    assert summary["headline_metrics"]["suspect_false_negative_rate_within_hard"] == 0.75
    assert summary["headline_metrics"]["likely_true_hard_rate_within_hard"] == 0.25

    candidates = [json.loads(line) for line in (output_dir / "candidates.jsonl").read_text(encoding="utf-8").splitlines()]
    labels = {row["run_id"]: row["primary_label"] for row in candidates}
    assert labels == {
        "world": "suspect_world_contract",
        "answer": "suspect_answer_contract",
        "judge": "suspect_judge_false_negative",
        "hard": "likely_true_hard",
        "incomplete": "insufficient_evidence",
    }

    world_row = next(row for row in candidates if row["run_id"] == "world")
    assert world_row["summary_source"] == "next_director_metrics"
    assert "ambiguity_type1_suspected" in world_row["evidence_codes"]

    judge_row = next(row for row in candidates if row["run_id"] == "judge")
    assert "solve_equals_answer_but_correct_false" in judge_row["evidence_codes"]

    review_queue = [json.loads(line) for line in (output_dir / "review_queue.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(review_queue) == 4
    assert review_queue[0]["review_priority"] == "high"

    summary_md = (output_dir / "summary.md").read_text(encoding="utf-8")
    assert "observed_hard_case_rate" in summary_md
    assert "suspect_judge_false_negative" in summary_md


def test_run_path_hardcase_audit_batch_dir_completed_only_filters_running(tmp_path: Path) -> None:
    batch_dir = tmp_path / "batch_x"
    runs_dir = batch_dir / "runs"
    output_dir = tmp_path / "audit_batch"

    completed_run = runs_dir / "run_completed"
    make_run(completed_run, run_id="completed", stop_reason="reach_max_rounds(6)")
    make_candidate(
        completed_run,
        round_id=2,
        step=2,
        answer_contract_report=make_clean_answer_contract_report(),
        ambiguity_report={"type1_suspected": False, "wellposed_votes": {"true": 3, "false": 0, "null": 0}},
        next_director_summary=None,
        solve_rows=[
            make_solve_row(answer="\\boxed{ref}", solve="\\boxed{wrong}", correct=False, expression_equivalent="no"),
            make_solve_row(answer="\\boxed{ref}", solve="\\boxed{ref}", correct=True, expression_equivalent="yes"),
            make_solve_row(answer="\\boxed{ref}", solve="\\boxed{ref}", correct=True, expression_equivalent="yes"),
        ],
    )

    running_run = runs_dir / "run_running"
    make_run(running_run, run_id="running", stop_reason=None)
    make_candidate(
        running_run,
        round_id=2,
        step=2,
        answer_contract_report=make_clean_answer_contract_report(),
        ambiguity_report=None,
        next_director_summary=None,
        solve_rows=[
            make_solve_row(answer="\\boxed{ref}", solve="\\boxed{partial}", correct=False, finish_reason="length"),
        ],
    )

    write_json(
        batch_dir / "batch_manifest.json",
        {
            "batch_id": "batch_x",
        },
    )
    write_jsonl(
        batch_dir / "batch_results.jsonl",
        [
            {
                "run_id": "completed",
                "status": "success",
                "error": "",
            }
        ],
    )

    summary = run_path_hardcase_audit(batch_dir=batch_dir, output_dir=output_dir, completed_only=True)
    assert summary["counts"]["total_candidates"] == 1
    candidates = [json.loads(line) for line in (output_dir / "candidates.jsonl").read_text(encoding="utf-8").splitlines()]
    assert candidates[0]["run_id"] == "completed"
