import json
from pathlib import Path


def _write_single_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")


def test_multi_strong_all_solvers_run_path(monkeypatch, tmp_path: Path) -> None:
    from agenqa.graph.state import AgentState, Decision, KQARecord
    from agenqa.nodes.evaluators import solve as solve_eval

    calls: list[tuple[str, bool]] = []

    def _fake_solve_once(_agent_conf, _state, tier, *, solve_conf_override=None, enable_path: bool, timeout_seconds=None):
        calls.append((tier, enable_path))
        solve_dir = tmp_path / "solve"
        out = solve_dir / f"solve_{tier}.jsonl"
        _write_single_jsonl(out, {"correct": True, "token_ratio": 0.1, "metrics": {"kq_tokens": 10, "completion_tokens": 1}})

        ht = None
        if enable_path:
            ht = solve_dir / f"solve_path_{tier}.jsonl"
            _write_single_jsonl(
                ht, {"correct": True, "token_ratio": 0.2, "metrics": {"kq_tokens": 20, "completion_tokens": 2}}
            )
        return True, 0.1, out, solve_dir, ht

    monkeypatch.setattr(solve_eval, "_solve_once", _fake_solve_once)

    agent_conf = {
        "consensus": {"mode": "always", "answer_judge": "llm"},
        "solvers": {"medium": [{"generator": {}}], "strong": [{"generator": {}}, {"generator": {}}]},
    }
    state = AgentState(
        run_id="t",
        artifacts_dir=tmp_path,
        qa_idx=1,
        last_decision=Decision(operation="extend", params={"question_type": "Derivation"}),
    )
    state.history.append(KQARecord(qa_idx=1, question="Q", answer="A", question_type="Derivation"))

    out = solve_eval.solve_dual(agent_conf, state)
    new_state = out.state if hasattr(out, "state") else out

    assert ("strong_0", True) in calls
    assert ("strong_1", True) in calls

    # Ensure solver_index contains path results for all strong solvers.
    assert new_state.solver_index[1][1]["path"]["strong_0"].correct is True
    assert new_state.solver_index[1][1]["path"]["strong_1"].correct is True
