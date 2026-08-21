import json
from pathlib import Path

import pytest


def _write_single_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")


def test_apply_expression_judge_uses_explicit_expression_judge_generator_for_derivation(monkeypatch, tmp_path: Path) -> None:
    # Ensure: even when agent.symbolic_only is false, consensus.answer_judge=llm enables expression judge for Derivation.
    from agenqa.graph.state import AgentState, Decision
    from agenqa.nodes.evaluators import solve as solve_eval

    solve_path = tmp_path / "solve_strong_0.jsonl"
    _write_single_jsonl(
        solve_path,
        {
            "known": "K",
            "question": "Q",
            "answer": r"\boxed{a=b=c}",
            "solve": r"\boxed{a=b}",
            "correct": False,
            "metrics": {"kq_tokens": 100, "completion_tokens": 10},
        },
    )

    monkeypatch.setattr(
        solve_eval,
        "load_expression_judge_generator",
        lambda _conf: {"service_type": "private_endpoint", "api_base": "http://localhost/v1", "model_name": "judge"},
    )
    monkeypatch.setattr(solve_eval, "run_expression_equivalence_judge", lambda *_a, **_k: (True, "ok"))

    agent_conf = {
        "agent": {"symbolic_only": False},
        "consensus": {"answer_judge": "llm"},
        "solvers": {"strong": [{"generator": {"model_name": "dummy"}}]},
    }
    state = AgentState(run_id="t", artifacts_dir=tmp_path, qa_idx=1, last_decision=Decision(operation="extend"))

    ok, diff = solve_eval._apply_expression_judge(
        agent_conf, state, "strong", solve_path, numeric_ok=False, numeric_diff=None, question_type="Derivation"
    )
    assert ok is True
    assert diff == pytest.approx(0.1)

    # Confirm it wrote back expression_judge metadata and overrode correct.
    row = json.loads(solve_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["correct"] is True
    assert row["correct_numeric"] is False
    assert row["expression_judge"]["strong"]["equivalent"] == "yes"


def test_apply_expression_judge_records_soft_failure_when_strict_is_legacy_config(monkeypatch, tmp_path: Path) -> None:
    from agenqa.graph.state import AgentState, Decision
    from agenqa.nodes.evaluators import solve as solve_eval

    solve_path = tmp_path / "solve_strong_0.jsonl"
    _write_single_jsonl(
        solve_path,
        {"known": "K", "question": "Q", "answer": r"\boxed{1}", "solve": r"\boxed{1}", "correct": True},
    )

    monkeypatch.setattr(solve_eval, "load_expression_judge_generator", lambda _conf: None)

    def _boom(*_a, **_k):
        raise RuntimeError("judge down")

    monkeypatch.setattr(solve_eval, "run_expression_equivalence_judge", _boom)

    agent_conf = {
        "agent": {"symbolic_only": False},
        "consensus": {"answer_judge": "llm", "answer_judge_strict": True},
        "solvers": {"strong": [{"generator": {"model_name": "dummy"}}]},
    }
    state = AgentState(run_id="t", artifacts_dir=tmp_path, qa_idx=1, last_decision=Decision(operation="extend"))

    ok, diff = solve_eval._apply_expression_judge(
        agent_conf, state, "strong", solve_path, numeric_ok=True, numeric_diff=0.01, question_type="Derivation"
    )
    assert ok is None
    assert diff == pytest.approx(0.01)
    row = json.loads(solve_path.read_text(encoding="utf-8"))
    assert row["expression_judge"]["strong"]["status"] == "failed"


def test_apply_expression_judge_records_missing_generator(monkeypatch, tmp_path: Path) -> None:
    from agenqa.graph.state import AgentState, Decision
    from agenqa.nodes.evaluators import solve as solve_eval

    solve_path = tmp_path / "solve_strong_0.jsonl"
    _write_single_jsonl(
        solve_path,
        {"known": "K", "question": "Q", "answer": r"\boxed{1}", "solve": r"\boxed{1}", "correct": True},
    )

    monkeypatch.setattr(solve_eval, "load_expression_judge_generator", lambda _conf: None)

    agent_conf = {
        "agent": {"symbolic_only": False},
        "consensus": {"answer_judge": "llm"},
        "solvers": {"strong": [{"generator": {"model_name": "dummy"}}]},
    }
    state = AgentState(run_id="t", artifacts_dir=tmp_path, qa_idx=1, last_decision=Decision(operation="extend"))

    ok, diff = solve_eval._apply_expression_judge(
        agent_conf, state, "strong", solve_path, numeric_ok=True, numeric_diff=0.01, question_type="Derivation"
    )
    assert ok is None
    assert diff == pytest.approx(0.01)
    row = json.loads(solve_path.read_text(encoding="utf-8"))
    assert row["expression_judge"]["strong"]["status"] == "failed"
    assert "no generator configured" in row["expression_judge"]["strong"]["failure_reason"]
