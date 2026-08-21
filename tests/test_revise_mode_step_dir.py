from __future__ import annotations

from pathlib import Path

import pytest

from agenqa.graph.state import AgentState, Decision, KQARecord


def test_solve_passes_revise_mode_to_compute_step_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import agenqa.nodes.evaluators.solve as solve_mod

    state = AgentState(run_id="r", artifacts_dir=tmp_path, rounds=1, qa_idx=0)
    state.history.append(KQARecord(qa_idx=0, known="k", question="q", answer="\\\\boxed{1}"))
    state.last_decision = Decision(operation="revise", params={"revise_mode": "correctness"})

    called = {}

    def fake_compute_step_dir(root, node, qa_idx, round_idx, revise_mode=None):  # noqa: ANN001, ANN201
        called["revise_mode"] = revise_mode
        return tmp_path / "step_dir"

    monkeypatch.setattr(solve_mod, "compute_step_dir", fake_compute_step_dir)

    def fake_dump_latest_kqa_jsonl(_state, path):  # noqa: ANN001, ANN201
        path.write_text('{"known":"k","question":"q","answer":"\\\\boxed{1}"}\n', encoding="utf-8")
        return path

    monkeypatch.setattr(solve_mod, "dump_latest_kqa_jsonl", fake_dump_latest_kqa_jsonl)

    class _DummyPromptTemplate:
        template = ""

    class DummySolverRunner:
        def __init__(self, config):  # noqa: ANN001
            self.config = config
            self.prompt_template = _DummyPromptTemplate()

        def run(self, _kqa_path, output_path, append=False, concurrency=1):  # noqa: ANN001, ANN201
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                '{"correct": false, "token_ratio": null, "solve": "x", "service_id": "s", "model": "m"}\n',
                encoding="utf-8",
            )
            return output_path

    monkeypatch.setattr(solve_mod, "SolverRunner", DummySolverRunner)

    solve_mod._solve_once(  # type: ignore[attr-defined]
        agent_conf={},
        state=state,
        tier="strong",
        enable_path=False,
        timeout_seconds=1,
    )
    assert called.get("revise_mode") == "correctness"


def test_consensus_passes_revise_mode_to_compute_step_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import agenqa.nodes.evaluators.consensus as cons_mod

    state = AgentState(run_id="r", artifacts_dir=tmp_path, rounds=2, qa_idx=0)
    state.history.append(KQARecord(qa_idx=0, known="k", question="A. x\nB. y\nC. z\nD. w", answer="A"))
    state.last_decision = Decision(operation="revise", params={"revise_mode": "correctness"})

    called = {}

    def fake_compute_step_dir(root, node, qa_idx, round_idx, revise_mode=None):  # noqa: ANN001, ANN201
        called["revise_mode"] = revise_mode
        return tmp_path / "step_dir"

    monkeypatch.setattr(cons_mod, "compute_step_dir", fake_compute_step_dir)

    cons_mod.compute_consensus(  # type: ignore[attr-defined]
        agent_conf={"solvers": {"strong": [{}, {}]}},
        state=state,
    )
    assert called.get("revise_mode") == "correctness"
