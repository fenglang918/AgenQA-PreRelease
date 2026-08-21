from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from agenqa.graph.state import AgentState, KQARecord
from agenqa.nodes.evaluators.consensus import _solver_visible_question_for_judge as _consensus_question
from agenqa.nodes.evaluators.solve import _solver_visible_question_for_judge as _solve_question


def _make_state() -> AgentState:
    td = TemporaryDirectory()
    state = AgentState(run_id="test", artifacts_dir=Path(td.name), qa_idx=1)
    state._tmpdir_for_test = td  # type: ignore[attr-defined]
    state.history.append(
        KQARecord(
            paper_id="p1",
            step=1,
            known="{}",
            question="Core question body",
            world_contract_text="**World Contract**\n- L3.foo: bar",
            answer="\\boxed{x}",
            chain="k1,q1,a1",
        )
    )
    return state


def test_solve_judge_question_uses_solver_visible_delivery_from_state() -> None:
    state = _make_state()
    row = {"question": "Core question body", "world_contract_text": "**World Contract**\n- L3.foo: bar"}

    question = _solve_question(state=state, row=row)

    assert question == "Core question body\n\n**World Contract**\n- L3.foo: bar"


def test_consensus_judge_question_prefers_precomposed_solver_question() -> None:
    state = _make_state()
    solver_rows = [
        (
            0,
            {
                "question": "Core question body",
                "world_contract_text": "**World Contract**\n- L3.foo: bar",
                "question_for_solver": "Precomposed final question\n\n**World Contract**\n- L3.foo: bar",
            },
        )
    ]

    question = _consensus_question(state=state, solver_rows=solver_rows)

    assert question == "Precomposed final question\n\n**World Contract**\n- L3.foo: bar"
