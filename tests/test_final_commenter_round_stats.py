import json
from pathlib import Path

from agenqa.graph.state import AgentState, SolverResult
from agenqa.nodes.evaluators.final_commenter import final_commenter_node


def test_final_commenter_writes_strong_solver_round_stats_markdown(tmp_path: Path) -> None:
    state = AgentState(run_id="run_test", artifacts_dir=tmp_path)
    state.update_solver_index(
        step=0,
        round=1,
        target="edge",
        tier="strong",
        result=SolverResult(correct=True, model="gpt-a", service_id="svc-a", question_well_posed=True),
    )
    state.update_solver_index(
        step=0,
        round=1,
        target="path",
        tier="strong",
        result=SolverResult(correct=False, model="gpt-a", service_id="svc-a", question_well_posed=False),
    )
    state.update_solver_index(
        step=0,
        round=1,
        target="edge",
        tier="strong_1",
        result=SolverResult(correct=False, model="gpt-b", service_id="svc-b", question_well_posed=False),
    )
    state.update_solver_index(
        step=0,
        round=1,
        target="path",
        tier="strong_1",
        result=SolverResult(correct=True, model="gpt-b", service_id="svc-b", question_well_posed=True),
    )
    state.update_solver_index(
        step=1,
        round=2,
        target="edge",
        tier="strong",
        result=SolverResult(correct=True, model="gpt-a", service_id="svc-a", question_well_posed=True),
    )
    state.update_solver_index(
        step=1,
        round=2,
        target="path",
        tier="strong",
        result=SolverResult(correct=True, model="gpt-a", service_id="svc-a", question_well_posed=True),
    )

    agent_conf = {
        "final_commenter": {
            "enabled": False,
        }
    }
    final_commenter_node(agent_conf, state)

    report_path = tmp_path / "00_Summary" / "strong_solver_round_stats.md"
    json_path = tmp_path / "00_Summary" / "strong_solver_round_stats.json"
    assert report_path.exists()
    assert json_path.exists()

    content = report_path.read_text(encoding="utf-8")
    assert "# Strong Solver Round Stats" in content
    assert "### Round 1" in content
    assert "### Round 2" in content
    assert "#### Strong Summary" in content
    assert "| edge | 1 | 1 | 0 |" in content
    assert "| path | 1 | 1 | 0 |" in content
    assert "| 1 | 0 | strong | gpt-a | svc-a | correct | true | incorrect | false | 1 | 0 | 0 | 0 | 1 | 0 |" in content
    assert "| 1 | 0 | strong_1 | gpt-b | svc-b | incorrect | false | correct | true | 0 | 1 | 0 | 1 | 0 | 0 |" in content
    assert "| 2 | 1 | strong | gpt-a | svc-a | correct | true | correct | true | 1 | 0 | 0 | 1 | 0 | 0 |" in content

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(payload["rows"]) == 3
    assert payload["rounds"][0]["round"] == 1
    assert payload["rounds"][0]["strong_summary"]["edge"] == {
        "correct_count": 1,
        "incorrect_count": 1,
        "unknown_count": 0,
    }
    assert payload["rounds"][0]["strong_summary"]["path"] == {
        "correct_count": 1,
        "incorrect_count": 1,
        "unknown_count": 0,
    }
    assert payload["rounds"][0]["solver_rows"][0]["edge_well_posed"] in {"true", "false", "unknown"}
