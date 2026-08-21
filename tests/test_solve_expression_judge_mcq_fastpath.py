import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agenqa.graph.state import AgentState
from agenqa.nodes.evaluators.solve import _apply_expression_judge


class TestSolveExpressionJudgeMCQJudge(unittest.TestCase):
    def _mk_state(self, td: str) -> AgentState:
        return AgentState(run_id="test", artifacts_dir=Path(td), qa_idx=1)

    def _write_solve_jsonl(self, path: Path, *, ans: str, pred: str) -> None:
        row = {
            "known": "",
            "question": "",
            "answer": ans,
            "solve": pred,
            "correct": False,
            "metrics": {"kq_tokens": 10.0, "completion_tokens": 20.0},
        }
        path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    def test_fastpath_yes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            solve_path = Path(td) / "solve.jsonl"
            self._write_solve_jsonl(solve_path, ans="\\boxed{A}", pred="\\boxed{A}")
            conf = {"agent": {"symbolic_only_question_types": ["MCQ"]}, "consensus": {"answer_judge": "llm"}}
            with patch("agenqa.nodes.evaluators.solve.load_expression_judge_generator", return_value={"model": "dummy"}):
                with patch(
                    "agenqa.nodes.evaluators.solve.run_expression_equivalence_judge",
                    return_value=(True, "ok"),
                ):
                    ok, token_ratio = _apply_expression_judge(
                        conf,
                        self._mk_state(td),
                        "strong",
                        solve_path,
                        numeric_ok=False,
                        numeric_diff=None,
                        question_type="MCQ",
                    )
            self.assertTrue(ok)
            self.assertEqual(token_ratio, 2.0)
            row = json.loads(solve_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertTrue(row.get("correct"))
            self.assertIn("expression_judge", row)
            self.assertIn("strong", row["expression_judge"])

    def test_judge_no(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            solve_path = Path(td) / "solve.jsonl"
            self._write_solve_jsonl(solve_path, ans="\\boxed{A}", pred="\\boxed{B}")
            conf = {"agent": {"symbolic_only_question_types": ["MCQ"]}, "consensus": {"answer_judge": "llm"}}
            with patch("agenqa.nodes.evaluators.solve.load_expression_judge_generator", return_value={"model": "dummy"}):
                with patch(
                    "agenqa.nodes.evaluators.solve.run_expression_equivalence_judge",
                    return_value=(False, "no"),
                ):
                    ok, token_ratio = _apply_expression_judge(
                        conf,
                        self._mk_state(td),
                        "strong",
                        solve_path,
                        numeric_ok=True,
                        numeric_diff=1.0,
                        question_type="MCQ",
                    )
            self.assertFalse(ok)
            self.assertIsNone(token_ratio)
            row = json.loads(solve_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertFalse(row.get("correct"))


if __name__ == "__main__":
    unittest.main()
