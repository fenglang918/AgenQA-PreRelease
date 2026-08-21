import tempfile
import unittest
from pathlib import Path

from agenqa.graph.state import AgentState, KQARecord
from agenqa.nodes.op_revise import _infer_question_type


class TestReviseInheritsLockedQuestionType(unittest.TestCase):
    def _mk_state(self, step: int, rec: KQARecord) -> AgentState:
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        state = AgentState(run_id="test", artifacts_dir=Path(td.name), qa_idx=step)
        state.history.append(rec)
        return state

    def test_locked_question_type_wins_over_qa_heuristic(self) -> None:
        conf = {
            "agent": {
                "track": "unified",
                "question_type_policy": {"no_mcq_from_step": 2, "allowed_question_types": ["MCQ", "Derivation"]},
            }
        }
        # Step=2: only Derivation allowed (no MCQ, no Numeric).
        rec = KQARecord(
            step=2,
            known="{}",
            question="A. foo\nB. bar\n请计算并给出 abs_tol=1e-3 的数值答案",
            answer="\\boxed{0.1}",
            question_type="Derivation",
        )
        state = self._mk_state(2, rec)
        self.assertEqual(_infer_question_type(conf, state, 2), "Derivation")

    def test_locked_question_type_policy_violation_fails_fast(self) -> None:
        conf = {
            "agent": {
                "track": "unified",
                "question_type_policy": {"no_mcq_from_step": 2, "allowed_question_types": ["MCQ", "Derivation"]},
            }
        }
        # Step=2 disallows MCQ.
        rec = KQARecord(step=2, known="{}", question="A. foo\nB. bar", answer="\\boxed{A}", question_type="MCQ")
        state = self._mk_state(2, rec)
        with self.assertRaises(ValueError):
            _infer_question_type(conf, state, 2)

    def test_locked_question_type_from_constraints_is_supported(self) -> None:
        conf = {
            "agent": {
                "track": "unified",
                "question_type_policy": {"no_mcq_from_step": 2, "allowed_question_types": ["MCQ", "Derivation"]},
            }
        }
        rec = KQARecord(
            step=2,
            known="{}",
            question="A. foo\nB. bar\nabs_tol=1e-3",
            answer="\\boxed{0.1}",
            question_type=None,
            question_type_constraints={"locked_question_type": "Derivation"},
        )
        state = self._mk_state(2, rec)
        self.assertEqual(_infer_question_type(conf, state, 2), "Derivation")


if __name__ == "__main__":
    unittest.main()
