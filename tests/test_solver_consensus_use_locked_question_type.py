import tempfile
import unittest
from pathlib import Path

from agenqa.graph.state import AgentState, KQARecord
from agenqa.nodes.evaluators.solve import _infer_question_type_for_solver
from agenqa.nodes.evaluators.consensus import _infer_question_type as _infer_question_type_for_consensus


class TestSolverConsensusUseLockedQuestionType(unittest.TestCase):
    def _mk_state(self, step: int, rec: KQARecord) -> AgentState:
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        state = AgentState(run_id="test", artifacts_dir=Path(td.name), qa_idx=step)
        state.history.append(rec)
        return state

    def test_solver_prefers_locked_question_type_over_qa_heuristic(self) -> None:
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
            question_type="Derivation",
        )
        state = self._mk_state(2, rec)
        self.assertEqual(_infer_question_type_for_solver(conf, state), "Derivation")

    def test_solver_policy_violation_fails_fast(self) -> None:
        conf = {
            "agent": {
                "track": "unified",
                "question_type_policy": {"no_mcq_from_step": 2, "allowed_question_types": ["MCQ", "Derivation"]},
            }
        }
        rec = KQARecord(step=2, known="{}", question="abs_tol=1e-3", answer="\\boxed{0.1}", question_type="Numeric")
        state = self._mk_state(2, rec)
        with self.assertRaises(ValueError):
            _infer_question_type_for_solver(conf, state)

    def test_consensus_prefers_locked_question_type_over_qa_heuristic(self) -> None:
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
            question_type="Derivation",
        )
        state = self._mk_state(2, rec)
        self.assertEqual(_infer_question_type_for_consensus(conf, state), "Derivation")

    def test_consensus_policy_violation_fails_fast(self) -> None:
        conf = {
            "agent": {
                "track": "unified",
                "question_type_policy": {"no_mcq_from_step": 2, "allowed_question_types": ["MCQ", "Derivation"]},
            }
        }
        rec = KQARecord(step=2, known="{}", question="abs_tol=1e-3", answer="\\boxed{0.1}", question_type="Numeric")
        state = self._mk_state(2, rec)
        with self.assertRaises(ValueError):
            _infer_question_type_for_consensus(conf, state)


if __name__ == "__main__":
    unittest.main()
