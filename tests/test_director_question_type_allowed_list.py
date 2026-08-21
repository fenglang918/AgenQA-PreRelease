import tempfile
import unittest
from pathlib import Path

from agenqa.graph.state import AgentState
import agenqa.nodes.director as director


class TestDirectorQuestionTypeAllowedList(unittest.TestCase):
    def _mk_state(self, step: int) -> AgentState:
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        return AgentState(run_id="test", artifacts_dir=Path(td.name), qa_idx=step)

    def test_default_behavior_unchanged(self) -> None:
        conf = {
            "agent": {
                "track": "unified",
                "question_type_policy": {"no_mcq_from_step": 2},
            }
        }
        s0 = self._mk_state(0)  # next_step=1
        self.assertEqual(director._allowed_question_types_for_next_step(conf, s0), ["MCQ", "Derivation", "Numeric"])
        s1 = self._mk_state(1)  # next_step=2
        self.assertEqual(director._allowed_question_types_for_next_step(conf, s1), ["Derivation", "Numeric"])

    def test_whitelist_filters_base_allowed(self) -> None:
        conf = {
            "agent": {
                "track": "unified",
                "question_type_policy": {"no_mcq_from_step": 2, "allowed_question_types": ["MCQ", "Derivation"]},
            }
        }
        s0 = self._mk_state(0)  # base=MCQ/Derivation/Numeric -> effective=MCQ/Derivation
        self.assertEqual(director._allowed_question_types_for_next_step(conf, s0), ["MCQ", "Derivation"])
        s1 = self._mk_state(1)  # base=Derivation/Numeric -> effective=Derivation
        self.assertEqual(director._allowed_question_types_for_next_step(conf, s1), ["Derivation"])

    def test_whitelist_derivation_only(self) -> None:
        conf = {
            "agent": {
                "track": "unified",
                "question_type_policy": {"no_mcq_from_step": 2, "allowed_question_types": ["Derivation"]},
            }
        }
        s0 = self._mk_state(0)
        s1 = self._mk_state(1)
        self.assertEqual(director._allowed_question_types_for_next_step(conf, s0), ["Derivation"])
        self.assertEqual(director._allowed_question_types_for_next_step(conf, s1), ["Derivation"])

    def test_invalid_question_type_fails_fast(self) -> None:
        conf = {
            "agent": {
                "track": "unified",
                "question_type_policy": {"no_mcq_from_step": 2, "allowed_question_types": ["Foo"]},
            }
        }
        s0 = self._mk_state(0)
        with self.assertRaises(ValueError):
            director._allowed_question_types_for_next_step(conf, s0)

    def test_empty_intersection_fails_fast(self) -> None:
        conf = {
            "agent": {
                "track": "unified",
                "question_type_policy": {"no_mcq_from_step": 2, "allowed_question_types": ["MCQ"]},
            }
        }
        s1 = self._mk_state(1)  # next_step=2 -> base=Derivation/Numeric
        with self.assertRaises(ValueError):
            director._allowed_question_types_for_next_step(conf, s1)


if __name__ == "__main__":
    unittest.main()
