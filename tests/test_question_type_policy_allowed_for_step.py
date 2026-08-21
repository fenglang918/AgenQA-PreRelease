import unittest

from agenqa.nodes.utils import allowed_question_types_for_step


class TestQuestionTypePolicyAllowedForStep(unittest.TestCase):
    def test_default_behavior_matches_next_step_semantics(self) -> None:
        conf = {"agent": {"track": "unified", "question_type_policy": {"no_mcq_from_step": 2}}}
        # step=1: MCQ allowed
        self.assertEqual(allowed_question_types_for_step(conf, 1), ["MCQ", "Derivation", "Numeric"])
        # step=2+: MCQ disallowed by default
        self.assertEqual(allowed_question_types_for_step(conf, 2), ["Derivation", "Numeric"])

    def test_whitelist_filters_base_allowed(self) -> None:
        conf = {
            "agent": {
                "track": "unified",
                "question_type_policy": {"no_mcq_from_step": 2, "allowed_question_types": ["MCQ", "Derivation"]},
            }
        }
        self.assertEqual(allowed_question_types_for_step(conf, 1), ["MCQ", "Derivation"])
        # step=2 base=Derivation/Numeric -> effective=Derivation
        self.assertEqual(allowed_question_types_for_step(conf, 2), ["Derivation"])

    def test_invalid_question_type_fails_fast(self) -> None:
        conf = {
            "agent": {
                "track": "unified",
                "question_type_policy": {"no_mcq_from_step": 2, "allowed_question_types": ["Foo"]},
            }
        }
        with self.assertRaises(ValueError):
            allowed_question_types_for_step(conf, 1)

    def test_empty_intersection_fails_fast(self) -> None:
        conf = {
            "agent": {
                "track": "unified",
                "question_type_policy": {"no_mcq_from_step": 2, "allowed_question_types": ["MCQ"]},
            }
        }
        # step=2 base excludes MCQ -> empty intersection
        with self.assertRaises(ValueError):
            allowed_question_types_for_step(conf, 2)


if __name__ == "__main__":
    unittest.main()
