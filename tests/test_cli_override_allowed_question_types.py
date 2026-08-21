import unittest

import cli


class TestCliOverrideAllowedQuestionTypes(unittest.TestCase):
    def test_override_writes_allowed_question_types(self) -> None:
        config = {"agent": {"question_type_policy": {"no_mcq_from_step": 2}}}
        cli._override_allowed_question_types(config, ["MCQ", "Derivation"])
        self.assertEqual(config["agent"]["question_type_policy"]["allowed_question_types"], ["MCQ", "Derivation"])

    def test_override_off_removes_key(self) -> None:
        config = {"agent": {"question_type_policy": {"allowed_question_types": ["MCQ", "Numeric"]}}}
        cli._override_allowed_question_types(config, ["off"])
        self.assertNotIn("allowed_question_types", config["agent"]["question_type_policy"])

    def test_override_parses_comma_separated(self) -> None:
        config = {"agent": {"question_type_policy": {}}}
        cli._override_allowed_question_types(config, ["MCQ,Derivation"])
        self.assertEqual(config["agent"]["question_type_policy"]["allowed_question_types"], ["MCQ", "Derivation"])

    def test_invalid_type_raises(self) -> None:
        config = {"agent": {"question_type_policy": {}}}
        with self.assertRaises(ValueError):
            cli._override_allowed_question_types(config, ["Foo"])


if __name__ == "__main__":
    unittest.main()
