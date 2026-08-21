import unittest

from agenqa.nodes.utils import is_symbolic_only_for_question_type


class TestSymbolicOnlyQuestionTypes(unittest.TestCase):
    def test_global_symbolic_only_always_true(self) -> None:
        conf = {"agent": {"symbolic_only": True}}
        self.assertTrue(is_symbolic_only_for_question_type(conf, "MCQ"))
        self.assertTrue(is_symbolic_only_for_question_type(conf, "Derivation"))
        self.assertTrue(is_symbolic_only_for_question_type(conf, "Numeric"))

    def test_per_qtype_symbolic_only(self) -> None:
        conf = {"agent": {"symbolic_only": False, "symbolic_only_question_types": ["Derivation"]}}
        self.assertFalse(is_symbolic_only_for_question_type(conf, "MCQ"))
        self.assertTrue(is_symbolic_only_for_question_type(conf, "Derivation"))
        self.assertFalse(is_symbolic_only_for_question_type(conf, "Numeric"))

    def test_per_qtype_multiple(self) -> None:
        conf = {"agent": {"symbolic_only_question_types": ["MCQ", "Derivation"]}}
        self.assertTrue(is_symbolic_only_for_question_type(conf, "MCQ"))
        self.assertTrue(is_symbolic_only_for_question_type(conf, "Derivation"))
        self.assertFalse(is_symbolic_only_for_question_type(conf, "Numeric"))

    def test_string_form(self) -> None:
        conf = {"agent": {"symbolic_only_question_types": "Derivation"}}
        self.assertTrue(is_symbolic_only_for_question_type(conf, "derive"))


if __name__ == "__main__":
    unittest.main()
