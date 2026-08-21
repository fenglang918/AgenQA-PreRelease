import unittest

from agenqa.nodes.evaluators.numeric_judge import parse_numeric_judge_output


class TestNumericJudgeStrict(unittest.TestCase):
    def test_parses_boolean_equivalent_true(self) -> None:
        text = """```json
{"equivalent": true, "reason": "ok"}
```"""
        eq, reason = parse_numeric_judge_output(text)
        self.assertTrue(eq)
        self.assertEqual(reason, "ok")

    def test_parses_boolean_equivalent_false(self) -> None:
        eq, reason = parse_numeric_judge_output('{"equivalent": false}')
        self.assertFalse(eq)
        self.assertIsNone(reason)

    def test_rejects_missing_equivalent(self) -> None:
        with self.assertRaises(ValueError):
            parse_numeric_judge_output('{"reason": "x"}')

    def test_rejects_non_boolean_equivalent(self) -> None:
        with self.assertRaises(ValueError):
            parse_numeric_judge_output('{"equivalent": "yes"}')
