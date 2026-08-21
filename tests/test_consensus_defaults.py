import unittest

from agenqa.nodes.evaluators import consensus as consensus_mod


class TestConsensusDefaults(unittest.TestCase):
    def test_consensus_mode_defaults_to_always_when_multi_strong(self) -> None:
        self.assertEqual(consensus_mod._consensus_mode({}, 2), "always")
        self.assertEqual(consensus_mod._consensus_mode({}, 1), "none")

    def test_consensus_mode_accepts_disabled_aliases(self) -> None:
        self.assertEqual(consensus_mod._consensus_mode({"consensus": {"mode": "none"}}, 2), "none")
        self.assertEqual(consensus_mod._consensus_mode({"consensus": {"mode": "disabled"}}, 2), "none")
        self.assertEqual(consensus_mod._consensus_mode({"consensus": {"mode": "off"}}, 2), "none")

    def test_consensus_mode_rejects_lazy(self) -> None:
        with self.assertRaises(ValueError):
            consensus_mod._consensus_mode({"consensus": {"mode": "lazy"}}, 2)

    def test_answer_judge_defaults_numeric_to_llm(self) -> None:
        self.assertEqual(consensus_mod._answer_judge_mode({}, "Numeric"), "llm")
        self.assertEqual(consensus_mod._answer_judge_mode({}, "Derivation"), "llm")
        self.assertEqual(consensus_mod._answer_judge_mode({}, "MCQ"), "normalize")
