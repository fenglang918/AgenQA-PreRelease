import unittest


class TestFormatKnownForSolver(unittest.TestCase):
    def test_v2_excludes_seed_meta_by_default(self) -> None:
        from agenqa.domain.known_utils import format_known_for_solver  # type: ignore

        known = {
            "schema_version": 2,
            "episode_seed": {"subject": "materials", "keywords": ["pcm", "crosstalk"]},
            "premise_bank": [{"id": "p1", "text": "A=1"}, {"id": "p2", "statement": "B=2"}],
            "fact_bank": [{"id": "f1", "statement": "C=3"}],
            "step_certs": [{"step": 1, "key_fact_id": "f1", "cert_text": "derived C"}],
        }

        txt = format_known_for_solver(known)
        self.assertIn("Premise_0: A=1", txt)
        self.assertIn("Premise_1: B=2", txt)
        self.assertIn("Fact_0: C=3", txt)
        self.assertIn("Cert_0:", txt)
        self.assertNotIn("Subject:", txt)
        self.assertNotIn("Keywords:", txt)

    def test_v2_includes_seed_meta_when_enabled(self) -> None:
        from agenqa.domain.known_utils import format_known_for_solver  # type: ignore

        known = {
            "schema_version": 2,
            "episode_seed": {"subject": "materials", "keywords": ["pcm", "crosstalk"]},
            "premise_bank": [{"id": "p1", "text": "A=1"}],
            "fact_bank": [],
            "step_certs": [],
        }

        txt = format_known_for_solver(known, include_seed_meta=True)
        self.assertIn("Subject: materials", txt)
        self.assertIn("Keywords: pcm, crosstalk", txt)
        self.assertIn("Premise_0: A=1", txt)

    def test_v1_fallback(self) -> None:
        from agenqa.domain.known_utils import format_known_for_solver  # type: ignore

        known = {
            "known_0": "base",
            "background": ["b1"],
            "history": [{"question_0": "q0", "answer_0": "a0"}],
            "derived_facts": ["d1"],
        }

        txt = format_known_for_solver(known)
        self.assertIn("Known_0: base", txt)
        self.assertIn("Background_0: b1", txt)
        self.assertIn("Q_0: q0", txt)
        self.assertIn("A_0: a0", txt)
        self.assertIn("DerivedFact_0: d1", txt)


if __name__ == "__main__":
    unittest.main()
