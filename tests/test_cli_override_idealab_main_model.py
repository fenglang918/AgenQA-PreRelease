import logging
import unittest

import cli


class TestOverrideIdealabMainModel(unittest.TestCase):
    def test_override_does_not_raise_for_strong_list_and_dict_targets(self) -> None:
        logger = logging.getLogger("test")
        config = {
            "init": {
                "generator": {"model_name": "old", "generation": {"temperature": 0.3}},
                "paper_brief": {"generator": {"model_name": "old", "generation": {"temperature": 0.3}}},
            },
            "director": {"generator": {"model_name": "old", "generation": {"temperature": 0.3}}},
            "operators": {
                "extend": {"generator": {"model_name": "old", "generation": {"temperature": 0.3}}},
                "revise": {"generator": {"model_name": "old", "generation": {"temperature": 0.3}}},
            },
            "solvers": {
                "strong": [
                    {
                        "generator": {"model_name": "old", "generation": {"temperature": 0.3}},
                    }
                ]
            },
        }

        target = "gpt-5-mini-0807-global"
        cli._override_idealab_main_model(config, target, logger=logger)

        self.assertEqual(config["init"]["generator"]["model_name"], target)
        self.assertEqual(config["init"]["paper_brief"]["generator"]["model_name"], target)
        self.assertEqual(config["director"]["generator"]["model_name"], target)
        self.assertEqual(config["operators"]["extend"]["generator"]["model_name"], target)
        self.assertEqual(config["operators"]["revise"]["generator"]["model_name"], target)
        self.assertEqual(config["solvers"]["strong"][0]["generator"]["model_name"], target)

        # gpt-5-mini family: explicit temperature must be removed.
        self.assertNotIn("temperature", config["init"]["generator"]["generation"])
        self.assertNotIn("temperature", config["init"]["paper_brief"]["generator"]["generation"])
        self.assertNotIn("temperature", config["director"]["generator"]["generation"])
        self.assertNotIn("temperature", config["operators"]["extend"]["generator"]["generation"])
        self.assertNotIn("temperature", config["operators"]["revise"]["generator"]["generation"])
        self.assertNotIn("temperature", config["solvers"]["strong"][0]["generator"]["generation"])
