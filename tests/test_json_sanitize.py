import json
import unittest

from infra.text.json_sanitize import sanitize_json_text


class TestJsonSanitize(unittest.TestCase):
    def test_escapes_stray_quotes_inside_string(self) -> None:
        raw = '{\n  "Solution": "DELFI 属于"无似然"（Likelihood-Free）推理方法"\n}\n'
        fixed = sanitize_json_text(raw)
        obj = json.loads(fixed)
        self.assertEqual(obj["Solution"], 'DELFI 属于"无似然"（Likelihood-Free）推理方法')

    def test_keeps_valid_json_unchanged(self) -> None:
        raw = '{ "a": "b", "c": 1, "d": ["x", "y"] }'
        fixed = sanitize_json_text(raw)
        self.assertEqual(json.loads(raw), json.loads(fixed))
