import unittest

from agenqa.nodes.evaluators.consensus import _detect_status


class TestConsensusDetectStatusTool(unittest.TestCase):
    def test_tool_exec_failure_is_error(self) -> None:
        row = {
            "tool": {
                "used": True,
                "code": "print('x')",
                "value": None,
                "exec": {"success": False, "error": "boom"},
            }
        }
        self.assertEqual(_detect_status(row), "request_failed")

    def test_tool_success_but_no_value_is_parse_error(self) -> None:
        row = {
            "tool": {
                "used": True,
                "code": "print('x')",
                "value": None,
                "exec": {"success": True, "output": "not json"},
            }
        }
        self.assertEqual(_detect_status(row), "parse_failed")

    def test_tool_claimed_but_missing_code_is_parse_error(self) -> None:
        row = {"tool": {"used": True, "code": "   "}}
        self.assertEqual(_detect_status(row), "parse_failed")
