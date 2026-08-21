import sys
import tempfile
import unittest
from pathlib import Path
from string import Template

from infra.data.io import write_jsonl, read_jsonl

from agenqa.skills.solver_tool import SolverToolConfig, SolverToolRunner


class _FakeSession:
    def __init__(self, content: str) -> None:
        self._content = content
        self.model_name = "fake-model"
        self.service_id = "fake-service"

    def chat(self, messages, **kwargs):  # noqa: ANN001, ANN201
        return {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": self._content},
                    "finish_reason": "stop",
                }
            ]
        }

    def extract_text(self, response, default=""):  # noqa: ANN001, ANN201
        try:
            return response["choices"][0]["message"]["content"]
        except Exception:
            return default


def _make_runner_with_fake_session(content: str) -> SolverToolRunner:
    runner = SolverToolRunner.__new__(SolverToolRunner)
    runner.config = SolverToolConfig(
        generator={},
        prompt_path=Path("src/agenqa/prompts/solver_tool.py"),
        prompt_text="Known:$known\nQuestion:$question\n",
        lang="zh",
        timeout_seconds=2.0,
        memory_limit_mb=256,
        temp_dir=tempfile.gettempdir(),
        python_bin=sys.executable,
    )
    runner.session = _FakeSession(content)
    runner._chat_args = {}
    runner.prompt_text = runner.config.prompt_text or ""
    runner.prompt_template = Template(runner.prompt_text)
    return runner


class TestSolverToolPolicy(unittest.TestCase):
    def test_render_prompt_accepts_answer_output_spec_prompt_var(self) -> None:
        runner = _make_runner_with_fake_session("{}")
        runner.config.prompt_text = "Known:$known\nQuestion:$question\nSpec:$answer_output_spec\n"
        runner.config.prompt_vars = {"answer_output_spec": "OUTPUT_SPEC"}
        runner.prompt_text = runner.config.prompt_text or ""
        runner.prompt_template = Template(runner.prompt_text)

        prompt = runner._render_prompt("k", "q")

        self.assertIn("OUTPUT_SPEC", prompt)
        self.assertNotIn("$answer_output_spec", prompt)

    def test_tool_exec_failure_makes_row_invalid(self) -> None:
        content = """```json
{
  "Answer": "\\\\boxed{1}",
  "SolverReasoning": null,
  "QuestionWellPosed": true,
  "CorrectnessFeedback": "ok",
  "DifficultyFeedback": null,
  "KeyConclusion": null,
  "ToolUsed": true,
  "ToolName": "python_executor",
  "ToolCode": "raise RuntimeError('boom')",
  "ToolNotes": ""
}
```"""
        runner = _make_runner_with_fake_session(content)

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            kqa_path = td_path / "kqa.jsonl"
            out_path = td_path / "out.jsonl"
            write_jsonl(
                [{"known": "k", "question": "q", "answer": "\\\\boxed{1}"}],
                kqa_path,
                schema=None,
                append=False,
            )

            runner.run(kqa_path, out_path, append=False)
            row = next(read_jsonl(out_path, schema=None))

        self.assertFalse(row["correct"])
        self.assertEqual(row["solve"], "")
        self.assertEqual(row.get("error"), "tool_exec_failed")
        self.assertTrue(row["tool"]["used"])
        self.assertIsInstance(row["tool"]["exec"], dict)
        self.assertFalse(row["tool"]["exec"]["success"])

    def test_tool_claimed_but_missing_code_is_invalid(self) -> None:
        content = """```json
{
  "Answer": "\\\\boxed{1}",
  "SolverReasoning": null,
  "QuestionWellPosed": true,
  "CorrectnessFeedback": "ok",
  "DifficultyFeedback": null,
  "KeyConclusion": null,
  "ToolUsed": true,
  "ToolName": "python_executor",
  "ToolCode": "",
  "ToolNotes": ""
}
```"""
        runner = _make_runner_with_fake_session(content)

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            kqa_path = td_path / "kqa.jsonl"
            out_path = td_path / "out.jsonl"
            write_jsonl(
                [{"known": "k", "question": "q", "answer": "\\\\boxed{1}"}],
                kqa_path,
                schema=None,
                append=False,
            )

            runner.run(kqa_path, out_path, append=False)
            row = next(read_jsonl(out_path, schema=None))

        self.assertFalse(row["correct"])
        self.assertEqual(row["solve"], "")
        self.assertEqual(row.get("error"), "tool_exec_failed")
        self.assertTrue(row["tool"]["used"])

    def test_no_tool_uses_reported_answer(self) -> None:
        content = """```json
{
  "Answer": "\\\\boxed{1}",
  "SolverReasoning": null,
  "QuestionWellPosed": true,
  "CorrectnessFeedback": "ok",
  "DifficultyFeedback": null,
  "KeyConclusion": null,
  "ToolUsed": false,
  "ToolName": "python_executor",
  "ToolCode": "",
  "ToolNotes": ""
}
```"""
        runner = _make_runner_with_fake_session(content)

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            kqa_path = td_path / "kqa.jsonl"
            out_path = td_path / "out.jsonl"
            write_jsonl(
                [{"known": "k", "question": "q", "answer": "\\\\boxed{1}"}],
                kqa_path,
                schema=None,
                append=False,
            )

            runner.run(kqa_path, out_path, append=False)
            row = next(read_jsonl(out_path, schema=None))

        # correctness is decided later by solve-stage LLM judge (no programmatic judge here)
        self.assertIsNone(row["correct"])
        self.assertEqual(row["solve"], "\\boxed{1}")
