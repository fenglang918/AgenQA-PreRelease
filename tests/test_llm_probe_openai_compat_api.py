import io
import unittest
from contextlib import redirect_stdout
from unittest import mock


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> dict:
        return self._payload


class _FakeStreamingResponse:
    def __init__(self, *, lines: list[str], status_code: int = 200) -> None:
        self.status_code = status_code
        self._lines = lines
        self.text = ""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def iter_lines(self, decode_unicode: bool = True):
        for line in self._lines:
            yield line


class TestOpenAICompatProbeTool(unittest.TestCase):
    def test_extract_message_text_falls_back_to_reasoning_content(self) -> None:
        from infra.llm.probes.test_openai_compat_api import extract_message_text

        resp = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "OK_FROM_REASONING",
                    }
                }
            ]
        }
        self.assertEqual(extract_message_text(resp), "OK_FROM_REASONING")

    def test_consume_sse_stream_supports_reasoning_content(self) -> None:
        from infra.llm.probes.test_openai_compat_api import consume_sse_stream

        class _Resp:
            def iter_lines(self, decode_unicode: bool = True):
                yield (
                    'data: {"choices":[{"index":0,"delta":{"reasoning_content":"OK_FROM_STREAM"},"finish_reason":null}]}'
                )
                yield "data: [DONE]"

        content, reasoning = consume_sse_stream(_Resp())
        self.assertEqual(content, "")
        self.assertEqual(reasoning, "OK_FROM_STREAM")

    def test_cmd_responses_extracts_output_text(self) -> None:
        from infra.llm.probes import test_openai_compat_api as probe

        fake = _FakeResponse(
            200,
            {
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "HELLO_FROM_RESPONSES"}],
                    }
                ]
            },
        )

        with mock.patch("infra.llm.probes.test_openai_compat_api.requests.post", return_value=fake):
            result = probe.cmd_responses(
                "https://example.com/v1",
                "sk-test",
                model="gpt-x",
                file_path=None,
                instructions=None,
            )

        self.assertTrue(result["success"])
        self.assertIn("HELLO_FROM_RESPONSES", result.get("output_preview", ""))

    def test_cmd_stream_returns_reasoning_text_when_content_empty(self) -> None:
        from infra.llm.probes import test_openai_compat_api as probe

        lines = [
            'data: {"choices":[{"index":0,"delta":{"reasoning_content":"OK_FROM_STREAM"},"finish_reason":null}]}',
            "data: [DONE]",
        ]
        fake = _FakeStreamingResponse(lines=lines)

        with mock.patch("infra.llm.probes.test_openai_compat_api.requests.post", return_value=fake):
            buf = io.StringIO()
            with redirect_stdout(buf):
                result = probe.cmd_stream("https://example.com/v1", "sk-test", model="qwen-thinking")

        self.assertTrue(result["success"])
        self.assertEqual(result["content_text"], "")
        self.assertEqual(result["reasoning_text"], "OK_FROM_STREAM")
        self.assertEqual(result["printed"], "reasoning_content")


if __name__ == "__main__":
    unittest.main()
