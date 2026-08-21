import os
import unittest
from unittest import mock


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self) -> dict:
        return self._payload


class TestSimpleApiClientHttpErrors(unittest.TestCase):
    def setUp(self) -> None:
        self._old_env = dict(os.environ)
        os.environ["SCICLONE_HTTP_RETRY_TIMES"] = "1"

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._old_env)

    def test_non_200_raises_with_gateway_error_message(self) -> None:
        from infra.llm.service_client import _SimpleApiClient  # type: ignore

        client = _SimpleApiClient(
            base_url="https://example.com/v1",
            api_key="sk-invalid",
            timeout=1,
            stream=False,
            extra_headers=None,
            default_headers=None,
        )

        fake = _FakeResponse(
            400,
            {"status": 400, "error": {"error": "无效的api key"}},
            text="bad request",
        )

        with mock.patch("requests.post", return_value=fake):
            with self.assertRaises(RuntimeError) as ctx:
                client.chat([{"role": "user", "content": "hi"}], model="gemini-3-pro-preview")

        msg = str(ctx.exception)
        self.assertIn("status=400", msg)
        self.assertIn("无效的api key", msg)

    def test_http_200_with_embedded_status_raises(self) -> None:
        from infra.llm.service_client import _SimpleApiClient  # type: ignore

        client = _SimpleApiClient(
            base_url="https://example.com/v1",
            api_key="sk-invalid",
            timeout=1,
            stream=False,
            extra_headers=None,
            default_headers=None,
        )

        # Some gateways incorrectly return HTTP 200 while embedding the actual status in JSON.
        fake = _FakeResponse(
            200,
            {"status": 400, "error": {"message": "无效的api key"}},
            text="ok-but-error",
        )

        with mock.patch("requests.post", return_value=fake):
            with self.assertRaises(RuntimeError) as ctx:
                client.chat([{"role": "user", "content": "hi"}], model="claude_sonnet4_5")

        msg = str(ctx.exception)
        self.assertIn("status=400", msg)
        self.assertIn("无效的api key", msg)

    def test_claude_gateway_error_not_silently_returned(self) -> None:
        from infra.llm.service_client import LLMServiceSession  # type: ignore

        session = LLMServiceSession(
            service_id="test",
            model_name="claude_sonnet4_5",
            base_url="https://idealab.alibaba-inc.com/api/openai/v1",
            api_key="sk-invalid",
            default_headers={},
            timeout=1,
            stream=True,
            extra_headers=None,
            api_channel="idealab",
        )

        # Force the OpenAI-compat path to fail, so the Claude gateway error would have been returned previously.
        session._client.chat = mock.Mock(side_effect=RuntimeError("Streaming chat got empty content"))  # type: ignore[attr-defined]

        claude_err = {
            "status": 400,
            "error": {"message": "无效的api key"},
            "_gateway": {"api": "anthropic_messages", "url": "https://idealab.alibaba-inc.com/api/code/v1/messages"},
        }

        with mock.patch.dict(
            os.environ,
            {"SCICLONE_CLAUDE_MESSAGES_GATEWAY": "1", "SCICLONE_CLAUDE_PREFER_MESSAGES": "1"},
        ):
            with mock.patch("infra.llm.service_client._anthropic_messages_call", return_value=claude_err):
                with self.assertRaises(RuntimeError) as ctx:
                    session.chat([{"role": "user", "content": "hi"}], model="claude_sonnet4_5")

        msg = str(ctx.exception)
        self.assertIn("status=400", msg)
        self.assertIn("无效的api key", msg)

    def test_extract_text_falls_back_to_reasoning_content(self) -> None:
        from infra.llm.service_client import LLMServiceSession  # type: ignore

        session = LLMServiceSession(
            service_id="test",
            model_name="qwen3-next-80b-a3b-thinking",
            base_url="https://example.com/api/openai/v1",
            api_key="sk-test",
            default_headers={},
            timeout=1,
            stream=False,
            extra_headers=None,
        )

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
        self.assertEqual(session.extract_text(resp, default=""), "OK_FROM_REASONING")

    def test_streaming_reasoning_content_is_preserved(self) -> None:
        from infra.llm.service_client import LLMServiceSession  # type: ignore

        class _FakeStreamingResponse:
            def __init__(self) -> None:
                self.status_code = 200
                self.headers = {}

            def iter_lines(self, decode_unicode: bool = True):
                yield 'data: {"id":"x","object":"chat.completion.chunk","created":0,"model":"qwen3-next-80b-a3b-thinking","choices":[{"index":0,"delta":{"reasoning_content":"OK_FROM_STREAM"},"finish_reason":null}]}'
                yield "data: [DONE]"

            def close(self) -> None:
                return None

        session = LLMServiceSession(
            service_id="test",
            model_name="qwen3-next-80b-a3b-thinking",
            base_url="https://example.com/api/openai/v1",
            api_key="sk-test",
            default_headers={},
            timeout=1,
            stream=True,
            extra_headers=None,
        )

        with mock.patch("requests.post", return_value=_FakeStreamingResponse()):
            resp = session.chat([{"role": "user", "content": "hi"}], model="qwen3-next-80b-a3b-thinking")

        self.assertEqual(session.extract_text(resp, default=""), "OK_FROM_STREAM")
