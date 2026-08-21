from __future__ import annotations

import os
from unittest import mock

from infra.config.config_loader import load_config_with_base
from infra.llm.service_client import LLMServiceSession, _ResponsesApiClient, _first_nonempty_text


class _Response:
    status_code = 200
    text = ""

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def json(self) -> dict:
        return self._payload


def test_responses_text_can_follow_reasoning_item() -> None:
    payload = {
        "output": [
            {"type": "reasoning", "content": []},
            {"type": "message", "content": [{"type": "output_text", "text": "VISIBLE"}]},
        ]
    }
    assert _first_nonempty_text(payload) == "VISIBLE"

    session = LLMServiceSession(
        service_id="test",
        model_name="gpt-test",
        base_url="https://api.example.com/v1",
        api_key="test-key",
        default_headers={},
        timeout=1,
        stream=False,
        extra_headers=None,
        api_style="responses",
    )
    assert session.extract_text(payload) == "VISIBLE"


def test_responses_client_maps_max_tokens(monkeypatch) -> None:
    seen: dict = {}

    def fake_post(_url, **kwargs):
        seen.update(kwargs["json"])
        return _Response(
            {"output": [{"type": "message", "content": [{"type": "output_text", "text": "OK"}]}]}
        )

    monkeypatch.setattr("requests.post", fake_post)
    client = _ResponsesApiClient(
        base_url="https://api.example.com/v1",
        api_key="test-key",
        timeout=1,
        extra_headers=None,
        default_headers=None,
    )
    client.chat([{"role": "user", "content": "hi"}], model="gpt-test", max_tokens=123)
    assert seen["max_output_tokens"] == 123
    assert "max_tokens" not in seen


def test_executable_prompt_modules_use_in_repo_common_contracts() -> None:
    from agenqa.prompts.executable_draft_step import EXECUTABLE_DRAFT_STEP_V1
    from agenqa.prompts.executable_extract import EXECUTABLE_EXTRACT_V1
    from agenqa.prompts.executable_revise_step import EXECUTABLE_REVISE_STEP_V1
    from agenqa.prompts.executable_test_inputs import EXECUTABLE_TEST_INPUTS_V1

    assert all(
        text.strip()
        for text in (
            EXECUTABLE_DRAFT_STEP_V1,
            EXECUTABLE_EXTRACT_V1,
            EXECUTABLE_REVISE_STEP_V1,
            EXECUTABLE_TEST_INPUTS_V1,
        )
    )


def test_portable_config_resolves_standard_openai_key(monkeypatch) -> None:
    with mock.patch.dict(
        os.environ,
        {
            "OPENAI_API_KEY": "test-key",
            "AGENQA_API_BASE": "https://api.openai.com/v1",
            "AGENQA_MODEL": "gpt-test",
        },
        clear=False,
    ):
        config = load_config_with_base("config/agent_openai.yaml")
    generator = config["init"]["generator"]
    assert generator["api_key"] == "test-key"
    assert generator["api_base"] == "https://api.openai.com/v1"
    assert generator["model_name"] == "gpt-test"


def test_portable_config_loads_demo_pdf_and_compiles_real_graph() -> None:
    from agenqa.graph.builder import build_graph
    from infra.input_adapters.paper_input_loader import load_one_paper_like_record

    with mock.patch.dict(
        os.environ,
        {"OPENAI_API_KEY": "test-key", "AGENQA_MODEL": "gpt-test"},
        clear=False,
    ):
        config = load_config_with_base("config/agent_openai.yaml")
        record = load_one_paper_like_record(config)
        graph = build_graph(config)

    assert len(record["text"]) > 1000
    assert "Thickness perturbation" in record["text"]
    assert graph is not None
