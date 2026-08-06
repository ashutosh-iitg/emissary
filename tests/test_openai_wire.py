"""The OpenAI-compatible wire adapter — mocked at the SDK client, never the
network. This is the wire kimi/deepseek/gemini/vllm all share with openai.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest

from emissary import ProviderError, parse_spec
from emissary.wire.openai_wire import call_text, call_tool

TOOL = {"name": "record", "description": "d", "input_schema": {"type": "object"}}


def _response(*, finish_reason="tool_calls", tool_calls=None, text=None, model="kimi-k3"):
    message = SimpleNamespace(tool_calls=tool_calls, content=text)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
        model=model,
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, prompt_tokens_details=None),
    )


def _mock_client(response=None, side_effect=None):
    client = MagicMock()
    if side_effect is not None:
        client.chat.completions.create.side_effect = side_effect
    else:
        client.chat.completions.create.return_value = response
    return patch("openai.OpenAI", return_value=client)


def test_call_tool_returns_the_parsed_tool_arguments():
    call = SimpleNamespace(function=SimpleNamespace(arguments=json.dumps({"a": 1})))
    response = _response(tool_calls=[call])
    with _mock_client(response):
        out = call_tool(parse_spec("kimi"), system="s", blocks=[{"text": "d"}], tool=TOOL)

    assert out.payload == {"a": 1}
    assert out.provider == "kimi"


def test_blocks_are_concatenated_with_no_cache_control():
    call = SimpleNamespace(function=SimpleNamespace(arguments="{}"))
    response = _response(tool_calls=[call])
    with _mock_client(response) as ctor:
        call_tool(
            parse_spec("kimi"),
            system="s",
            blocks=[{"text": "doc", "cache": True}, {"text": "instr", "cache": False}],
            tool=TOOL,
        )

        sent = ctor.return_value.chat.completions.create.call_args.kwargs["messages"]
        assert sent[1]["content"] == "doc\n\ninstr"


def test_no_tool_call_raises():
    response = _response(tool_calls=None, finish_reason="stop")
    with _mock_client(response), pytest.raises(ProviderError, match="no record call"):
        call_tool(parse_spec("kimi"), system="s", blocks=[{"text": "d"}], tool=TOOL)


def test_unparseable_tool_arguments_are_not_retryable():
    """The model answered, just unusably — retrying elsewhere would be
    shopping for a provider whose JSON happens to parse."""
    call = SimpleNamespace(function=SimpleNamespace(arguments="not json"))
    response = _response(tool_calls=[call])
    with _mock_client(response), pytest.raises(ProviderError) as caught:
        call_tool(parse_spec("kimi"), system="s", blocks=[{"text": "d"}], tool=TOOL)

    assert not caught.value.retryable


def test_server_errors_are_retryable_client_errors_are_not():
    request = httpx.Request("POST", "https://api.moonshot.ai/v1")

    import openai

    overloaded = openai.APIStatusError(
        "overloaded", response=httpx.Response(503, request=request), body=None
    )
    with _mock_client(side_effect=overloaded), pytest.raises(ProviderError) as caught:
        call_tool(parse_spec("kimi"), system="s", blocks=[{"text": "d"}], tool=TOOL)
    assert caught.value.retryable

    bad_request = openai.APIStatusError(
        "bad request", response=httpx.Response(400, request=request), body=None
    )
    with _mock_client(side_effect=bad_request), pytest.raises(ProviderError) as caught:
        call_tool(parse_spec("kimi"), system="s", blocks=[{"text": "d"}], tool=TOOL)
    assert not caught.value.retryable


def test_call_text_returns_the_message_content():
    response = _response(text="hello there")
    with _mock_client(response):
        out = call_text(
            parse_spec("kimi"), system="s", messages=[{"role": "user", "content": "hi"}]
        )

    assert out.payload == "hello there"


def test_strict_is_only_sent_for_first_party_providers():
    call = SimpleNamespace(function=SimpleNamespace(arguments="{}"))
    response = _response(tool_calls=[call])

    with _mock_client(response) as ctor:
        call_tool(parse_spec("openai:gpt-5"), system="s", blocks=[{"text": "d"}], tool=TOOL)
        sent_openai = ctor.return_value.chat.completions.create.call_args.kwargs["tools"][0][
            "function"
        ]
    assert sent_openai["strict"] is True

    with _mock_client(response) as ctor:
        call_tool(parse_spec("kimi"), system="s", blocks=[{"text": "d"}], tool=TOOL)
        sent_kimi = ctor.return_value.chat.completions.create.call_args.kwargs["tools"][0][
            "function"
        ]
    assert "strict" not in sent_kimi


def test_vllm_uses_a_placeholder_key_when_none_is_set(monkeypatch):
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    call = SimpleNamespace(function=SimpleNamespace(arguments="{}"))
    response = _response(tool_calls=[call])

    with patch("openai.OpenAI", return_value=MagicMock()) as ctor:
        ctor.return_value.chat.completions.create.return_value = response
        call_tool(parse_spec("vllm:qwen3-8b"), system="s", blocks=[{"text": "d"}], tool=TOOL)

    assert ctor.call_args.kwargs["api_key"] == "not-required"
    assert ctor.call_args.kwargs["base_url"] == "http://localhost:8000/v1"
