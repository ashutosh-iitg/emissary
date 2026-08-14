"""The Anthropic wire adapter — mocked at the SDK client, never the network."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest

from emissary import ProviderError, parse_spec
from emissary.llm.wire.anthropic import call_tool

TOOL = {"name": "record", "description": "d", "input_schema": {"type": "object"}}


def _response(*, stop_reason="tool_use", content=None, model="claude-opus-5", cache_read=0):
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=content or [],
        model=model,
        usage=SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=cache_read),
    )


def _mock_client(response=None, side_effect=None):
    client = MagicMock()
    if side_effect is not None:
        client.messages.create.side_effect = side_effect
    else:
        client.messages.create.return_value = response
    return patch("anthropic.Anthropic", return_value=client)


def test_call_tool_returns_the_tool_arguments_and_usage():
    response = _response(
        content=[SimpleNamespace(type="tool_use", name="record", input={"a": 1})],
        cache_read=7,
    )
    with _mock_client(response):
        out = call_tool(
            parse_spec("anthropic"),
            system="s",
            blocks=[{"text": "doc", "cache": True}, {"text": "instr", "cache": False}],
            tool=TOOL,
        )

    assert out.payload == {"a": 1}
    assert out.provider == "anthropic"
    assert out.cached_input_tokens == 7


def test_cache_marked_blocks_get_an_ephemeral_breakpoint():
    response = _response(content=[SimpleNamespace(type="tool_use", name="record", input={})])
    with _mock_client(response) as ctor:
        call_tool(
            parse_spec("anthropic"),
            system="s",
            blocks=[{"text": "doc", "cache": True}, {"text": "instr", "cache": False}],
            tool=TOOL,
        )

        sent = ctor.return_value.messages.create.call_args.kwargs["messages"][0]["content"]
        assert sent[0]["cache_control"] == {"type": "ephemeral"}
        assert "cache_control" not in sent[1]


def test_a_refusal_is_retryable():
    response = _response(stop_reason="refusal")
    with _mock_client(response), pytest.raises(ProviderError) as caught:
        call_tool(parse_spec("anthropic"), system="s", blocks=[{"text": "d"}], tool=TOOL)

    assert caught.value.retryable


def test_no_matching_tool_call_raises():
    response = _response(content=[])
    with _mock_client(response), pytest.raises(ProviderError, match="no record call"):
        call_tool(parse_spec("anthropic"), system="s", blocks=[{"text": "d"}], tool=TOOL)


def test_server_errors_are_retryable_client_errors_are_not():
    request = httpx.Request("POST", "https://api.anthropic.com")

    import anthropic

    overloaded = anthropic.APIStatusError(
        "overloaded", response=httpx.Response(529, request=request), body=None
    )
    with _mock_client(side_effect=overloaded), pytest.raises(ProviderError) as caught:
        call_tool(parse_spec("anthropic"), system="s", blocks=[{"text": "d"}], tool=TOOL)
    assert caught.value.retryable

    bad_request = anthropic.APIStatusError(
        "bad request", response=httpx.Response(400, request=request), body=None
    )
    with _mock_client(side_effect=bad_request), pytest.raises(ProviderError) as caught:
        call_tool(parse_spec("anthropic"), system="s", blocks=[{"text": "d"}], tool=TOOL)
    assert not caught.value.retryable


def test_a_connection_error_is_retryable():
    import anthropic

    request = httpx.Request("POST", "https://api.anthropic.com")
    with (
        _mock_client(side_effect=anthropic.APIConnectionError(request=request)),
        pytest.raises(ProviderError) as caught,
    ):
        call_tool(parse_spec("anthropic"), system="s", blocks=[{"text": "d"}], tool=TOOL)

    assert caught.value.retryable
