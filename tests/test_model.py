from unittest.mock import patch

import pytest

from emissary import PROVIDERS, CapabilityError, Provider, ProviderError, parse_spec
from emissary.llm.credentials import Unauthenticated
from emissary.llm.decision import FinalOutput, ModelCapabilities, ModelResult, ToolDefinition, Usage
from emissary.llm.messages import TextBlock, UserMessage
from emissary.llm.model import FallbackModelCaller, call_model

MESSAGES = (UserMessage((TextBlock("hello"),)),)
RESULT = ModelResult(FinalOutput(text="done"), "anthropic", "claude", Usage(1, 1))
TOOL = ToolDefinition("lookup", "Look up.", {"type": "object"})


def test_call_model_gates_credentials_before_wire_dispatch(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with (
        patch("emissary.llm.wire.anthropic.call_model") as wire,
        pytest.raises(ProviderError, match="ANTHROPIC_API_KEY"),
    ):
        call_model(parse_spec("anthropic"), system="s", messages=MESSAGES)

    wire.assert_not_called()


def test_call_model_dispatches_by_wire_and_returns_normalized_result(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")

    with patch("emissary.llm.wire.anthropic.call_model", return_value=RESULT) as wire:
        assert call_model(parse_spec("anthropic"), system="s", messages=MESSAGES) is RESULT

    wire.assert_called_once()


def test_fallback_caller_retries_only_availability_failures():
    caller = FallbackModelCaller(parse_spec("anthropic"), parse_spec("kimi"))
    outcomes = [ProviderError("overloaded", retryable=True), RESULT]

    with patch("emissary.llm.model.call_model", side_effect=outcomes) as called:
        assert caller(system="s", messages=MESSAGES) is RESULT
    assert called.call_count == 2

    with (
        patch(
            "emissary.llm.model.call_model", side_effect=ProviderError("malformed", retryable=False)
        ) as called,
        pytest.raises(ProviderError, match="malformed"),
    ):
        caller(system="s", messages=MESSAGES)
    assert called.call_count == 1


def test_fallback_caller_does_not_repeat_the_same_provider():
    caller = FallbackModelCaller(parse_spec("anthropic"), parse_spec("anthropic"))

    with (
        patch(
            "emissary.llm.model.call_model", side_effect=ProviderError("down", retryable=True)
        ) as called,
        pytest.raises(ProviderError),
    ):
        caller(system="s", messages=MESSAGES, tools=(TOOL,))

    assert called.call_count == 1


def test_unsupported_tool_capability_fails_before_wire_dispatch(monkeypatch):
    monkeypatch.setitem(
        PROVIDERS,
        "textonly",
        Provider(
            wire="openai",
            credential=Unauthenticated(),
            default_model="plain",
            capabilities=ModelCapabilities(),
        ),
    )

    with (
        patch("emissary.llm.wire.openai_compatible.call_model") as wire,
        pytest.raises(CapabilityError, match="does not support tool calling"),
    ):
        call_model(parse_spec("textonly"), system="s", messages=MESSAGES, tools=(TOOL,))

    wire.assert_not_called()
