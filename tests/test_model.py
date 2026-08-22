from unittest.mock import patch

import pytest

from emissary import PROVIDERS, CapabilityError, Provider, ProviderError, parse_spec
from emissary.llm import retry
from emissary.llm.credentials import Unauthenticated
from emissary.llm.decision import FinalOutput, ModelCapabilities, ModelResult, ToolDefinition, Usage
from emissary.llm.messages import TextBlock, UserMessage
from emissary.llm.model import FallbackModelCaller, call_model

MESSAGES = (UserMessage((TextBlock("hello"),)),)
RESULT = ModelResult(FinalOutput(text="done"), "anthropic", "claude", Usage(1, 1))
TOOL = ToolDefinition("lookup", "Look up.", {"type": "object"})


class Recorder:
    def __init__(self):
        self.text = []
        self.thinking = []

    def on_text(self, delta):
        self.text.append(delta)

    def on_thinking(self, delta):
        self.thinking.append(delta)


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


def test_fallback_caller_retries_only_availability_failures(monkeypatch):
    monkeypatch.setattr(retry, "RETRY_DELAYS", ())
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


def test_fallback_caller_does_not_repeat_the_same_provider(monkeypatch):
    monkeypatch.setattr(retry, "RETRY_DELAYS", ())
    caller = FallbackModelCaller(parse_spec("anthropic"), parse_spec("anthropic"))

    with (
        patch(
            "emissary.llm.model.call_model", side_effect=ProviderError("down", retryable=True)
        ) as called,
        pytest.raises(ProviderError),
    ):
        caller(system="s", messages=MESSAGES, tools=(TOOL,))

    assert called.call_count == 1


def test_streaming_failure_after_a_delta_is_not_retried_or_fallen_back(monkeypatch, caplog):
    """Once visible output escapes, another attempt would append a second,
    contradictory answer to the same sink. Availability is less important
    than never presenting a stitched-together answer as one model turn."""
    monkeypatch.setattr(retry, "RETRY_DELAYS", ())
    caller = FallbackModelCaller(parse_spec("anthropic"), parse_spec("kimi"))
    sink = Recorder()

    def partial_failure(*args, **kwargs):
        kwargs["sink"].on_text("partial")
        raise ProviderError("stream disconnected", retryable=True)

    with (
        patch("emissary.llm.model.call_model", side_effect=partial_failure) as called,
        caplog.at_level("WARNING", logger="emissary.llm.model"),
        pytest.raises(ProviderError, match="after streaming output"),
    ):
        caller(system="s", messages=MESSAGES, sink=sink)

    assert called.call_count == 1
    assert sink.text == ["partial"]
    assert "retry suppressed" in caplog.text


def test_streaming_failure_before_any_delta_can_fall_back(monkeypatch):
    """No sink state escaped, so retrying remains observationally safe."""
    monkeypatch.setattr(retry, "RETRY_DELAYS", ())
    caller = FallbackModelCaller(parse_spec("anthropic"), parse_spec("kimi"))
    sink = Recorder()
    calls = 0

    def fail_then_answer(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ProviderError("stream disconnected", retryable=True)
        kwargs["sink"].on_text("complete")
        return RESULT

    with patch("emissary.llm.model.call_model", side_effect=fail_then_answer):
        assert caller(system="s", messages=MESSAGES, sink=sink) is RESULT

    assert calls == 2
    assert sink.text == ["complete"]


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
