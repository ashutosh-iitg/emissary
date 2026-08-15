"""Streaming observes a turn; it never changes what the turn returns (ADR-0022).

The load-bearing property is that a streamed call and an unstreamed one produce
the *same* `ModelResult`. Everything downstream — the event log, replay,
fallback — consumes that value, so a streaming path that normalized separately
would let the two drift apart silently.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from emissary import parse_spec
from emissary.llm.decision import ModelSettings, ToolCalls
from emissary.llm.messages import TextBlock, UserMessage
from emissary.llm.streaming import StreamSink
from emissary.llm.wire import anthropic, gemini, openai_compatible

MESSAGES = (UserMessage((TextBlock("count the letters"),)),)


class Recorder:
    """A sink that satisfies the protocol and remembers what it saw."""

    def __init__(self) -> None:
        self.text: list[str] = []
        self.thinking: list[str] = []

    def on_text(self, delta: str) -> None:
        self.text.append(delta)

    def on_thinking(self, delta: str) -> None:
        self.thinking.append(delta)


def test_recorder_satisfies_the_protocol():
    assert isinstance(Recorder(), StreamSink)


# --- Anthropic -------------------------------------------------------------


def _anthropic_final(*, thinking=True):
    content = [SimpleNamespace(type="text", text="six")]
    if thinking:
        content.insert(0, SimpleNamespace(type="thinking", thinking="counting", signature="sig-1"))
    return SimpleNamespace(
        stop_reason="end_turn",
        content=content,
        model="claude",
        usage=SimpleNamespace(input_tokens=5, output_tokens=2, cache_read_input_tokens=0),
    )


def _anthropic_stream_client(events, final):
    stream = MagicMock()
    stream.__iter__ = lambda self: iter(events)
    stream.get_final_message.return_value = final
    manager = MagicMock()
    manager.__enter__ = lambda self: stream
    manager.__exit__ = lambda self, *exc: False
    client = MagicMock()
    client.messages.stream.return_value = manager
    return client


def test_anthropic_streams_text_and_thinking_deltas_then_returns_one_result():
    events = [
        SimpleNamespace(
            type="content_block_delta",
            delta=SimpleNamespace(type="thinking_delta", thinking="count"),
        ),
        SimpleNamespace(
            type="content_block_delta", delta=SimpleNamespace(type="thinking_delta", thinking="ing")
        ),
        SimpleNamespace(
            type="content_block_delta", delta=SimpleNamespace(type="text_delta", text="six")
        ),
        SimpleNamespace(type="message_stop"),
    ]
    sink = Recorder()

    with patch(
        "anthropic.Anthropic", return_value=_anthropic_stream_client(events, _anthropic_final())
    ):
        result = anthropic.call_model(
            parse_spec("anthropic"), system="s", messages=MESSAGES, sink=sink
        )

    assert sink.thinking == ["count", "ing"]
    assert sink.text == ["six"]
    assert result.decision.text == "six"
    # Streaming must not cost the signature — the next turn is rejected without it.
    assert result.reasoning.blocks[0]["signature"] == "sig-1"


def test_anthropic_streamed_and_unstreamed_results_are_identical():
    final = _anthropic_final()
    plain = MagicMock()
    plain.messages.create.return_value = final

    with patch("anthropic.Anthropic", return_value=plain):
        unstreamed = anthropic.call_model(parse_spec("anthropic"), system="s", messages=MESSAGES)

    with patch("anthropic.Anthropic", return_value=_anthropic_stream_client([], final)):
        streamed = anthropic.call_model(
            parse_spec("anthropic"), system="s", messages=MESSAGES, sink=Recorder()
        )

    assert streamed == unstreamed


def test_omitting_the_sink_does_not_open_a_stream():
    """Existing callers must keep issuing exactly the request they issue today."""
    client = MagicMock()
    client.messages.create.return_value = _anthropic_final()

    with patch("anthropic.Anthropic", return_value=client):
        anthropic.call_model(parse_spec("anthropic"), system="s", messages=MESSAGES)

    client.messages.stream.assert_not_called()
    client.messages.create.assert_called_once()


# --- OpenAI-compatible -----------------------------------------------------


def _openai_final(*, reasoning=None):
    message = SimpleNamespace(content="six", tool_calls=None)
    if reasoning is not None:
        message.reasoning_content = reasoning
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        model="deepseek-v4-pro",
        usage=SimpleNamespace(prompt_tokens=4, completion_tokens=3, prompt_tokens_details=None),
    )


def _openai_stream_client(events, final):
    stream = MagicMock()
    stream.__iter__ = lambda self: iter(events)
    stream.get_final_completion.return_value = final
    manager = MagicMock()
    manager.__enter__ = lambda self: stream
    manager.__exit__ = lambda self, *exc: False
    client = MagicMock()
    client.chat.completions.stream.return_value = manager
    return client


def _chunk(*, content=None, reasoning=None):
    delta = SimpleNamespace(content=content, reasoning_content=reasoning)
    return SimpleNamespace(
        type="chunk", chunk=SimpleNamespace(choices=[SimpleNamespace(delta=delta)])
    )


def test_openai_streams_reasoning_content_and_keeps_it_on_the_result():
    """The SDK accumulator has no schema for this vendor field, so the wire
    accumulates it itself — losing it here would 400 the *next* turn."""
    events = [
        _chunk(reasoning="count"),
        _chunk(reasoning="ing"),
        _chunk(content="six"),
    ]
    sink = Recorder()

    # The final completion deliberately omits `reasoning_content`, standing in
    # for an accumulator that dropped the unknown field.
    with patch("openai.OpenAI", return_value=_openai_stream_client(events, _openai_final())):
        result = openai_compatible.call_model(
            parse_spec("deepseek"), system="s", messages=MESSAGES, sink=sink
        )

    assert sink.thinking == ["count", "ing"]
    assert sink.text == ["six"]
    assert result.thinking == "counting"
    assert result.reasoning.blocks == ({"reasoning_content": "counting"},)


def test_openai_streamed_tool_calls_come_back_whole():
    """Argument fragments are meaningless mid-stream, so they are not sunk —
    the accumulated completion is what becomes the decision."""
    call = SimpleNamespace(
        id="one", function=SimpleNamespace(name="lookup", arguments='{"term": "a"}')
    )
    final = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=None, tool_calls=[call]),
                finish_reason="tool_calls",
            )
        ],
        model="deepseek-v4-pro",
        usage=SimpleNamespace(prompt_tokens=4, completion_tokens=3, prompt_tokens_details=None),
    )

    with patch("openai.OpenAI", return_value=_openai_stream_client([], final)):
        result = openai_compatible.call_model(
            parse_spec("deepseek"), system="s", messages=MESSAGES, sink=Recorder()
        )

    assert isinstance(result.decision, ToolCalls)
    assert result.decision.calls[0].arguments == {"term": "a"}


# --- Gemini ----------------------------------------------------------------


def _gemini_part(**kwargs):
    base = {"text": None, "thought": None, "thought_signature": None, "function_call": None}
    return SimpleNamespace(**{**base, **kwargs})


def _gemini_chunk(*parts, finish="STOP"):
    return SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=list(parts), role="model"), finish_reason=finish
            )
        ],
        usage_metadata=SimpleNamespace(
            prompt_token_count=5, candidates_token_count=2, cached_content_token_count=1
        ),
        model_version="gemini-3.6-flash",
    )


def test_gemini_merges_chunks_without_losing_thought_signatures():
    """Gemini's SDK accumulates nothing, so this wire merges chunks itself —
    the one place a streamed turn could disagree with an unstreamed one."""
    chunks = [
        _gemini_chunk(_gemini_part(text="count", thought=True, thought_signature=b"sig")),
        _gemini_chunk(_gemini_part(text="ing", thought=True)),
        _gemini_chunk(_gemini_part(text="six")),
    ]
    client = MagicMock()
    client.models.generate_content_stream.return_value = iter(chunks)
    sink = Recorder()

    with patch("google.genai.Client", return_value=client):
        result = gemini.call_model(parse_spec("gemini"), system="s", messages=MESSAGES, sink=sink)

    assert sink.thinking == ["count", "ing"]
    assert sink.text == ["six"]
    assert result.decision.text == "six"
    assert result.thinking == "counting"
    assert result.reasoning.blocks[0]["thought_signature"] == "c2ln"


def test_gemini_streamed_result_matches_the_unstreamed_one():
    parts = (
        _gemini_part(text="counting", thought=True, thought_signature=b"sig"),
        _gemini_part(text="six"),
    )
    client = MagicMock()
    client.models.generate_content.return_value = _gemini_chunk(*parts)
    client.models.generate_content_stream.return_value = iter([_gemini_chunk(*parts)])

    with patch("google.genai.Client", return_value=client):
        unstreamed = gemini.call_model(parse_spec("gemini"), system="s", messages=MESSAGES)
        streamed = gemini.call_model(
            parse_spec("gemini"), system="s", messages=MESSAGES, sink=Recorder()
        )

    assert streamed == unstreamed


# --- Shared contract -------------------------------------------------------


def test_a_sink_that_raises_is_not_swallowed():
    """A frozen UI with no error anywhere is worse than a loud failure the
    caller can fix. The caller owns the sink (ADR-0022)."""

    class Broken:
        def on_text(self, delta: str) -> None:
            raise RuntimeError("sink is broken")

        def on_thinking(self, delta: str) -> None:
            pass

    events = [
        SimpleNamespace(
            type="content_block_delta", delta=SimpleNamespace(type="text_delta", text="six")
        )
    ]

    with (
        patch(
            "anthropic.Anthropic",
            return_value=_anthropic_stream_client(events, _anthropic_final()),
        ),
        pytest.raises(RuntimeError, match="sink is broken"),
    ):
        anthropic.call_model(parse_spec("anthropic"), system="s", messages=MESSAGES, sink=Broken())


def test_call_model_passes_the_sink_through_the_neutral_boundary(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    from emissary.llm import call_model

    sink = Recorder()
    with patch("emissary.llm.wire.anthropic.call_model") as wire:
        call_model(
            parse_spec("anthropic"),
            system="s",
            messages=MESSAGES,
            settings=ModelSettings(),
            sink=sink,
        )

    assert wire.call_args.kwargs["sink"] is sink
