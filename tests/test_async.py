"""The async boundary is a shell over the same translation as the sync one.

The property that matters is equality: given the same recorded response, the
sync and async paths must produce the same `ModelResult`. They share
`_request` and `_normalize` precisely so they cannot drift, and these tests are
what would catch it if someone re-implemented either.
"""

import json
import math
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from emissary import ProviderError, parse_spec
from emissary.llm import retry
from emissary.llm.decision import FinalOutput, ToolCalls, ToolDefinition
from emissary.llm.messages import TextBlock, UserMessage
from emissary.llm.model import AsyncFallbackModelCaller, AsyncSpecModelCaller, acall_model
from emissary.llm.streaming import AsyncStreamSink
from emissary.llm.wire import anthropic, gemini, openai_compatible

MESSAGES = (UserMessage((TextBlock("count the letters"),)),)
TOOLS = (
    ToolDefinition(
        name="lookup",
        description="Look up a term.",
        input_schema={"type": "object", "properties": {"term": {"type": "string"}}},
    ),
)


class AsyncRecorder:
    def __init__(self) -> None:
        self.text: list[str] = []
        self.thinking: list[str] = []

    async def on_text(self, delta: str) -> None:
        self.text.append(delta)

    async def on_thinking(self, delta: str) -> None:
        self.thinking.append(delta)


async def _aiter(items):
    for item in items:
        yield item


def test_async_recorder_satisfies_the_async_protocol():
    assert isinstance(AsyncRecorder(), AsyncStreamSink)


# --- Anthropic -------------------------------------------------------------


def _anthropic_response():
    return SimpleNamespace(
        stop_reason="end_turn",
        content=[
            SimpleNamespace(type="thinking", thinking="counting", signature="sig-1"),
            SimpleNamespace(type="text", text="six"),
        ],
        model="claude",
        usage=SimpleNamespace(input_tokens=5, output_tokens=2, cache_read_input_tokens=0),
    )


async def test_anthropic_async_and_sync_agree_on_the_same_response():
    response = _anthropic_response()

    sync_client = MagicMock()
    sync_client.messages.create.return_value = response
    async_client = MagicMock()
    async_client.messages.create = AsyncMock(return_value=response)

    with patch("anthropic.Anthropic", return_value=sync_client):
        expected = anthropic.call_model(parse_spec("anthropic"), system="s", messages=MESSAGES)
    with patch("anthropic.AsyncAnthropic", return_value=async_client):
        actual = await anthropic.acall_model(parse_spec("anthropic"), system="s", messages=MESSAGES)

    assert actual == expected
    # Including the part that a re-implementation would most easily lose.
    assert actual.reasoning.blocks[0]["signature"] == "sig-1"


async def test_anthropic_async_sends_the_same_request_as_sync():
    """Divergence in request building is as damaging as divergence in parsing,
    and silent — the model simply answers a different question."""
    response = _anthropic_response()
    sync_client = MagicMock()
    sync_client.messages.create.return_value = response
    async_client = MagicMock()
    async_client.messages.create = AsyncMock(return_value=response)

    with patch("anthropic.Anthropic", return_value=sync_client):
        anthropic.call_model(parse_spec("anthropic"), system="s", messages=MESSAGES, tools=TOOLS)
    with patch("anthropic.AsyncAnthropic", return_value=async_client):
        await anthropic.acall_model(
            parse_spec("anthropic"), system="s", messages=MESSAGES, tools=TOOLS
        )

    assert (
        async_client.messages.create.await_args.kwargs
        == sync_client.messages.create.call_args.kwargs
    )


async def test_anthropic_async_streaming_awaits_the_sink():
    events = [
        SimpleNamespace(
            type="content_block_delta",
            delta=SimpleNamespace(type="thinking_delta", thinking="count"),
        ),
        SimpleNamespace(
            type="content_block_delta", delta=SimpleNamespace(type="text_delta", text="six")
        ),
    ]

    stream = MagicMock()
    stream.__aiter__ = lambda self: _aiter(events)
    stream.get_final_message = AsyncMock(return_value=_anthropic_response())
    manager = MagicMock()
    manager.__aenter__ = AsyncMock(return_value=stream)
    manager.__aexit__ = AsyncMock(return_value=False)
    client = MagicMock()
    client.messages.stream.return_value = manager

    sink = AsyncRecorder()
    with patch("anthropic.AsyncAnthropic", return_value=client):
        result = await anthropic.acall_model(
            parse_spec("anthropic"), system="s", messages=MESSAGES, sink=sink
        )

    assert sink.thinking == ["count"]
    assert sink.text == ["six"]
    assert result.decision.text == "six"


# --- OpenAI-compatible -----------------------------------------------------


def _openai_response(*, reasoning=None):
    message = SimpleNamespace(content="six", tool_calls=None)
    if reasoning is not None:
        message.reasoning_content = reasoning
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        model="deepseek-v4-pro",
        usage=SimpleNamespace(prompt_tokens=4, completion_tokens=3, prompt_tokens_details=None),
    )


async def test_openai_async_and_sync_agree_on_the_same_response():
    response = _openai_response(reasoning="counting")

    sync_client = MagicMock()
    sync_client.chat.completions.create.return_value = response
    async_client = MagicMock()
    async_client.chat.completions.create = AsyncMock(return_value=response)

    with patch("openai.OpenAI", return_value=sync_client):
        expected = openai_compatible.call_model(
            parse_spec("deepseek"), system="s", messages=MESSAGES
        )
    with patch("openai.AsyncOpenAI", return_value=async_client):
        actual = await openai_compatible.acall_model(
            parse_spec("deepseek"), system="s", messages=MESSAGES
        )

    assert actual == expected
    assert actual.reasoning.blocks == ({"reasoning_content": "counting"},)


async def test_openai_async_normalizes_tool_calls():
    call = SimpleNamespace(
        id="one", function=SimpleNamespace(name="lookup", arguments=json.dumps({"term": "a"}))
    )
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=None, tool_calls=[call]),
                finish_reason="tool_calls",
            )
        ],
        model="deepseek-v4-pro",
        usage=SimpleNamespace(prompt_tokens=4, completion_tokens=3, prompt_tokens_details=None),
    )
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)

    with patch("openai.AsyncOpenAI", return_value=client):
        result = await openai_compatible.acall_model(
            parse_spec("deepseek"), system="s", messages=MESSAGES, tools=TOOLS
        )

    assert isinstance(result.decision, ToolCalls)
    assert result.decision.calls[0].arguments == {"term": "a"}


async def test_openai_async_errors_stay_classified_for_fallback():
    """The fallback policy reads `retryable`; losing that on the async path
    would turn a recoverable outage into a hard failure (ADR-0002)."""
    import openai

    client = MagicMock()
    client.chat.completions.create = AsyncMock(
        side_effect=openai.APIConnectionError(request=MagicMock())
    )

    with (
        patch("openai.AsyncOpenAI", return_value=client),
        pytest.raises(ProviderError) as caught,
    ):
        await openai_compatible.acall_model(parse_spec("deepseek"), system="s", messages=MESSAGES)

    assert caught.value.retryable is True


# --- Gemini ----------------------------------------------------------------


def _gemini_part(**kwargs):
    base = {"text": None, "thought": None, "thought_signature": None, "function_call": None}
    return SimpleNamespace(**{**base, **kwargs})


def _gemini_response(*parts):
    return SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=list(parts), role="model"), finish_reason="STOP"
            )
        ],
        usage_metadata=SimpleNamespace(
            prompt_token_count=5, candidates_token_count=2, cached_content_token_count=1
        ),
        model_version="gemini-3.6-flash",
    )


async def test_gemini_async_and_sync_agree_on_the_same_response():
    parts = (
        _gemini_part(text="counting", thought=True, thought_signature=b"sig"),
        _gemini_part(text="six"),
    )
    response = _gemini_response(*parts)

    client = MagicMock()
    client.models.generate_content.return_value = response
    client.aio.models.generate_content = AsyncMock(return_value=response)

    with patch("google.genai.Client", return_value=client):
        expected = gemini.call_model(parse_spec("gemini"), system="s", messages=MESSAGES)
        actual = await gemini.acall_model(parse_spec("gemini"), system="s", messages=MESSAGES)

    assert actual == expected
    assert actual.reasoning.blocks[0]["thought_signature"] == "c2ln"


async def test_gemini_async_streaming_merges_chunks_and_awaits_the_sink():
    """`generate_content_stream` on the `.aio` namespace is a coroutine that
    returns an async iterator, so it must be awaited *before* iterating —
    a placement no mock-free reading of the code would catch."""
    chunks = [
        _gemini_response(_gemini_part(text="count", thought=True, thought_signature=b"sig")),
        _gemini_response(_gemini_part(text="ing", thought=True)),
        _gemini_response(_gemini_part(text="six")),
    ]
    client = MagicMock()
    client.aio.models.generate_content_stream = AsyncMock(return_value=_aiter(chunks))

    sink = AsyncRecorder()
    with patch("google.genai.Client", return_value=client):
        result = await gemini.acall_model(
            parse_spec("gemini"), system="s", messages=MESSAGES, sink=sink
        )

    assert sink.thinking == ["count", "ing"]
    assert sink.text == ["six"]
    assert result.decision.text == "six"
    assert result.reasoning.blocks[0]["thought_signature"] == "c2ln"


async def test_openai_async_streaming_accumulates_reasoning_and_awaits_the_sink():
    events = [
        SimpleNamespace(
            type="chunk",
            chunk=SimpleNamespace(
                choices=[
                    SimpleNamespace(delta=SimpleNamespace(content=None, reasoning_content="count"))
                ]
            ),
        ),
        SimpleNamespace(
            type="chunk",
            chunk=SimpleNamespace(
                choices=[
                    SimpleNamespace(delta=SimpleNamespace(content="six", reasoning_content=None))
                ]
            ),
        ),
    ]
    stream = MagicMock()
    stream.__aiter__ = lambda self: _aiter(events)
    # Stands in for an accumulator that dropped the vendor field.
    stream.get_final_completion = AsyncMock(return_value=_openai_response())
    manager = MagicMock()
    manager.__aenter__ = AsyncMock(return_value=stream)
    manager.__aexit__ = AsyncMock(return_value=False)
    client = MagicMock()
    client.chat.completions.stream.return_value = manager

    sink = AsyncRecorder()
    with patch("openai.AsyncOpenAI", return_value=client):
        result = await openai_compatible.acall_model(
            parse_spec("deepseek"), system="s", messages=MESSAGES, sink=sink
        )

    assert sink.thinking == ["count"]
    assert sink.text == ["six"]
    assert result.reasoning.blocks == ({"reasoning_content": "count"},)


# --- Neutral boundary ------------------------------------------------------


async def test_acall_model_gates_credentials_and_capabilities_before_dispatch(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with (
        patch("emissary.llm.wire.anthropic.acall_model") as wire,
        pytest.raises(ProviderError, match="ANTHROPIC_API_KEY"),
    ):
        await acall_model(parse_spec("anthropic"), system="s", messages=MESSAGES)

    wire.assert_not_called()


async def test_async_spec_caller_satisfies_the_async_caller_protocol(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    caller = AsyncSpecModelCaller(parse_spec("anthropic"))
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=_anthropic_response())

    with patch("anthropic.AsyncAnthropic", return_value=client):
        result = await caller(system="s", messages=MESSAGES)

    assert isinstance(result.decision, FinalOutput)


async def test_async_fallback_retries_only_availability_failures(monkeypatch):
    monkeypatch.setattr(retry, "RETRY_DELAYS", ())
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("MOONSHOT_API_KEY", "test")
    caller = AsyncFallbackModelCaller(parse_spec("anthropic"), parse_spec("kimi"))
    result = SimpleNamespace()

    with patch(
        "emissary.llm.model.acall_model",
        new=AsyncMock(side_effect=[ProviderError("overloaded", retryable=True), result]),
    ) as called:
        assert await caller(system="s", messages=MESSAGES) is result
    assert called.await_count == 2

    with (
        patch(
            "emissary.llm.model.acall_model",
            new=AsyncMock(side_effect=ProviderError("malformed", retryable=False)),
        ) as called,
        pytest.raises(ProviderError, match="malformed"),
    ):
        await caller(system="s", messages=MESSAGES)
    assert called.await_count == 1


async def test_async_streaming_failure_after_a_delta_is_not_retried(monkeypatch, caplog):
    monkeypatch.setattr(retry, "RETRY_DELAYS", ())
    caller = AsyncFallbackModelCaller(parse_spec("anthropic"), parse_spec("kimi"))
    sink = AsyncRecorder()

    async def partial_failure(*args, **kwargs):
        await kwargs["sink"].on_text("partial")
        raise ProviderError("stream disconnected", retryable=True)

    with (
        patch(
            "emissary.llm.model.acall_model", new=AsyncMock(side_effect=partial_failure)
        ) as called,
        caplog.at_level("WARNING", logger="emissary.llm.model"),
        pytest.raises(ProviderError, match="after streaming output"),
    ):
        await caller(system="s", messages=MESSAGES, sink=sink)

    assert called.await_count == 1
    assert sink.text == ["partial"]
    assert "retry suppressed" in caplog.text


async def test_acall_tool_returns_the_same_payload_as_the_sync_call(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    from emissary.llm.calls import acall_tool, call_tool

    tool = {"name": "extract", "description": "Extract.", "input_schema": {"type": "object"}}
    response = SimpleNamespace(
        stop_reason="tool_use",
        content=[SimpleNamespace(type="tool_use", name="extract", input={"value": 1})],
        model="claude",
        usage=SimpleNamespace(input_tokens=3, output_tokens=1, cache_read_input_tokens=0),
    )
    sync_client = MagicMock()
    sync_client.messages.create.return_value = response
    async_client = MagicMock()
    async_client.messages.create = AsyncMock(return_value=response)

    with patch("anthropic.Anthropic", return_value=sync_client):
        expected = call_tool(parse_spec("anthropic"), tool=tool, system="s", blocks=())
    with patch("anthropic.AsyncAnthropic", return_value=async_client):
        actual = await acall_tool(parse_spec("anthropic"), tool=tool, system="s", blocks=())

    assert actual == expected


async def test_acall_choice_scores_from_logprobs_like_the_sync_call():
    from emissary.llm.calls import acall_choice, call_choice

    position = SimpleNamespace(
        token="SAFE",
        logprob=math.log(0.9),
        top_logprobs=[
            SimpleNamespace(token="SAFE", logprob=math.log(0.9)),
            SimpleNamespace(token="FLAG", logprob=math.log(0.1)),
        ],
    )
    response = SimpleNamespace(
        choices=[SimpleNamespace(logprobs=SimpleNamespace(content=[position]))],
        model="qwen",
        usage=SimpleNamespace(prompt_tokens=2, completion_tokens=1),
    )
    sync_client = MagicMock()
    sync_client.chat.completions.create.return_value = response
    async_client = MagicMock()
    async_client.chat.completions.create = AsyncMock(return_value=response)

    with patch("openai.OpenAI", return_value=sync_client):
        expected = call_choice(parse_spec("vllm:qwen"), labels=["SAFE", "FLAG"], system="s")
    with patch("openai.AsyncOpenAI", return_value=async_client):
        actual = await acall_choice(parse_spec("vllm:qwen"), labels=["SAFE", "FLAG"], system="s")

    assert actual == expected
    assert actual.probabilities["SAFE"] == pytest.approx(0.9)
