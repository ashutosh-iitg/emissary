"""The native Anthropic Messages API adapter."""

from typing import Any

from ..decision import (
    FinalOutput,
    ModelResult,
    ModelSettings,
    ReasoningState,
    Refusal,
    ToolCall,
    ToolCalls,
    ToolDefinition,
    Usage,
)
from ..errors import ProviderError, retryable_status
from ..messages import AssistantMessage, Message, TextBlock, ToolMessage, UserMessage
from ..provider import MAX_TOKENS, Spec
from ..result import CallResult
from ..streaming import AsyncStreamSink, StreamSink
from .thinking import thinking_kwargs

WIRE = "anthropic"
"""Tags reasoning state this wire issues, so no other wire replays it."""

THINKING_TYPES = ("thinking", "redacted_thinking")
"""Block types that carry signed reasoning and must be echoed back unchanged."""


def _anthropic_messages(messages: tuple[Message, ...]) -> list[dict[str, Any]]:
    translated: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, UserMessage):
            translated.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": block.text,
                            **({"cache_control": {"type": "ephemeral"}} if block.cache else {}),
                        }
                        for block in message.content
                    ],
                }
            )
        elif isinstance(message, AssistantMessage):
            content: list[dict[str, Any]] = []
            # Thinking blocks lead the turn and carry a signature the API
            # verifies. Dropping or reordering them provokes a 400 as surely as
            # editing one would, so they go back first and untouched.
            if message.reasoning is not None and message.reasoning.wire == WIRE:
                content.extend(dict(block) for block in message.reasoning.blocks)
            if message.text:
                content.append({"type": "text", "text": message.text})
            content.extend(
                {
                    "type": "tool_use",
                    "id": call.id,
                    "name": call.name,
                    "input": call.arguments,
                }
                for call in message.tool_calls
            )
            translated.append({"role": "assistant", "content": content})
        elif isinstance(message, ToolMessage):
            translated.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": message.call_id,
                            "content": message.content,
                        }
                    ],
                }
            )
    return translated


def call_model(
    spec: Spec,
    *,
    system: str,
    messages: tuple[Message, ...],
    tools: tuple[ToolDefinition, ...] = (),
    settings: ModelSettings | None = None,
    sink: StreamSink | None = None,
) -> ModelResult:
    """Execute one provider-neutral conversational model turn.

    With a `sink`, deltas are reported as they arrive and the SDK's accumulated
    final message is normalized by the same code path as an unstreamed call —
    so the two cannot disagree about what the model said (ADR-0022).
    """
    import anthropic

    kwargs = _request(spec, system, messages, tools, settings)
    try:
        response = _stream(anthropic, kwargs, sink) if sink else _create(anthropic, kwargs)
    except anthropic.APIStatusError as exc:
        raise _status_error(spec, exc) from exc
    except anthropic.APIConnectionError as exc:
        raise ProviderError(f"{spec}: could not reach the API ({exc})", retryable=True) from exc

    return _normalize(spec, response)


async def acall_model(
    spec: Spec,
    *,
    system: str,
    messages: tuple[Message, ...],
    tools: tuple[ToolDefinition, ...] = (),
    settings: ModelSettings | None = None,
    sink: AsyncStreamSink | None = None,
) -> ModelResult:
    """`call_model` on the async client — same request, same normalisation.

    Holds no logic of its own on purpose: everything that could disagree with
    the sync path lives in `_request` and `_normalize`, which both call.
    """
    import anthropic

    kwargs = _request(spec, system, messages, tools, settings)
    try:
        response = (
            await _astream(anthropic, kwargs, sink) if sink else await _acreate(anthropic, kwargs)
        )
    except anthropic.APIStatusError as exc:
        raise _status_error(spec, exc) from exc
    except anthropic.APIConnectionError as exc:
        raise ProviderError(f"{spec}: could not reach the API ({exc})", retryable=True) from exc

    return _normalize(spec, response)


def _status_error(spec: Spec, exc) -> ProviderError:
    return ProviderError(
        f"{spec}: {exc.status_code} {exc.message}", retryable=retryable_status(exc.status_code)
    )


def _request(
    spec: Spec,
    system: str,
    messages: tuple[Message, ...],
    tools: tuple[ToolDefinition, ...],
    settings: ModelSettings | None,
) -> dict[str, Any]:
    """Everything the API needs, built once for every shell that sends it."""
    configured = settings or ModelSettings()
    kwargs: dict[str, Any] = {
        "model": spec.model,
        "max_tokens": configured.max_output_tokens or MAX_TOKENS,
        "system": system,
        "messages": _anthropic_messages(messages),
    }
    if tools and configured.tool_choice != "none":
        kwargs["tools"] = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            for tool in tools
        ]
        if configured.tool_choice == "required":
            kwargs["tool_choice"] = {"type": "any"}
    kwargs.update(thinking_kwargs(spec, configured.thinking))
    return kwargs


def _create(anthropic, kwargs: dict[str, Any]):
    return anthropic.Anthropic().messages.create(**kwargs)


async def _acreate(anthropic, kwargs: dict[str, Any]):
    return await anthropic.AsyncAnthropic().messages.create(**kwargs)


async def _astream(anthropic, kwargs: dict[str, Any], sink: AsyncStreamSink):
    async with anthropic.AsyncAnthropic().messages.stream(**kwargs) as stream:
        async for event in stream:
            if event.type != "content_block_delta":
                continue
            if event.delta.type == "text_delta":
                await sink.on_text(event.delta.text)
            elif event.delta.type == "thinking_delta":
                await sink.on_thinking(event.delta.thinking)
        return await stream.get_final_message()


def _stream(anthropic, kwargs: dict[str, Any], sink: StreamSink):
    """Drain the stream for observation, then hand back the whole message.

    `get_final_message()` reassembles thinking blocks with their signatures
    intact, which hand-accumulating the deltas would not — the deltas carry the
    text but not the signature.
    """
    with anthropic.Anthropic().messages.stream(**kwargs) as stream:
        for event in stream:
            if event.type != "content_block_delta":
                continue
            if event.delta.type == "text_delta":
                sink.on_text(event.delta.text)
            elif event.delta.type == "thinking_delta":
                sink.on_thinking(event.delta.thinking)
        return stream.get_final_message()


def _normalize(spec: Spec, response) -> ModelResult:
    if response.stop_reason == "refusal":
        decision = Refusal("the model declined this request")
    else:
        calls = tuple(
            ToolCall(part.id, part.name, dict(part.input))
            for part in response.content
            if part.type == "tool_use"
        )
        text = "".join(part.text for part in response.content if part.type == "text")
        if calls:
            decision = ToolCalls(calls, text=text or None)
        elif text:
            decision = FinalOutput(text=text)
        else:
            raise ProviderError(f"{spec}: response contained no usable decision")

    thinking_blocks = tuple(
        _thinking_block(part) for part in response.content if part.type in THINKING_TYPES
    )
    usage = response.usage
    return ModelResult(
        decision=decision,
        provider=spec.name,
        model=response.model,
        usage=Usage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cached_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        ),
        finish_reason=response.stop_reason,
        thinking="\n".join(block["thinking"] for block in thinking_blocks if block.get("thinking"))
        or None,
        reasoning=(ReasoningState(wire=WIRE, blocks=thinking_blocks) if thinking_blocks else None),
    )


def _thinking_block(part) -> dict[str, Any]:
    """A thinking block as JSON, keeping exactly the fields the API verifies.

    `redacted_thinking` has no readable text at all — only `data` — which is
    why the round-trip cannot be reconstructed from `ModelResult.thinking`.
    """
    if part.type == "redacted_thinking":
        return {"type": "redacted_thinking", "data": part.data}
    return {"type": "thinking", "thinking": part.thinking, "signature": part.signature}


def call_tool(
    spec: Spec,
    *,
    system: str,
    blocks: tuple[TextBlock, ...],
    tool: dict[str, Any],
    effort: str | None = None,
) -> CallResult:
    """One structured call that must answer by invoking `tool`."""
    import anthropic

    try:
        response = anthropic.Anthropic().messages.create(
            **_tool_request(spec, system, blocks, tool, effort)
        )
    except anthropic.APIStatusError as exc:
        raise _status_error(spec, exc) from exc
    except anthropic.APIConnectionError as exc:
        raise ProviderError(f"{spec}: could not reach the API ({exc})", retryable=True) from exc

    return _normalize_tool(spec, response, tool)


async def acall_tool(
    spec: Spec,
    *,
    system: str,
    blocks: tuple[TextBlock, ...],
    tool: dict[str, Any],
    effort: str | None = None,
) -> CallResult:
    """`call_tool` on the async client — same request, same validation."""
    import anthropic

    try:
        response = await anthropic.AsyncAnthropic().messages.create(
            **_tool_request(spec, system, blocks, tool, effort)
        )
    except anthropic.APIStatusError as exc:
        raise _status_error(spec, exc) from exc
    except anthropic.APIConnectionError as exc:
        raise ProviderError(f"{spec}: could not reach the API ({exc})", retryable=True) from exc

    return _normalize_tool(spec, response, tool)


def _tool_request(
    spec: Spec,
    system: str,
    blocks: tuple[TextBlock, ...],
    tool: dict[str, Any],
    effort: str | None,
) -> dict[str, Any]:
    return {
        "model": spec.model,
        "max_tokens": MAX_TOKENS,
        "thinking": {"type": "adaptive"},
        **({"output_config": {"effort": effort}} if effort else {}),
        "system": system,
        "tools": [tool],
        "tool_choice": {"type": "tool", "name": tool["name"]},
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": block.text,
                        **({"cache_control": {"type": "ephemeral"}} if block.cache else {}),
                    }
                    for block in blocks
                ],
            }
        ],
    }


def _normalize_tool(spec: Spec, response, tool: dict[str, Any]) -> CallResult:
    # Checked before reading `content`: a refusal is a 200 with an empty or
    # partial body, and indexing it is the standard way this breaks.
    if response.stop_reason == "refusal":
        raise ProviderError(f"{spec}: the model declined this request", retryable=True)

    for part in response.content:
        if part.type == "tool_use" and part.name == tool["name"]:
            usage = response.usage
            return CallResult(
                payload=dict(part.input),
                provider=spec.name,
                model=response.model,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cached_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            )
    raise ProviderError(f"{spec}: no {tool['name']} call (stop_reason={response.stop_reason})")
