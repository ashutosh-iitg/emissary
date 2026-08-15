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
) -> ModelResult:
    """Execute one provider-neutral conversational model turn."""
    import anthropic

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

    try:
        response = anthropic.Anthropic().messages.create(**kwargs)
    except anthropic.APIStatusError as exc:
        raise ProviderError(
            f"{spec}: {exc.status_code} {exc.message}",
            retryable=retryable_status(exc.status_code),
        ) from exc
    except anthropic.APIConnectionError as exc:
        raise ProviderError(f"{spec}: could not reach the API ({exc})", retryable=True) from exc

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

    content = [
        {
            "type": "text",
            "text": block.text,
            **({"cache_control": {"type": "ephemeral"}} if block.cache else {}),
        }
        for block in blocks
    ]
    try:
        response = anthropic.Anthropic().messages.create(
            model=spec.model,
            max_tokens=MAX_TOKENS,
            thinking={"type": "adaptive"},
            **({"output_config": {"effort": effort}} if effort else {}),
            system=system,
            tools=[tool],
            tool_choice={"type": "tool", "name": tool["name"]},
            messages=[{"role": "user", "content": content}],
        )
    except anthropic.APIStatusError as exc:
        raise ProviderError(
            f"{spec}: {exc.status_code} {exc.message}",
            retryable=retryable_status(exc.status_code),
        ) from exc
    except anthropic.APIConnectionError as exc:
        raise ProviderError(f"{spec}: could not reach the API ({exc})", retryable=True) from exc

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
