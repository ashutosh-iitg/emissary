"""Model-visible messages, derived from the event log (ADR-0011).

The log is the single source of conversation state. This module is the only
place that turns events into messages, and the only place that turns run values
into the JSON-native payloads events carry. Everything here is pure: replay
(ADR-0015) depends on it.
"""

import json
from collections.abc import Iterable
from dataclasses import asdict
from typing import Any

from ..llm.decision import (
    FinalOutput,
    ModelDecision,
    ModelResult,
    Refusal,
    ToolCall,
    ToolCalls,
    Usage,
)
from ..llm.messages import AssistantMessage, Message, TextBlock, ToolMessage, UserMessage
from .context import ContextOp
from .events import RunEvent
from .tools import ToolResult


def _call_data(call: ToolCall) -> dict[str, Any]:
    return {"id": call.id, "name": call.name, "arguments": call.arguments}


def _call_from_data(data: dict[str, Any]) -> ToolCall:
    return ToolCall(data["id"], data["name"], data["arguments"])


def _decision_data(decision: ModelDecision) -> dict[str, Any]:
    # `kind` is a stable wire token, not the class name: renaming a Python class
    # must not invalidate recorded fixtures.
    if isinstance(decision, ToolCalls):
        return {
            "kind": "tool_calls",
            "calls": [_call_data(call) for call in decision.calls],
            "text": decision.text,
        }
    if isinstance(decision, FinalOutput):
        return {"kind": "final_output", "text": decision.text, "value": decision.value}
    if isinstance(decision, Refusal):
        return {"kind": "refusal", "reason": decision.reason}
    raise TypeError(f"unknown model decision {type(decision).__name__}")


def user_message_data(message: UserMessage) -> dict[str, Any]:
    return {"content": [{"text": block.text, "cache": block.cache} for block in message.content]}


def model_result_data(result: ModelResult) -> dict[str, Any]:
    """Everything a replaying caller needs to rebuild this ``ModelResult``."""
    return {
        "decision": _decision_data(result.decision),
        "provider": result.provider,
        "model": result.model,
        "usage": {
            "input_tokens": result.usage.input_tokens,
            "output_tokens": result.usage.output_tokens,
            "cached_input_tokens": result.usage.cached_input_tokens,
        },
        "finish_reason": result.finish_reason,
    }


def tool_result_data(call: ToolCall, result: ToolResult) -> dict[str, Any]:
    """`status` is duplicated outside `result` so a redacting sink can drop the
    payload and still report severity."""
    # Normalising through JSON keeps an in-memory projection byte-identical to
    # one taken after a persistence round trip: a tuple in `content` renders the
    # same but is a different object once reloaded.
    payload = json.loads(json.dumps(asdict(result), sort_keys=True))
    return {"call_id": call.id, "tool": call.name, "status": result.status, "result": payload}


def context_op_data(op: ContextOp) -> dict[str, Any]:
    return {
        "start": op.start,
        "end": op.end,
        "reason": op.reason,
        "replacement": [message_to_data(message) for message in op.replacement],
    }


def decision_from_data(data: dict[str, Any]) -> ModelDecision:
    kind = data["kind"]
    if kind == "tool_calls":
        return ToolCalls(
            tuple(_call_from_data(call) for call in data["calls"]), text=data.get("text")
        )
    if kind == "final_output":
        return FinalOutput(data["text"], data["value"])
    if kind == "refusal":
        return Refusal(data["reason"])
    raise ValueError(f"unknown recorded decision kind {kind!r}")


def model_result_from_data(data: dict[str, Any]) -> ModelResult:
    """Rebuild a recorded turn — the inverse of ``model_result_data``."""
    usage = data["usage"]
    return ModelResult(
        decision=decision_from_data(data["decision"]),
        provider=data["provider"],
        model=data["model"],
        usage=Usage(usage["input_tokens"], usage["output_tokens"], usage["cached_input_tokens"]),
        finish_reason=data["finish_reason"],
    )


def tool_result_from_data(data: dict[str, Any]) -> ToolResult:
    """Rebuild a recorded outcome. ``artifacts`` returns as a list from JSON."""
    return ToolResult(
        status=data["status"],
        summary=data["summary"],
        content=data["content"],
        artifacts=tuple(data["artifacts"]),
        timed_out=data["timed_out"],
        retryable=data["retryable"],
    )


def message_to_data(message: Message) -> dict[str, Any]:
    if isinstance(message, UserMessage):
        return {"type": "user", **user_message_data(message)}
    if isinstance(message, AssistantMessage):
        return {
            "type": "assistant",
            "text": message.text,
            "tool_calls": [_call_data(call) for call in message.tool_calls],
        }
    if isinstance(message, ToolMessage):
        return {
            "type": "tool",
            "call_id": message.call_id,
            "tool_name": message.tool_name,
            "content": message.content,
        }
    raise TypeError(f"unknown message type {type(message).__name__}")


def message_from_data(data: dict[str, Any]) -> Message:
    kind = data["type"]
    if kind == "user":
        return _user_message(data)
    if kind == "assistant":
        return AssistantMessage(
            data["text"], tuple(_call_from_data(call) for call in data["tool_calls"])
        )
    if kind == "tool":
        return ToolMessage(data["call_id"], data["tool_name"], data["content"])
    raise ValueError(f"unknown persisted message type {kind!r}")


def _user_message(data: dict[str, Any]) -> UserMessage:
    return UserMessage(tuple(TextBlock(b["text"], b["cache"]) for b in data["content"]))


def _assistant_message(data: dict[str, Any]) -> AssistantMessage | None:
    """A refusal and a value-only final output project nothing.

    A refusal reason is harness-authored prose, not something the model said; a
    structured value has no faithful rendering as assistant text. Both remain
    fully recorded in the log.
    """
    kind = data["kind"]
    if kind == "tool_calls":
        return AssistantMessage(
            text=data.get("text") or None,
            tool_calls=tuple(_call_from_data(c) for c in data["calls"]),
        )
    if kind == "final_output" and data["text"]:
        return AssistantMessage(text=data["text"])
    return None


def _tool_message(data: dict[str, Any]) -> ToolMessage:
    return ToolMessage(
        call_id=data["call_id"],
        tool_name=data["tool"],
        content=json.dumps(data["result"], sort_keys=True),
    )


def _reject_orphans(surface: list[Message]) -> None:
    issued = {
        call.id
        for message in surface
        if isinstance(message, AssistantMessage)
        for call in message.tool_calls
    }
    for message in surface:
        if isinstance(message, ToolMessage) and message.call_id not in issued:
            raise ValueError(f"context op orphaned tool result {message.call_id!r} from its call")


def _apply_op(surface: list[Message], data: dict[str, Any]) -> list[Message]:
    start, end = data["start"], data["end"]
    if not 0 <= start <= end <= len(surface):
        raise ValueError(
            f"context op range [{start}, {end}) does not fit a surface of {len(surface)}"
        )
    replacement = [message_from_data(message) for message in data["replacement"]]
    applied = surface[:start] + replacement + surface[end:]
    _reject_orphans(applied)
    return applied


def derive_messages(events: Iterable[RunEvent], *, apply_ops: bool = True) -> tuple[Message, ...]:
    """Fold the log into the model-visible surface.

    Events are consumed in order; this deliberately does not sort by sequence,
    because an out-of-order log is a defect to surface rather than paper over.
    ``apply_ops=False`` yields the pre-compaction history, for audit only.
    """
    surface: list[Message] = []
    for event in events:
        if event.kind == "user_message":
            surface.append(_user_message(event.data))
        elif event.kind == "model_call_completed":
            message = _assistant_message(event.data["decision"])
            if message is not None:
                surface.append(message)
        elif event.kind == "tool_call_completed":
            surface.append(_tool_message(event.data))
        elif event.kind == "context_compacted" and apply_ops:
            surface = _apply_op(surface, event.data)
    return tuple(surface)


__all__ = [
    "context_op_data",
    "decision_from_data",
    "derive_messages",
    "message_from_data",
    "message_to_data",
    "model_result_data",
    "model_result_from_data",
    "tool_result_data",
    "tool_result_from_data",
    "user_message_data",
]
