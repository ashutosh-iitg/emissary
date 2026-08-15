"""Selection of provider-neutral conversation state for each model turn."""

from dataclasses import dataclass
from typing import Protocol

from ..llm.messages import AssistantMessage, Message, ToolMessage


@dataclass(frozen=True)
class ContextOp:
    """Replace ``surface[start:end]``; an empty replacement is a drop.

    Logged before it is applied, so the trajectory records every omission
    (ADR-0011). Indices address the projected surface the policy was handed,
    which is stable only because the log is append-only: folding the events
    before the op reproduces exactly that surface.
    """

    start: int
    end: int
    reason: str
    replacement: tuple[Message, ...] = ()

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError("context op range must be non-negative and ordered")
        if not self.reason:
            raise ValueError("context op reason must not be empty")


class ContextPolicy(Protocol):
    def plan(self, messages: tuple[Message, ...]) -> tuple[ContextOp, ...]: ...


class CompleteHistory:
    def plan(self, messages: tuple[Message, ...]) -> tuple[ContextOp, ...]:
        return ()


class RecentHistory:
    """Keep the task and a trailing window, never splitting a call from its result."""

    def __init__(self, max_messages: int):
        if max_messages <= 0:
            raise ValueError("max_messages must be positive")
        self.max_messages = max_messages

    def plan(self, messages: tuple[Message, ...]) -> tuple[ContextOp, ...]:
        if len(messages) <= self.max_messages:
            return ()
        keep_from = (
            len(messages)
            if self.max_messages == 1
            else _pair_safe_start(messages, len(messages) - (self.max_messages - 1))
        )
        if keep_from <= 1:
            return ()
        return (ContextOp(1, keep_from, f"recent_history:{self.max_messages}"),)


def _pair_safe_start(messages: tuple[Message, ...], start: int) -> int:
    """Walk back so a leading tool result never loses the call that issued it.

    The walk can push the kept window above ``max_messages``; that is the point,
    and it corrects itself once the pair scrolls out together.
    """
    if isinstance(messages[start], ToolMessage):
        call_id = messages[start].call_id
        while start > 1:
            start -= 1
            candidate = messages[start]
            if isinstance(candidate, AssistantMessage) and any(
                call.id == call_id for call in candidate.tool_calls
            ):
                break
    return start


__all__ = ["CompleteHistory", "ContextOp", "ContextPolicy", "RecentHistory"]
