"""Provider-neutral conversation values translated only by wire adapters."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .decision import ReasoningState, ToolCall


@dataclass(frozen=True)
class TextBlock:
    text: str
    cache: bool = False

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("text must not be empty")


@dataclass(frozen=True)
class UserMessage:
    content: tuple[TextBlock, ...]

    def __post_init__(self) -> None:
        if not self.content:
            raise ValueError("user message content must not be empty")


@dataclass(frozen=True)
class AssistantMessage:
    """A turn the model produced, as it must be replayed.

    `reasoning` is opaque provider state (ADR-0018): the wire that issued it
    sends it back unchanged, every other wire ignores it. Nothing here reads
    it, and nothing may edit it — a modified signature is rejected by the API
    as surely as a missing one.
    """

    text: str | None = None
    tool_calls: tuple["ToolCall", ...] = ()
    reasoning: "ReasoningState | None" = None

    def __post_init__(self) -> None:
        if not self.text and not self.tool_calls:
            raise ValueError("assistant message needs text or tool calls")


@dataclass(frozen=True)
class ToolMessage:
    call_id: str
    tool_name: str
    content: str

    def __post_init__(self) -> None:
        if not self.call_id:
            raise ValueError("call_id must not be empty")
        if not self.tool_name:
            raise ValueError("tool_name must not be empty")


Message = UserMessage | AssistantMessage | ToolMessage

__all__ = [
    "AssistantMessage",
    "Message",
    "TextBlock",
    "ToolMessage",
    "UserMessage",
]
