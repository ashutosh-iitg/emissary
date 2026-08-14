"""Selection of provider-neutral conversation state for each model turn."""

from typing import Protocol

from .messages import AssistantMessage, Message, ToolMessage


class ContextPolicy(Protocol):
    def select(self, messages: tuple[Message, ...]) -> tuple[Message, ...]: ...


class CompleteHistory:
    def select(self, messages: tuple[Message, ...]) -> tuple[Message, ...]:
        return messages


class RecentHistory:
    def __init__(self, max_messages: int):
        if max_messages <= 0:
            raise ValueError("max_messages must be positive")
        self.max_messages = max_messages

    def select(self, messages: tuple[Message, ...]) -> tuple[Message, ...]:
        if len(messages) <= self.max_messages:
            return messages
        if self.max_messages == 1:
            return messages[:1]
        start = len(messages) - (self.max_messages - 1)
        if isinstance(messages[start], ToolMessage):
            call_id = messages[start].call_id
            while start > 1:
                start -= 1
                candidate = messages[start]
                if isinstance(candidate, AssistantMessage) and any(
                    call.id == call_id for call in candidate.tool_calls
                ):
                    break
        return messages[:1] + messages[start:]


__all__ = ["CompleteHistory", "ContextPolicy", "RecentHistory"]
