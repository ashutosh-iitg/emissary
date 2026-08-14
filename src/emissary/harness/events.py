"""Canonical, ordered facts emitted by an agent run."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol


@dataclass(frozen=True)
class RunEvent:
    run_id: str
    sequence: int
    kind: str
    data: dict[str, Any]
    occurred_at: datetime


class EventSink(Protocol):
    def emit(self, event: RunEvent) -> None: ...


class InMemoryEventSink:
    def __init__(self) -> None:
        self.events: list[RunEvent] = []

    def emit(self, event: RunEvent) -> None:
        self.events.append(event)


class NullEventSink:
    def emit(self, event: RunEvent) -> None:
        pass


def new_event(run_id: str, sequence: int, kind: str, **data: Any) -> RunEvent:
    return RunEvent(run_id, sequence, kind, data, datetime.now(UTC))


__all__ = ["EventSink", "InMemoryEventSink", "NullEventSink", "RunEvent"]
