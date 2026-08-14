"""Optional local storage for versioned run records, not durable execution."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from ..harness.events import RunEvent
from ..harness.state import RunResult, RunStatus, StopReason
from ..llm.decision import FinalOutput, ToolCall, Usage
from ..llm.messages import AssistantMessage, Message, TextBlock, ToolMessage, UserMessage

SCHEMA_VERSION = 1


class RunStore(Protocol):
    def save(self, result: RunResult) -> None: ...

    def load(self, run_id: str) -> RunResult | None: ...


def _message_to_data(message: Message) -> dict[str, Any]:
    if isinstance(message, UserMessage):
        return {
            "type": "user",
            "content": [{"text": block.text, "cache": block.cache} for block in message.content],
        }
    if isinstance(message, AssistantMessage):
        return {
            "type": "assistant",
            "text": message.text,
            "tool_calls": [
                {"id": call.id, "name": call.name, "arguments": call.arguments}
                for call in message.tool_calls
            ],
        }
    return {
        "type": "tool",
        "call_id": message.call_id,
        "tool_name": message.tool_name,
        "content": message.content,
    }


def _message_from_data(data: dict[str, Any]) -> Message:
    if data["type"] == "user":
        return UserMessage(
            tuple(TextBlock(block["text"], block["cache"]) for block in data["content"])
        )
    if data["type"] == "assistant":
        calls = tuple(
            ToolCall(call["id"], call["name"], call["arguments"]) for call in data["tool_calls"]
        )
        return AssistantMessage(data["text"], calls)
    if data["type"] == "tool":
        return ToolMessage(data["call_id"], data["tool_name"], data["content"])
    raise ValueError(f"unknown persisted message type {data['type']!r}")


def serialize_run(result: RunResult) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": result.run_id,
        "status": result.status.value,
        "stop_reason": result.stop_reason.value,
        "output": (
            {"text": result.output.text, "value": result.output.value}
            if result.output is not None
            else None
        ),
        "usage": {
            "input_tokens": result.usage.input_tokens,
            "output_tokens": result.usage.output_tokens,
            "cached_input_tokens": result.usage.cached_input_tokens,
        },
        "messages": [_message_to_data(message) for message in result.messages],
        "events": [
            {
                "run_id": event.run_id,
                "sequence": event.sequence,
                "kind": event.kind,
                "data": event.data,
                "occurred_at": event.occurred_at.isoformat(),
            }
            for event in result.events
        ],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def deserialize_run(payload: str) -> RunResult:
    data = json.loads(payload)
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported run schema version {data.get('schema_version')!r}")
    output = data["output"]
    usage = data["usage"]
    return RunResult(
        run_id=data["run_id"],
        status=RunStatus(data["status"]),
        stop_reason=StopReason(data["stop_reason"]),
        output=FinalOutput(output["text"], output["value"]) if output is not None else None,
        usage=Usage(usage["input_tokens"], usage["output_tokens"], usage["cached_input_tokens"]),
        messages=tuple(_message_from_data(message) for message in data["messages"]),
        events=tuple(
            RunEvent(
                event["run_id"],
                event["sequence"],
                event["kind"],
                event["data"],
                datetime.fromisoformat(event["occurred_at"]),
            )
            for event in data["events"]
        ),
    )


class SQLiteRunStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
            )

    def save(self, result: RunResult) -> None:
        payload = serialize_run(result)
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "INSERT OR REPLACE INTO runs (run_id, payload) VALUES (?, ?)",
                (result.run_id, payload),
            )

    def load(self, run_id: str) -> RunResult | None:
        with sqlite3.connect(self.path) as connection:
            row = connection.execute(
                "SELECT payload FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return deserialize_run(row[0]) if row else None


__all__ = [
    "SCHEMA_VERSION",
    "RunStore",
    "SQLiteRunStore",
    "deserialize_run",
    "serialize_run",
]
