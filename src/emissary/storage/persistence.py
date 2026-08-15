"""Optional local storage for versioned run records, not durable execution."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Protocol

from ..harness.events import RunEvent
from ..harness.state import RunResult, RunStatus, StopReason
from ..llm.decision import FinalOutput, Usage

SCHEMA_VERSION = 2


class RunStore(Protocol):
    def save(self, result: RunResult) -> None: ...

    def load(self, run_id: str) -> RunResult | None: ...


def serialize_run(result: RunResult) -> str:
    """Messages are not stored: they are derived from ``events`` (ADR-0011)."""
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
