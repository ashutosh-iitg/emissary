"""Memory exposed to the agent as ordinary harness tools.

Each factory closes over one application-scoped store and returns a `Tool`, so
memory goes through the same validation, circuit breaking and event log as any
other tool call. Only explicit facts and scratch notes are written here, in the
response path; episodes and inferences are written after the run by
consolidation, where they can carry evidence.
"""

from collections.abc import Mapping
from typing import Any

from ..harness.tools import Tool
from .records import Fact
from .stores import EpisodeStore, FactStore, Scratchpad, VectorStore

_TEXT = {"type": "string", "minLength": 1}

_MATCHES_SCHEMA = {
    "type": "object",
    "properties": {
        "matches": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "text": {"type": "string"},
                    "score": {"type": "number"},
                    "metadata": {"type": "object"},
                },
                "required": ["id", "text", "score", "metadata"],
            },
        }
    },
    "required": ["matches"],
}


def _object(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def remember_fact_tool(store: FactStore) -> Tool:
    def remember(text: str) -> dict[str, str]:
        fact = Fact(text, "explicit")
        store.add(fact)
        return {"id": fact.id}

    return Tool(
        name="remember_fact",
        description=(
            "Save something the user explicitly told you about themselves, such as a "
            "preference or a goal. Do not save your own guesses about them."
        ),
        input_schema=_object({"text": _TEXT}, ["text"]),
        execute=remember,
        side_effect="local",
    )


def take_note_tool(pad: Scratchpad) -> Tool:
    def note(key: str, value: str) -> dict[str, str]:
        pad.write(key, value)
        return {"key": key}

    return Tool(
        name="take_note",
        description=(
            "Keep a note for the rest of this session: hints already given, what you are "
            "waiting for, or a hunch to check. Notes are forgotten when the session ends."
        ),
        input_schema=_object({"key": _TEXT, "value": _TEXT}, ["key", "value"]),
        execute=note,
        side_effect="local",
    )


def recall_episodes_tool(store: EpisodeStore, *, max_limit: int = 5) -> Tool:
    def recall(limit: int) -> dict[str, Any]:
        return {
            "episodes": [
                {"id": e.id, "summary": e.summary, "occurred_at": e.occurred_at.isoformat()}
                for e in store.recent(limit)
            ]
        }

    return Tool(
        name="recall_episodes",
        description=(
            "Look back at the most recent notable moments with this user, newest first. "
            "Use it when a past experience would change what you do next."
        ),
        input_schema=_object(
            {"limit": {"type": "integer", "minimum": 1, "maximum": max_limit}}, ["limit"]
        ),
        execute=recall,
    )


def vector_search_tool(
    store: VectorStore,
    *,
    name: str,
    description: str,
    filters_schema: dict[str, Any] | None = None,
    max_limit: int = 10,
) -> Tool:
    """Wrap any vector store as a search tool.

    `filters_schema` is the application's contract for what the model may filter
    on; it is validated like any tool input, so a filter it does not declare is
    rejected before the store is touched.
    """
    properties: dict[str, Any] = {
        "query": _TEXT,
        "limit": {"type": "integer", "minimum": 1, "maximum": max_limit},
    }
    if filters_schema is not None:
        properties["filters"] = filters_schema

    def search(query: str, limit: int, filters: Mapping[str, Any] | None = None) -> Any:
        matches = store.search(query, limit=limit, filters=filters or {})
        return {
            "matches": [
                {"id": m.id, "text": m.text, "score": m.score, "metadata": dict(m.metadata)}
                for m in matches
            ]
        }

    return Tool(
        name=name,
        description=description,
        input_schema=_object(properties, ["query", "limit"]),
        output_schema=_MATCHES_SCHEMA,
        execute=search,
    )


__all__ = [
    "recall_episodes_tool",
    "remember_fact_tool",
    "take_note_tool",
    "vector_search_tool",
]
