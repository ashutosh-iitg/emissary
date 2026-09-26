"""Storage contracts the application implements; emissary holds no state.

Each store is already scoped by the application (one user, one session), so
no method takes an owner id — a tool cannot be talked into reading someone
else's memory because it has no way to name them.

Only what emissary itself calls is declared here. Deleting on a user's
request, or indexing documents into a vector store, is the application's
business and goes through its own code.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from .records import Episode, Fact, Procedure


class FactStore(Protocol):
    def facts(self) -> tuple[Fact, ...]: ...

    def add(self, fact: Fact) -> None:
        """Insert, or replace the fact with the same id."""
        ...

    def remove(self, fact_id: str) -> None: ...


class EpisodeStore(Protocol):
    def recent(self, limit: int) -> tuple[Episode, ...]:
        """Newest first."""
        ...

    def add(self, episode: Episode) -> None: ...


class ProcedureStore(Protocol):
    def procedures(self) -> tuple[Procedure, ...]: ...

    def save(self, procedure: Procedure) -> None:
        """Insert, or replace the procedure with the same name."""
        ...


class Scratchpad(Protocol):
    """Working memory that outlives one run but not the session: hints given,
    what the agent is waiting for, a hypothesis to check. Never promoted to
    long-term memory directly — only through consolidation's evidence rules.
    """

    def notes(self) -> Mapping[str, str]: ...

    def write(self, key: str, value: str) -> None: ...


@dataclass(frozen=True)
class Match:
    id: str
    text: str
    score: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


class VectorStore(Protocol):
    """Any vector database. Embedding the query is the store's concern: only the
    application knows which model indexed it. A store may use `emissary.embed`
    for that; the protocol takes text so emissary never picks the model (ADR-0026).
    """

    def search(
        self, query: str, *, limit: int, filters: Mapping[str, Any]
    ) -> tuple[Match, ...]: ...


__all__ = [
    "EpisodeStore",
    "FactStore",
    "Match",
    "ProcedureStore",
    "Scratchpad",
    "VectorStore",
]
