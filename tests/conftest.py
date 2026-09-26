"""In-memory stores standing in for an application's storage in memory tests."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import pytest

from emissary.memory import Episode, Fact, Match, Procedure


class DictFactStore:
    def __init__(self, *facts: Fact):
        self._facts = {fact.id: fact for fact in facts}

    def facts(self) -> tuple[Fact, ...]:
        return tuple(self._facts.values())

    def add(self, fact: Fact) -> None:
        self._facts[fact.id] = fact

    def remove(self, fact_id: str) -> None:
        self._facts.pop(fact_id, None)


class ListEpisodeStore:
    def __init__(self, *episodes: Episode):
        self._episodes = list(episodes)

    def recent(self, limit: int) -> tuple[Episode, ...]:
        newest_first = sorted(self._episodes, key=lambda e: e.occurred_at, reverse=True)
        return tuple(newest_first[:limit])

    def add(self, episode: Episode) -> None:
        self._episodes.append(episode)


class DictProcedureStore:
    def __init__(self, *procedures: Procedure):
        self._by_name = {procedure.name: procedure for procedure in procedures}

    def procedures(self) -> tuple[Procedure, ...]:
        return tuple(self._by_name.values())

    def save(self, procedure: Procedure) -> None:
        self._by_name[procedure.name] = procedure


class DictScratchpad:
    def __init__(self, **notes: str):
        self._notes = dict(notes)

    def notes(self) -> Mapping[str, str]:
        return dict(self._notes)

    def write(self, key: str, value: str) -> None:
        self._notes[key] = value


@dataclass
class RecordingVectorStore:
    matches: tuple[Match, ...] = ()
    queries: list[dict[str, Any]] = field(default_factory=list)

    def search(self, query: str, *, limit: int, filters: Mapping[str, Any]) -> tuple[Match, ...]:
        self.queries.append({"query": query, "limit": limit, "filters": dict(filters)})
        return self.matches[:limit]


@pytest.fixture
def facts() -> DictFactStore:
    return DictFactStore()


@pytest.fixture
def episodes() -> ListEpisodeStore:
    return ListEpisodeStore()


@pytest.fixture
def procedures() -> DictProcedureStore:
    return DictProcedureStore()
