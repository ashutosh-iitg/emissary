"""The memory a run starts with, and how it reaches the model.

Kept small on purpose: this sits in front of every response, so it reads a
compact profile — facts, active strategies, session notes and a few recent
episodes. Anything larger is fetched on demand through the memory tools.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta

from ..harness.agent import Agent
from .records import Episode, Fact, Procedure
from .stores import EpisodeStore, FactStore, ProcedureStore, Scratchpad


@dataclass(frozen=True)
class Recall:
    facts: tuple[Fact, ...] = ()
    episodes: tuple[Episode, ...] = ()
    procedures: tuple[Procedure, ...] = ()
    notes: Mapping[str, str] = field(default_factory=dict)


def load_recall(
    *,
    facts: FactStore | None = None,
    episodes: EpisodeStore | None = None,
    procedures: ProcedureStore | None = None,
    scratchpad: Scratchpad | None = None,
    episode_limit: int = 1,
    inferred_max_age: timedelta | None = None,
    now: datetime | None = None,
) -> Recall:
    """`inferred_max_age` hides inferred facts nothing has supported within that window.

    They stay in the store (a person reviewing memory still sees them) and come
    back once consolidation reaffirms them. The window is the application's to
    choose: how fast a conclusion goes stale depends on who is being remembered.
    """
    return Recall(
        facts=_current(facts.facts(), inferred_max_age, now) if facts else (),
        episodes=episodes.recent(episode_limit) if episodes else (),
        procedures=procedures.procedures() if procedures else (),
        notes=dict(scratchpad.notes()) if scratchpad else {},
    )


def _current(
    facts: tuple[Fact, ...], max_age: timedelta | None, now: datetime | None
) -> tuple[Fact, ...]:
    if max_age is None:
        return facts
    cutoff = (now or datetime.now(UTC)) - max_age
    return tuple(f for f in facts if f.source == "explicit" or f.last_supported_at >= cutoff)


def with_memory(agent: Agent, recall: Recall) -> Agent:
    """Return the agent with its memory appended after its own instructions.

    Memory goes into the system prompt rather than the user turn so that the
    conversation history an application persists holds only what was said.
    The agent's own instructions stay first and unedited: they are the global
    rules, and no memory may override them.
    """
    sections = [
        _section("What you know about the user", [_fact_line(f) for f in recall.facts]),
        _section(
            "Strategies that have worked for this user",
            [p.text for p in recall.procedures if p.active],
        ),
        _section("Recent moments", [e.summary for e in recall.episodes]),
        _section("Notes from this session", [f"{k}: {v}" for k, v in recall.notes.items()]),
    ]
    memory = "\n\n".join(section for section in sections if section)
    if not memory:
        return agent
    return replace(agent, instructions=f"{agent.instructions}\n\n{memory}")


def _fact_line(fact: Fact) -> str:
    return f"(inferred) {fact.text}" if fact.source == "inferred" else fact.text


def _section(title: str, lines: list[str]) -> str:
    if not lines:
        return ""
    return "\n".join([f"## {title}", *(f"- {line}" for line in lines)])


__all__ = ["Recall", "load_recall", "with_memory"]
