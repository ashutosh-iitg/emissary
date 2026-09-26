"""What long-term memory holds, as immutable values.

Semantic memory is `Fact`, episodic is `Episode`, procedural is `Procedure`.
Measured state (scores, mastery) is deliberately absent: it belongs in the
application's structured model, which the agent reaches through its own tools.
"""

import uuid
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Literal


def _new_id() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class Fact:
    """`explicit` is what the user said; `inferred` is a conclusion drawn from episodes.

    No confidence score: a model's self-reported confidence is not calibrated
    (the reason `call_choice` exists), so evidence is what an inference carries.
    """

    text: str
    source: Literal["explicit", "inferred"]
    evidence: tuple[str, ...] = ()
    id: str = field(default_factory=_new_id)
    recorded_at: datetime = field(default_factory=_now)
    last_supported_at: datetime = field(default_factory=_now)

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("fact text must not be empty")
        if self.source not in ("explicit", "inferred"):
            raise ValueError(f"unknown fact source {self.source!r}")
        if self.source == "inferred" and not self.evidence:
            raise ValueError("an inferred fact needs the evidence (episode ids) behind it")

    def reaffirmed_by(self, episode: "Episode") -> "Fact":
        if self.source != "inferred":
            raise ValueError("only an inferred fact is renewed by evidence")
        return replace(
            self,
            evidence=(*self.evidence, episode.id),
            last_supported_at=episode.occurred_at,
        )


@dataclass(frozen=True)
class Episode:
    """A bounded, notable event. `refs` point at the application's own records of it."""

    summary: str
    occurred_at: datetime
    refs: tuple[str, ...] = ()
    id: str = field(default_factory=_new_id)

    def __post_init__(self) -> None:
        if not self.summary:
            raise ValueError("episode summary must not be empty")


@dataclass(frozen=True)
class Procedure:
    """A user-specific strategy. Global rules are not procedures: they are the
    agent's instructions, versioned in the application's code and never written
    at runtime.
    """

    name: str
    text: str
    evidence: tuple[str, ...] = ()
    active: bool = False

    def __post_init__(self) -> None:
        if not self.name or not self.text:
            raise ValueError("procedure needs a name and text")

    def supported_by(self, episode_id: str, *, promote_at: int) -> "Procedure":
        if episode_id in self.evidence:
            return self
        evidence = (*self.evidence, episode_id)
        return replace(self, evidence=evidence, active=self.active or len(evidence) >= promote_at)


__all__ = ["Episode", "Fact", "Procedure"]
