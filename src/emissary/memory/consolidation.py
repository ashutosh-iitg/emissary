"""Turning a finished conversation into long-term memory, after the response.

Split along Rule 5: `consolidate` asks the model the judgment questions (was
anything notable, what does it suggest, what does it contradict), and
`apply_consolidation` decides in code what is stored. The rules that keep a
single conversation from rewriting who the user is live in the second half:
nothing is inferred without an episode as evidence, what the user said is never
retracted by an inference, and a strategy is active only once enough separate
episodes support it.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from jsonschema import Draft202012Validator, ValidationError

from ..llm import calls
from ..llm.messages import AssistantMessage, Message, TextBlock, ToolMessage, UserMessage
from ..llm.provider import Spec
from .recall import Recall
from .records import Episode, Fact, Procedure
from .stores import EpisodeStore, FactStore, ProcedureStore

DEFAULT_PROMOTE_AT = 3
"""Separate episodes needed before a strategy is used. A starting point, not a
measurement: tune it once real consolidations can be reviewed."""

_STRINGS = {"type": "array", "items": {"type": "string", "minLength": 1}}

CONSOLIDATION_TOOL: dict[str, Any] = {
    "name": "record_consolidation",
    "description": "Record what this conversation should add to long-term memory.",
    "input_schema": {
        "type": "object",
        "properties": {
            "episode": {
                "type": ["string", "null"],
                "description": "One-paragraph summary of the notable moment, or null.",
            },
            "inferred_facts": _STRINGS,
            "reaffirmed_fact_ids": _STRINGS,
            "contradicted_fact_ids": _STRINGS,
            "strategies": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "minLength": 1},
                        "text": {"type": "string", "minLength": 1},
                    },
                    "required": ["name", "text"],
                    "additionalProperties": False,
                },
            },
        },
        "required": [
            "episode",
            "inferred_facts",
            "reaffirmed_fact_ids",
            "contradicted_fact_ids",
            "strategies",
        ],
        "additionalProperties": False,
    },
}

SYSTEM_PROMPT = """\
You maintain an assistant's long-term memory of one user. Read the conversation \
and decide what, if anything, is worth remembering. Most conversations contain \
nothing notable; then set `episode` to null and leave every list empty.

- `episode`: a short summary of a notable moment — a breakthrough, a repeated \
mistake, an approach that clearly worked or failed, a strong reaction. Say what \
happened and what followed.
- `inferred_facts`: durable conclusions this conversation supports. Not things the \
user stated about themselves (those are already saved), not scores, not guesses \
from a single ambiguous reply.
- `reaffirmed_fact_ids`: ids of listed inferred facts this conversation shows are \
still true.
- `contradicted_fact_ids`: ids of listed facts this conversation clearly contradicts.
- `strategies`: ways of acting that demonstrably helped this user. Reuse the name of \
a listed strategy when it is the same approach.

Call `record_consolidation` exactly once."""


@dataclass(frozen=True)
class Consolidation:
    episode: Episode | None
    inferred_facts: tuple[str, ...]
    contradicted_fact_ids: tuple[str, ...]
    strategies: tuple[tuple[str, str], ...]
    reaffirmed_fact_ids: tuple[str, ...] = ()

    @classmethod
    def of(
        cls,
        *,
        episode_summary: str | None,
        occurred_at: datetime,
        inferred_facts: tuple[str, ...] = (),
        contradicted_fact_ids: tuple[str, ...] = (),
        strategies: tuple[tuple[str, str], ...] = (),
        reaffirmed_fact_ids: tuple[str, ...] = (),
    ) -> "Consolidation":
        episode = Episode(episode_summary, occurred_at) if episode_summary else None
        return cls(episode, inferred_facts, contradicted_fact_ids, strategies, reaffirmed_fact_ids)


def consolidate(
    spec: Spec,
    *,
    transcript: tuple[Message, ...],
    recall: Recall,
    occurred_at: datetime | None = None,
) -> Consolidation:
    result = calls.call_tool(
        spec, system=SYSTEM_PROMPT, blocks=_blocks(transcript, recall), tool=CONSOLIDATION_TOOL
    )
    return _parse(result.payload, occurred_at or datetime.now(UTC))


async def aconsolidate(
    spec: Spec,
    *,
    transcript: tuple[Message, ...],
    recall: Recall,
    occurred_at: datetime | None = None,
) -> Consolidation:
    result = await calls.acall_tool(
        spec, system=SYSTEM_PROMPT, blocks=_blocks(transcript, recall), tool=CONSOLIDATION_TOOL
    )
    return _parse(result.payload, occurred_at or datetime.now(UTC))


def apply_consolidation(
    consolidation: Consolidation,
    *,
    facts: FactStore,
    episodes: EpisodeStore,
    procedures: ProcedureStore | None = None,
    promote_at: int = DEFAULT_PROMOTE_AT,
) -> None:
    episode = consolidation.episode
    if episode is None:
        return
    episodes.add(episode)
    _revise_facts(consolidation, episode, facts)
    if procedures is not None:
        _support_strategies(consolidation, episode, procedures, promote_at)


def _revise_facts(consolidation: Consolidation, episode: Episode, facts: FactStore) -> None:
    known = {fact.id: fact for fact in facts.facts()}
    for fact_id in consolidation.contradicted_fact_ids:
        fact = known.get(fact_id)
        if fact is not None and fact.source == "inferred":
            facts.remove(fact_id)
    doubted = set(consolidation.contradicted_fact_ids)
    for fact_id in consolidation.reaffirmed_fact_ids:
        fact = None if fact_id in doubted else known.get(fact_id)
        if fact is not None and fact.source == "inferred":
            facts.add(fact.reaffirmed_by(episode))
    for text in consolidation.inferred_facts:
        facts.add(
            Fact(text, "inferred", evidence=(episode.id,), last_supported_at=episode.occurred_at)
        )


def _support_strategies(
    consolidation: Consolidation, episode: Episode, store: ProcedureStore, promote_at: int
) -> None:
    existing = {procedure.name: procedure for procedure in store.procedures()}
    for name, text in consolidation.strategies:
        procedure = existing.get(name) or Procedure(name, text)
        store.save(procedure.supported_by(episode.id, promote_at=promote_at))


def _blocks(transcript: tuple[Message, ...], recall: Recall) -> tuple[TextBlock, ...]:
    facts = [f"- [{f.id}] ({f.source}) {f.text}" for f in recall.facts]
    strategies = [
        f"- {p.name}: {p.text} ({'active' if p.active else 'candidate'})" for p in recall.procedures
    ]
    return (
        TextBlock("# Known facts\n" + ("\n".join(facts) or "(none)")),
        TextBlock("# Known strategies\n" + ("\n".join(strategies) or "(none)")),
        TextBlock("# Conversation\n" + "\n".join(_render(m) for m in transcript)),
    )


def _render(message: Message) -> str:
    if isinstance(message, UserMessage):
        return "User: " + " ".join(block.text for block in message.content)
    if isinstance(message, AssistantMessage):
        called = ", ".join(call.name for call in message.tool_calls)
        return f"Assistant: {message.text or ''}" + (f" [called {called}]" if called else "")
    if isinstance(message, ToolMessage):
        return f"Tool {message.tool_name}: {message.content}"
    raise TypeError(f"unknown message type {type(message).__name__}")


def _parse(payload: dict[str, Any], occurred_at: datetime) -> Consolidation:
    """The payload is model output, so it is checked before anything is stored."""
    try:
        Draft202012Validator(CONSOLIDATION_TOOL["input_schema"]).validate(payload)
    except ValidationError as exc:
        raise ValueError(f"malformed consolidation: {exc.message}") from exc
    return Consolidation.of(
        episode_summary=payload["episode"],
        occurred_at=occurred_at,
        inferred_facts=tuple(payload["inferred_facts"]),
        contradicted_fact_ids=tuple(payload["contradicted_fact_ids"]),
        reaffirmed_fact_ids=tuple(payload["reaffirmed_fact_ids"]),
        strategies=tuple((s["name"], s["text"]) for s in payload["strategies"]),
    )


__all__ = [
    "CONSOLIDATION_TOOL",
    "DEFAULT_PROMOTE_AT",
    "Consolidation",
    "aconsolidate",
    "apply_consolidation",
    "consolidate",
]
