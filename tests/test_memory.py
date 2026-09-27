from datetime import UTC, datetime, timedelta

import pytest
from conftest import DictScratchpad, ListEpisodeStore, RecordingVectorStore

from emissary.harness.agent import Agent
from emissary.harness.tools import LocalToolExecutor, ToolContext
from emissary.llm.decision import ToolCall
from emissary.memory import (
    Episode,
    Fact,
    Match,
    Procedure,
    Recall,
    load_recall,
    recall_episodes_tool,
    remember_fact_tool,
    take_note_tool,
    vector_search_tool,
    with_memory,
)

MONDAY = datetime(2026, 9, 21, tzinfo=UTC)
TUESDAY = datetime(2026, 9, 22, tzinfo=UTC)


def call(tool, **arguments):
    executor = LocalToolExecutor()
    return executor.execute(ToolCall("c1", tool.name, arguments), tool, ToolContext("run"))


def test_an_inferred_fact_cannot_exist_without_the_episodes_behind_it():
    """A hunch from one moment must not become a permanent trait with nothing to revisit."""
    with pytest.raises(ValueError, match="evidence"):
        Fact("struggles with denominators", source="inferred")


def test_a_procedure_becomes_active_only_once_enough_episodes_support_it():
    candidate = Procedure("use-songs", "Teach new letters through a short song.")

    once = candidate.supported_by("ep1", promote_at=2)
    twice = once.supported_by("ep2", promote_at=2)

    assert not once.active
    assert twice.active
    assert twice.evidence == ("ep1", "ep2")


def test_the_same_episode_counts_once_as_support():
    candidate = Procedure("use-songs", "Sing it.").supported_by("ep1", promote_at=2)

    assert not candidate.supported_by("ep1", promote_at=2).active


def test_remember_fact_stores_what_the_learner_said_as_explicit(facts):
    outcome = call(remember_fact_tool(facts), text="I love dinosaurs")

    assert outcome.status == "success"
    (stored,) = facts.facts()
    assert stored.text == "I love dinosaurs"
    assert stored.source == "explicit"


def test_take_note_keeps_a_session_hypothesis_in_the_scratchpad_only(facts):
    pad = DictScratchpad()

    call(take_note_tool(pad), key="hints_given", value="1")

    assert pad.notes() == {"hints_given": "1"}
    assert facts.facts() == ()


def test_recall_episodes_returns_the_newest_within_the_limit():
    store = ListEpisodeStore(Episode("Counted stars", MONDAY), Episode("Moon shapes", TUESDAY))

    outcome = call(recall_episodes_tool(store, max_limit=5), limit=1)

    assert [episode["summary"] for episode in outcome.content["episodes"]] == ["Moon shapes"]


def test_recall_episodes_refuses_more_than_its_ceiling():
    outcome = call(recall_episodes_tool(ListEpisodeStore(), max_limit=3), limit=10)

    assert outcome.status == "error"


def test_vector_search_passes_the_application_defined_filters_through():
    store = RecordingVectorStore((Match("m1", "A kite story", 0.9, {"programme": "stories"}),))
    tool = vector_search_tool(
        store,
        name="search_stories",
        description="Find stories the learner has heard.",
        api_scope="internal",
        filters_schema={
            "type": "object",
            "properties": {"programme": {"type": "string"}},
            "additionalProperties": False,
        },
    )

    outcome = call(tool, query="kite", limit=3, filters={"programme": "stories"})

    assert store.queries == [{"query": "kite", "limit": 3, "filters": {"programme": "stories"}}]
    assert outcome.content["matches"][0]["text"] == "A kite story"


def test_vector_search_rejects_filters_outside_the_declared_schema():
    tool = vector_search_tool(
        RecordingVectorStore(),
        name="search_stories",
        description="Find stories.",
        api_scope="internal",
        filters_schema={"type": "object", "additionalProperties": False},
    )

    outcome = call(tool, query="kite", limit=3, filters={"child_id": "someone-else"})

    assert outcome.status == "error"


def test_with_memory_keeps_the_global_instructions_first_and_unchanged():
    agent = Agent("tutor", "Never reveal an answer before a hint.")

    remembered = with_memory(agent, Recall(facts=(Fact("Loves dinosaurs", "explicit"),)))

    assert remembered.instructions.startswith("Never reveal an answer before a hint.")
    assert "Loves dinosaurs" in remembered.instructions
    assert agent.instructions == "Never reveal an answer before a hint."


def test_candidate_procedures_do_not_reach_the_model():
    candidate = Procedure("use-songs", "Teach letters through songs.", evidence=("ep1",))
    active = Procedure("draw-it", "Draw the number.", evidence=("ep1", "ep2"), active=True)

    instructions = with_memory(Agent("t", "Base."), Recall(procedures=(candidate, active)))

    assert "Draw the number." in instructions.instructions
    assert "Teach letters through songs." not in instructions.instructions


def test_inferred_facts_are_labelled_so_the_model_treats_them_as_hypotheses():
    inferred = Fact("Finds 'b' and 'd' confusing", "inferred", evidence=("ep1",))

    instructions = with_memory(Agent("t", "Base."), Recall(facts=(inferred,))).instructions

    assert "(inferred) Finds 'b' and 'd' confusing" in instructions


def test_an_empty_recall_leaves_the_agent_as_it_was():
    agent = Agent("t", "Base.")

    assert with_memory(agent, Recall()) is agent


def test_an_inferred_fact_nothing_has_supported_lately_is_left_out(facts):
    """Young learners change fast; a months-old conclusion must not keep steering."""
    stale = Fact("Cannot count past five", "inferred", evidence=("ep0",), last_supported_at=MONDAY)
    facts.add(stale)

    recall = load_recall(
        facts=facts, inferred_max_age=timedelta(days=30), now=MONDAY + timedelta(days=31)
    )

    assert recall.facts == ()
    assert facts.facts() == (stale,)


def test_what_the_learner_said_never_goes_stale(facts):
    said = Fact("I love dinosaurs", "explicit", last_supported_at=MONDAY)
    facts.add(said)

    recall = load_recall(
        facts=facts, inferred_max_age=timedelta(days=30), now=MONDAY + timedelta(days=365)
    )

    assert recall.facts == (said,)


def test_without_a_max_age_every_fact_is_recalled(facts):
    facts.add(Fact("Old hunch", "inferred", evidence=("ep0",), last_supported_at=MONDAY))

    assert len(load_recall(facts=facts).facts) == 1


def test_load_recall_reads_only_the_stores_it_is_given(facts):
    facts.add(Fact("Loves dinosaurs", "explicit"))

    recall = load_recall(facts=facts)

    assert recall.facts == facts.facts()
    assert recall.episodes == ()
