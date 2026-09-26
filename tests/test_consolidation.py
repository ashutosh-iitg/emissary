from datetime import UTC, datetime

import pytest
from conftest import DictFactStore, DictProcedureStore, ListEpisodeStore

from emissary.llm import calls
from emissary.llm.messages import AssistantMessage, TextBlock, UserMessage
from emissary.llm.provider import parse_spec
from emissary.llm.result import CallResult
from emissary.memory import (
    Consolidation,
    Fact,
    Procedure,
    Recall,
    aconsolidate,
    apply_consolidation,
    consolidate,
)

NOW = datetime(2026, 9, 27, tzinfo=UTC)
TRANSCRIPT = (
    UserMessage((TextBlock("Is it b or d?"),)),
    AssistantMessage(text="Let's sing the letter song and see."),
)


def payload(**overrides):
    base = {
        "episode": None,
        "inferred_facts": [],
        "reaffirmed_fact_ids": [],
        "contradicted_fact_ids": [],
        "strategies": [],
    }
    return {**base, **overrides}


@pytest.fixture
def model(monkeypatch):
    requests = []

    def respond_with(result_payload):
        def fake_call_tool(spec, **kwargs):
            requests.append(kwargs)
            return CallResult(result_payload, "fake", "scripted", 1, 1, 0)

        monkeypatch.setattr(calls, "call_tool", fake_call_tool)
        return requests

    return respond_with


def test_consolidate_sends_the_transcript_and_current_memory_to_the_model(model):
    requests = model(payload())
    known = Fact("Mixes up b and d", "inferred", evidence=("old",))

    consolidate(parse_spec("anthropic"), transcript=TRANSCRIPT, recall=Recall(facts=(known,)))

    sent = "\n".join(block.text for block in requests[0]["blocks"])
    assert "Is it b or d?" in sent
    assert known.id in sent


def test_session_notes_reach_consolidation_as_unverified_hunches(model):
    """A hunch noted mid-session is only judged if the judging step can see it."""
    requests = model(payload())

    consolidate(
        parse_spec("anthropic"),
        transcript=TRANSCRIPT,
        recall=Recall(notes={"hunch": "confuses b and d"}),
    )

    sent = "\n".join(block.text for block in requests[0]["blocks"])
    assert "# Session notes (unverified)" in sent
    assert "hunch: confuses b and d" in sent


def test_malformed_model_output_is_refused_before_anything_can_be_stored(model):
    model({"episode": "Something happened"})

    with pytest.raises(ValueError, match="malformed consolidation"):
        consolidate(parse_spec("anthropic"), transcript=TRANSCRIPT, recall=Recall())


async def test_async_consolidation_reads_the_same_contract(monkeypatch):
    async def fake_acall_tool(spec, **kwargs):
        return CallResult(payload(episode="Counted to ten."), "fake", "scripted", 1, 1, 0)

    monkeypatch.setattr(calls, "acall_tool", fake_acall_tool)

    result = await aconsolidate(parse_spec("anthropic"), transcript=TRANSCRIPT, recall=Recall())

    assert result.episode.summary == "Counted to ten."


def test_nothing_notable_produces_no_episode(model):
    model(payload())

    result = consolidate(parse_spec("anthropic"), transcript=TRANSCRIPT, recall=Recall())

    assert result.episode is None


def test_an_episode_carries_the_inferences_it_supports(model):
    model(
        payload(
            episode="Sang the letter song; told b from d twice.",
            inferred_facts=["Songs help with letter shapes"],
        )
    )

    result = consolidate(
        parse_spec("anthropic"), transcript=TRANSCRIPT, recall=Recall(), occurred_at=NOW
    )

    assert result.episode.summary == "Sang the letter song; told b from d twice."
    assert result.episode.occurred_at == NOW
    assert result.inferred_facts == ("Songs help with letter shapes",)


def test_apply_records_the_episode_as_evidence_for_inferred_facts():
    facts, episodes = DictFactStore(), ListEpisodeStore()
    outcome = Consolidation.of(
        episode_summary="Told b from d after a song.",
        occurred_at=NOW,
        inferred_facts=("Songs help with letter shapes",),
    )

    apply_consolidation(outcome, facts=facts, episodes=episodes)

    (episode,) = episodes.recent(1)
    (fact,) = facts.facts()
    assert fact.source == "inferred"
    assert fact.evidence == (episode.id,)


def test_contradiction_removes_inferred_facts_but_never_what_the_learner_said():
    inferred = Fact("Cannot count past five", "inferred", evidence=("ep0",))
    explicit = Fact("I hate counting", "explicit")
    facts = DictFactStore(inferred, explicit)
    outcome = Consolidation.of(
        episode_summary="Counted to eight.",
        occurred_at=NOW,
        contradicted_fact_ids=(inferred.id, explicit.id),
    )

    apply_consolidation(outcome, facts=facts, episodes=ListEpisodeStore())

    assert facts.facts() == (explicit,)


def test_reaffirming_an_inferred_fact_renews_it_with_the_new_evidence():
    old = datetime(2026, 6, 1, tzinfo=UTC)
    fact = Fact("Songs help with letters", "inferred", evidence=("ep0",), last_supported_at=old)
    facts = DictFactStore(fact)
    outcome = Consolidation.of(
        episode_summary="Sang the b/d song again; got both right.",
        occurred_at=NOW,
        reaffirmed_fact_ids=(fact.id,),
    )

    apply_consolidation(outcome, facts=facts, episodes=ListEpisodeStore())

    (renewed,) = facts.facts()
    assert renewed.id == fact.id
    assert renewed.last_supported_at == NOW
    assert renewed.evidence == ("ep0", outcome.episode.id)


def test_reaffirmation_cannot_touch_what_the_learner_said():
    said = Fact("I hate counting", "explicit", last_supported_at=datetime(2026, 1, 1, tzinfo=UTC))
    facts = DictFactStore(said)

    apply_consolidation(
        Consolidation.of(
            episode_summary="Counted.", occurred_at=NOW, reaffirmed_fact_ids=(said.id,)
        ),
        facts=facts,
        episodes=ListEpisodeStore(),
    )

    assert facts.facts() == (said,)


def test_strategies_are_promoted_only_after_enough_supporting_episodes():
    procedures = DictProcedureStore()
    song = (("use-songs", "Teach letters through a short song."),)

    def consolidate_once():
        apply_consolidation(
            Consolidation.of(episode_summary="A song helped.", occurred_at=NOW, strategies=song),
            facts=DictFactStore(),
            episodes=ListEpisodeStore(),
            procedures=procedures,
            promote_at=3,
        )

    consolidate_once()
    consolidate_once()
    (still_candidate,) = procedures.procedures()
    assert not still_candidate.active

    consolidate_once()
    (promoted,) = procedures.procedures()
    assert promoted.active


def test_a_supported_strategy_keeps_its_original_wording():
    """Wording drifts between consolidations; the evidence attaches to the named strategy."""
    existing = Procedure("use-songs", "Teach letters through a short song.", evidence=("ep0",))
    procedures = DictProcedureStore(existing)

    apply_consolidation(
        Consolidation.of(
            episode_summary="Song again.",
            occurred_at=NOW,
            strategies=(("use-songs", "Sing everything."),),
        ),
        facts=DictFactStore(),
        episodes=ListEpisodeStore(),
        procedures=procedures,
    )

    (procedure,) = procedures.procedures()
    assert procedure.text == "Teach letters through a short song."
    assert len(procedure.evidence) == 2


def test_without_an_episode_nothing_is_inferred_or_promoted():
    facts, procedures = DictFactStore(), DictProcedureStore()
    outcome = Consolidation(
        episode=None,
        inferred_facts=("Loves maths",),
        contradicted_fact_ids=(),
        strategies=(("use-songs", "Sing."),),
    )

    apply_consolidation(outcome, facts=facts, episodes=ListEpisodeStore(), procedures=procedures)

    assert facts.facts() == ()
    assert procedures.procedures() == ()


def test_a_fact_both_contradicted_and_reaffirmed_is_removed():
    """Contradiction wins: keeping a doubted conclusion is the costlier mistake."""
    fact = Fact("Cannot count past five", "inferred", evidence=("ep0",))
    facts = DictFactStore(fact)

    apply_consolidation(
        Consolidation.of(
            episode_summary="Counted to eight, then to four.",
            occurred_at=NOW,
            contradicted_fact_ids=(fact.id,),
            reaffirmed_fact_ids=(fact.id,),
        ),
        facts=facts,
        episodes=ListEpisodeStore(),
    )

    assert facts.facts() == ()
