"""The retry-then-fallback policy: how long we wait, who we ask, and who hears.

Every test here is about a failure an operator has to be able to explain after
the fact. The ladder decides how much of a rate limit or an overload the caller
absorbs before a *different model* answers, and the warnings are the only
record that the answer did not come from the model that was requested.

The suite-wide fixture in `conftest.py` empties the ladder so other tests do
not wait it out; this module restores the shipped constant, because a policy
tested against a fixture's idea of its own delays is not tested at all.
"""

import logging

import pytest

from emissary import ProviderError, parse_spec
from emissary.llm import retry

SHIPPED_DELAYS = retry.RETRY_DELAYS
"""Captured at import, before `conftest.py` empties it for the rest of the suite."""

PRIMARY = parse_spec("anthropic")
FALLBACK = parse_spec("kimi")


@pytest.fixture(autouse=True)
def shipped_ladder(monkeypatch):
    monkeypatch.setattr(retry, "RETRY_DELAYS", SHIPPED_DELAYS)


class Attempts:
    """Replays a scripted outcome per provider and records who was asked."""

    def __init__(self, **by_provider):
        self._by_provider = by_provider
        self.seen: list[str] = []

    def __call__(self, spec):
        self.seen.append(spec.name)
        outcome = self._by_provider[spec.name].pop(0)
        if isinstance(outcome, ProviderError):
            raise outcome
        return outcome


class AsyncAttempts(Attempts):
    async def __call__(self, spec):
        return Attempts.__call__(self, spec)


class Clock:
    """Stands in for `time.sleep` so the ladder is asserted, not waited out."""

    def __init__(self):
        self.slept: list[float] = []

    def __call__(self, delay):
        self.slept.append(delay)


class AsyncClock(Clock):
    async def __call__(self, delay):
        Clock.__call__(self, delay)


def _unavailable(message="overloaded"):
    return ProviderError(message, retryable=True)


def test_the_primary_absorbs_the_whole_ladder_before_another_model_is_asked():
    """The point of waiting at all: rate limits and overloads clear in tens of
    seconds, so the provider the caller chose gets four attempts across 100s
    before anyone settles for a different one."""
    attempt = Attempts(anthropic=[_unavailable()] * 4, kimi=["fallback answer"])
    clock = Clock()

    result = retry.call_with_fallback(PRIMARY, FALLBACK, attempt, sleep=clock)

    assert result == "fallback answer"
    assert attempt.seen == ["anthropic"] * 4 + ["kimi"]
    assert clock.slept == [10.0, 30.0, 60.0]


def test_the_fallback_gets_one_attempt_and_no_ladder_of_its_own():
    """A fallback that fails too is an outage across two providers — a thing an
    operator must see now, not after a second 100 seconds of waiting."""
    attempt = Attempts(
        anthropic=[_unavailable("primary down")] * 4, kimi=[_unavailable("also down")]
    )
    clock = Clock()

    with pytest.raises(ProviderError) as failure:
        retry.call_with_fallback(PRIMARY, FALLBACK, attempt, sleep=clock)

    assert attempt.seen.count("kimi") == 1
    assert clock.slept == [10.0, 30.0, 60.0]
    # Both named: "the fallback failed" alone sends an operator to the wrong
    # status page.
    assert "primary down" in str(failure.value)
    assert "also down" in str(failure.value)


def test_recovering_partway_up_the_ladder_never_reaches_the_fallback():
    """The ladder exists to keep the caller on the model it chose. Answering on
    attempt three must not silently downgrade anyone to a different model."""
    attempt = Attempts(anthropic=[_unavailable(), _unavailable(), "recovered"], kimi=[])
    clock = Clock()

    assert retry.call_with_fallback(PRIMARY, FALLBACK, attempt, sleep=clock) == "recovered"
    assert "kimi" not in attempt.seen
    assert clock.slept == [10.0, 30.0]


def test_a_non_retryable_failure_spends_no_time_and_asks_nobody_else():
    """A malformed payload is not an outage: the model answered, just unusably.
    Repeating the call cannot change the answer, and asking a second provider
    is shopping for a parseable one — so this must cost zero seconds."""
    attempt = Attempts(anthropic=[ProviderError("tool arguments were not valid JSON")], kimi=[])
    clock = Clock()

    with pytest.raises(ProviderError, match="valid JSON"):
        retry.call_with_fallback(PRIMARY, FALLBACK, attempt, sleep=clock)

    assert attempt.seen == ["anthropic"]
    assert clock.slept == []


def test_the_ladder_still_runs_when_no_fallback_is_configured():
    """Retrying is about riding out a transient outage, which has nothing to do
    with whether a second provider happens to be configured."""
    attempt = Attempts(anthropic=[_unavailable()] * 4)
    clock = Clock()

    with pytest.raises(ProviderError):
        retry.call_with_fallback(PRIMARY, None, attempt, sleep=clock)

    assert attempt.seen == ["anthropic"] * 4
    assert clock.slept == [10.0, 30.0, 60.0]


def test_the_same_provider_named_twice_is_not_a_fallback():
    attempt = Attempts(anthropic=[_unavailable()] * 4)

    with pytest.raises(ProviderError):
        retry.call_with_fallback(PRIMARY, parse_spec("anthropic"), attempt, sleep=Clock())

    assert attempt.seen == ["anthropic"] * 4


def test_every_retry_and_the_switch_are_logged_at_warning(caplog):
    """A silent fallback is the failure this policy exists to prevent: a caller
    reads an answer and believes it came from the model it asked for. The
    record has to name both models, or it explains nothing."""
    attempt = Attempts(anthropic=[_unavailable()] * 4, kimi=["fallback answer"])

    with caplog.at_level(logging.WARNING, logger="emissary.llm.retry"):
        retry.call_with_fallback(PRIMARY, FALLBACK, attempt, sleep=Clock())

    assert all(record.levelno == logging.WARNING for record in caplog.records)
    # Three retries plus the switch — nothing about this run is unrecorded.
    assert len(caplog.records) == 4

    switch = caplog.records[-1].getMessage()
    assert str(PRIMARY) in switch and str(FALLBACK) in switch
    assert "different model than the one requested" in switch


def test_a_call_that_simply_works_says_nothing(caplog):
    """The warnings only mean something if a healthy call is silent. A policy
    that logged on every call would train an operator to filter it out, which
    is the same as not logging the fallback at all."""
    attempt = Attempts(anthropic=["fine"])
    clock = Clock()

    with caplog.at_level(logging.WARNING, logger="emissary.llm.retry"):
        assert retry.call_with_fallback(PRIMARY, FALLBACK, attempt, sleep=clock) == "fine"

    assert caplog.records == []
    assert clock.slept == []


async def test_the_async_path_retries_and_falls_back_identically(caplog):
    attempt = AsyncAttempts(anthropic=[_unavailable()] * 4, kimi=["fallback answer"])
    clock = AsyncClock()

    with caplog.at_level(logging.WARNING, logger="emissary.llm.retry"):
        result = await retry.acall_with_fallback(PRIMARY, FALLBACK, attempt, sleep=clock)

    assert result == "fallback answer"
    assert attempt.seen == ["anthropic"] * 4 + ["kimi"]
    assert clock.slept == [10.0, 30.0, 60.0]
    assert len(caplog.records) == 4


async def test_the_async_path_does_not_retry_an_unusable_answer():
    attempt = AsyncAttempts(anthropic=[ProviderError("not a JSON object")], kimi=[])
    clock = AsyncClock()

    with pytest.raises(ProviderError, match="JSON object"):
        await retry.acall_with_fallback(PRIMARY, FALLBACK, attempt, sleep=clock)

    assert attempt.seen == ["anthropic"]
    assert clock.slept == []
