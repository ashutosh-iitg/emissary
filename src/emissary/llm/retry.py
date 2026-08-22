"""The retry-then-fallback policy, stated once for everything that needs it.

Three shells used to carry their own copy of it — the tool-forced call, the
conversational call, and its async twin. That was tolerable while the policy
was four lines of `retryable` branching. A timed retry ladder, a distinct
warning per outcome, and a deliberate asymmetry between primary and fallback
is no longer four lines, and three copies would drift into three different
ideas of what an outage looks like.

Nothing here is configurable. The delays are a property of the policy rather
than of a caller, and a wrapper shared across projects that grew a settings
object for them would be the settings framework this package refuses to become.
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable

from .errors import ProviderError
from .provider import Spec

logger = logging.getLogger(__name__)
"""Deliberately **not** given a `NullHandler`.

A library normally installs one so an unconfigured application sees nothing.
Here that would be backwards: with no handler configured, `logging.lastResort`
puts WARNING and above on stderr, which is exactly what a degraded call should
do. An application that configures logging captures these like any other record.
"""

RETRY_DELAYS: tuple[float, ...] = (10.0, 30.0, 60.0)
"""Seconds to wait before each retry of the primary — four attempts in all.

Long for a request-scoped retry, and deliberately so: these exist for rate
limits and provider overloads, which clear on the order of tens of seconds. A
millisecond-scale ladder would spend every attempt inside the same outage and
report failure just as fast, having achieved nothing but added load.
"""


def _should_fall_back(primary: Spec, fallback: Spec, failure: ProviderError) -> bool:
    """Whether a different provider could plausibly answer where this one didn't.

    A non-retryable failure stops here: the model answered, just unusably, and
    asking someone else is shopping for a parseable answer rather than
    recovering from an outage. A "fallback" naming the same provider is not one.
    """
    return failure.retryable and fallback.name != primary.name


def _warn_retrying(spec: Spec, failure: ProviderError, number: int, delay: float) -> None:
    logger.warning(
        "%s: attempt %d of %d failed, retrying in %gs — %s",
        spec,
        number,
        len(RETRY_DELAYS) + 1,
        delay,
        failure,
    )


def _warn_falling_back(primary: Spec, fallback: Spec, failure: ProviderError) -> None:
    """The line an operator must not be able to miss.

    A silent fallback means a caller reads an answer believing it came from the
    model it asked for. Naming both specs is the whole point of the record.
    """
    logger.warning(
        "%s: exhausted %d attempts over %gs, falling back to %s — this answer comes "
        "from a different model than the one requested. Last failure: %s",
        primary,
        len(RETRY_DELAYS) + 1,
        sum(RETRY_DELAYS),
        fallback,
        failure,
    )


def call_with_fallback[Result](
    primary: Spec,
    fallback: Spec | None,
    attempt: Callable[[Spec], Result],
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> Result:
    """`attempt` on `primary` up the retry ladder, then **once** on `fallback`.

    `sleep` is injected only so tests can assert the ladder without waiting it
    out; no production caller passes it.
    """
    try:
        return _with_retries(primary, attempt, sleep)
    except ProviderError as first:
        if fallback is None or not _should_fall_back(primary, fallback, first):
            raise
        _warn_falling_back(primary, fallback, first)
        try:
            # One attempt, no second ladder: the primary already spent the
            # outage window, and a fallback that fails too is a condition an
            # operator must see now, not after another two minutes of waiting.
            return attempt(fallback)
        except ProviderError as second:
            # Both named, because "the fallback failed" without saying what the
            # primary did sends an operator to the wrong status page.
            raise ProviderError(f"{first}; fallback {second}") from second


async def acall_with_fallback[Result](
    primary: Spec,
    fallback: Spec | None,
    attempt: Callable[[Spec], Awaitable[Result]],
    *,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> Result:
    """`call_with_fallback` on the async client — same ladder, same warnings."""
    try:
        return await _awith_retries(primary, attempt, sleep)
    except ProviderError as first:
        if fallback is None or not _should_fall_back(primary, fallback, first):
            raise
        _warn_falling_back(primary, fallback, first)
        try:
            return await attempt(fallback)
        except ProviderError as second:
            raise ProviderError(f"{first}; fallback {second}") from second


def _with_retries[Result](
    spec: Spec, attempt: Callable[[Spec], Result], sleep: Callable[[float], None]
) -> Result:
    for number, delay in enumerate(RETRY_DELAYS, start=1):
        try:
            return attempt(spec)
        except ProviderError as failure:
            # A non-retryable failure ends the ladder wherever it happens:
            # repeating a call the provider already answered badly cannot
            # change the answer, and waiting 100s to re-learn that is worse.
            if not failure.retryable:
                raise
            _warn_retrying(spec, failure, number, delay)
            sleep(delay)
    return attempt(spec)


async def _awith_retries[Result](
    spec: Spec,
    attempt: Callable[[Spec], Awaitable[Result]],
    sleep: Callable[[float], Awaitable[None]],
) -> Result:
    for number, delay in enumerate(RETRY_DELAYS, start=1):
        try:
            return await attempt(spec)
        except ProviderError as failure:
            if not failure.retryable:
                raise
            _warn_retrying(spec, failure, number, delay)
            await sleep(delay)
    return await attempt(spec)


__all__ = ["RETRY_DELAYS", "acall_with_fallback", "call_with_fallback"]
