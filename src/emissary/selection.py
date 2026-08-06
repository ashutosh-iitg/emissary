"""Two independent conveniences built on `calls.py`: plain env-based spec
resolution, and one-shot fallback orchestration.

Neither reads anything but `os.environ` — no settings framework, no config
file. A caller with its own config source (stria's Django settings, for
instance) resolves a raw string itself and passes it to `parse_spec`, then
uses `call_tool_with_fallback` directly with two already-resolved `Spec`s.
"""

import os
from typing import Any

from .calls import call_tool
from .errors import ProviderError
from .provider import Spec, parse_spec
from .result import CallResult
from .wire.types import Block


def resolve_spec(value: str | None = None, *, env_var: str, default: str) -> Spec:
    """Explicit override, then `env_var`, then `default`. All plain strings,
    each parsed as `"provider"` or `"provider:model"`."""
    return parse_spec(value or os.environ.get(env_var) or default)


def call_tool_with_fallback(
    primary: Spec,
    fallback: Spec | None,
    *,
    system: str,
    blocks: list[Block],
    tool: dict[str, Any],
    effort: str | None = None,
) -> CallResult:
    """One attempt on `primary`, one attempt on `fallback` if `primary` fails
    in a way `fallback` could plausibly answer.

    Only `ProviderError.retryable` failures fall back — connection errors,
    rate limits, overloads, refusals. A malformed payload or a missing
    credential does not, and neither does anything downstream that raises a
    non-retryable `ProviderError` before returning: retrying elsewhere would
    be shopping for a provider whose answer happens to be usable, not
    recovering from an outage.

    One attempt on the fallback, not a chain — a second failure is a
    condition the caller should see, not another silent retry.
    """
    try:
        return call_tool(primary, system=system, blocks=blocks, tool=tool, effort=effort)
    except ProviderError as first:
        if not first.retryable or fallback is None or fallback.name == primary.name:
            raise
        try:
            return call_tool(fallback, system=system, blocks=blocks, tool=tool, effort=effort)
        except ProviderError as second:
            # Both named, because "the fallback failed" without saying what
            # the primary did sends an operator to the wrong status page.
            raise ProviderError(f"{first}; fallback {second}") from second
