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
from .messages import TextBlock
from .provider import Spec, parse_spec
from .result import CallResult
from .retry import call_with_fallback


def resolve_spec(value: str | None = None, *, env_var: str, default: str) -> Spec:
    """Explicit override, then `env_var`, then `default`. All plain strings,
    each parsed as `"provider"` or `"provider:model"`."""
    return parse_spec(value or os.environ.get(env_var) or default)


def call_tool_with_fallback(
    primary: Spec,
    fallback: Spec | None,
    *,
    system: str,
    blocks: tuple[TextBlock | dict, ...],
    tool: dict[str, Any],
    effort: str | None = None,
) -> CallResult:
    """The retry ladder on `primary`, then one attempt on `fallback`.

    `primary` is retried on the `RETRY_DELAYS` ladder; only then does a
    different provider get asked, once. Both the retries and the switch are
    logged at WARNING — a fallback nobody noticed is a caller reading an answer
    from a model it did not choose.

    Only `ProviderError.retryable` failures retry or fall back — connection
    errors, rate limits, overloads, refusals. A malformed payload or a missing
    credential does neither: the model answered, just unusably, and asking
    again or asking elsewhere would be shopping for a parseable answer rather
    than recovering from an outage.
    """
    return call_with_fallback(
        primary,
        fallback,
        lambda spec: call_tool(spec, system=system, blocks=blocks, tool=tool, effort=effort),
    )
