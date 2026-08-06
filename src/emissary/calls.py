"""The two public entry points — one structured, one plain — dispatched by wire.

Everything above this speaks a `Spec` and never learns which wire adapter
answered. `provider.py` holds the registry; `wire/` holds the two wire
formats; this holds the choice between them.
"""

from typing import Any

from .errors import ProviderError
from .provider import Spec, key_present
from .result import CallResult
from .wire import anthropic_wire, openai_wire
from .wire.anthropic_wire import Block, Message

__all__ = ["Block", "Message", "call_text", "call_tool"]


def call_tool(
    spec: Spec,
    *,
    system: str,
    blocks: list[Block],
    tool: dict[str, Any],
    effort: str | None = None,
) -> CallResult:
    """One structured call that must answer by invoking `tool`.

    `blocks` are concatenated user content; on the Anthropic wire, a block
    with `cache=True` gets an ephemeral prompt-cache breakpoint. `effort` is
    Anthropic-only (`output_config.effort`) and is ignored on the
    OpenAI-compatible wire, which has no equivalent.
    """
    if not key_present(spec):
        raise ProviderError(f"{spec.provider.key_env} is not set for provider {spec.name!r}")
    if spec.provider.wire == "anthropic":
        return anthropic_wire.call_tool(
            spec, system=system, blocks=blocks, tool=tool, effort=effort
        )
    return openai_wire.call_tool(spec, system=system, blocks=blocks, tool=tool)


def call_text(spec: Spec, *, system: str, messages: list[Message]) -> CallResult:
    """One plain call, no forced tool — text in, text out."""
    if not key_present(spec):
        raise ProviderError(f"{spec.provider.key_env} is not set for provider {spec.name!r}")
    if spec.provider.wire == "anthropic":
        return anthropic_wire.call_text(spec, system=system, messages=messages)
    return openai_wire.call_text(spec, system=system, messages=messages)
