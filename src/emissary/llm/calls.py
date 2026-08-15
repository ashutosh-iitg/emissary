"""The public entry point, dispatched by wire.

Everything above this speaks a `Spec` and never learns which wire adapter
answered. `provider.py` holds the registry; `wire/` holds the two wire
formats; this holds the choice between them.
"""

from typing import Any

from .errors import ProviderError
from .messages import TextBlock
from .prompt import Prompt, build_prompt
from .provider import Spec, key_present
from .result import CallResult, ChoiceResult
from .wire import WIRES, anthropic

__all__ = ["acall_choice", "acall_tool", "call_choice", "call_tool"]


def _require_scoring(spec: Spec) -> None:
    """Gated on the capability, not on one wire's name: Anthropic and Gemini
    both lack logprobs, and a name check would have silently admitted the third
    wire when it was added (ADR-0004)."""
    if not spec.provider.capabilities.logprobs:
        raise ProviderError(
            f"{spec}: this provider exposes no logprobs, so it cannot be scored — use an "
            "OpenAI-compatible provider such as 'vllm:<model>' or 'openai'"
        )


def _require_credential(spec: Spec) -> None:
    if not key_present(spec):
        raise ProviderError(
            f"{spec.provider.credential.describe()} is not configured for provider {spec.name!r}"
        )


def call_tool(
    spec: Spec,
    *,
    tool: dict[str, Any],
    prompt: Prompt | None = None,
    system: str | None = None,
    blocks: tuple[TextBlock | dict[str, Any], ...] = (),
    effort: str | None = None,
) -> CallResult:
    """One structured call that must answer by invoking `tool`.

    Takes either a `prompt` or the older `system`/`blocks` pair. Blocks are
    concatenated user content; on the Anthropic wire, a block with `cache=True`
    gets an ephemeral prompt-cache breakpoint. `effort` is Anthropic-only
    (`output_config.effort`) and is ignored on the OpenAI-compatible wire,
    which has no equivalent.
    """
    request = build_prompt(prompt, system, tuple(blocks))
    _require_credential(spec)
    wire = WIRES[spec.provider.wire]
    if not hasattr(wire, "call_tool"):
        raise ProviderError(f"{spec}: this provider does not serve tool-forced calls")
    # `effort` is Anthropic-only; passing it elsewhere would be an unknown kwarg.
    extra = {"effort": effort} if wire is anthropic else {}
    return wire.call_tool(spec, system=request.system, blocks=request.blocks, tool=tool, **extra)


async def acall_tool(
    spec: Spec,
    *,
    tool: dict[str, Any],
    prompt: Prompt | None = None,
    system: str | None = None,
    blocks: tuple[TextBlock | dict[str, Any], ...] = (),
    effort: str | None = None,
) -> CallResult:
    """`call_tool` on the async client, with the same admission rules."""
    request = build_prompt(prompt, system, tuple(blocks))
    _require_credential(spec)
    wire = WIRES[spec.provider.wire]
    if not hasattr(wire, "acall_tool"):
        raise ProviderError(f"{spec}: this provider does not serve tool-forced calls")
    extra = {"effort": effort} if wire is anthropic else {}
    return await wire.acall_tool(
        spec, system=request.system, blocks=request.blocks, tool=tool, **extra
    )


def call_choice(
    spec: Spec,
    *,
    labels: list[str],
    prompt: Prompt | None = None,
    system: str | None = None,
    blocks: tuple[TextBlock | dict[str, Any], ...] = (),
) -> ChoiceResult:
    """Score one exchange against a fixed label set, from the model's logprobs.

    **OpenAI-compatible wire only.** The Anthropic Messages API exposes no
    token logprobs — there is no parameter for it and no way to derive one, so
    a Claude-hosted model cannot be scored this way. Point this at a locally
    served open-weight model (`vllm:<model>`) or at OpenAI. Asking a model to
    report its own confidence is *not* an equivalent fallback: self-reported
    confidence is not calibrated, and thresholding it only looks like
    measurement.
    """
    request = build_prompt(prompt, system, tuple(blocks))
    _require_scoring(spec)
    _require_credential(spec)
    return WIRES[spec.provider.wire].call_choice(
        spec, system=request.system, blocks=request.blocks, labels=labels
    )


async def acall_choice(
    spec: Spec,
    *,
    labels: list[str],
    prompt: Prompt | None = None,
    system: str | None = None,
    blocks: tuple[TextBlock | dict[str, Any], ...] = (),
) -> ChoiceResult:
    """`call_choice` on the async client, with the same admission rules."""
    request = build_prompt(prompt, system, tuple(blocks))
    _require_scoring(spec)
    _require_credential(spec)
    return await WIRES[spec.provider.wire].acall_choice(
        spec, system=request.system, blocks=request.blocks, labels=labels
    )
