"""The public entry point, dispatched by wire.

Everything above this speaks a `Spec` and never learns which wire adapter
answered. `provider.py` holds the registry; `wire/` holds the two wire
formats; this holds the choice between them.
"""

from typing import Any

from .errors import ProviderError
from .provider import Spec, key_present
from .result import CallResult, ChoiceResult
from .wire import anthropic, openai_compatible
from .wire.types import Block

__all__ = ["Block", "call_choice", "call_tool"]


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
        return anthropic.call_tool(spec, system=system, blocks=blocks, tool=tool, effort=effort)
    return openai_compatible.call_tool(spec, system=system, blocks=blocks, tool=tool)


def call_choice(spec: Spec, *, system: str, blocks: list[Block], labels: list[str]) -> ChoiceResult:
    """Score one exchange against a fixed label set, from the model's logprobs.

    **OpenAI-compatible wire only.** The Anthropic Messages API exposes no
    token logprobs — there is no parameter for it and no way to derive one, so
    a Claude-hosted model cannot be scored this way. Point this at a locally
    served open-weight model (`vllm:<model>`) or at OpenAI. Asking a model to
    report its own confidence is *not* an equivalent fallback: self-reported
    confidence is not calibrated, and thresholding it only looks like
    measurement.
    """
    if spec.provider.wire == "anthropic":
        raise ProviderError(
            f"{spec}: the Anthropic API exposes no logprobs, so this provider cannot be "
            "scored — use an OpenAI-compatible provider such as 'vllm:<model>' or 'openai'"
        )
    if not key_present(spec):
        raise ProviderError(f"{spec.provider.key_env} is not set for provider {spec.name!r}")
    return openai_compatible.call_choice(spec, system=system, blocks=blocks, labels=labels)
