"""The public entry point, dispatched by wire.

Everything above this speaks a `Spec` and never learns which wire adapter
answered. `provider.py` holds the registry; `wire/` holds the wire formats;
this holds the choice between them.
"""

from collections.abc import Sequence
from typing import Any, Literal

from .errors import CapabilityError, ProviderError
from .messages import TextBlock
from .prompt import Prompt, build_prompt
from .provider import Spec, key_present
from .result import CallResult, ChoiceResult, EmbeddingResult, OcrResult
from .wire import WIRES, anthropic

__all__ = [
    "EmbeddingInput",
    "acall_choice",
    "acall_tool",
    "aembed",
    "aocr",
    "call_choice",
    "call_tool",
    "embed",
    "ocr",
]

EmbeddingInput = Literal["query", "document"]
"""Which side of a retrieval the text is on. Asymmetric models embed the two
differently, and a vector index is only as good as that choice being right."""

OCR_URL_SCHEMES = ("http://", "https://", "data:image/")


def _require_scoring(spec: Spec) -> None:
    """Gated on the capability, not on one wire's name: Anthropic and Gemini
    both lack logprobs, and a name check would have silently admitted the third
    wire when it was added (ADR-0004)."""
    capabilities = spec.provider.capabilities
    if not (capabilities.logprobs or capabilities.calibrated_choice):
        raise ProviderError(
            f"{spec}: this provider exposes no logprobs, so it cannot be scored — use "
            "'typesafe' (Jev), or an OpenAI-compatible provider such as 'vllm:<model>' or 'openai'"
        )


def _require_credential(spec: Spec) -> None:
    if not key_present(spec):
        raise ProviderError(
            f"{spec.provider.credential.describe()} is not configured for provider {spec.name!r}"
        )


def _require_chat(spec: Spec) -> None:
    if not spec.provider.capabilities.chat:
        raise CapabilityError(f"{spec}: this provider does not hold a conversation")


def _require_endpoint(spec: Spec) -> None:
    """A provider addressed only by env var has nowhere to go without it — and the
    OpenAI SDK, given no base URL, would quietly send the request to OpenAI."""
    provider = spec.provider
    if provider.base_url_env and not provider.resolved_base_url():
        raise ProviderError(f"{provider.base_url_env} is not configured for provider {spec.name!r}")


def _admit_embedding(spec: Spec, texts: Sequence[str], input_type: EmbeddingInput | None) -> None:
    provider = spec.provider
    if not provider.capabilities.embeddings:
        raise CapabilityError(f"{spec}: this provider does not serve embeddings")
    if input_type is not None and provider.embedding_dialect == "none":
        raise CapabilityError(
            f"{spec}: this provider cannot embed by input_type; pass input_type=None"
        )
    if not texts or not all(texts):
        raise ProviderError(f"{spec}: embed needs at least one text, and no empty ones")
    _require_credential(spec)
    _require_endpoint(spec)


def _admit_ocr(spec: Spec, image_url: str) -> None:
    if not spec.provider.capabilities.ocr:
        raise CapabilityError(f"{spec}: this provider does not serve OCR")
    if not image_url.startswith(OCR_URL_SCHEMES):
        raise ProviderError(
            f"{spec}: image_url must be an http(s) URL or a data:image/...;base64 URI"
        )
    _require_credential(spec)


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
    _require_chat(spec)
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
    _require_chat(spec)
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


def embed(
    spec: Spec, texts: Sequence[str], *, input_type: EmbeddingInput | None = None
) -> EmbeddingResult:
    """One vector per text, in order, from an embedding model.

    Never retried on or fallen back to another provider by anything here: vectors
    from two models do not share a space, so a "fallback" embedding would sit in
    the index looking valid and match nothing (ADR-0026). `input_type` is refused,
    not dropped, where the provider cannot express it.
    """
    _admit_embedding(spec, texts, input_type)
    return WIRES[spec.provider.wire].embed(spec, texts=tuple(texts), input_type=input_type)


async def aembed(
    spec: Spec, texts: Sequence[str], *, input_type: EmbeddingInput | None = None
) -> EmbeddingResult:
    """`embed` on the async client, with the same admission rules."""
    _admit_embedding(spec, texts, input_type)
    return await WIRES[spec.provider.wire].aembed(spec, texts=tuple(texts), input_type=input_type)


def ocr(spec: Spec, *, image_url: str) -> OcrResult:
    """Transcribe one page image to Markdown.

    `image_url` is an http(s) URL the provider can fetch, or a
    `data:image/<type>;base64,...` URI for bytes the caller holds.
    """
    _admit_ocr(spec, image_url)
    return WIRES[spec.provider.wire].ocr(spec, image_url=image_url)


async def aocr(spec: Spec, *, image_url: str) -> OcrResult:
    """`ocr` on the async client, with the same admission rules."""
    _admit_ocr(spec, image_url)
    return await WIRES[spec.provider.wire].aocr(spec, image_url=image_url)
