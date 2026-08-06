"""The native Anthropic Messages API adapter."""

from typing import Any

from ..errors import ProviderError
from ..provider import MAX_TOKENS, Provider, Spec
from ..result import CallResult

Block = dict[str, Any]
"""`{"text": str, "cache": bool}`. `cache=True` marks the ephemeral prompt-cache
breakpoint — put it on content a caller will resend across many calls (e.g. a
document a tool-calling loop re-sends per section)."""

Message = dict[str, Any]
"""`{"role": "user" | "assistant", "content": str}`."""


def _client(provider: Provider):
    import anthropic

    return anthropic.Anthropic()


def call_tool(
    spec: Spec,
    *,
    system: str,
    blocks: list[Block],
    tool: dict[str, Any],
    effort: str | None = None,
) -> CallResult:
    """One structured call that must answer by invoking `tool`."""
    import anthropic

    client = _client(spec.provider)
    content = [
        {
            "type": "text",
            "text": block["text"],
            **({"cache_control": {"type": "ephemeral"}} if block.get("cache") else {}),
        }
        for block in blocks
    ]
    try:
        response = client.messages.create(
            model=spec.model,
            max_tokens=MAX_TOKENS,
            thinking={"type": "adaptive"},
            **({"output_config": {"effort": effort}} if effort else {}),
            system=system,
            tools=[tool],
            tool_choice={"type": "tool", "name": tool["name"]},
            messages=[{"role": "user", "content": content}],
        )
    except anthropic.APIStatusError as exc:
        raise ProviderError(
            f"{spec}: {exc.status_code} {exc.message}",
            retryable=exc.status_code in (408, 409, 429) or exc.status_code >= 500,
        ) from exc
    except anthropic.APIConnectionError as exc:
        raise ProviderError(f"{spec}: could not reach the API ({exc})", retryable=True) from exc

    # Checked before reading `content`: a refusal is a 200 with an empty or
    # partial body, and indexing it is the standard way this breaks.
    if response.stop_reason == "refusal":
        raise ProviderError(f"{spec}: the model declined this request", retryable=True)

    for part in response.content:
        if part.type == "tool_use" and part.name == tool["name"]:
            usage = response.usage
            return CallResult(
                payload=dict(part.input),
                provider=spec.name,
                model=response.model,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cached_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            )
    raise ProviderError(f"{spec}: no {tool['name']} call (stop_reason={response.stop_reason})")


def call_text(spec: Spec, *, system: str, messages: list[Message]) -> CallResult:
    """One plain call, no forced tool — text in, text out."""
    import anthropic

    client = _client(spec.provider)
    try:
        response = client.messages.create(
            model=spec.model,
            max_tokens=MAX_TOKENS,
            thinking={"type": "adaptive"},
            system=system,
            messages=[{"role": m["role"], "content": m["content"]} for m in messages],
        )
    except anthropic.APIStatusError as exc:
        raise ProviderError(
            f"{spec}: {exc.status_code} {exc.message}",
            retryable=exc.status_code in (408, 409, 429) or exc.status_code >= 500,
        ) from exc
    except anthropic.APIConnectionError as exc:
        raise ProviderError(f"{spec}: could not reach the API ({exc})", retryable=True) from exc

    if response.stop_reason == "refusal":
        raise ProviderError(f"{spec}: the model declined this request", retryable=True)

    text = "".join(part.text for part in response.content if part.type == "text")
    if not text:
        raise ProviderError(f"{spec}: no text content (stop_reason={response.stop_reason})")

    usage = response.usage
    return CallResult(
        payload=text,
        provider=spec.name,
        model=response.model,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cached_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
    )
