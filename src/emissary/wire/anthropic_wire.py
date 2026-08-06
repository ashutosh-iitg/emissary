"""The native Anthropic Messages API adapter."""

from typing import Any

from ..errors import ProviderError, retryable_status
from ..provider import MAX_TOKENS, Spec
from ..result import CallResult
from .types import Block


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

    content = [
        {
            "type": "text",
            "text": block["text"],
            **({"cache_control": {"type": "ephemeral"}} if block.get("cache") else {}),
        }
        for block in blocks
    ]
    try:
        response = anthropic.Anthropic().messages.create(
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
            retryable=retryable_status(exc.status_code),
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
