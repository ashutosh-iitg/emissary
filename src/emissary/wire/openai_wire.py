"""The OpenAI-compatible chat-completions adapter — openai, kimi, deepseek,
gemini, and vllm all speak this wire."""

import json
import os
from typing import Any

from ..errors import ProviderError, retryable_status
from ..provider import MAX_TOKENS, Spec
from ..result import CallResult
from .types import Block


def call_tool(spec: Spec, *, system: str, blocks: list[Block], tool: dict[str, Any]) -> CallResult:
    import openai

    provider = spec.provider
    api_key = os.environ.get(provider.key_env) if provider.key_env else None
    base_url = provider.resolved_base_url()
    client = openai.OpenAI(
        # vLLM's OpenAI-compatible server doesn't validate the key, but the
        # SDK still requires a non-empty string to construct a client.
        api_key=api_key or "not-required",
        **({"base_url": base_url} if base_url else {}),
    )

    function = {
        "name": tool["name"],
        "description": tool["description"],
        "parameters": tool["input_schema"],
    }
    if provider.strict:
        function["strict"] = True

    # No cache breakpoint on this wire — every block is concatenated into one
    # user message, unlike the Anthropic wire's cache-marked content blocks.
    user_content = "\n\n".join(block["text"] for block in blocks)

    try:
        response = client.chat.completions.create(
            model=spec.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            tools=[{"type": "function", "function": function}],
            tool_choice={"type": "function", "function": {"name": tool["name"]}},
            **{provider.max_tokens_field: MAX_TOKENS},
        )
    except openai.APIStatusError as exc:
        raise ProviderError(
            f"{spec}: {exc.status_code} {exc}", retryable=retryable_status(exc.status_code)
        ) from exc
    except openai.APIConnectionError as exc:
        raise ProviderError(f"{spec}: could not reach the API ({exc})", retryable=True) from exc

    choice = response.choices[0]
    calls = choice.message.tool_calls or []
    if not calls:
        raise ProviderError(f"{spec}: no {tool['name']} call (finish={choice.finish_reason})")

    try:
        payload = json.loads(calls[0].function.arguments)
    except json.JSONDecodeError as exc:
        # Not retryable: the model answered, just unusably. Falling back here
        # would be shopping for a provider whose JSON happens to parse.
        raise ProviderError(f"{spec}: tool arguments were not valid JSON ({exc})") from exc

    usage = response.usage
    details = getattr(usage, "prompt_tokens_details", None)
    return CallResult(
        payload=payload,
        provider=spec.name,
        model=response.model,
        input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        cached_input_tokens=getattr(details, "cached_tokens", 0) or 0,
    )
