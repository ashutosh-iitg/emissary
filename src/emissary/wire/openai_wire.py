"""The OpenAI-compatible chat-completions adapter — openai, kimi, deepseek,
gemini, and vllm all speak this wire."""

import json
import os
from typing import Any

from ..errors import ProviderError
from ..provider import MAX_TOKENS, Provider, Spec
from ..result import CallResult
from .anthropic_wire import Block, Message


def _client(provider: Provider):
    import openai

    api_key = os.environ.get(provider.key_env) if provider.key_env else None
    return openai.OpenAI(
        # vLLM's OpenAI-compatible server doesn't validate the key, but the
        # SDK still requires a non-empty string to construct a client.
        api_key=api_key or "not-required",
        **({"base_url": provider.resolved_base_url()} if provider.resolved_base_url() else {}),
    )


def call_tool(spec: Spec, *, system: str, blocks: list[Block], tool: dict[str, Any]) -> CallResult:
    import openai

    provider = spec.provider
    client = _client(provider)

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
            f"{spec}: {exc.status_code} {exc}",
            retryable=exc.status_code in (408, 409, 429) or exc.status_code >= 500,
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


def call_text(spec: Spec, *, system: str, messages: list[Message]) -> CallResult:
    import openai

    provider = spec.provider
    client = _client(provider)

    try:
        response = client.chat.completions.create(
            model=spec.model,
            messages=[
                {"role": "system", "content": system},
                *({"role": m["role"], "content": m["content"]} for m in messages),
            ],
            **{provider.max_tokens_field: MAX_TOKENS},
        )
    except openai.APIStatusError as exc:
        raise ProviderError(
            f"{spec}: {exc.status_code} {exc}",
            retryable=exc.status_code in (408, 409, 429) or exc.status_code >= 500,
        ) from exc
    except openai.APIConnectionError as exc:
        raise ProviderError(f"{spec}: could not reach the API ({exc})", retryable=True) from exc

    choice = response.choices[0]
    text = choice.message.content or ""
    if not text:
        raise ProviderError(f"{spec}: no text content (finish={choice.finish_reason})")

    usage = response.usage
    details = getattr(usage, "prompt_tokens_details", None)
    return CallResult(
        payload=text,
        provider=spec.name,
        model=response.model,
        input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        cached_input_tokens=getattr(details, "cached_tokens", 0) or 0,
    )
