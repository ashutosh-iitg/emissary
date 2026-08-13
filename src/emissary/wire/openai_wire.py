"""The OpenAI-compatible chat-completions adapter — openai, kimi, deepseek,
gemini, and vllm all speak this wire."""

import json
import math
import os
from typing import Any

from ..errors import ProviderError, retryable_status
from ..provider import MAX_TOKENS, Spec
from ..result import CallResult, ChoiceResult
from .types import Block

TOP_LOGPROBS = 20
"""How many alternatives to ask for at the scored position.

The labels must all appear among them or the score is unusable, and a label
whose first token is rare can sit well down the list. 20 is the ceiling the
OpenAI-compatible wire accepts, and the cost is negligible — one position.
"""


def _client(spec: Spec):
    import openai

    provider = spec.provider
    api_key = os.environ.get(provider.key_env) if provider.key_env else None
    base_url = provider.resolved_base_url()
    return openai.OpenAI(
        # vLLM's OpenAI-compatible server doesn't validate the key, but the
        # SDK still requires a non-empty string to construct a client.
        api_key=api_key or "not-required",
        **({"base_url": base_url} if base_url else {}),
    )


def _status_error(spec: Spec, exc) -> ProviderError:
    return ProviderError(
        f"{spec}: {exc.status_code} {exc}", retryable=retryable_status(exc.status_code)
    )


def call_tool(spec: Spec, *, system: str, blocks: list[Block], tool: dict[str, Any]) -> CallResult:
    import openai

    provider = spec.provider
    client = _client(spec)

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
        raise _status_error(spec, exc) from exc
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


def call_choice(spec: Spec, *, system: str, blocks: list[Block], labels: list[str]) -> ChoiceResult:
    """One constrained single-token call, scored from the model's own logprobs.

    Generates **one** token and reads the alternatives considered at that
    position, keeping the mass that landed on `labels` and renormalising over
    it. The result is a genuine probability from the model's distribution —
    not a number the model was asked to report about itself, which is not
    calibrated and not what a threshold should be turned on.

    `labels` must be distinguishable by their **first token**: they are matched
    against the sampled alternatives by prefix, so `["SAFE", "FLAG"]` works and
    `["FLAG_A", "FLAG_B"]` does not — both would collapse onto `FLAG`.
    """
    import openai

    provider = spec.provider
    if not labels:
        raise ProviderError(f"{spec}: call_choice needs at least one label")

    user_content = "\n\n".join(block["text"] for block in blocks)
    extra: dict[str, Any] = {}
    if provider.guided_choice:
        # vLLM only — constrains decoding to the label set so the sampled
        # token cannot be anything else. The score is read the same way with
        # or without it; this just removes the off-label case.
        extra["extra_body"] = {"guided_choice": labels}

    client = _client(spec)
    try:
        response = client.chat.completions.create(
            model=spec.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
            logprobs=True,
            top_logprobs=TOP_LOGPROBS,
            **{provider.max_tokens_field: 1},
            **extra,
        )
    except openai.APIStatusError as exc:
        raise _status_error(spec, exc) from exc
    except openai.APIConnectionError as exc:
        raise ProviderError(f"{spec}: could not reach the API ({exc})", retryable=True) from exc

    choice = response.choices[0]
    content = getattr(choice.logprobs, "content", None) if choice.logprobs else None
    if not content:
        # The server accepted `logprobs` and returned none — scoring is
        # impossible, and a caller thresholding a made-up number is worse than
        # a caller that stops. Not retryable: another attempt answers the same.
        raise ProviderError(f"{spec}: no logprobs returned; this model cannot be scored")

    return ChoiceResult(
        probabilities=_label_probabilities(spec, content[0], labels),
        provider=spec.name,
        model=response.model,
        input_tokens=getattr(response.usage, "prompt_tokens", 0) or 0,
        output_tokens=getattr(response.usage, "completion_tokens", 0) or 0,
        cached_input_tokens=0,
    )


def _label_probabilities(spec: Spec, position, labels: list[str]) -> dict[str, float]:
    """Mass that landed on each label at one token position, renormalised.

    Renormalising over just the labels is the point: the raw probabilities
    also cover whitespace, casing variants, and everything else the model
    might have said, and a threshold is only meaningful against the choice the
    caller actually posed.
    """
    alternatives = list(getattr(position, "top_logprobs", None) or [])
    if not alternatives:
        alternatives = [position]

    mass = dict.fromkeys(labels, 0.0)
    for alternative in alternatives:
        token = (alternative.token or "").strip().upper()
        if not token:
            continue
        for label in labels:
            # Prefix match in both directions: a label may tokenise to several
            # tokens (we see the first), or a single token may carry the whole
            # label plus punctuation.
            upper = label.upper()
            if token.startswith(upper) or upper.startswith(token):
                mass[label] += math.exp(alternative.logprob)
                break

    total = sum(mass.values())
    if total <= 0.0:
        raise ProviderError(
            f"{spec}: none of {labels} appeared in the top {TOP_LOGPROBS} tokens; "
            "the model answered something else entirely"
        )
    return {label: value / total for label, value in mass.items()}
