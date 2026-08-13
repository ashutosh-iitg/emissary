"""Logprob-based scoring — mocked at the SDK client, never the network.

**The live path is unverified.** These tests pin the arithmetic and the wire
shape against a hand-built logprob payload; no test here has ever talked to a
real vLLM server. Treat a green run as "the extraction is correct given this
response shape", not as "scoring works against a served model".
"""

import math
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from emissary import ProviderError, parse_spec
from emissary.calls import call_choice as gated_call_choice
from emissary.wire.openai_wire import call_choice

BLOCKS = [{"text": "an exchange", "cache": False}]


def _alt(token, probability):
    return SimpleNamespace(token=token, logprob=math.log(probability))


def _response(alternatives, model="qwen3-8b"):
    position = SimpleNamespace(token=alternatives[0].token, top_logprobs=alternatives)
    return SimpleNamespace(
        choices=[SimpleNamespace(logprobs=SimpleNamespace(content=[position]))],
        model=model,
        usage=SimpleNamespace(prompt_tokens=120, completion_tokens=1),
    )


def _mock_client(response=None, side_effect=None):
    client = MagicMock()
    if side_effect is not None:
        client.chat.completions.create.side_effect = side_effect
    else:
        client.chat.completions.create.return_value = response
    return patch("openai.OpenAI", return_value=client)


def test_probabilities_are_renormalised_over_the_labels_only():
    """The raw distribution also covers whitespace and casing variants; a
    threshold is only meaningful against the choice the caller posed."""
    response = _response([_alt("FLAG", 0.30), _alt("SAFE", 0.10), _alt("\n", 0.60)])

    with _mock_client(response):
        out = call_choice(
            parse_spec("vllm:qwen3-8b"), system="s", blocks=BLOCKS, labels=["SAFE", "FLAG"]
        )

    assert out.probability("FLAG") == pytest.approx(0.75)
    assert out.probability("SAFE") == pytest.approx(0.25)
    assert out.label == "FLAG"


def test_casing_and_punctuation_variants_are_matched_to_their_label():
    response = _response([_alt(" safe", 0.5), _alt("SAFE.", 0.2), _alt("FLAG", 0.3)])

    with _mock_client(response):
        out = call_choice(
            parse_spec("vllm:qwen3-8b"), system="s", blocks=BLOCKS, labels=["SAFE", "FLAG"]
        )

    assert out.probability("SAFE") == pytest.approx(0.7)


def test_only_one_token_is_generated():
    """Cost is the whole point of the screening stage — one output token."""
    response = _response([_alt("SAFE", 1.0)])

    with _mock_client(response) as ctor:
        call_choice(parse_spec("vllm:qwen3-8b"), system="s", blocks=BLOCKS, labels=["SAFE", "FLAG"])
        sent = ctor.return_value.chat.completions.create.call_args.kwargs

    assert sent["max_tokens"] == 1
    assert sent["logprobs"] is True
    assert sent["top_logprobs"] == 20


def test_guided_choice_is_sent_to_vllm_and_withheld_from_openai():
    """`guided_choice` is a vLLM extension — a vendor endpoint rejects it as an
    unknown parameter."""
    response = _response([_alt("SAFE", 1.0)])

    with _mock_client(response) as ctor:
        call_choice(parse_spec("vllm:qwen3-8b"), system="s", blocks=BLOCKS, labels=["SAFE", "FLAG"])
        sent = ctor.return_value.chat.completions.create.call_args.kwargs
    assert sent["extra_body"] == {"guided_choice": ["SAFE", "FLAG"]}

    with _mock_client(response) as ctor:
        call_choice(parse_spec("openai:gpt-5"), system="s", blocks=BLOCKS, labels=["SAFE", "FLAG"])
        sent = ctor.return_value.chat.completions.create.call_args.kwargs
    assert "extra_body" not in sent


def test_no_label_in_the_alternatives_is_an_error_not_a_guess():
    """A caller thresholding a made-up number is worse than a caller that
    stops."""
    response = _response([_alt("MAYBE", 0.6), _alt("hmm", 0.4)])

    with _mock_client(response), pytest.raises(ProviderError, match="appeared in the top"):
        call_choice(parse_spec("vllm:qwen3-8b"), system="s", blocks=BLOCKS, labels=["SAFE", "FLAG"])


def test_a_model_that_returns_no_logprobs_cannot_be_scored():
    response = SimpleNamespace(
        choices=[SimpleNamespace(logprobs=None)],
        model="qwen3-8b",
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
    )

    with _mock_client(response), pytest.raises(ProviderError, match="cannot be scored"):
        call_choice(parse_spec("vllm:qwen3-8b"), system="s", blocks=BLOCKS, labels=["SAFE", "FLAG"])


def test_anthropic_is_refused_with_the_reason_and_the_alternative():
    """The Anthropic API exposes no logprobs at all — this is a permanent
    property of the provider, so the error names what to use instead."""
    with pytest.raises(ProviderError, match="no logprobs"):
        gated_call_choice(
            parse_spec("anthropic"), system="s", blocks=BLOCKS, labels=["SAFE", "FLAG"]
        )
