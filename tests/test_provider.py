"""Spec parsing, the provider registry, and credential presence.

No test here reaches a network — this is routing and configuration only.
"""

import pytest

from emissary import PROVIDERS, ProviderError, key_present, parse_spec


def test_a_bare_provider_uses_its_verified_default_model():
    assert parse_spec("kimi").model == "kimi-k3"
    assert parse_spec("anthropic").model == "claude-opus-5"


def test_a_model_can_be_named_inline():
    spec = parse_spec("kimi:kimi-k2.6")

    assert (spec.name, spec.model) == ("kimi", "kimi-k2.6")


def test_a_provider_with_no_verified_default_must_be_given_a_model():
    """Rather than shipping a guessed model ID: endpoints are stable, but
    model IDs move faster, and a plausible-looking one that resolves to
    nothing is a worse failure than an explicit error up front."""
    with pytest.raises(ProviderError, match="name one as"):
        parse_spec("openai")

    assert parse_spec("openai:some-model").model == "some-model"
    assert parse_spec("vllm:my-local-model").model == "my-local-model"


def test_an_unknown_provider_is_refused_with_the_list():
    with pytest.raises(ProviderError, match="kimi"):
        parse_spec("llama")


def test_providers_outnumber_wires_by_more_than_two_to_one():
    """The reason this wrapper is small enough to be worth having: it is a few
    adapters and a table, not one integration per vendor.

    A native wire is earned only when the compatibility layer loses a capability
    the harness needs (ADR-0020) — Gemini's dropped thought signatures did;
    DeepSeek and Kimi, whose first-party API *is* OpenAI-compatible, did not.
    If this ratio ever approaches 1:1 the abstraction has stopped paying for
    itself and should be reconsidered rather than extended.
    """
    wires = {provider.wire for provider in PROVIDERS.values()}

    assert len(PROVIDERS) >= 2 * len(wires) + 1
    assert PROVIDERS["anthropic"].wire == "anthropic"
    assert PROVIDERS["gemini"].wire == PROVIDERS["vertex"].wire == "gemini"
    assert {PROVIDERS[name].wire for name in ("openai", "kimi", "deepseek", "vllm")} == {"openai"}


def test_only_first_party_providers_claim_strict_function_calling():
    """Conservative for the compatibility layers: unverified there, and being
    wrong costs nothing beyond a rejected tool call."""
    assert PROVIDERS["anthropic"].strict
    assert PROVIDERS["openai"].strict
    assert not PROVIDERS["kimi"].strict
    assert not PROVIDERS["deepseek"].strict
    # Gemini is deliberately absent: `strict` is an OpenAI-wire function-calling
    # flag and means nothing on `generateContent`.
    assert not PROVIDERS["vllm"].strict


def test_a_hosted_provider_needs_its_key_present(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert key_present(parse_spec("anthropic")) is False

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert key_present(parse_spec("anthropic")) is True


def test_vllm_needs_no_key_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    assert key_present(parse_spec("vllm:qwen3-8b")) is True


def test_vllm_base_url_is_read_from_env_at_call_time(monkeypatch):
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    assert PROVIDERS["vllm"].resolved_base_url() == "http://localhost:8000/v1"

    monkeypatch.setenv("VLLM_BASE_URL", "http://gpu-box:9000/v1")
    assert PROVIDERS["vllm"].resolved_base_url() == "http://gpu-box:9000/v1"
