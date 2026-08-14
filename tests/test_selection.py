"""`resolve_spec`'s override>env>default precedence, and the fallback policy —
what falls back, and, more importantly, what does not.
"""

from unittest.mock import patch

import pytest

from emissary import CallResult, ProviderError, parse_spec, resolve_spec
from emissary.llm.selection import call_tool_with_fallback

PAYLOAD = {"title": "A law"}


def result(provider="anthropic", model="claude-opus-5"):
    return CallResult(PAYLOAD, provider, model, 10, 5, 0)


def _call(primary="anthropic", fallback=None):
    return call_tool_with_fallback(
        parse_spec(primary),
        parse_spec(fallback) if fallback else None,
        system="s",
        blocks=[{"text": "d", "cache": True}],
        tool={"name": "t"},
    )


class TestResolveSpec:
    def test_default_is_used_when_nothing_else_is_set(self, monkeypatch):
        monkeypatch.delenv("SOME_LLM_PROVIDER", raising=False)
        assert str(resolve_spec(env_var="SOME_LLM_PROVIDER", default="anthropic")) == (
            "anthropic:claude-opus-5"
        )

    def test_env_var_overrides_the_default(self, monkeypatch):
        monkeypatch.setenv("SOME_LLM_PROVIDER", "kimi:kimi-k2.6")
        assert str(resolve_spec(env_var="SOME_LLM_PROVIDER", default="anthropic")) == (
            "kimi:kimi-k2.6"
        )

    def test_an_explicit_value_overrides_everything(self, monkeypatch):
        monkeypatch.setenv("SOME_LLM_PROVIDER", "kimi")
        assert (
            resolve_spec("deepseek", env_var="SOME_LLM_PROVIDER", default="anthropic").name
            == "deepseek"
        )


class TestFallback:
    def test_an_availability_failure_reaches_the_fallback(self):
        outcomes = [ProviderError("overloaded", retryable=True), result("kimi", "kimi-k3")]

        with patch("emissary.llm.selection.call_tool", side_effect=outcomes):
            answered = _call(fallback="kimi")

        assert answered.provider == "kimi"

    def test_a_non_retryable_failure_does_not_reach_the_fallback(self):
        """A malformed payload or a missing credential is the same on every
        provider. Trying another one only spends money to fail twice."""
        with (
            patch(
                "emissary.llm.selection.call_tool",
                side_effect=ProviderError("tool arguments were not valid JSON"),
            ) as called,
            pytest.raises(ProviderError),
        ):
            _call(fallback="kimi")

        assert called.call_count == 1

    def test_it_makes_one_fallback_attempt_and_then_stops(self):
        """Not a chain. A second failure is a condition the caller should
        see, and a wrapper that keeps trying turns an outage into a slow,
        expensive silence."""
        with (
            patch(
                "emissary.llm.selection.call_tool",
                side_effect=ProviderError("overloaded", retryable=True),
            ) as called,
            pytest.raises(ProviderError),
        ):
            _call(fallback="kimi")

        assert called.call_count == 2

    def test_both_failures_are_named_in_the_error(self):
        outcomes = [
            ProviderError("anthropic:claude-opus-5: 529 overloaded", retryable=True),
            ProviderError("kimi:kimi-k3: 429 rate limited", retryable=True),
        ]

        with (
            patch("emissary.llm.selection.call_tool", side_effect=outcomes),
            pytest.raises(ProviderError) as caught,
        ):
            _call(fallback="kimi")

        assert "anthropic" in str(caught.value)
        assert "kimi" in str(caught.value)

    def test_no_fallback_configured_means_one_attempt(self):
        with (
            patch(
                "emissary.llm.selection.call_tool",
                side_effect=ProviderError("overloaded", retryable=True),
            ) as called,
            pytest.raises(ProviderError),
        ):
            _call()

        assert called.call_count == 1

    def test_the_same_provider_named_twice_is_not_a_fallback(self):
        with (
            patch(
                "emissary.llm.selection.call_tool",
                side_effect=ProviderError("overloaded", retryable=True),
            ) as called,
            pytest.raises(ProviderError),
        ):
            _call(primary="anthropic", fallback="anthropic")

        assert called.call_count == 1

    def test_a_successful_primary_call_returns_its_payload_untouched(self):
        """The wrapper returns the payload whatever it contains — judging it
        is the caller's job, not a reason to call again."""
        with patch("emissary.llm.selection.call_tool", return_value=result()) as called:
            answered = _call()

        assert called.call_count == 1
        assert answered.payload == PAYLOAD
