"""The provider registry — every backend this package knows how to reach, and how.

Six providers, **two wire formats**. Claude speaks the Anthropic Messages API;
OpenAI, Kimi, DeepSeek, Gemini, and a locally-hosted vLLM server all speak
OpenAI-compatible chat completions. So this is two adapters and a table, not
six integrations.

Base URLs and key variables for the hosted providers were verified against
each provider's own documentation. Model IDs move faster than endpoints;
where a provider's current default could not be confirmed with confidence,
`default_model` is left `None` and the caller must name one explicitly rather
than be handed a guess.
"""

import os
from dataclasses import dataclass

from .errors import ProviderError


@dataclass(frozen=True)
class Provider:
    wire: str
    key_env: str | None = None
    # Whether `key_env` must actually be set for `key_present` to pass. False
    # for vLLM: it has a key var for deployments that put auth in front of
    # it, but an unauthenticated local server — the default — needs none.
    key_required: bool = True
    base_url_env: str | None = None
    base_url: str | None = None
    default_model: str | None = None
    # Whether the provider honours strict function calling. Conservative for
    # the OpenAI-compatible layers: unverified there, and being wrong costs
    # nothing beyond a rejected tool call, never a silently wrong one.
    strict: bool = False
    # OpenAI moved newer models to `max_completion_tokens`; the
    # compatibility layers still take `max_tokens`.
    max_tokens_field: str = "max_tokens"

    def resolved_base_url(self) -> str | None:
        """The base URL to call, reading an env var for providers whose
        address isn't a fixed constant (a local vLLM server has no vendor
        endpoint to hardcode)."""
        if self.base_url_env:
            return os.environ.get(self.base_url_env) or self.base_url
        return self.base_url


PROVIDERS: dict[str, Provider] = {
    "anthropic": Provider(
        wire="anthropic",
        key_env="ANTHROPIC_API_KEY",
        default_model="claude-opus-5",
        strict=True,
    ),
    "openai": Provider(
        wire="openai",
        key_env="OPENAI_API_KEY",
        default_model=None,
        strict=True,
        max_tokens_field="max_completion_tokens",
    ),
    "kimi": Provider(
        wire="openai",
        key_env="MOONSHOT_API_KEY",
        base_url="https://api.moonshot.ai/v1",
        default_model="kimi-k3",
    ),
    "deepseek": Provider(
        wire="openai",
        key_env="DEEPSEEK_API_KEY",
        base_url="https://api.deepseek.com",
        default_model="deepseek-v4-pro",
    ),
    "gemini": Provider(
        wire="openai",
        key_env="GEMINI_API_KEY",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        default_model="gemini-3.6-flash",
    ),
    "vllm": Provider(
        wire="openai",
        # Not required: vLLM's OpenAI-compatible server doesn't authenticate
        # by default. When a deployment does put a key in front of it, set
        # VLLM_API_KEY and it's sent like any other provider's credential.
        key_env="VLLM_API_KEY",
        key_required=False,
        base_url_env="VLLM_BASE_URL",
        base_url="http://localhost:8000/v1",
        default_model=None,
    ),
}

MAX_TOKENS = 16000


@dataclass(frozen=True)
class Spec:
    """A resolved choice of provider and model. Parsed from `provider[:model]`."""

    name: str
    model: str

    @property
    def provider(self) -> Provider:
        return PROVIDERS[self.name]

    def __str__(self) -> str:
        return f"{self.name}:{self.model}"


def parse_spec(spec: str, default_model: str | None = None) -> Spec:
    """`"kimi"` or `"kimi:kimi-k2.6"` -> a Spec."""
    name, _, model = spec.partition(":")
    name = name.strip().lower()
    if name not in PROVIDERS:
        raise ProviderError(f"unknown provider {name!r}; have {sorted(PROVIDERS)}")

    chosen = model.strip() or default_model or PROVIDERS[name].default_model
    if not chosen:
        raise ProviderError(
            f"no default model for {name!r} — name one as '{name}:<model-id>'. "
            "Defaults are only set where the provider's current model ID was verified."
        )
    return Spec(name=name, model=chosen)


def key_present(spec: Spec) -> bool:
    """Whether the selected provider has a credential, without spending a call.

    A provider marked `key_required=False` (an unauthenticated local vLLM
    server, by default) is always "present" — there is nothing to configure.
    """
    provider = spec.provider
    if not provider.key_required:
        return True
    return bool(os.environ.get(provider.key_env)) if provider.key_env else True


__all__ = [
    "MAX_TOKENS",
    "PROVIDERS",
    "Provider",
    "Spec",
    "key_present",
    "parse_spec",
]
