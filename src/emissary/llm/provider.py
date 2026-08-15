"""The provider registry — every backend this package knows how to reach, and how.

Seven providers, **three wire formats**. Claude speaks the Anthropic Messages
API; Gemini and Vertex speak `generateContent`; OpenAI, Kimi, DeepSeek, and a
locally-hosted vLLM server all speak OpenAI-compatible chat completions. So
this is three adapters and a table, not seven integrations — and a provider
only earns its own adapter when the compatibility layer loses a capability the
harness needs (ADR-0020).

Base URLs and key variables for the hosted providers were verified against
each provider's own documentation. Model IDs move faster than endpoints;
where a provider's current default could not be confirmed with confidence,
`default_model` is left `None` and the caller must name one explicitly rather
than be handed a guess.
"""

import os
from dataclasses import dataclass, field

from .credentials import ApiKey, Credential, GoogleADC, Unauthenticated
from .decision import ModelCapabilities
from .errors import ProviderError


@dataclass(frozen=True)
class Provider:
    wire: str
    # How this backend proves who it is (ADR-0021). A collaborator rather than
    # an env var name, because Vertex authenticates with ADC and addresses
    # models by project and region — which two fields cannot express.
    credential: Credential = field(default_factory=lambda: Unauthenticated())
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
    # Whether the server honours vLLM's `guided_choice` extension, which
    # constrains decoding to a fixed label set. Only vLLM implements it —
    # sending it to a vendor endpoint is an unknown-parameter error. Scoring
    # works without it (the logprobs are read the same way either way); it
    # just guarantees the sampled token is one of the labels.
    guided_choice: bool = False
    # Which thinking dialect this provider speaks (ADR-0019). A neutral name;
    # `llm/wire/thinking.py` owns the payload it maps to. Left "none" wherever
    # the provider's parameter could not be confirmed against its own docs —
    # same rule as `default_model`, for the same reason.
    thinking_dialect: str = "none"
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)

    @property
    def key_env(self) -> str | None:
        """The variable an operator must set, where there is exactly one."""
        return getattr(self.credential, "env_var", None)

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
        credential=ApiKey("ANTHROPIC_API_KEY"),
        default_model="claude-opus-5",
        strict=True,
        thinking_dialect="anthropic",
        capabilities=ModelCapabilities(tool_calling=True, parallel_tool_calls=True, thinking=True),
    ),
    "openai": Provider(
        wire="openai",
        credential=ApiKey("OPENAI_API_KEY"),
        default_model=None,
        strict=True,
        max_tokens_field="max_completion_tokens",
        capabilities=ModelCapabilities(
            tool_calling=True, parallel_tool_calls=True, structured_output=True, logprobs=True
        ),
    ),
    "kimi": Provider(
        wire="openai",
        credential=ApiKey("MOONSHOT_API_KEY"),
        base_url="https://api.moonshot.ai/v1",
        default_model="kimi-k3",
        # K3 always reasons and is tuned by `reasoning_effort`; it has no off
        # switch, so `thinking="off"` fails rather than being ignored.
        thinking_dialect="effort",
        capabilities=ModelCapabilities(
            tool_calling=True, parallel_tool_calls=True, logprobs=True, thinking=True
        ),
    ),
    "deepseek": Provider(
        wire="openai",
        credential=ApiKey("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        default_model="deepseek-v4-pro",
        thinking_dialect="deepseek",
        capabilities=ModelCapabilities(
            tool_calling=True, parallel_tool_calls=True, logprobs=True, thinking=True
        ),
    ),
    # Native `generateContent`, not the OpenAI-compatible shim: that layer
    # drops `thought_signature`, and Gemini 3+ rejects a multi-turn tool call
    # without it, which is every turn but the first of an agent run.
    "gemini": Provider(
        wire="gemini",
        credential=ApiKey("GEMINI_API_KEY"),
        default_model="gemini-3.6-flash",
        thinking_dialect="gemini",
        capabilities=ModelCapabilities(tool_calling=True, parallel_tool_calls=True, thinking=True),
    ),
    # The same wire and model family reached through GCP: ADC instead of a key,
    # project and region instead of a base URL. No default model — Vertex model
    # IDs are region-dependent, so naming one here would be a guess.
    "vertex": Provider(
        wire="gemini",
        credential=GoogleADC("GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION"),
        default_model=None,
        thinking_dialect="gemini",
        capabilities=ModelCapabilities(tool_calling=True, parallel_tool_calls=True, thinking=True),
    ),
    "vllm": Provider(
        wire="openai",
        # Not required: vLLM's OpenAI-compatible server doesn't authenticate
        # by default. When a deployment does put a key in front of it, set
        # VLLM_API_KEY and it's sent like any other provider's credential.
        credential=Unauthenticated("VLLM_API_KEY"),
        base_url_env="VLLM_BASE_URL",
        base_url="http://localhost:8000/v1",
        default_model=None,
        guided_choice=True,
        capabilities=ModelCapabilities(tool_calling=True, parallel_tool_calls=True, logprobs=True),
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

    Named for the API-key case that covers most providers, but it delegates to
    whatever `Credential` the provider carries — ADC for Vertex, always-true
    for an unauthenticated local vLLM server. The name is kept because both
    consumers import it (ADR-0021).
    """
    return spec.provider.credential.available()


__all__ = [
    "MAX_TOKENS",
    "PROVIDERS",
    "Provider",
    "Spec",
    "key_present",
    "parse_spec",
]
