"""The sole provider-neutral model-call boundary used by agent runtimes."""

from dataclasses import dataclass
from typing import Protocol

from .decision import ModelResult, ModelSettings, ToolDefinition
from .errors import CapabilityError, ProviderError
from .messages import Message
from .provider import Spec, key_present
from .wire import WIRES


class ModelCaller(Protocol):
    def __call__(
        self,
        *,
        system: str,
        messages: tuple[Message, ...],
        tools: tuple[ToolDefinition, ...] = (),
        settings: ModelSettings | None = None,
    ) -> ModelResult: ...


def call_model(
    spec: Spec,
    *,
    system: str,
    messages: tuple[Message, ...],
    tools: tuple[ToolDefinition, ...] = (),
    settings: ModelSettings | None = None,
) -> ModelResult:
    """Execute one normalized model turn through the selected wire."""
    if not key_present(spec):
        raise ProviderError(
            f"{spec.provider.credential.describe()} is not configured for provider {spec.name!r}"
        )
    if tools and not spec.provider.capabilities.tool_calling:
        raise CapabilityError(f"{spec}: this provider does not support tool calling")
    if (
        settings is not None
        and settings.thinking != "default"
        and not spec.provider.capabilities.thinking
    ):
        raise CapabilityError(f"{spec}: this provider does not support thinking control")
    return WIRES[spec.provider.wire].call_model(
        spec, system=system, messages=messages, tools=tools, settings=settings
    )


@dataclass(frozen=True)
class SpecModelCaller:
    spec: Spec

    def __call__(
        self,
        *,
        system: str,
        messages: tuple[Message, ...],
        tools: tuple[ToolDefinition, ...] = (),
        settings: ModelSettings | None = None,
    ) -> ModelResult:
        return call_model(
            self.spec, system=system, messages=messages, tools=tools, settings=settings
        )


@dataclass(frozen=True)
class FallbackModelCaller:
    primary: Spec
    fallback: Spec | None = None

    def __call__(
        self,
        *,
        system: str,
        messages: tuple[Message, ...],
        tools: tuple[ToolDefinition, ...] = (),
        settings: ModelSettings | None = None,
    ) -> ModelResult:
        try:
            return call_model(
                self.primary,
                system=system,
                messages=messages,
                tools=tools,
                settings=settings,
            )
        except ProviderError as first:
            if (
                not first.retryable
                or self.fallback is None
                or self.fallback.name == self.primary.name
            ):
                raise
            try:
                return call_model(
                    self.fallback,
                    system=system,
                    messages=messages,
                    tools=tools,
                    settings=settings,
                )
            except ProviderError as second:
                raise ProviderError(f"{first}; fallback {second}") from second


__all__ = ["FallbackModelCaller", "ModelCaller", "SpecModelCaller", "call_model"]
