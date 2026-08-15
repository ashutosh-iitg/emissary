"""The sole provider-neutral model-call boundary used by agent runtimes."""

from dataclasses import dataclass
from typing import Protocol

from .decision import ModelResult, ModelSettings, ToolDefinition
from .errors import CapabilityError, ProviderError
from .messages import Message
from .provider import Spec, key_present
from .streaming import StreamSink
from .wire import WIRES


class ModelCaller(Protocol):
    def __call__(
        self,
        *,
        system: str,
        messages: tuple[Message, ...],
        tools: tuple[ToolDefinition, ...] = (),
        settings: ModelSettings | None = None,
        sink: StreamSink | None = None,
    ) -> ModelResult: ...


def call_model(
    spec: Spec,
    *,
    system: str,
    messages: tuple[Message, ...],
    tools: tuple[ToolDefinition, ...] = (),
    settings: ModelSettings | None = None,
    sink: StreamSink | None = None,
) -> ModelResult:
    """Execute one normalized model turn through the selected wire.

    A `sink` streams deltas as they arrive; the result is the same complete
    `ModelResult` either way (ADR-0022).
    """
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
        spec, system=system, messages=messages, tools=tools, settings=settings, sink=sink
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
        sink: StreamSink | None = None,
    ) -> ModelResult:
        return call_model(
            self.spec, system=system, messages=messages, tools=tools, settings=settings, sink=sink
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
        sink: StreamSink | None = None,
    ) -> ModelResult:
        try:
            return call_model(
                self.primary,
                system=system,
                messages=messages,
                tools=tools,
                settings=settings,
                sink=sink,
            )
        except ProviderError as first:
            if (
                not first.retryable
                or self.fallback is None
                or self.fallback.name == self.primary.name
            ):
                raise
            try:
                # The sink may already have seen a partial answer from the
                # primary. That is the caller's to reconcile: emissary cannot
                # unsay deltas, and hiding the second attempt's stream would be
                # worse than a visible restart.
                return call_model(
                    self.fallback,
                    system=system,
                    messages=messages,
                    tools=tools,
                    settings=settings,
                    sink=sink,
                )
            except ProviderError as second:
                raise ProviderError(f"{first}; fallback {second}") from second


__all__ = ["FallbackModelCaller", "ModelCaller", "SpecModelCaller", "call_model"]
