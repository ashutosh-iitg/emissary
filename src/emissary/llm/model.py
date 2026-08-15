"""The sole provider-neutral model-call boundary used by agent runtimes."""

from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Protocol

from .decision import ModelResult, ModelSettings, ToolDefinition
from .errors import CapabilityError, ProviderError
from .messages import Message
from .provider import Spec, key_present
from .streaming import AsyncStreamSink, StreamSink
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


class AsyncModelCaller(Protocol):
    def __call__(
        self,
        *,
        system: str,
        messages: tuple[Message, ...],
        tools: tuple[ToolDefinition, ...] = (),
        settings: ModelSettings | None = None,
        sink: AsyncStreamSink | None = None,
    ) -> Awaitable[ModelResult]: ...


def _gate(spec: Spec, tools, settings) -> None:
    """Preconditions shared by both shells, checked before any request.

    Kept in one place so an async caller cannot quietly acquire different
    admission rules than the sync one.
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
    _gate(spec, tools, settings)
    return WIRES[spec.provider.wire].call_model(
        spec, system=system, messages=messages, tools=tools, settings=settings, sink=sink
    )


async def acall_model(
    spec: Spec,
    *,
    system: str,
    messages: tuple[Message, ...],
    tools: tuple[ToolDefinition, ...] = (),
    settings: ModelSettings | None = None,
    sink: AsyncStreamSink | None = None,
) -> ModelResult:
    """`call_model` on the async client, with the same admission rules."""
    _gate(spec, tools, settings)
    return await WIRES[spec.provider.wire].acall_model(
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
class AsyncSpecModelCaller:
    spec: Spec

    async def __call__(
        self,
        *,
        system: str,
        messages: tuple[Message, ...],
        tools: tuple[ToolDefinition, ...] = (),
        settings: ModelSettings | None = None,
        sink: AsyncStreamSink | None = None,
    ) -> ModelResult:
        return await acall_model(
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


@dataclass(frozen=True)
class AsyncFallbackModelCaller:
    """The async twin of `FallbackModelCaller`.

    The policy is duplicated rather than shared: bridging four lines of
    `retryable` branching across the sync/async divide costs more indirection
    than it saves (ADR-0023). A change to one belongs in the other.
    """

    primary: Spec
    fallback: Spec | None = None

    async def __call__(
        self,
        *,
        system: str,
        messages: tuple[Message, ...],
        tools: tuple[ToolDefinition, ...] = (),
        settings: ModelSettings | None = None,
        sink: AsyncStreamSink | None = None,
    ) -> ModelResult:
        try:
            return await acall_model(
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
                return await acall_model(
                    self.fallback,
                    system=system,
                    messages=messages,
                    tools=tools,
                    settings=settings,
                    sink=sink,
                )
            except ProviderError as second:
                raise ProviderError(f"{first}; fallback {second}") from second


__all__ = [
    "AsyncFallbackModelCaller",
    "AsyncModelCaller",
    "AsyncSpecModelCaller",
    "FallbackModelCaller",
    "ModelCaller",
    "SpecModelCaller",
    "acall_model",
    "call_model",
]
