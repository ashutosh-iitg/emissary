"""The sole provider-neutral model-call boundary used by agent runtimes."""

import logging
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Protocol

from .decision import ModelResult, ModelSettings, ToolDefinition
from .errors import CapabilityError, ProviderError
from .messages import Message
from .provider import Spec, key_present
from .retry import acall_with_fallback, call_with_fallback
from .streaming import (
    AsyncStreamSink,
    AsyncTrackingStreamSink,
    StreamSink,
    TrackingStreamSink,
)
from .wire import WIRES

logger = logging.getLogger(__name__)


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
        tracked = TrackingStreamSink(sink) if sink is not None else None

        def attempt(spec: Spec) -> ModelResult:
            try:
                return call_model(
                    spec,
                    system=system,
                    messages=messages,
                    tools=tools,
                    settings=settings,
                    sink=tracked,
                )
            except ProviderError as failure:
                if tracked is None or not tracked.emitted or not failure.retryable:
                    raise
                logger.warning(
                    "%s: stream failed after output reached the sink; retry suppressed to "
                    "prevent a second answer being appended to the same turn — %s",
                    spec,
                    failure,
                )
                raise ProviderError(
                    f"{spec}: failed after streaming output; retry suppressed ({failure})"
                ) from failure

        return call_with_fallback(
            self.primary,
            self.fallback,
            attempt,
        )


@dataclass(frozen=True)
class AsyncFallbackModelCaller:
    """The async twin of `FallbackModelCaller`.

    Both defer to `retry.py`: a timed ladder and its warnings are substantial
    policy, and duplicating them across sync, async, and tool-forced calls would
    let the three paths drift.
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
        tracked = AsyncTrackingStreamSink(sink) if sink is not None else None

        async def attempt(spec: Spec) -> ModelResult:
            try:
                return await acall_model(
                    spec,
                    system=system,
                    messages=messages,
                    tools=tools,
                    settings=settings,
                    sink=tracked,
                )
            except ProviderError as failure:
                if tracked is None or not tracked.emitted or not failure.retryable:
                    raise
                logger.warning(
                    "%s: stream failed after output reached the sink; retry suppressed to "
                    "prevent a second answer being appended to the same turn — %s",
                    spec,
                    failure,
                )
                raise ProviderError(
                    f"{spec}: failed after streaming output; retry suppressed ({failure})"
                ) from failure

        return await acall_with_fallback(
            self.primary,
            self.fallback,
            attempt,
        )


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
