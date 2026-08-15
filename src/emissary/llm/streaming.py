"""Observing a model turn as it arrives (ADR-0022).

Streaming is an *additional channel*, never a different answer. Passing a sink
to `call_model` changes when the caller learns things, not what `ModelResult`
finally says — which is why the runner, projection, replay, and fallback know
nothing about any of this.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class StreamSink(Protocol):
    """Receives deltas as the model produces them.

    Called synchronously from inside the wire's read loop, so a slow
    implementation slows the read. **Exceptions are not caught**: they
    propagate and the turn is lost, even though the request has already been
    billed. Swallowing them would leave a display frozen with no error
    recorded anywhere, which is the harder failure to diagnose.

    Only text and reasoning are streamed. Tool-call arguments arrive as JSON
    fragments that mean nothing until complete, so they reach the caller whole
    on `ModelResult.decision` instead.
    """

    def on_text(self, delta: str) -> None:
        """A fragment of the model's visible answer."""
        ...

    def on_thinking(self, delta: str) -> None:
        """A fragment of the model's reasoning text, where the provider shows it."""
        ...


@runtime_checkable
class AsyncStreamSink(Protocol):
    """The same channel for `acall_model`, awaited rather than called.

    A separate protocol rather than a reused one: the reason to stream from
    async code is usually to forward deltas somewhere that must be awaited — a
    websocket, a queue — and a sync-only sink would force the caller to buffer
    or to spawn tasks that reorder the output (ADR-0023).

    Exceptions propagate here too, for the reason given on `StreamSink`.
    """

    async def on_text(self, delta: str) -> None:
        """A fragment of the model's visible answer."""
        ...

    async def on_thinking(self, delta: str) -> None:
        """A fragment of the model's reasoning text, where the provider shows it."""
        ...


__all__ = ["AsyncStreamSink", "StreamSink"]
