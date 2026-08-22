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
    or to spawn tasks that reorder the output.

    Exceptions propagate here too, for the reason given on `StreamSink`.
    """

    async def on_text(self, delta: str) -> None:
        """A fragment of the model's visible answer."""
        ...

    async def on_thinking(self, delta: str) -> None:
        """A fragment of the model's reasoning text, where the provider shows it."""
        ...


class TrackingStreamSink:
    """Forwards deltas while remembering whether visible state escaped.

    Retry orchestration needs exactly this fact. Before the first delta, a
    disconnected stream is indistinguishable from any other failed request and
    can safely be attempted again. Afterwards, retrying would append a second
    answer to a sink that cannot retract the first one.
    """

    def __init__(self, sink: StreamSink):
        self.sink = sink
        self.emitted = False

    def on_text(self, delta: str) -> None:
        self.emitted = True
        self.sink.on_text(delta)

    def on_thinking(self, delta: str) -> None:
        self.emitted = True
        self.sink.on_thinking(delta)


class AsyncTrackingStreamSink:
    """The awaited twin of `TrackingStreamSink`."""

    def __init__(self, sink: AsyncStreamSink):
        self.sink = sink
        self.emitted = False

    async def on_text(self, delta: str) -> None:
        self.emitted = True
        await self.sink.on_text(delta)

    async def on_thinking(self, delta: str) -> None:
        self.emitted = True
        await self.sink.on_thinking(delta)


__all__ = [
    "AsyncStreamSink",
    "AsyncTrackingStreamSink",
    "StreamSink",
    "TrackingStreamSink",
]
