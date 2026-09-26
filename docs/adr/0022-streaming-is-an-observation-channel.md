# ADR-0022: Streaming Is an Observation Channel, Not a Second Result Type

**Date:** 2026-08-15  
**Status:** accepted  
**Deciders:** Project maintainers

## Context

Callers want tokens as they arrive — a terminal that prints, a UI that fills in,
an operator watching a long agent turn. The obvious implementation is a second
call shape that returns an iterator, and it is the wrong one here.

`call_model` returns `ModelResult`, and four things downstream depend on that
being a complete value: the runner records it as one `model_call_completed`
event (ADR-0011), the projection derives messages from that event, replay
rebuilds it from a fixture (ADR-0015), and the fallback policy inspects the
raised error to decide whether to try a second provider (ADR-0002). An
iterator-returning variant would either duplicate all four paths or force them
to consume a stream before they could act — and a fallback cannot be decided
until the stream is drained anyway, because the failure may arrive mid-stream.

There is also a reasoning constraint. ADR-0018 requires the opaque reasoning
state to survive verbatim, and it is only complete when the response is. A
design where the caller assembles the answer from deltas would put the
signature round-trip in the caller's hands, which is exactly where it must not
be.

Meanwhile all three SDKs already accumulate for us: Anthropic's
`messages.stream()` exposes `get_final_message()`, OpenAI's
`chat.completions.stream()` exposes `get_final_completion()`, and both return
an object of the same shape the non-streaming call returns.

## Decision

Streaming is an **observer**, not a return type. `call_model` takes an optional
sink and still returns exactly one complete `ModelResult`.

```python
class StreamSink(Protocol):
    def on_text(self, delta: str) -> None: ...
    def on_thinking(self, delta: str) -> None: ...
```

Passing a sink switches the wire to its streaming API; omitting it leaves the
request byte-identical to today. Nothing else in the package changes — the
runner, projection, replay, persistence, and fallback are untouched, and they
remain untouched by construction rather than by care.

Two deltas, not more. `on_text` and `on_thinking` are what a human watches;
tool-call arguments stream as JSON fragments that are meaningless until
complete, and a sink that received them could only buffer them, which is what
the SDK accumulator already does.

**Normalisation is shared.** Each wire's streaming path drains the stream for
observation, asks the SDK for the accumulated final response, and passes it to
the *same* `_normalize` the non-streaming path uses. A streamed turn and an
unstreamed one therefore cannot disagree about what the model said — the
divergence a hand-written accumulator would eventually introduce is not
possible, because there is only one normaliser.

**On the OpenAI-compatible wire, `reasoning_content` is accumulated by hand**
and passed into `_normalize` explicitly. It is a vendor extension, and the
SDK's accumulator makes no promise to preserve fields outside its own schema.
Relying on it would risk silently losing the value ADR-0018 exists to protect,
and the loss would surface as a provider 400 on the next turn rather than
anywhere near this code.

**A sink that raises is not caught.** The exception propagates and the turn is
lost. Swallowing it would leave a UI silently frozen with no error anywhere,
which is worse than a loud failure the caller can fix — and consistent with
failing loud elsewhere in this package. The cost is real: the request has
already been billed when the sink runs. Documented on the protocol.

## Alternatives Considered

- **`stream_model()` returning an iterator of deltas:** the caller must
  assemble the result, which puts reasoning-state round-tripping outside the
  wire and gives the runner a second shape to record. Rejected.
- **Return `(ModelResult, Iterator)`:** a result that is not yet valid until
  the iterator is drained, with no type-level signal saying so. Rejected.
- **A callback per SDK event type:** exposes provider vocabulary
  (`content_block_delta`, `ChunkEvent`) through a neutral protocol, which is
  the coupling `llm/wire/` exists to prevent. Rejected.
- **Hand-rolled accumulation on every wire:** needed only where an SDK offers
  none. Doing it everywhere would mean three chances to disagree with the
  non-streaming path about the same response. Rejected.
- **Catching sink exceptions and continuing:** considered seriously, because
  the response is already paid for. Rejected on the grounds that a silent
  observer failure is undebuggable, and the caller owns the sink.

## Consequences

### Positive

- The entire harness, evaluation, and persistence stack is unaffected —
  streaming is invisible above the wire.
- A streamed and an unstreamed turn produce identical `ModelResult`s, enforced
  by sharing one normaliser rather than by review.
- The sink is a two-method protocol, so a caller can implement it inline.

### Negative

- Streaming buys latency-to-first-token, not memory: the accumulated response
  is still held whole. A caller streaming a very long output pays for both.
- A sink is called synchronously inside the wire, so a slow sink slows the
  read loop.

### Risks

- The Gemini SDK has no accumulator, so that wire merges chunks itself — the
  one place where a streamed and unstreamed response could diverge. Mitigated
  by tests asserting that a streamed turn yields the same decision *and* the
  same `thought_signature` blocks as the unstreamed one.
- `sink` widens the `ModelCaller` protocol, so external implementations must
  accept it. Kept keyword-only with a default so existing callers still
  satisfy it.
