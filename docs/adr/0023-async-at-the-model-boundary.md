# ADR-0023: Async at the Model Boundary, Not (Yet) in the Runner

**Date:** 2026-08-16  
**Status:** accepted  
**Deciders:** Project maintainers  
**Amends:** ADR-0005

## Context

Callers want concurrency. stria extracts many documents and would run them in
parallel; doom scores many candidates and would batch them; anyone embedding an
agent in an async web framework needs to not block the loop. Today every entry
point is synchronous.

"Add async" is two changes of very different size, and conflating them is how
this goes wrong.

**The model boundary** — `call_model`, `call_tool`, `call_choice` — is a thin
I/O shell around pure translation. All three SDKs ship async clients that mirror
their sync API exactly: `AsyncAnthropic`, `AsyncOpenAI`, and the Gemini client's
`.aio` namespace, each with the same `create` / `stream` / `generate_content`
shapes. Making these async is a shell change over code that already exists.

**The runner** is a state machine that interleaves model calls and tool calls
across turns. An async version cannot be a shell change: either the loop is
duplicated — two implementations of retries, circuit breaking, limits, and
terminal conditions, guaranteed to drift — or it is restructured into a
sans-I/O core that yields effects for a driver to perform. That restructure
rewrites `runner.py` and every runner test.

ADR-0005 chose a bounded synchronous runner first, deliberately. Nothing has
happened to invalidate that; what has happened is that the *boundary* below it
now has three wires, and each one added multiplies the cost of getting the
sync/async split wrong later.

## Decision

**Async the whole LLM boundary now. Leave the runner synchronous.**

Every public call gains an `a`-prefixed sibling — `acall_model`, `acall_tool`,
`acall_choice`, plus `AsyncModelCaller`, `AsyncSpecModelCaller`, and
`AsyncFallbackModelCaller`. The prefix follows the convention stria already
lives with in Django (`aget`, `acreate`).

**The pair must share everything except the await.** Each wire is factored so
that request construction and response normalisation are pure functions called
by both shells:

```
_request(spec, …) -> dict        # pure
_normalize(spec, response, …)    # pure
call_model(...)                  # sync shell
acall_model(...)                 # async shell
```

This is not tidiness. A duplicated `_normalize` is how a sync turn and an async
turn come to disagree about what the model said, and the disagreement would be
invisible until a trajectory recorded through one path is replayed through the
other. The same discipline already keeps streaming honest (ADR-0022); async is
the third shell over the same two functions.

**Async streaming gets its own sink type.** `AsyncStreamSink` declares
`async def on_text` / `async def on_thinking`. The synchronous `StreamSink`
is not reused, because the reason to stream from async code is usually to
forward deltas somewhere that must be awaited — a websocket, a queue — and a
sync-only sink would force the caller to buffer or to spawn tasks that reorder
the output. Two protocols, each unambiguous about what it may do.

**The runner stays sync, and this ADR records what changing that costs.** When
an async runner is wanted, the answer is *not* a parallel `arun` beside `run`.
It is: extract the loop into a generator that yields `CallModel` and
`ExecuteTool` effects and receives their outcomes, then write a thin sync driver
and a thin async driver over it. One policy, two I/O shells — the same shape
this ADR applies to the wires. That is its own decision, with its own ADR.

## Alternatives Considered

- **Async runner in the same change:** the honest scope is a rewrite of
  `runner.py` plus every runner test, landing alongside six new wire functions.
  Too large to review as one slice, and the runner refactor is strictly easier
  once an async caller exists to depend on — the ordering is forced, not
  preferred. Deferred.
- **`asyncio.to_thread` around the sync calls:** one line, no new code, and
  wrong. It burns a thread per in-flight request, so the concurrency ceiling
  becomes the thread pool rather than the event loop, and it cannot stream
  without hopping every delta across a thread boundary. Rejected.
- **Async-only, with sync callers wrapping via `asyncio.run`:** would break
  both consumers, which are synchronous today, and `asyncio.run` cannot be
  called from inside a running loop — so it would fail exactly in the async
  contexts it claims to serve. Rejected.
- **Generate the sync API from the async one** (the unasync approach some SDKs
  take): removes duplication at the cost of a build step and generated source
  in a package whose whole claim is being small enough to read. Rejected.
- **Reuse `StreamSink` for async, awaiting the return value if it is
  awaitable:** duck-typing that makes the protocol's contract unstateable and
  fails confusingly when a sink returns something else. Rejected.

## Consequences

### Positive

- Callers can run many extractions or classifications concurrently on one
  thread, which is the concurrency both consumers would actually use.
- Async agent loops become possible to build *on top of* emissary even while
  the bundled runner is synchronous, because the caller protocol they need is
  the async one this ADR adds.
- Factoring `_request` and `_normalize` out of the shells makes each wire's
  translation testable without any client at all, and leaves the sans-I/O
  runner refactor a strictly smaller job than it is today.

### Negative

- The public surface roughly doubles at the boundary: six new functions and
  three new protocols. Mitigated by every one of them being a shell — the
  logic they share is the logic that was already there.
- Two sink protocols where a reader might expect one.
- `FallbackModelCaller` now exists twice. The fallback *policy* is four lines
  of branching on `retryable`; duplicating it is cheaper than the indirection
  needed to share it across the sync/async divide, but it is duplication and a
  change to one must be made to the other.

### Risks

- A future contributor adds a field to one shell and not the other. Mitigated
  structurally — the shells hold no logic to diverge in — and by tests
  asserting the sync and async paths return equal `ModelResult`s from the same
  recorded response.
- Async tests need an event loop. `pytest-asyncio` becomes a dev dependency;
  it does not enter the runtime dependency set, and no test may reach a
  network, exactly as before.
