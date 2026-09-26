# ADR-0024: One Loop, Two Drivers — a Sans-I/O Runner Core

**Date:** 2026-08-16  
**Status:** accepted  
**Deciders:** Project maintainers  
**Amends:** ADR-0005 · **Completes:** ADR-0023

## Context

ADR-0023 made the model boundary async and left the runner synchronous,
recording that an async runner is *not* a parallel `arun` beside `run`. This
decides what it is instead.

`runner.py` is 171 lines holding every policy the harness has: turn limits,
token limits, batch-validate-before-execute ordering, per-tool circuit
breaking, idempotent retries, approval resolution, six terminal conditions, and
the re-fold that keeps the model-visible surface coming from one code path.
None of that is I/O. Three things in it are:

- `caller(...)` — one model turn
- `executor.validate(...)` — admission checking, which a remote executor
  performs remotely
- `executor.execute(...)` — the tool itself

Copying the loop for async would duplicate all the policy to gain the three
awaits. Every future change — a new limit, a new terminal condition — would
have to be made twice, and the second copy is the one that gets forgotten. The
bug that follows is invisible in review: two runners that agree on the tests
written for the first one.

The wires already solved the same problem. ADR-0022 and ADR-0023 factored each
adapter into a pure `_request`/`_normalize` core with sync, async, streaming,
and async-streaming shells over it. That shape is available here for the same
reason it was available there.

## Decision

**The loop becomes a generator that yields effects and receives their
outcomes. `run` and `arun` are drivers over it.**

```python
Effect = CallModel | ValidateTool | ExecuteTool

def agent_machine(agent, task, *, run_id, ...) -> Generator[Effect, Any, RunResult]
```

The machine holds all state and all policy. It never calls a caller, an
executor, or anything awaitable. A driver receives an effect, performs it, and
sends the outcome back. Both drivers are the same seven lines apart from one
`await`.

Four decisions inside that shape:

**Errors travel by `throw`, not by a result union.** When a `ProviderError`
escapes the caller, the driver throws it into the generator, which catches it
around the `yield`. The machine's error handling therefore reads exactly as it
does today — `try: … except ProviderError:` — rather than becoming a match on
an outcome wrapper. The alternative spreads one control-flow concern across
both driver and machine.

**Validation is an effect, not a pure helper.** `ToolExecutor.validate` is on
the protocol beside `execute`, and a sandboxed or remote executor performs both
remotely. Treating it as pure would put I/O back in the machine for the one
executor that happens to be local today. Keeping it an effect also preserves
the batch-validate-before-execute ordering: the machine yields every
`ValidateTool` for a turn before yielding any `ExecuteTool`, which a driver
cannot reorder.

**Event emission stays inside the machine.** `EventSink.emit` returns `None`
and is synchronous, so an async driver can call it without blocking on
anything awaitable — the same standing as a `StreamSink` under ADR-0022. Making
emission a fourth effect would buy async sinks, which nothing asks for, at the
cost of a yield per event. If an `AsyncEventSink` is ever wanted, `Emit`
becomes an additive effect and the drivers grow one branch each. Recorded here
so that is a decision rather than a discovery.

**Approval stays a synchronous collaborator.** An approver that must ask a
human is already served by the designed path: no approver means `PAUSE`, the
run stops with `APPROVAL_REQUIRED`, and the caller resumes later. Adding an
`Approve` effect to let an async runner await a UI would duplicate a mechanism
that exists.

**`run_id` moves into the machine's signature.** `uuid.uuid4()` is
nondeterminism, and a pure core should not contain it; the drivers generate it.
This also lets a test drive the machine with a fixed id and get byte-stable
events.

## Alternatives Considered

- **Copy the loop into `arun`:** the outcome ADR-0023 already rejected. Two
  implementations of six terminal conditions, guaranteed to drift, and the
  drift is invisible because the tests were written against the first copy.
  Rejected.
- **`asyncio.to_thread(run, ...)`:** one line, and it does give an async-looking
  API. But the whole run occupies a thread for its full duration — twelve turns
  of network latency — so concurrency is capped by the thread pool, which is
  precisely what an async caller was added to escape. Rejected.
- **An `AgentLoop` class with overridable `_call_model` hooks:** the classic
  template-method answer. It shares the policy, but the async subclass must
  return awaitables from methods the sync base calls directly, so either the
  base awaits (breaking sync) or the subclass blocks (breaking async). The
  colour of the function leaks through inheritance. Rejected.
- **Make everything an effect, including emission and approval:** maximally
  pure and genuinely tidy, but it adds two effect types and two driver branches
  to serve no current caller. Deferred, with the extension point named above.
- **Rewrite the loop as an explicit state machine with a `step()` method:**
  more portable than a generator and testable without driving, but it turns
  local variables into fields and control flow into a transition table — a
  large legibility cost for a loop that is genuinely sequential. Rejected.

## Consequences

### Positive

- One implementation of every policy. A new limit or terminal condition is
  written once and both drivers inherit it.
- The machine is testable with no caller, no executor, and no event loop: send
  it effects' outcomes and assert the events it emits. Deterministic given a
  fixed `run_id`.
- `arun` is genuinely thin, so reviewing it is reviewing seven lines rather
  than a second runner.
- ADR-0015 replay keeps working unchanged, because it supplies a caller and an
  executor to `run` exactly as before.

### Negative

- Reading the loop now requires understanding generator `send`/`throw`, which
  is a real cost for a contributor who has not met the pattern. Mitigated by
  keeping the machine's body a straight-line translation of today's loop —
  every policy line is where it was, with `yield` in place of the call.
- Three modules where there was one (`effects.py`, `machine.py`, `runner.py`).
- A driver that fails to send an outcome back hangs the machine rather than
  raising. Both bundled drivers are exhaustive over the effect union.

### Risks

- **Behaviour drift during the move.** This is the whole risk, and the
  mitigation is that `tests/test_runner.py` is not modified: the existing
  suite pins event order, terminal statuses, retry counts, circuit behaviour,
  and the exact `ToolMessage` rendering, and it must pass untouched. A new
  async suite asserts `arun` produces the identical event trajectory for the
  same scripted inputs.
- An effect union that grows past a handful would make the drivers no longer
  thin, which would be the signal that this shape has stopped paying for
  itself.
