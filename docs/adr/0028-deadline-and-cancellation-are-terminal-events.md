# ADR-0028: The Deadline and Cancellation Are Terminal Events

**Date:** 2026-09-28  
**Status:** accepted  
**Deciders:** Project maintainers  
**Amends:** ADR-0024

## Context

Every run limit ends in a typed `RunResult` with a logged `run_stopped` — except
`max_duration_seconds`, which raised `TimeoutError` out of both drivers, and
async task cancellation, which propagated with no terminal event. Both left the
event log, the canonical trajectory (ADR-0008), ending mid-turn, and contradicted
the architecture's "`RunResult` always includes status and stop reason".

## Decision

**The deadline is a budget.** A driver that finds it spent throws `RunTimedOut`
into the machine, which stops with `STOPPED / MAX_DURATION`. Like every count
budget, it is checked before dispatch: it stops new work and never discards a
result that has already returned. `arun` additionally bounds each await, and
converts only its own timer's expiry — a `TimeoutError` raised by a caller or
tool propagates as before.

`RunTimedOut` subclasses `RunCancelled`: a timeout is a cancellation by the
clock, and the subclass lets every existing catch site and driver tuple carry
it unchanged, with one mapping in the machine choosing the status.

**Async cancellation records, then propagates.** On `CancelledError`, `arun`
throws `RunCancelled` into the machine so the log ends on `run_stopped`
(`cancelled`), then re-raises. It does not return a `RunResult`.

**ADR-0024 corrections.** The effect union is four (`WaitRetry` joined with
idempotent retries), and drivers now raise on an effect they do not know
rather than falling through to `ExecuteTool`. A fixed `run_id` makes the log
stable apart from `occurred_at`; the machine still reads the clock there.

## Alternatives Considered

- **Keep `TimeoutError`:** the only limit whose end the log cannot show.
  Rejected.
- **Swallow `CancelledError` and return a `CANCELLED` result:** gives async
  parity with `run`, but breaks `TaskGroup` and outer `asyncio.timeout`, which
  rely on seeing the cancellation. Rejected.
- **A `cancel_event` parameter on `arun`:** duplicates task cancellation, the
  mechanism async callers already have, and no caller asks for it. Rejected.
- **A separate `RunTimedOut` exception:** adds it to five `except` clauses for
  no difference in behaviour. Rejected in favour of the subclass.

## Consequences

- Callers that caught `TimeoutError` from `run`/`arun` must read
  `result.stop_reason is StopReason.MAX_DURATION` instead. Breaking; check
  `stria` and `doom`.
- A run whose last synchronous effect overran the deadline but returned a final
  answer now completes rather than raising.
