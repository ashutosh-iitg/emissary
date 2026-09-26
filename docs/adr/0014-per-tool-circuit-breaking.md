# ADR-0014: Per-Tool Circuit Breaking

**Date:** 2026-08-15  
**Status:** accepted  
**Deciders:** Project maintainers

## Context

The runner's only failure budget is global: `max_consecutive_tool_errors`
across all tools. One flaky tool can therefore burn the whole run's error
budget — or, worse, alternate with successful calls to other tools so the
consecutive counter resets and the run loops on a broken tool until
`max_turns`. The failure signal exists per call but is not accumulated per
tool.

## Decision

The runner tracks consecutive failures per tool name. When a tool crosses its
threshold (a per-tool limit in `RunLimits`, with a sane default), its circuit
opens for the remainder of the run: subsequent calls to it are not executed and
return an error observation telling the model the tool is unavailable, and a
`tool_circuit_opened` event records the transition. The global consecutive-error
limit remains as a backstop. Circuits do not reset within a run.

## Alternatives Considered

- **Global counter only (status quo):** cannot isolate one bad tool; permits
  the alternating-success loop described above. Rejected.
- **Withdraw the tool's schema from subsequent turns:** stops the model
  proposing it, but silently mutates the model-visible action space mid-run;
  under ADR-0011 that is a logged context mutation with its own design
  questions. Deferred, not chosen for v1.
- **Half-open retry after cooldown:** classic circuit-breaker refinement, but
  within a bounded synchronous run there is no meaningful cooldown clock.
  Rejected for v1.

## Consequences

### Positive

- A single flaky tool degrades gracefully instead of consuming the run.
- The model gets an explicit, actionable observation and can route around the
  dead tool.

### Negative

- Another limit to configure; defaults must be forgiving enough not to trip on
  legitimate model-driven retry-after-fix patterns.

### Risks

- If the model's task depends entirely on the broken tool, an open circuit
  converts a fast failure into several wasted turns before `max_turns`;
  evaluation metrics should surface runs that continue past an open circuit
  without progress.
