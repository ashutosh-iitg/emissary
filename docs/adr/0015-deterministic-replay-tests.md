# ADR-0015: Deterministic Replay Tests from Recorded Event Logs

**Date:** 2026-08-15  
**Status:** accepted  
**Deciders:** Project maintainers

## Context

The runner is tested with hand-scripted model turns, which verify the state
machine branch by branch but encode the author's expectation of a trajectory,
not a real one. Once ADR-0011 makes the event log complete — every model
decision, tool call, and result reconstructable from events — a recorded log
is a full, replayable specimen of runner behavior. Without replay coverage, a
change to projection or control flow can alter what the model would have seen
in past runs while every scripted unit test stays green.

## Decision

Versioned event-log fixtures become regression tests. A replay harness feeds a
recorded log's model decisions through a replaying `ModelCaller` and its tool
results through a replaying `ToolExecutor`, re-runs the state machine, and
asserts the produced events, projected messages, and terminal outcome match
the recording exactly. Fixtures carry the event-schema version; a schema change
requires migrating or consciously retiring fixtures, never silently skipping
them. Replay tests are network-free like all others in this repository.

## Alternatives Considered

- **Scripted unit tests only (status quo):** verifies branches, not
  trajectories; blind to projection regressions. Rejected as sufficient.
- **Live-model end-to-end tests:** nondeterministic and network-bound,
  violating the repository's no-network test rule. Rejected.
- **Snapshot only the terminal `RunResult`:** misses mid-run divergence — two
  different trajectories can reach the same terminal state. Rejected.

## Consequences

### Positive

- Runner and projection changes are checked against complete real
  trajectories, not just authored expectations.
- Fixtures double as executable documentation of what runs actually look like.

### Negative

- Fixture maintenance: intentional behavior changes require regenerating
  recordings, and the diff review must distinguish intended from accidental
  divergence.

### Risks

- Fixtures rot if event schemas change casually; the schema-version gate on
  fixtures is the enforcement point, and retiring a fixture must be a visible
  reviewed act, not a skip.
