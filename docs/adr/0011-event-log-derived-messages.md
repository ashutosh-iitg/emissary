# ADR-0011: Derive Model-Visible Messages from the Event Log

**Date:** 2026-08-15  
**Status:** accepted  
**Deciders:** Project maintainers

## Context

ADR-0008 made events the canonical trajectory, but the runner does not honor it
for conversation state. `harness/runner.py` hand-threads `messages` as a mutated
tuple that is simultaneously the durable record and the model-visible view, and
`RecentHistory` trims tail history with no event recording what was dropped or
why. Two sources of truth exist, and one of them loses information silently.

This blocks the reliability work that depends on a complete factual record:
compaction (blueprint Step 8) must replace a range of history auditably;
checkpoint/resume must reconstruct exact model-visible state; deterministic
replay must re-derive what the model saw. Each becomes a retrofit if built on
the hand-threaded tuple. DeepSeek Harness independently converged on the same
answer — its `Session` is an append-only event log and `deriveMessages()` is a
pure cached projection, under the invariant "model-visible means logged."

## Decision

The run's append-only event log is the single source of truth for conversation
state. Events carry the content required for reconstruction — the task, each
model decision, each tool call and its result — not only metadata about them.
The model-visible message sequence is produced by a pure projection function
(`derive_messages`) over the log; no independently mutated message collection
exists in the runner.

Two invariants follow:

1. **Model-visible means logged.** Nothing reaches a model request that cannot
   be reconstructed from the event log alone.
2. **Omission is an event.** A context policy that trims or replaces history
   does so through a logged operation recording what was dropped or summarized
   and why. Silent trimming is a contract violation.

`RunResult.messages` remains in the public contract but is computed by the same
projection, so callers observe no behavioral change on the happy path.

## Alternatives Considered

- **Keep the hand-threaded tuple (status quo):** least immediate work, but
  preserves dual sources of truth, contradicts ADR-0008, and converts Step 8
  into a rewrite of runner internals under a live compaction feature. Rejected.
- **Keep both, cross-validate log against tuple:** detects divergence but
  institutionalizes the duplication that causes it, and doubles the surface
  every new message kind must update. Rejected.
- **Full event-sourcing framework (snapshots, upcasting, replay engine):**
  machinery for problems this library does not have; the log lives in memory
  within one bounded run and the storage layer's claims are unchanged. Rejected.

## Consequences

### Positive

- One factual record serves debugging, evaluation, compaction, and any future
  checkpoint/resume — they stop needing separate state plumbing.
- Context policies become auditable: every trim is visible in the trajectory.
- Deterministic replay tests become possible from recorded logs alone.

### Negative

- Event payloads grow from metadata to content-bearing, so the event schema
  takes on version discipline for message-bearing kinds.
- Projection runs per turn; acceptable at bounded-run scale, and cacheable by
  sequence number if it ever measures as a cost.

### Risks

- Content in events sharpens the ADR-0008 leakage risk; redaction remains sink
  policy, and the in-run log keeps full fidelity because projection requires it.
- A projection bug silently alters what the model sees; the test suite must
  assert exact projected messages for every event pattern (golden tests), not
  merely terminal outcomes.
