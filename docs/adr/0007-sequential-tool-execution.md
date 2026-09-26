# ADR-0007: Execute Tool-call Batches Sequentially First

**Date:** 2026-08-14  
**Status:** accepted  
**Deciders:** Project maintainers

## Context

Providers can emit multiple tool calls, but parallel execution is unsafe when calls have ordering dependencies or side effects.

## Decision

Preserve all calls and IDs, validate the full batch first, then execute sequentially in response order. Concurrency requires a later ADR and explicit independence policy.

## Alternatives Considered

- **Execute concurrently by default:** lower latency, but unsafe without effect/dependency knowledge. Rejected.
- **Keep only the first call:** simple, but corrupts the model's decision. Rejected.

## Consequences

### Positive

- Deterministic effects and future-compatible data model.

### Negative

- Independent calls take longer.

### Risks

- Models may assume parallel semantics; observations preserve ordered completion explicitly.

