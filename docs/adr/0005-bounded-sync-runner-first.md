# ADR-0005: Build a Bounded Synchronous Runner First

**Date:** 2026-08-14  
**Status:** accepted  
**Deciders:** Project maintainers

## Context

The package and current callers are synchronous. Agent loops need control semantics before concurrency or graph infrastructure.

## Decision

Build one finite synchronous state machine first, with explicit turn and tool budgets. Add async only for a demonstrated consumer requirement.

## Alternatives Considered

- **Async-first:** enables concurrency, but doubles lifecycle complexity before the state contract is proven. Rejected for Phase 1.
- **Graph-first:** provides composition, but imposes a framework before a useful loop exists. Rejected.

## Consequences

### Positive

- Small testable kernel aligned with existing consumers.

### Negative

- No streamed turns or concurrent tools initially.

### Risks

- Sync contracts could impede async later; state and event types therefore remain transport-neutral.

