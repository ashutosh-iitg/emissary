# ADR-0009: Enforce SOLID Through Dependency Rules

**Date:** 2026-08-14  
**Status:** accepted  
**Deciders:** Project maintainers

## Context

The harness will gain model, tool, state, policy, context, and persistence concerns. Without explicit boundaries, a runner class can become a framework-sized coordinator that changes for every feature.

## Decision

Apply SOLID as enforceable dependency direction: modules have one owner/reason to change; extension occurs through narrow protocols at external boundaries; implementations pass common contract tests; clients depend on capability-specific interfaces; high-level orchestration receives model, executor, policy, context, and event dependencies.

## Alternatives Considered

- **One configurable runner class:** fewer files, but accumulates unrelated policy and infrastructure. Rejected.
- **Abstraction for every class:** superficially “SOLID,” but increases indirection without substitution needs. Rejected.

## Consequences

### Positive

- Provider, execution, storage, and context implementations can change independently.
- Core logic remains directly testable.

### Negative

- Boundaries require deliberate contract ownership and tests.

### Risks

- Ceremony and interface proliferation; protocols are introduced only where two implementations or a test substitute exist.

