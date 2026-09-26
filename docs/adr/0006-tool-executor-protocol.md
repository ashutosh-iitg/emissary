# ADR-0006: Isolate Tool Execution Behind a Protocol

**Date:** 2026-08-14  
**Status:** accepted  
**Deciders:** Project maintainers

## Context

Local callables, sandboxes, MCP tools, and durable activities have different execution mechanics but the runner needs one policy-level operation.

## Decision

The runner depends on a narrow `ToolExecutor` protocol. A local executor is the first implementation.

## Alternatives Considered

- **Invoke callables directly in the runner:** minimal initially, but mixes orchestration, validation, policy, and effects. Rejected.
- **Universal plugin framework:** extensible, but speculative and configuration-heavy. Rejected.

## Consequences

### Positive

- Execution environments are substitutable through contract tests.

### Negative

- One indirection in the common local path.

### Risks

- An oversized protocol would violate interface segregation; keep it to execution only.

