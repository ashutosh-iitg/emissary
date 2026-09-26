# ADR-0004: Fail Explicitly on Unsupported Capabilities

**Date:** 2026-08-14  
**Status:** accepted  
**Deciders:** Project maintainers

## Context

Models sharing an API shape do not necessarily support the same tools, structured output, parallel calls, or logprobs.

## Decision

Declare conservative capabilities and reject unsupported requested operations explicitly. Runtime incompatibilities become typed non-retryable errors.

## Alternatives Considered

- **Assume every wire feature works:** simple, but converts capability mismatch into remote failure. Rejected.
- **Silently emulate missing features:** broadens apparent support, but produces incomparable and potentially unsafe semantics. Rejected.

## Consequences

### Positive

- “Any model” has an honest, inspectable meaning.

### Negative

- Conservative declarations may require explicit overrides for capable models.

### Risks

- Capability metadata can drift; wire tests and documentation must evolve together.

