# ADR-0001: Emissary Is the Sole Model Boundary

**Date:** 2026-08-14  
**Status:** accepted  
**Deciders:** Project maintainers

## Context

The harness must work with every model supported by emissary without acquiring provider-specific branches or SDK values.

## Decision

All harness inference uses the public emissary model caller. Only wire adapters import provider SDKs.

## Alternatives Considered

- **Direct SDK calls in the runner:** expose more provider features, but couple control flow to providers and duplicate translation. Rejected.
- **Framework-specific model objects:** accelerate one integration, but make that framework the real abstraction boundary. Rejected.

## Consequences

### Positive

- New models on existing wires require no harness changes.
- Runner tests use a fake caller without SDKs or network.

### Negative

- Provider features require normalization before harness use.

### Risks

- Leaks can reappear through convenience code; architecture import tests will forbid them.

