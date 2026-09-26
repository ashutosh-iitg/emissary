# ADR-0003: Use Discriminated Model Decisions

**Date:** 2026-08-14  
**Status:** accepted  
**Deciders:** Project maintainers

## Context

A model turn may complete, request tools, or refuse. Optional fields permit contradictory and partially initialized states.

## Decision

Represent decisions as the disjoint union `FinalOutput | ToolCalls | Refusal`, wrapped by one provenance-bearing `ModelResult`.

## Alternatives Considered

- **One dataclass with optional fields:** compact, but invalid combinations become runtime convention. Rejected.
- **Raw provider dictionaries:** flexible, but moves provider parsing into every caller. Rejected.

## Consequences

### Positive

- Runner handling is exhaustive and each variant validates its own invariants.

### Negative

- More small public types.

### Risks

- Variant proliferation; add one only for a genuinely distinct control outcome.

