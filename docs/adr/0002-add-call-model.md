# ADR-0002: Add call_model and Preserve Specialized APIs

**Date:** 2026-08-14  
**Status:** accepted  
**Deciders:** Project maintainers

## Context

`call_tool` forces one structured tool and `call_choice` reads logprobs. An agent turn must also represent final output and multiple tool calls.

## Decision

Add a general normalized `call_model` operation. Preserve `call_tool`, `call_choice`, and their result contracts.

## Alternatives Considered

- **Widen `call_tool` into a result union:** fewer entry points, but breaks consumers' guaranteed dictionary payload. Rejected.
- **Implement agents by repeatedly forcing one synthetic tool:** avoids a new wire operation, but cannot faithfully represent native final output or multiple calls. Rejected.

## Consequences

### Positive

- Existing consumers remain sound.
- Agent semantics map directly to provider capabilities.

### Negative

- Adapters expose one additional operation.

### Risks

- Shared helper refactors could change old behavior; existing tests remain mandatory compatibility gates.

