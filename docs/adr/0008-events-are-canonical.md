# ADR-0008: Use Events as the Canonical Trajectory

**Date:** 2026-08-14  
**Status:** accepted  
**Deciders:** Project maintainers

## Context

Agent failures require reconstruction of model turns, actions, approvals, outcomes, limits, and usage. Ad-hoc logs are neither typed nor reliable evaluation input.

## Decision

Every state transition emits an immutable ordered event. The event stream is the source for traces and trajectory evaluation.

## Alternatives Considered

- **Logs only:** easy to add, but unstable, hard to redact, and not machine-verifiable. Rejected.
- **Persist full state snapshots only:** resumable, but obscures causal transitions. Rejected.

## Consequences

### Positive

- Debugging, metrics, and evaluation share one factual record.

### Negative

- Event schemas require version discipline.

### Risks

- Sensitive content leakage; content is optional and redaction is sink policy.

