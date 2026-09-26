# ADR-0012: Idempotency Keys Gate State-Changing Tool Retries

**Date:** 2026-08-15  
**Status:** accepted  
**Deciders:** Project maintainers

## Context

Retries are currently off for state-changing tools, so the harness's only
answer to a transient failure in an external effect is to surface the error and
let the model or the run absorb it. Enabling retries without protection would
duplicate effects — the exact failure mode the fallback policy (one
availability-only attempt, never replaying completed tools) was designed to
prevent.

## Decision

A tool may opt into retries only by declaring idempotency. The executor derives
a stable idempotency key from `run_id` and the tool call's `id` — identical
across attempts of the same call — and passes it to the tool, whose
implementation (or its downstream service) uses it to deduplicate. A tool that
does not declare idempotency is never retried by the harness, regardless of how
transient the failure looks. Every retry attempt emits an event carrying the
key and attempt number.

## Alternatives Considered

- **Retry transient-looking failures blindly:** simple, but a timeout is
  indistinguishable from a slow success, so blind retry double-executes
  effects. Rejected.
- **Never retry anything (status quo):** safe but brittle; one network blip in
  an otherwise idempotent tool fails work the harness could have completed.
  Rejected as the permanent answer.
- **Attempt-scoped keys (key varies per attempt):** defeats deduplication —
  the downstream cannot recognize the retry as the same logical operation.
  Rejected.

## Consequences

### Positive

- Transient tool failures become recoverable without risking duplicate effects.
- The safety property stays structural: no idempotency declaration, no retry.

### Negative

- Idempotency becomes the tool author's obligation; the harness can pass the
  key but cannot verify the tool honors it.

### Risks

- A tool falsely declaring idempotency shifts the failure from "no retry" to
  "silent duplicate"; tool contract tests must exercise duplicate delivery of
  the same key.
