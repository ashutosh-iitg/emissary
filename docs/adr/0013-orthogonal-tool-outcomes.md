# ADR-0013: Report Orthogonal Tool Outcomes Independently

**Date:** 2026-08-15  
**Status:** accepted  
**Deciders:** Project maintainers

## Context

`ToolResult.status` is a single severity axis: `success | warning | error`.
Upcoming work adds outcomes that are not severities — timeout (blueprint
Step 4), retryability (ADR-0012), truncation of oversized output (Step 8).
Folding these into `status` conflates independent dimensions: a timed-out call
that still wrote partial state is not the same "error" as a validation
rejection, and consumers matching on status strings cannot tell them apart.
DeepSeek Harness states this as a hard defensive rule — one of its postmortems
traces a real incident to a misclassified child failure folded into a parent
status.

## Decision

Each independent outcome dimension gets its own field on `ToolResult`.
`status` remains the single severity axis and never grows new values for
non-severity facts. First applications: an orthogonal `timed_out: bool` when
Step 4's timeout work lands, and `retryable: bool` alongside ADR-0012. Events
mirror the same fields, so trajectories preserve the distinction.

## Alternatives Considered

- **Expand the status enum (`timeout`, `retryable_error`, …):** every new
  dimension multiplies the enum combinatorially and breaks every existing
  consumer match. Rejected.
- **Encode qualifiers in the summary text:** human-readable but not
  machine-checkable; policy and evaluation code would parse prose. Rejected.

## Consequences

### Positive

- Policy, evaluation, and circuit-breaking logic can react to exactly the
  dimension they care about.
- Existing consumers matching on `status` keep working unchanged.

### Negative

- `ToolResult` grows fields over time and each needs a documented default that
  keeps old constructors valid.

### Risks

- Field proliferation without discipline; a new field must represent a genuinely
  independent dimension, not a convenience flag — the test is whether it can
  co-occur with every status value.
