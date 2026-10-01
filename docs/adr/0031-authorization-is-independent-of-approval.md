# ADR-0031: Authorization Is Independent of Approval

**Date:** 2026-10-01

**Status:** accepted

**Deciders:** Project maintainers

**Amends:** ADR-0024

## Context

`approval="never"` returned ALLOW immediately, and nothing asked whether this
principal may make this exact request. Tools reached through a source are
third-party code; the harness cannot assume the registered catalog is what the
user is entitled to call.

## Decision

**Authorization is its own effect, `AuthorizeTool`.** The application supplies
an `Authorizer` (sync or async) and a trusted `AuthorizationContext`
(principal, tenant, nonsecret references). Model arguments cannot set either.

**Sequence.** Resolve, snapshot and schema-validate the whole batch; authorize
the whole batch before any execution; resolve approval; then authorize again
immediately before every attempt, retries included. `approval="never"` never
skips authorization, and approval cannot override a denial. A denial in the
batch stops it with no effects. After execution starts, batches are not
transactions and no atomicity is claimed.

**The request is a snapshot.** `InvocationRequest` holds canonical JSON bytes,
not a dict, so no callback can change what runs. Callbacks receive detached
views; the built-in dispatcher rebuilds arguments from the snapshot and
verifies the decision (digest, attempt, expiry) immediately before delegating.
The digest covers source, fingerprint, arguments, trusted metadata, principal,
tenant and run/call identity; the decision additionally binds attempt, expiry
and policy version.

**Fail closed.** A policy that raises, returns a decision for another request,
or returns an expired one stops the run (`authorization_error`). A denial is
`authorization_denied`.

**Compatible default.** With no authorizer, the machine delegates the
registered catalog (`AllowRegisteredTools`) without an effect or new events, so
existing logs and golden fixtures are unchanged. Runs with tool sources record
that delegation on `tools_prepared`. This is not resource authorization.

**Replay.** `RecordedAuthorizer` answers from recorded `authorization_resolved`
events, so replay needs no live policy, credential or server.

## Limits

Injected executors and direct callables remain trusted extension code. Exact
arguments do not resolve backend state races: contextual constraints must also
be enforced atomically by the API or database performing the operation. Signed
remote grants are a later optional adapter.
