# ADR-0021: Credentials Are a Strategy, Not an Environment Variable Name

**Date:** 2026-08-15  
**Status:** accepted  
**Deciders:** Project maintainers

## Context

`Provider` encodes authentication as two fields — `key_env: str | None` and
`key_required: bool` — and `key_present(spec)` answers "can this call be made"
by reading that one variable. The design assumed every provider authenticates
with a single API key held in the environment, which was true for all six
entries in the table.

Vertex AI breaks the assumption. It authenticates with Application Default
Credentials — a service-account file, workload identity, or `gcloud` user
credentials — and addresses the model by GCP project and region rather than by
a base URL. There is no key variable to check, and "is a credential available"
is a question only the Google auth library can answer.

The pressure to bolt this on is obvious and wrong: a `vertex_project_env`
field, a `uses_adc` boolean, and a branch inside `key_present`. That is one
special case now and a second when the next non-key provider appears, in a
table whose value is that it has no special cases.

`key_present` also carries a contract worth preserving: it answers **without
spending a call**, so `selection.resolve_spec` can pick a provider and
`call_tool_with_fallback` can skip an unreachable one before paying for a
round trip.

## Decision

Replace the two fields with one collaborator.

```python
class Credential(Protocol):
    def available(self) -> bool: ...   # obtainable now, without a network call
    def token(self) -> str | None: ...  # the value, where an SDK needs it passed
    def describe(self) -> str: ...      # what to configure, for error messages
```

**Amendment, on implementation.** The protocol shipped with three methods, not
one. `token()` exists because the OpenAI SDK requires the key as a constructor
argument — it previously read `provider.key_env` from the environment itself,
and removing the field without replacing that access would have broken every
OpenAI-compatible provider. `describe()` exists because the error message
`"{key_env} is not set"` had no source once the field was gone. Both stay
inert data accessors: `GoogleADC.token()` returns `None` precisely because the
Google SDK must resolve and refresh ADC itself, and handing it a captured
string would expire mid-run.

Implementations live beside the provider table:

- `ApiKey(env_var)` — present when the variable is set. Covers Anthropic,
  OpenAI, Kimi, DeepSeek, Gemini.
- `Unauthenticated(env_var=None)` — always present. Covers vLLM, which does not
  authenticate by default but keeps an optional variable for deployments that
  put auth in front of it.
- `GoogleADC(project_env, location_env)` — present when a project is resolvable
  and the auth library finds credentials. Covers Vertex.

`Provider.credential: Credential` replaces `key_env` and `key_required`;
`key_present(spec)` keeps its name, signature, and no-network contract, and
becomes `spec.provider.credential.available()`.

The name `key_present` is kept deliberately. It is exported from `emissary` and
`emissary.llm` and used by both consumers; renaming it to `credential_present`
would be a breaking change to two repositories in exchange for a word. The
docstring carries the correction.

Credentials stay a *check*, not a *client factory*. Each wire still builds its
own SDK client, because how a credential is presented is an SDK detail and
belongs behind `llm/wire/` per ADR-0009. `Credential` answers one question.

## Alternatives Considered

- **Add Vertex-specific fields to `Provider`:** the special case this ADR
  exists to avoid, and it puts GCP vocabulary in a table five other providers
  read. Rejected.
- **Make `Credential` produce the client or the auth header:** would pull
  `google-auth` and SDK construction into the neutral layer, inverting the
  dependency direction ADR-0009 fixes and making `tests/test_architecture.py`
  fail by design. Rejected.
- **Drop the pre-flight check and let the call fail:** simpler, but discards
  the property that fallback selection can skip an unconfigured provider
  without spending a request — and turns a configuration mistake into a
  latency-and-billing event. Rejected.
- **Keep `key_env` and let `GoogleADC` set it to `None`:** `key_present` would
  then return `True` for Vertex whenever ADC is absent, reporting a provider as
  reachable when it is not — the silent-wrong-answer failure mode this package
  exists to prevent. Rejected.

## Consequences

### Positive

- Providers that authenticate differently are table entries, not branches.
- `key_present` keeps its exported name, signature, and no-network guarantee,
  so neither consumer changes.
- Vertex's project and region resolution has one owner instead of being spread
  between the table and the wire.

### Negative

- One more small protocol in a package that deliberately has few.
- `google-auth` is imported inside `GoogleADC.available()`, which is a
  provider-SDK import outside `llm/wire/` and therefore a hole in ADR-0009.
  It cannot move into a wire: `key_present` is reached from `provider.py`,
  and a provider importing a wire would invert the dependency direction. The
  exemption is made explicit in `tests/test_architecture.py` and pinned by a
  second test asserting it admits exactly one module and nothing more, so the
  hole cannot widen unnoticed.
- `Provider.key_env` survives as a read-only property over the credential.
  Kept because the field was public and used in error paths; it now returns
  `None` for Vertex, which is honest — there is no single variable.

### Risks

- `GoogleADC.available()` can be slow or can touch the metadata server in some
  environments, which would violate the no-network contract. Mitigated by
  checking only for a resolvable project and locally discoverable credentials,
  and by treating any exception as "not available" rather than propagating.
