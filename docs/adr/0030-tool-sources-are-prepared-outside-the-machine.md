# ADR-0030: Tool Sources Are Prepared Outside the Machine

**Date:** 2026-10-01

**Status:** accepted

**Deciders:** Project maintainers

**Amends:** ADR-0006, ADR-0024

## Context

Tools were a static tuple, and the machine built its registry from it. Tools
discovered from an MCP server need a connection, authentication and a
discovery round trip before the first model call, and they must be closed when
the run ends. A direct tool and an MCP tool are the same kind of capability:
MCP is a different discovery and invocation mechanism, not a different class
of tool.

## Decision

**`Agent.toolsets` holds `ToolSource`s, which are configuration until opened.**
`harness/tooling/preparation.py` opens every source on one exit stack, requires a
complete catalog or fails, freezes it into a `PreparedRun`, and closes in
reverse order — including when a later source fails. The machine receives the
finished registry and stays free of I/O (ADR-0024). The harness imports no MCP
code; it sees two protocols in `harness/tooling/sources.py`.

**Origin is stored beside the tool, not parsed from its name.** `ToolOrigin`
(source, original name, endpoint, contract fingerprint) is what authorization
and routing consult. An alias is only a model-facing label.

**One machine for every mix.** Direct-only, source-only and mixed agents run
the same loop. A `RoutingToolExecutor` sends source tools to the bundled async
executor (no injected `idempotency_key`) and direct tools to the caller's
executor, so `executor=` customises direct tools only.

**Async tools need `arun`.** `AsyncLocalToolExecutor` awaits coroutine results.
`run` rejects sources and coroutine tools with a pointer to `arun`: a hidden
event loop per call is a worse surprise than an error.

**Preparation is part of the run.** Under `arun`, `run_started` is recorded
before discovery, preparation shares the run deadline, and failure ends the log
with `run_stopped / preparation_failed`. `prepare(agent)` plus `arun(...,
prepared=)` shares connections across runs; a prepared run must not cross
tenants or authentication identities.

**The MCP adapter lives in `emissary/mcp/`** behind the `emissary[mcp]` extra,
pinned to the tested SDK minor. It is the only place SDK types appear
(architecture test). Names are `namespace__tool`, with a sanitised prefix plus a
stable hash when provider limits require it; collisions fail preparation and
are never resolved by discovery order. Remote schemas keep their original form
for validation, only local `$ref`s and known dialects are accepted, and the
OpenAI-compatible wire is told not to force strict mode on them.

## Alternatives Considered

- **A separate MCP runner:** duplicates policy, retries and projection.
- **Only an MCP executor:** leaves discovery and lifetime to every client.
- **Discovery inside the machine:** puts I/O back into the policy core.

## Consequences

- A required source that is down stops the run before any model spend.
- Catalogs are frozen per run; there is no cross-run discovery cache.
- The effect union gains no I/O effect for preparation, only `AuthorizeTool`
  (ADR-0031).
