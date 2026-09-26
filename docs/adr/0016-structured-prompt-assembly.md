# ADR-0016: Structured Prompt Assembly

**Date:** 2026-08-15  
**Status:** superseded by [ADR-0017](0017-request-assembly.md)  
**Deciders:** Project maintainers

> **Superseded before implementation.** This ADR's central premise — that consumers
> need ordered *system-prompt* sections with `{{variable}}` interpolation — was
> disproved by reading the consumers. Their system prompts are entirely static and
> their real structure lives in the content blocks. ADR-0017 records the corrected
> decision. Kept for the reasoning trail.

## Context

Every consumer hand-rolls its prompts as string constants: stria's
`CRITERIA_SYSTEM`, `LAW_LAYER_SYSTEM`, and `RELATIONSHIPS_SYSTEM`; doom's
`SYSTEM_PROMPT` and `SCREENING_PROMPT`; the harness's `Agent.instructions` is a
raw string passed straight through to `call_model`. Nothing enforces shared
structure, section ordering, cache-prefix stability, or interpolation safety,
and the evaluation layer wants prompt fingerprints it currently cannot get.

DeepSeek Harness solves this with a `system-prompt` registry: named sections
concatenated in ascending numeric order bands (harness identity −100, persona
0, tool guidance 100–199), strict `{{variable}}` interpolation that throws on
unknown or valueless references ("fail loud beats shipping a malformed
prompt"), duplicate section names rejected, empty sections dropped, and
documented KV-cache consequences — the rendered prefix is cache-stable only
while sections render identically. Industry guidance converges on the same
shape: labeled sections separating role, constraints, context, and output
format, with prompt versioning treated as an engineering artifact.

## Decision

Add a provider-neutral prompt module to `emissary.llm` — the "prompt
generator" — built from immutable values:

- `PromptSection(name, order, text, cache=False)`: a named contribution.
  Duplicate names are rejected; sections render in ascending `order` with
  `name` as the deterministic tie-break (deliberately improving on DSH's
  registration-order tie-break, which its own docs list as a limitation).
- `Prompt(sections, variables)`: an immutable assembly. `render()` interpolates
  `{{variable}}` references strictly — an unknown or valueless reference
  raises — drops empty sections, and produces the provider-neutral inputs the
  existing APIs already accept (system text plus content blocks, with
  `cache=True` sections mapped to the existing `TextBlock`/block cache
  marking). Rendering is a pure function: same prompt, same output.
- `Prompt.fingerprint`: a stable content hash over the rendered form, recorded
  in run events and evaluation records so prompt changes are a visible,
  comparable experimental variable.

Adoption is compatible, not forced: `call_tool`, `call_choice`, and
`call_model` keep accepting plain strings; `Agent.instructions` accepts a
string or a `Prompt`. Consumers migrate when they want structure, caching, or
fingerprints. The harness composes its own contributions (instructions, tool
guidance) through the same type rather than string concatenation.

Explicitly out of scope, as product-scale machinery this library does not
need: DSH's plugin registry, assembly waterfall/event hooks, per-agent scoping
and section shadowing, and any template engine beyond strict variable
substitution.

## Alternatives Considered

- **Keep raw strings (status quo):** zero cost today, but no fingerprints, no
  cache discipline, and three consumers already diverging in structure.
  Rejected.
- **A template engine (Jinja et al.):** powerful, but stringly-typed logic in
  templates, a new dependency at a trust boundary, and injection surface —
  strict `{{var}}` substitution covers every current need. Rejected.
- **DSH-style registry with contribution/waterfall hooks:** the right answer
  for a plugin ecosystem; emissary has two consumers and explicit
  composition — a value object callers construct beats a mutable registry.
  Rejected.

## Consequences

### Positive

- One house shape for prompts across stria, doom, and the harness, with
  deterministic rendering and cache-aware prefixes.
- Prompt fingerprints make evaluation comparisons and event trajectories
  honest about which prompt actually ran.
- Strict interpolation converts silent template bugs into loud failures.

### Negative

- One more public surface to keep stable; migration of existing constants is
  real (if small) work in two consumer repos.

### Risks

- Scope creep toward a template engine; the boundary is that `Prompt` holds
  data and substitution only — any conditional or loop logic belongs in the
  Python that builds the sections.
- If the harness starts contributing sections (tool guidance) while consumers
  also pass full prompts, ordering conventions need documenting up front —
  order bands akin to DSH's are the likely answer and should be fixed in the
  implementation PR.
