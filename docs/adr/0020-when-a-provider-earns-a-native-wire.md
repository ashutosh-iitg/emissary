# ADR-0020: When a Provider Earns a Native Wire

**Date:** 2026-08-15  
**Status:** accepted  
**Deciders:** Project maintainers

## Context

`CLAUDE.md` records a load-bearing constraint: *two wire formats, not N
integrations* — "if it ever becomes six real integrations, the abstraction has
stopped paying for itself and should be reconsidered rather than extended."
That decision asks not to be re-opened without new information.

There is new information, and it points in two directions at once.

**Gemini has a real gap.** Gemini 3+ requires a `thought_signature` on
function calls in a multi-turn conversation and returns a 400 when it is
missing. The OpenAI-compatibility layer does not carry thought signatures.
Emissary's harness exists to drive multi-turn tool loops; through the
compatibility shim, it therefore cannot run an agent on Gemini at all. Vertex
compounds it: it is a different endpoint with GCP project/region addressing and
ADC credentials rather than an API key, which the current `Provider` table
cannot express (ADR-0021).

**DeepSeek and Kimi have no gap.** Their published, first-party API *is*
OpenAI-compatible — `https://api.deepseek.com/v1/chat/completions` and
`https://api.moonshot.ai/v1/chat/completions` are the native interfaces, not
shims over something else. Both additionally offer an Anthropic-compatible
endpoint, which is a second door onto the same model, not a third protocol.
Writing "native" adapters for them would produce two new modules that issue the
same HTTP request the existing adapter already issues.

Crucially, everything they need is *expressible on that wire and already
reachable through the SDK*: DeepSeek's `thinking: {"type": …}` rides in
`extra_body`, Kimi's `reasoning_effort` is a top-level field, and both return
`reasoning_content` on the standard message object. What emissary is missing is
not an adapter — it is per-provider request translation (ADR-0019) and
reasoning round-tripping (ADR-0018), both of which are changes *inside* the
existing adapter. A native wire would not have made either easier.

So the question is not "how many wires may we have" but "what evidence obliges
a new one".

## Decision

**A provider earns a native wire when the compatibility layer is lossy for a
capability the harness depends on — not when the vendor publishes one.**

Applying the criterion:

- **Gemini / Vertex: earns one.** Thought signatures are required for
  multi-turn tool calling and absent from the compatibility layer. Add
  `llm/wire/gemini.py` speaking `generateContent`, serving two providers
  (`gemini` for the Developer API, `vertex` for Vertex AI) that differ in
  credential and address, not in protocol.
- **DeepSeek, Kimi: do not.** No capability is lost. They stay on the
  OpenAI-compatible wire, and the work they actually need is provider-specific
  translation *within* that shared wire — a thinking dialect (ADR-0019) and
  `reasoning_content` round-tripping (ADR-0018) — not a new adapter.

Two supporting changes make a third wire additive rather than invasive:

- **A wire registry.** `llm/model.py` and `llm/calls.py` currently branch on
  `if spec.provider.wire == "anthropic"`. With three wires that branch is an
  open/closed violation that grows per wire. Replace it with
  `WIRES: dict[str, Wire]` keyed by `Provider.wire`, so adding a wire is a
  table entry.
- **Capability gating stands in for partial implementations.** Not every wire
  serves every call — Gemini has no logprobs, so `call_choice` is unreachable
  there. `ModelCapabilities` already gates before dispatch (ADR-0004), so the
  method is never reached rather than being present and raising. The precondition
  belongs at the boundary, not inside the adapter.

The resulting ratio is **three wires serving seven providers**. That is still a
table. The criterion above, not the number, is what keeps it one.

## Alternatives Considered

- **Write native adapters for DeepSeek and Kimi as requested:** produces
  duplicate code paths issuing identical requests, triples the surface that
  every future change to the OpenAI-compatible wire must be applied to, and
  buys no capability. This is exactly the "six real integrations" outcome
  `CLAUDE.md` warns against, reached without the justification that would
  excuse it. Rejected — the underlying need (`reasoning_content`) is met by
  ADR-0018 on the existing wire.
- **Use the Anthropic-compatible endpoints DeepSeek and Kimi publish:** would
  route them through `llm/wire/anthropic.py`, which is a table change, not a
  new adapter. Genuinely cheap, but it buys nothing today and costs a
  second way to reach the same model. Deferred until a capability difference
  appears; recorded here so it is not rediscovered as novel.
- **Keep Gemini on the compatibility layer and document the tool-loop gap:**
  leaves a provider in the table that the harness cannot actually drive.
  A provider that only works for single-shot calls is a trap for the next
  caller who reaches for it. Rejected.
- **Adopt a third-party multi-provider library instead of a third wire:** it
  would replace a package we understand with a dependency whose fallback and
  error-classification semantics we would have to re-derive — and those
  semantics are the reason this package exists. Rejected.

## Consequences

### Positive

- Gemini and Vertex become usable for the tool loop, which is emissary's
  purpose.
- The registry makes wire count a configuration fact rather than a branching
  cost, and removes an OCP violation already present at two wires.
- The criterion is written down, so the next "vendor X has its own API"
  request is answered by evidence rather than by preference.

### Negative

- A third adapter to keep current, with its own thinking, tool, and streaming
  mappings. The neutral contracts (`ModelResult`, `ModelDecision`) bound the
  blast radius, but the maintenance is real.
- `call_choice` is unavailable on Gemini, so the provider table is no longer
  uniform across all three call shapes.

### Risks

- The criterion is a judgement, and "lossy for a capability we depend on" can
  be argued into. Mitigated by requiring the specific evidence in the ADR that
  admits the wire — for Gemini, a documented 400.
- Three wires × three call shapes × sync/async/streaming is a combinatorial
  surface. Mitigated by keeping each wire's request-building and
  response-normalising pure and separate from its I/O shell, so the variants
  differ only in the shell.
