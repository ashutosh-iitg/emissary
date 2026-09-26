# ADR-0027: Jev Gets a Native Wire, Behind `call_choice`

**Date:** 2026-09-27  
**Status:** accepted  
**Deciders:** Project maintainers

## Context

TypeSafe AI's Jev is a "System One" model. It reads a state, answers typed questions
about it (yes/no, choice or score), and returns probabilities that were trained to be
calibrated. It is fast (70–500 ms) and bills input tokens only. It is not a chat model:
it is called at `POST https://api.typesafe.ai/v1/systemone` with
`{model, state, questions}` and returns `answers.<key>.probabilities`. Its docs are at
https://docs.typesafe.ai/api, verified 2026-09-27.

Emissary already has the matching contract. `call_choice` returns a probability over a
fixed label set, and its reason for existing is that a threshold needs a real
probability rather than a self-report. Until now only providers with `logprobs` could
serve it, and it was read off a single token, which is why labels had to differ in their
first token.

## Decision

- **A fourth wire, `llm/wire/typesafe.py`, serving only `call_choice`/`acall_choice`.**
  ADR-0020 grants a native wire when the compatibility layer is lossy. Here there is no
  compatibility layer to be lossy: the endpoint is not chat completions.
- **The mapping.** The system text becomes the question's `instructions`, the blocks
  become the `state`, and each label is a `choice` option described by itself. The call
  shape is unchanged, so stria and doom can point an existing `call_choice` at
  `typesafe` without code changes.
- **A new capability, `calibrated_choice`.** `call_choice` admits a provider with it or
  with `logprobs`. Jev does not return logprobs, and claiming it did would be the
  capability lie ADR-0004 exists to prevent.
- **No first-token rule on this wire.** Jev scores whole labels, so `["FLAG_VIOLENCE",
  "FLAG_SELF_HARM"]` is valid here and not on the logprob wire.
- **An answer over any other label set is a non-retryable error.** It cannot be
  thresholded against the choice posed, and asking again gives the same answer.
- **Plain `httpx`, no SDK.** TypeSafe publishes none, and one POST does not justify a
  dependency. `httpx` is declared directly and added to the architecture test's SDK set,
  so it stays inside `wire/`.
- **`chat=False`.** `call_model` and `call_tool` refuse `typesafe` with a
  `CapabilityError`.
- **The default model is `jev-latest`, TypeSafe's documented alias.**

## Alternatives Considered

- **Expose Jev's full question map (several questions, `noul`, `score`) as a new call
  type.** Richer, but no caller exists. The one real use, a SAFE/FLAG screen, is already
  `call_choice`. Rejected until a caller needs it.
- **Reach Jev through OpenRouter's listing.** Rejected: routing a typed-decision API
  through a chat router has no verified mapping for criteria or probabilities.
- **Reuse `logprobs=True` for Jev.** Rejected: it would admit Jev by claiming a
  capability it does not have.

## Consequences

### Positive

- `call_choice` gains a calibrated backend that needs no self-hosted GPU. It is the
  natural provider for doom's screen (`DOOM_SCREEN_PROVIDER=typesafe`).
- The providers-to-wires ratio test still holds (12 ≥ 2·4 + 1).

### Negative

- A fourth adapter, and the first whose HTTP code emissary owns rather than an SDK.

### Risks

- Labels are sent as their own descriptions. Jev may do better with real descriptions,
  and `call_choice` has no field for them yet. The question text is where a caller
  explains the labels meanwhile.
- The wire was tested against a mock transport built from TypeSafe's published examples.
  It is unverified against a live key.
