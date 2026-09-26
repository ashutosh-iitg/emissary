# ADR-0018: Reasoning Output Is Two Separate Concepts

**Date:** 2026-08-15  
**Status:** accepted  
**Deciders:** Project maintainers

## Context

"Log the model's thinking" reads like one feature. It is two, and conflating
them produces a wrapper that either loses audit data or breaks multi-turn tool
calling outright.

**Reasoning text** is human-readable prose the provider is willing to show:
Anthropic's summarized thinking blocks, DeepSeek's and Kimi's
`message.reasoning_content`, Gemini's thought parts under `includeThoughts`.
It exists for operators — logs, evaluation, debugging a trajectory. Nothing
downstream depends on it.

**Reasoning state** is a provider-issued payload that the *next request must
echo back or the call fails*. This is true on **all three wires**:

- **Anthropic** returns `thinking` blocks carrying a `signature` and
  `redacted_thinking` blocks carrying `data`. The API rejects tampering, and
  stripping the blocks from a follow-up turn can trigger ordering/signature
  400s.
- **Gemini 3+** rejects a tool-calling follow-up that omits
  `thought_signature` with a 400. The OpenAI-compatibility layer does not carry
  it at all.
- **DeepSeek V4 and Kimi K2.6+**, on the OpenAI-compatible wire, require
  `reasoning_content` to be present on every assistant message in the history
  once thinking mode is on: *"`reasoning_content` in thinking mode must be
  passed back to the API."* The first turn always succeeds; the second returns
  400. The documented root cause is generic — *most OpenAI-compatible clients
  silently strip unknown fields when they rebuild history* — and it is filed
  against many independent tools.

**Emissary is currently one of those clients.** `_openai_messages()` rebuilds
an assistant turn as `{"role": "assistant", "content": message.text}`, because
`AssistantMessage` has nowhere to keep `reasoning_content`. A DeepSeek or Kimi
agent loop therefore fails on its second turn. This is a live defect, not a
hypothetical.

This is not a display concern. It is a correctness obligation on exactly the
code path emissary's harness exists to drive — the multi-turn tool loop. A
design that treats reasoning as "text we might log" builds a harness that
cannot run a tool-calling agent on **any** provider that reasons.

The two also have opposite lifetimes. Reasoning text is safe to summarize,
truncate, redact, and drop. Reasoning state is safe to do none of those things
to; the only valid operations are *keep it exactly* or *do not send it at all*.

## Decision

Model them as two independent fields with different rules.

```python
@dataclass(frozen=True)
class ReasoningState:
    """Opaque provider-issued reasoning the next request must echo back.

    `blocks` is never inspected by anything outside the wire named in `wire`.
    """

    wire: str
    blocks: tuple[dict[str, Any], ...]
```

- `ModelResult.thinking: str | None` — reasoning text. Recorded in the
  `model_call_completed` event. **Never** projected into a model-visible
  message: a summary is not what the provider issued, and replaying it as if
  it were is the failure this ADR exists to prevent.
- `ModelResult.reasoning: ReasoningState | None` and
  `AssistantMessage.reasoning: ReasoningState | None` — reasoning state,
  carried through the event log and replayed by the wire verbatim.

Per wire, `blocks` holds whatever that wire must send back:

| wire | `blocks` contents | replayed as |
|---|---|---|
| anthropic | `thinking` / `redacted_thinking` blocks, signatures intact | leading content blocks of the assistant turn |
| gemini | parts carrying `thoughtSignature` | the assistant `Content` parts |
| openai-compatible | `[{"reasoning_content": "…"}]` | `reasoning_content` on the assistant message |

On the OpenAI-compatible wire the same string is both the readable text and the
round-trip obligation, so it is stored twice — once in `thinking`, once in
`reasoning.blocks`. That redundancy is deliberate: the two fields have opposite
rules, and a redaction policy that truncates `thinking` for a log must not
silently truncate the value the next request depends on.

`ReasoningState` is tagged with the **wire** that issued it, and a wire ignores
state it did not issue. That is what makes ADR-0002's single fallback attempt
safe: when an Anthropic call fails over to an OpenAI-compatible provider, the
Anthropic thinking blocks are dropped rather than forwarded as an unrecognised
payload. Within a wire the state is always kept, because dropping it there is
the thing that provokes the 400.

Recording reasoning text is unconditional — the provider volunteered it, and a
factual record never suppresses what arrived. Whether to *request* it is
ADR-0019.

## Alternatives Considered

- **One `thinking: str` field carrying both:** the signature is not text and
  cannot survive a string round-trip. It would work until the first
  tool-calling turn on Gemini 3+ or a signature-checked Anthropic follow-up,
  then fail as a provider 400 with no local explanation. Rejected.
- **Project reasoning text into `AssistantMessage.text`:** re-sends a summary
  in place of the real blocks. Corrupts the trajectory silently — the model
  sees prose it never wrote. Rejected.
- **Let the harness strip reasoning state before re-sending:** this is the
  current implicit behaviour and it is why the harness cannot support Gemini
  tool loops today. Rejected.
- **Type `blocks` per provider:** couples the neutral layer to SDK payload
  shapes, which is exactly what `llm/wire/` exists to contain. Opaque
  `dict` round-tripping keeps the dependency direction of ADR-0009 intact.
  Rejected.

## Consequences

### Positive

- Multi-turn tool calling becomes correct on Anthropic and possible on Gemini.
- Reasoning text is available for logging and evaluation without any risk of it
  re-entering the conversation.
- The wire tag makes cross-provider fallback safe by construction rather than
  by a rule someone must remember.

### Negative

- `AssistantMessage` grows a field that most consumers will never set, and
  whose contents they must not touch.
- Two similarly-named fields invite confusion. Mitigated by naming (`thinking`
  = text, `reasoning` = state) and by the docstring stating the round-trip
  obligation.

### Risks

- A wire that captures state but forgets to replay it fails only on the second
  turn of a tool loop, which is not covered by a single-call test. Mitigated by
  a two-turn wire test per wire asserting the blocks are sent back byte-identical.
