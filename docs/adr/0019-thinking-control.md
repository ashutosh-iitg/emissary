# ADR-0019: Provider-Neutral Thinking Control, Gated on Capability

**Date:** 2026-08-15  
**Status:** accepted  
**Deciders:** Project maintainers

## Context

ADR-0018 settled what comes *back*. This settles what goes *out*: how a caller
asks for thinking through one neutral interface when the providers disagree
about whether that is even a request-time choice.

- **Anthropic** takes a request parameter. `{"type": "disabled"}` suppresses
  thinking; `{"type": "adaptive"}` enables it; `display` selects
  `"omitted"` (default) or `"summarized"`, and on current models the text is
  omitted unless asked for. Four distinguishable request states, including
  "send nothing".
- **Gemini** takes `thinkingConfig` with `thinkingLevel` (Gemini 3+;
  `thinkingBudget` on older models) and `includeThoughts`. Also four states.
- **OpenAI-compatible servers** have no *standard* parameter, but the
  individual providers on that wire each define their own, and they disagree:
  - **DeepSeek** takes `thinking: {"type": "enabled" | "disabled"}` (through
    the SDK's `extra_body`) plus `reasoning_effort`.
  - **Kimi K3** always reasons and takes a top-level `reasoning_effort` of
    `low | high | max`; there is no way to turn it off. Kimi K2.6 takes
    `thinking.keep`, K2.5 took `thinking.type`.
  - **OpenAI** reasoning models take `reasoning_effort`.
  - **vLLM** serves whatever the loaded model does; nothing is guaranteed.

An earlier draft of this ADR keyed thinking support to the *wire* and concluded
the OpenAI-compatible wire could not express it at all. That was wrong: the
wire is a message format, and thinking control is a **provider** property that
varies *within* the wire — and, for Kimi, within a provider by model
generation. Keying it to the wire would have refused a request DeepSeek accepts.

So a neutral "make the model think" boolean would be a lie wherever it cannot
be turned off, and a neutral "don't think" would be an unenforceable promise on
those same providers — the worst kind, because a caller who sets it usually
means "do not bill me for reasoning tokens."

## Decision

One four-state setting, named for what the caller wants rather than for any
provider's parameter:

```python
Thinking = Literal["default", "off", "on", "visible"]

@dataclass(frozen=True)
class ModelSettings:
    ...
    thinking: Thinking = "default"
```

`default` is the default, so existing callers — stria and doom — keep their
current behaviour with no change and no new failure mode.

Translation is a **per-provider dialect**, selected by a neutral token on the
provider table and implemented inside `llm/wire/`:

```python
# provider.py — a name, not a payload shape
thinking_dialect: str = "none"
```

| dialect | providers | `off` | `on` | `visible` |
|---|---|---|---|---|
| `anthropic` | anthropic | `{"type": "disabled"}` | `{"type": "adaptive"}` | `adaptive` + `display: "summarized"` |
| `gemini` | gemini, vertex | `thinking_budget: 0` | `includeThoughts: false` | `includeThoughts: true` |
| `deepseek` | deepseek | `extra_body.thinking.type = "disabled"` | `= "enabled"` | `= "enabled"` |
| `effort` | kimi, openai | `CapabilityError` | *(omitted — always on)* | *(omitted — always on)* |
| `none` | vllm | `CapabilityError` | `CapabilityError` | `CapabilityError` |

The table holds a **name**; the wire owns the payload. This is the same split
as `Provider.wire` and it is what keeps ADR-0009's dependency direction intact:
`{"extra_body": {"thinking": …}}` is an OpenAI-SDK shape and must not appear in
the neutral layer.

`ModelCapabilities.thinking: bool` records whether the provider accepts a
thinking request at all. Any value the selected dialect cannot express raises
`CapabilityError`, consistent with ADR-0004: emissary refuses to accept an
instruction it cannot carry out rather than silently discarding it. This is
deliberately stricter than ignoring the setting, because every one of the three
explicit values is a promise about cost or disclosure. Note the asymmetry it
exposes honestly — under the `effort` dialect `off` fails, because those models
have no off switch.

**Amendment, on implementation.** The table above originally read "level `off`"
for Gemini. Introspecting the SDK showed `ThinkingLevel` has only
`MINIMAL | LOW | MEDIUM | HIGH` — there is no OFF member, and the documented
way to disable is `thinking_budget: 0`. That is the mapping now, with the
caveat recorded in `wire/thinking.py` that Gemini 3+ prefers `thinking_level`
and may not honour a zero budget. Named rather than hidden, because a
silently-ignored `off` is the exact failure this ADR exists to prevent.

Where a provider's dialect could not be confirmed against its own
documentation, it stays `none`. This follows the existing `default_model=None`
rule: a plausible-looking parameter that the server silently ignores is a worse
failure than an error at call time.

**Capture is unconditional and independent of this setting.** A server that
volunteers `reasoning_content` without being asked has stated a fact, and
ADR-0008's factual-record principle says we record it. So a DeepSeek call under
`default` still populates `ModelResult.thinking`. The setting shapes the
request; it never filters the response.

## Alternatives Considered

- **A boolean `thinking: bool`:** cannot express "provider default", so
  adopting it would force every existing call to choose a behaviour it never
  chose before. Rejected.
- **Pass provider-native config through (`thinking={"type": ...}`):** leaks
  SDK payload shapes into the neutral layer, breaking ADR-0009's dependency
  direction, and makes a spec un-portable across providers — the one property
  this package exists to provide. Rejected.
- **Three states, collapsing `default` into `on`:** `default` and `on` differ
  on the wire (sending nothing versus explicitly enabling adaptive thinking),
  and collapsing them would change behaviour for existing callers. Rejected.
- **Silently ignore unsupported values:** a caller asking for `off` to control
  spend would be billed for reasoning anyway, with nothing in the log to
  explain it. This is precisely the plausible-but-wrong failure the fallback
  policy already exists to prevent. Rejected.
- **Expose a token budget (`thinking_budget: int`):** Anthropic deprecated
  `budget_tokens` on 4.6+ and rejects it outright on current models; Gemini
  moved from `thinkingBudget` to `thinkingLevel` for Gemini 3+. A neutral
  integer budget would encode an interface both vendors are retiring.
  Rejected — revisit only if a consumer needs it.

## Consequences

### Positive

- One setting spans every provider, and its meaning is the same everywhere it
  is accepted.
- Existing consumers are untouched: `default` reproduces today's requests byte
  for byte.
- Asking for something a provider cannot do fails at the call boundary with a
  named error, not as a silent behaviour difference discovered in a bill.

### Negative

- Four states is more surface than a boolean, and the `default`/`on`
  distinction needs the table above to be understood.
- `off` is unavailable under the `effort` dialect, so a caller who wants
  guaranteed no-reasoning cannot use Kimi K3 for it. That is a true statement
  about that model, not a limitation this package invented.
- Kimi's dialect varies by model generation (K3 `reasoning_effort`, K2.6
  `thinking.keep`, K2.5 `thinking.type`) while the table is keyed by provider.
  The entry is set for the current default model; a caller naming an older Kimi
  explicitly may get a parameter that model does not honour.

### Risks

- Provider parameter names are moving (`thinkingBudget` → `thinkingLevel`,
  `budget_tokens` → `adaptive`). The neutral names above are chosen to outlast
  that churn, but each wire's mapping table will need revisiting per model
  generation. Confined to `llm/wire/`, which is where ADR-0009 puts it.
