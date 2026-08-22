# emissary: provider-agnostic LLM calls

---

# Rules — read these before every task and apply them throughout

These rules apply to every task in this project unless explicitly overridden.
Bias: caution over speed on non-trivial work. Use judgment on trivial tasks.

**These rules are shared verbatim with `stria` and `doom`.** They are copied
rather than referenced because each repo must stand alone when cloned. If you
change a rule, change it in all three — a rule that disagrees across repos is
worse than no rule.

## Rule 1 — Think Before Coding
State assumptions explicitly. If uncertain, ask rather than guess.
Present multiple interpretations when ambiguity exists.
Push back when a simpler approach exists.
Stop when confused. Name what's unclear.

## Rule 2 — Simplicity First
Minimum code that solves the problem. Nothing speculative.
No features beyond what was asked. No abstractions for single-use code.
Test: would a senior engineer say this is overcomplicated? If yes, simplify.

## Rule 3 — Surgical Changes
Touch only what you must. Clean up only your own mess.
Don't "improve" adjacent code, comments, or formatting.
Don't refactor what isn't broken. Match existing style.

## Rule 4 — Goal-Driven Execution
Define success criteria. Loop until verified.
Don't follow steps. Define success and iterate.
Strong success criteria let you loop independently.

## Rule 5 — Use the model only for judgment calls
Use me for: classification, drafting, summarization, extraction.
Do NOT use me for: routing, retries, deterministic transforms.
If code can answer, code answers.

## Rule 6 — Token budgets are not advisory
Per-task: 4,000 tokens. Per-session: 30,000 tokens.
These bind routine work. A long-running or genuinely high-complexity task —
multi-repo changes, extractions, anything needing sustained context — may exceed
them; say so when you pass, and why the task warranted it.
Never silently overrun, and never let a budget truncate work into something
half-done. If you are approaching budget on *routine* work, summarize and start
fresh — that is the case the budget exists for.

## Rule 7 — Surface conflicts, don't average them
If two patterns contradict, pick one (more recent / more tested).
Explain why. Flag the other for cleanup.
Don't blend conflicting patterns.

## Rule 8 — Read before you write
Before adding code, read exports, immediate callers, shared utilities.
"Looks orthogonal" is dangerous. If unsure why code is structured a way, ask.

## Rule 9 — Tests verify intent, not just behavior
Tests must encode WHY behavior matters, not just WHAT it does.
A test that can't fail when business logic changes is wrong.

## Rule 10 — Checkpoint after every significant step
Summarize what was done, what's verified, what's left.
Don't continue from a state you can't describe back.
If you lose track, stop and restate.

## Rule 11 — Match the codebase's conventions, even if you disagree
Conformance > taste inside the codebase.
If you genuinely think a convention is harmful, surface it. Don't fork silently.

## Rule 12 — Fail loud
"Completed" is wrong if anything was skipped silently.
"Tests pass" is wrong if any were skipped.
Default to surfacing uncertainty, not hiding it.

## Rule 13 — Read the architecture and summarise it whenever you lose sight of the big picture
Before diving into any isolated part, re-read the architecture section and state
in one sentence how the piece you are about to touch fits into the overall data
flow. If you cannot articulate that connection, stop and resolve it before
writing code. Use that connection to validate that local decisions (naming,
granularity, output shape) are consistent with the system's end-to-end contract,
not just locally convenient.

## Rule 14 — Every architecture decision must be defensible
Apply KISS, DRY, YAGNI, and SOLID throughout — not as a checklist but as a
standard of craft. Every non-trivial design choice (a new abstraction, a pattern,
a module boundary, a data model decision) must have a clear rationale grounded in
the system's actual constraints and the architecture it lives inside. "It's a
common pattern" is not a rationale. "It solves X because Y, and the alternative
has cost Z" is. If a decision cannot withstand a pointed question from a senior
engineer, it should not be made. Surface the trade-off, name the alternative you
rejected, and state why this system favors the chosen approach. Uninformed or
arbitrary decisions — even small ones — are not acceptable.

## Rule 15 — Good code is self-explanatory; bad code needs verbose comments
If a comment is explaining *what* the code does, the code is wrong — fix the
naming, the structure, or the decomposition instead. Reaching for a comment to
make a block legible is a signal, not a solution.

Comments are for the things the code genuinely cannot say:
- **Non-obvious catches** — why this order, why this guard, what breaks without it.
- **Assumptions made** — what we are relying on that the reader cannot see from here.
- **Docstrings on function, module, and API definitions** — contract, not narration.

Not required for simple functions. A docstring restating an obvious signature is
noise and should be deleted.

---

# Project context

**emissary** is a small wrapper over LLM APIs, shared by `stria` (legal-text
extraction) and `doom` (constitutional classifiers). It exists because both
needed the same thing — one structured call, several possible providers, a
credible fallback — and neither should own that code.

**It is a library, not a service.** No process, no state, no config file. It
reads `os.environ` and nothing else.

## The shape

```
llm/
  provider.py       provider registry, specs, and credential selection
  model.py          conversational sync/async dispatch and caller shells
  calls.py          tool-forced and scored calls
  retry.py          retry ladder, warnings, and one-shot fallback policy
  streaming.py      sync/async sink contracts and emission tracking
  wire/
    anthropic.py          native Messages API
    gemini.py             native generateContent API
    openai_compatible.py  OpenAI-compatible chat completions
    thinking.py           provider-neutral reasoning controls
harness/             bounded agent loop, effects, tools, and event projection
eval/                recorded evaluation and deterministic replay
storage/             run persistence
```

Data flow: caller builds a `Spec` → `model.call_model` or `calls.call_tool`
gates credentials and capabilities → the selected wire translates, calls, and
normalises the answer. Everything above `wire/` is provider-neutral; SDK
request and response vocabulary stays inside the wire adapters.

## Decisions — do not re-open without new information

**Three wire formats, not N integrations.** Anthropic speaks Messages; Gemini
and Vertex speak native `generateContent`; OpenAI, Kimi, DeepSeek, OpenRouter,
and vLLM speak OpenAI-compatible chat completions. So eight providers use
three adapters and a table. That ratio is the whole
justification for the package existing — if it ever becomes six real
integrations, the abstraction has stopped paying for itself and should be
reconsidered rather than extended.

**One call shape (`call_tool`), tool-forced.** A `call_text` existed briefly and
was deleted: no caller in either consuming project ever used it. Both want a
typed answer. Adding it back requires an actual caller, not an anticipated one —
its presence also forced `CallResult.payload` into a `dict | str` union that
made every consumer's indexing unsound.

**Retry the chosen provider, then fall back once — `retryable` only.**
Connection errors, rate limits, overloads and refusals climb a ladder on the
provider the caller asked for (`llm/retry.py`; `RETRY_DELAYS` = 10s, 30s, 60s,
four attempts in all), and only then does a *different* provider get asked,
**once**. A malformed payload or a missing credential does neither. That
distinction is the safety property this package exists to preserve: retrying
until an answer parses is shopping for a provider whose output happens to be
usable, which is exactly how a plausible-but-wrong result reaches a caller that
trusts it.

The delays are long on purpose — these failures clear in tens of seconds, and a
millisecond-scale ladder would spend every attempt inside the same outage. The
fallback gets no ladder of its own: the primary already spent the outage
window, and two providers failing is a condition an operator must see now
rather than after another two minutes of waiting.

**Nothing degrades silently.** Every retry and every switch is logged at
WARNING naming both specs, and the package installs no `NullHandler`, so an
application that configures no logging still gets them on stderr via
`logging.lastResort`. A fallback nobody noticed means a caller reading an
answer from a model it never chose. The policy lives in one module because
three copies of it would drift into three ideas of what an outage looks like.

**Streaming retries only before the first delta.** Once text or reasoning has
reached a sink, emissary cannot retract it. A failed streamed attempt therefore
becomes non-retryable after its first emitted delta; this is logged at WARNING.
Before any delta, the normal retry ladder still applies. This preserves actual
streaming without ever stitching two attempts into one apparent answer.

**No settings framework, ever.** `selection.resolve_spec` reads `os.environ`.
Callers with their own config source (stria reads Django settings first) resolve
their own string and call `parse_spec` + `call_tool_with_fallback` directly. A
wrapper shared across projects has no business knowing what Django is; the
two-function split is what keeps that true without forcing stria to give up its
settings-first behavior.

**`default_model=None` is deliberate.** Set only where a provider's current
model ID was actually verified against its documentation. Where it wasn't, the
caller must name one. A plausible-looking model ID that resolves to nothing is a
worse failure than an error at parse time.

**vLLM's base URL is read at call time** (`VLLM_BASE_URL`), not hardcoded — a
local server has no vendor endpoint to pin. `key_required=False` because vLLM
doesn't authenticate by default; the key var still exists for deployments that
put auth in front of it.

## Testing

`uv run pytest`. **No test may reach a network.** The wire adapters are tested
against a mocked SDK client; the fallback policy is tested by mocking the wire
dispatch. If a change makes that hard, the change is probably wrong.

## Consumers

`stria` (`intake/extract/client.py` — a thin Django-settings adapter) and `doom`
(`src/doom/judge.py`). Both depend on it as an editable path dependency. A
breaking change here means checking both before committing.
