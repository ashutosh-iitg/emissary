# emissary

A small, provider-agnostic wrapper over LLM APIs. One call shape, three wire
formats: the native Anthropic Messages API, Gemini's `generateContent`, and
OpenAI-compatible chat completions — which covers OpenAI, Kimi, DeepSeek, and
a locally hosted [vLLM](https://github.com/vllm-project/vllm) server. Three
adapters and a table, not seven integrations.

A provider only gets its own adapter when the compatibility layer loses
something the caller needs. Gemini earned one because that layer drops
`thought_signature`, which Gemini 3+ requires on every tool-calling turn after
the first. DeepSeek and Kimi did not: OpenAI-compatible *is* their first-party
API.

One call shape: a tool-forced structured call, because the callers that
exist all want a typed answer rather than prose.

```python
import emissary

spec = emissary.parse_spec("anthropic")  # or "kimi:kimi-k2.6", "vllm:my-model", ...

result = emissary.call_tool(
    spec,
    system="You are a careful reviewer.",
    prompt=emissary.Prompt(
        system="You extract structured data.",
        blocks=(
            emissary.TextBlock("<document>", cache=True),
            emissary.TextBlock("Summarize it."),
        ),
    ),
    tool={"name": "summarize", "description": "...", "input_schema": {...}},
)
print(result.payload)  # the tool's arguments, as a dict
```

`blocks` marked `cache=True` get an ephemeral prompt-cache breakpoint on the
Anthropic wire — put it on content resent across many calls. The
OpenAI-compatible wire has no equivalent and concatenates instead.

## Scoring (`call_choice`)

For classification where you want a **number to threshold** rather than a
verdict, `call_choice` generates one token and reads the probability the model
assigned it:

```python
result = emissary.call_choice(
    emissary.parse_spec("vllm:my-local-model"),
    system="Answer with exactly one word: SAFE or FLAG.",
    blocks=(emissary.TextBlock("<the thing to classify>"),),
    labels=["SAFE", "FLAG"],
)
result.probability("FLAG")  # 0.0-1.0, renormalised over the labels
result.label                # the most probable one
```

Cost is one output token. The score comes from the model's own distribution,
not from asking it to rate its confidence — self-reported confidence is not
calibrated, and thresholding it only looks like measurement.

**OpenAI-compatible wire only.** The Anthropic Messages API exposes no token
logprobs, so an `anthropic:` spec is refused with an error naming the
alternative. On vLLM the call also sends `guided_choice`, constraining
decoding to the label set; other providers get the same scoring without it.

Labels must differ in their **first token** — they're matched by prefix, so
`["SAFE", "FLAG"]` works and `["FLAG_A", "FLAG_B"]` does not.

## Providers

| name | wire | credential | notes |
|---|---|---|---|
| `anthropic` | anthropic | `ANTHROPIC_API_KEY` | default model `claude-opus-5` |
| `openai` | openai-compatible | `OPENAI_API_KEY` | no default model — name one |
| `kimi` | openai-compatible | `MOONSHOT_API_KEY` | default model `kimi-k3` |
| `deepseek` | openai-compatible | `DEEPSEEK_API_KEY` | default model `deepseek-v4-pro` |
| `gemini` | gemini | `GEMINI_API_KEY` | default model `gemini-3.6-flash` |
| `vertex` | gemini | Google ADC + `GOOGLE_CLOUD_PROJECT` | `GOOGLE_CLOUD_LOCATION` (default `global`), no default model |
| `vllm` | openai-compatible | `VLLM_API_KEY` (optional) | `VLLM_BASE_URL` (default `http://localhost:8000/v1`), no default model |

`vllm` points at any OpenAI-compatible server vLLM exposes for a locally
hosted, open-weight model — no credential required by default, since vLLM's
server doesn't authenticate unless you put something in front of it.

The Gemini SDK is an **optional extra** — it pulls in sixteen transitive
packages that an Anthropic-only or OpenAI-only caller never needs:

```bash
pip install 'emissary[gemini]'      # or: uv add 'emissary[gemini]'
```

Without it, `gemini` and `vertex` raise a `CapabilityError` naming the extra.
Every other provider works from the base install.

`vertex` reaches the same models as `gemini` through GCP: application default
credentials instead of an API key, and a project and region instead of a base
URL. `key_present(spec)` answers for either without spending a request.

## Thinking

`ModelSettings.thinking` is one neutral setting across every provider:

| value | meaning |
|---|---|
| `default` | send no thinking parameter — each provider's own behaviour |
| `off` | do not reason |
| `on` | reason, text not required |
| `visible` | reason and return the text |

Anything a provider cannot express raises `CapabilityError` rather than being
quietly dropped, because each explicit value is a promise about cost or
disclosure — Kimi K3 always reasons, so `off` fails there instead of billing
you for reasoning you asked to skip.

Two different things come back. `ModelResult.thinking` is readable text for
logs and evaluation. `ModelResult.reasoning` is opaque provider state that the
*next* request must carry back — signed Anthropic thinking blocks, Gemini
thought signatures, DeepSeek and Kimi `reasoning_content`. All three APIs
reject a follow-up turn that loses it, so the harness replays it verbatim and
never inspects it. It is tagged with the wire that issued it, so a fallback to
a different provider drops it rather than forwarding something unparseable.

## Streaming

Pass a sink to watch a turn arrive. The call still returns one complete
`ModelResult` — streaming is an observation channel, not a second result type
(ADR-0022), so nothing downstream of the wire changes:

```python
class Printer:
    def on_text(self, delta): print(delta, end="", flush=True)
    def on_thinking(self, delta): print(delta, end="", flush=True)

result = emissary.call_model(spec, system=..., messages=..., sink=Printer())
```

Omit `sink` and the request is byte-identical to a non-streaming one. Only text
and reasoning are streamed: tool-call arguments arrive as JSON fragments that
mean nothing until complete, so they reach you whole on `result.decision`.

A sink is called synchronously inside the read loop, and an exception from it
is **not** caught — a silently frozen display is harder to diagnose than a loud
failure.

## Async

Every call has an `a`-prefixed sibling — `acall_model`, `acall_tool`,
`acall_choice` — with matching `AsyncSpecModelCaller` and
`AsyncFallbackModelCaller`. Each is a shell over the *same* request-building
and response-normalising functions the sync path uses, so the two cannot
disagree about what the model said:

```python
import asyncio, emissary

async def score_all(candidates):
    spec = emissary.parse_spec("vllm:qwen")
    return await asyncio.gather(*(
        emissary.acall_choice(spec, labels=["SAFE", "FLAG"], system=..., blocks=(...,))
        for candidate in candidates
    ))
```

Async streaming takes an `AsyncStreamSink`, whose `on_text` / `on_thinking` are
awaited — the point of streaming from async code is usually to forward deltas
somewhere that must be awaited, which a sync-only sink could not do without
buffering or reordering.

**The bundled runner is still synchronous** (ADR-0005, amended by ADR-0023).
The async pieces above are the boundary an async agent loop would be built on,
not an async loop itself.

## Selection and fallback

`resolve_spec` is the plain "explicit override > env var > default" pattern:

```python
spec = emissary.resolve_spec(cli_arg, env_var="MY_APP_LLM_PROVIDER", default="anthropic")
```

`call_tool_with_fallback` makes one attempt on a
primary `Spec`, then one attempt on a fallback `Spec` — but only if the
primary failed in a way another provider could plausibly answer
(`ProviderError.retryable`: connection errors, rate limits, overloads,
refusals). A malformed payload or a missing credential never falls back —
retrying elsewhere would be shopping for a provider whose answer happens to
be usable, not recovering from an outage.

```python
result = emissary.call_tool_with_fallback(
    emissary.parse_spec("anthropic"),
    emissary.parse_spec("kimi"),
    system=system, blocks=blocks, tool=tool,
)
```

A caller with its own config source (a settings framework, a config file)
resolves its own raw provider string and calls `parse_spec` directly, rather
than using `resolve_spec` — this package only reads `os.environ`, nothing
else.

## Development

```bash
uv sync
uv run pytest
uv run ruff check .
```

No test reaches a network — the wire adapters are tested against a mocked
SDK client, and the fallback policy is tested by mocking the wire dispatch
itself.

## Agent harness

The harness runs a bounded model → tool → observation loop through the same
provider-neutral emissary caller. Harness code never imports a provider SDK,
so any model registered behind the Anthropic, Gemini, or OpenAI-compatible wire can be
used without changing the runner.

```python
import emissary


def add(a: int, b: int) -> dict:
    return {"sum": a + b}


agent = emissary.Agent(
    name="calculator",
    instructions="Use the available tools and return the answer.",
    tools=(
        emissary.Tool(
            name="add",
            description="Add two integers.",
            input_schema={
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
                "additionalProperties": False,
            },
            execute=add,
        ),
    ),
    limits=emissary.RunLimits(max_turns=6, max_tool_calls=4),
)

result = emissary.run(
    agent,
    "What is 19 + 23?",
    caller=emissary.SpecModelCaller(emissary.parse_spec("vllm:my-model")),
)

if result.status is emissary.RunStatus.COMPLETED:
    print(result.output)
else:
    print(result.stop_reason, result.events)
```

Tools are JSON-Schema validated before any call in a batch executes. Tools
marked `approval="always"` require an injected approver; without one the run
pauses before the effect. Every model call, proposed action, tool outcome, and
terminal transition is represented in the run's ordered event trajectory.

## Package layout

The root package re-exports the common API. Larger applications can import from
the responsibility-specific modules instead:

| Module | Responsibility |
|---|---|
| `emissary.llm` | Provider selection, normalized model calls, messages, decisions, and wire adapters |
| `emissary.harness` | Agent definitions, bounded execution, tools, policy, context, state, and events |
| `emissary.eval` | Deterministic run and trajectory evaluation |
| `emissary.storage` | Optional versioned run-record persistence |

## License

[MIT](LICENSE) © ashutosh-iitg
