# emissary

A small, provider-agnostic wrapper over LLM APIs. One call shape, two wire
formats: the native Anthropic Messages API, and OpenAI-compatible chat
completions — which covers OpenAI, Kimi, DeepSeek, Gemini, and a locally
hosted [vLLM](https://github.com/vllm-project/vllm) server. Two adapters and
a table, not five-or-six integrations.

One call shape: a tool-forced structured call, because the callers that
exist all want a typed answer rather than prose.

```python
import emissary

spec = emissary.parse_spec("anthropic")  # or "kimi:kimi-k2.6", "vllm:my-model", ...

result = emissary.call_tool(
    spec,
    system="You are a careful reviewer.",
    blocks=[{"text": "<document>", "cache": True}, {"text": "Summarize it.", "cache": False}],
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
    blocks=[{"text": "<the thing to classify>", "cache": False}],
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

| name | wire | key env | notes |
|---|---|---|---|
| `anthropic` | native | `ANTHROPIC_API_KEY` | default model `claude-opus-5` |
| `openai` | openai-compatible | `OPENAI_API_KEY` | no default model — name one |
| `kimi` | openai-compatible | `MOONSHOT_API_KEY` | default model `kimi-k3` |
| `deepseek` | openai-compatible | `DEEPSEEK_API_KEY` | default model `deepseek-v4-pro` |
| `gemini` | openai-compatible | `GEMINI_API_KEY` | default model `gemini-3.6-flash` |
| `vllm` | openai-compatible | `VLLM_API_KEY` (optional) | `VLLM_BASE_URL` (default `http://localhost:8000/v1`), no default model |

`vllm` points at any OpenAI-compatible server vLLM exposes for a locally
hosted, open-weight model — no credential required by default, since vLLM's
server doesn't authenticate unless you put something in front of it.

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
so any model registered behind the Anthropic or OpenAI-compatible wire can be
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

## License

[MIT](LICENSE) © ashutosh-iitg
