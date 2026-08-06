# emissary

A small, provider-agnostic wrapper over LLM APIs. One call shape, two wire
formats: the native Anthropic Messages API, and OpenAI-compatible chat
completions — which covers OpenAI, Kimi, DeepSeek, Gemini, and a locally
hosted [vLLM](https://github.com/vllm-project/vllm) server. Two adapters and
a table, not five-or-six integrations.

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

result = emissary.call_text(
    spec, system="You are terse.", messages=[{"role": "user", "content": "Hi"}]
)
print(result.payload)  # a string
```

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

`call_tool_with_fallback` / `call_text_with_fallback` make one attempt on a
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
