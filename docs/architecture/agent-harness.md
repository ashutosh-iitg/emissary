# Emissary Agent Harness Architecture

**Status:** Accepted for the bounded core; later workflow and durable-execution phases remain gated by demonstrated consumers  
**Date:** 2026-08-14  
**Scope:** Long-term target architecture and the compatibility-preserving path from the current LLM wrapper

## 1. Purpose

Emissary will provide a small, provider-agnostic runtime for building bounded agentic workflows and loops. The existing provider wrapper remains the standard and exclusive model-call interface used by the harness. Harness code must never import an Anthropic, OpenAI, or other model SDK, and must never branch on a provider name.

The architecture supports any model that can be represented by an emissary `Spec` and whose wire adapter can normalize the required capabilities. A model may support only a subset of capabilities; unsupported operations fail explicitly before execution rather than degrading silently.

This is the honest universality boundary: a new model on an existing wire should require only provider/model configuration; a model using a genuinely new protocol requires one new wire adapter. Neither case changes the harness. “Any model” cannot mean pretending that a text-only endpoint supports native tool calls, logprobs, or structured output that it does not expose.

## 2. Goals

1. Run model-directed tool loops with deterministic control and typed outcomes.
2. Keep all models and providers behind one normalized emissary interface.
3. Make actions, observations, limits, approvals, and stop reasons explicit.
4. Produce complete local event trajectories for debugging and evaluation.
5. Preserve the current `call_tool`, `call_choice`, fallback policy, and consumer contracts.
6. Remain useful as an embedded Python library without requiring a service or database.
7. Allow later async, durable, graph, MCP, and multi-agent adapters without putting those concerns in the core loop.

## 3. Non-goals for the first stable harness

- A hosted agent service or worker fleet.
- A graph DSL or general workflow scheduler.
- Exactly-once effects or transparent distributed recovery.
- Built-in shell, browser, filesystem, or network tools. (Exception: the opt-in improver's worktree-confined file tools, ADR-0029.)
- A prompt-template framework. (Memory and vector retrieval were a non-goal until ADR-0025.)
- A model router that guesses which model should handle a task.
- Multi-agent handoff abstractions before nested single-agent runs are proven.
- Reimplementing provider fallback in the runner.

## 4. Architectural invariants

These are release gates, not preferences.

1. **One model boundary:** all inference goes through public functions in `emissary.model`; only `emissary.wire` imports provider SDKs.
2. **No provider logic in the harness:** `agent`, `runner`, `tools`, `context`, and `events` cannot inspect `Spec.name`, wire names, or SDK objects.
3. **Capabilities fail loud:** unsupported tool use, structured output, logprobs, or other features raise a typed capability error before a call where possible.
4. **Model output is untrusted:** every decision, tool name, argument object, and final structured output is validated.
5. **The runner owns control:** the model proposes; the runner authorizes, executes, budgets, records, and stops.
6. **No invisible effects:** every attempted tool execution emits an event and has a stable call ID.
7. **No unbounded default:** every run has finite turn, physical model-attempt, logical tool-call, total tool-attempt, internal API-attempt, external API-attempt, serialized model-input, tool-result, and elapsed-time limits. Built-in callers count every provider retry and fallback attempt, and SDK retries are disabled. Read-only tools declare API access before registration. Idempotent tool retries use capped exponential backoff. Async waits are cancelled at the deadline; synchronous in-process effects require an application-owned isolated process for forced termination.
8. **No broad retries:** model fallback retains the existing availability-only policy; state-changing tools do not retry unless their tool policy opts in and supplies idempotency semantics.
9. **Serializable state model:** core state and events contain data, not callables or SDK values, even before durable storage exists.
10. **Network-free tests:** all core and wire behavior is testable with scripted model turns and mocked SDK clients.
11. **SOLID dependency direction:** each module has one reason to change; core policy depends on narrow protocols rather than SDKs or executors; implementations remain substitutable through shared contract tests; public interfaces stay capability-specific rather than accumulating optional methods; high-level runner policy owns and receives its dependencies explicitly.

## 5. System context

```text
Application (stria, doom, future consumers)
    │
    ├── existing structured operations ─► call_tool / call_choice
    │
    └── agentic operation ───────────────► run(agent, task, ...)
                                              │
                                              ▼
                                      Agent harness core
                                    runner/tools/context/events
                                              │
                                              ▼
                                      emissary.model.call_model
                                              │
                                  credential/fallback/capability policy
                                              │
              ┌──────────────┬────────────┴──────┬──────────────────┐
              ▼              ▼                   ▼                  ▼
        Anthropic wire   Gemini wire    OpenAI-compatible wire  TypeSafe wire
         (Messages)   (generateContent)  OpenAI/Kimi/DeepSeek/  (Jev decisions,
                                         OpenRouter/vLLM/…       call_choice only)
```

`call_tool` and `call_choice` remain public convenience contracts. They may share lower-level translation helpers with `call_model`, but their behavior does not become an agent-loop special case.

## 6. Package structure

```text
src/emissary/
  llm/              The model boundary — see CLAUDE.md for its modules
    wire/           anthropic, gemini, openai_compatible, typesafe, thinking
  harness/
    agent.py        Immutable Agent and RunLimits definitions
    machine.py      The bounded loop: all policy, no I/O (ADR-0024)
    effects.py      CallModel | ValidateTool | ExecuteTool | WaitRetry
    runner.py       run / arun: drivers that perform effects
    tools.py        Tool definitions, schemas, outcomes, registry, executor
    policy.py       Approval protocol
    context.py      ContextPolicy protocol and defaults
    events.py       RunEvent and EventSink protocol
    projection.py   Event log → model-visible messages (ADR-0011)
    state.py        RunResult, statuses, stop reasons
  eval/, storage/, memory/   Depend on the harness; it depends on none of them
```

Modules are introduced only when their first behavior lands. The list describes ownership, not a requirement to create empty scaffolding.

## 7. Standard model-call interface

### 7.1 Contract

The harness calls one provider-neutral operation:

```python
def call_model(
    spec: Spec,
    *,
    system: str,
    messages: list[Message],
    tools: list[ToolDefinition] = [],
    output_schema: dict[str, Any] | None = None,
    settings: ModelSettings | None = None,
) -> ModelResult:
    ...
```

The actual signature must avoid mutable defaults; the sketch emphasizes semantic inputs.

`ModelResult` contains:

```python
@dataclass(frozen=True)
class ModelResult:
    decision: FinalOutput | ToolCalls | Refusal
    provider: str
    model: str
    usage: Usage
    finish_reason: str | None
```

Decision variants are disjoint:

- `FinalOutput`: text and/or validated structured value, depending on the requested contract.
- `ToolCalls`: one or more calls with stable provider call ID, exact tool name, and argument object.
- `Refusal`: normalized refusal reason without pretending it is final success.

The result cannot simultaneously be final and request tools. If a provider emits both, the adapter follows a documented precedence rule or raises `ModelBehaviorError`; it never leaves interpretation to the runner.

### 7.2 Capability model

“Any model” means any model can be registered and used for operations it actually supports. It does not mean fabricating unavailable features.

```python
@dataclass(frozen=True)
class ModelCapabilities:
    tool_calling: bool
    parallel_tool_calls: bool
    structured_output: bool
    logprobs: bool
```

Capabilities begin as conservative provider defaults with optional model-level overrides in `Spec`. Runtime rejection by a nominally compatible endpoint is still normalized as a non-retryable capability/model-behavior error.

This avoids two failures:

- Assuming every OpenAI-compatible model implements every OpenAI feature.
- Hard-coding model families into the harness.

### 7.3 Conversation representation

Messages are provider-neutral tagged values, not raw provider dictionaries:

```text
UserMessage(content blocks)
AssistantMessage(text and/or proposed tool calls)
ToolMessage(call_id, tool_name, ToolResult)
```

Text is required initially. Content blocks leave a compatible extension point for image/audio/resource inputs without changing runner control flow. Wire adapters own conversion to provider message formats and preservation of provider-required tool call IDs.

### 7.4 Existing APIs

- `call_tool` retains forced single-tool structured behavior and `CallResult`.
- `call_choice` retains logprob classification and `ChoiceResult`.
- Neither is implemented by spinning up a runner.
- Shared private adapter helpers are allowed when semantics are identical.
- Breaking changes require checks in both current consumers.

## 8. Agent and tool model

### 8.1 Agent

An `Agent` is immutable configuration:

```python
Agent(
    name="reviewer",
    instructions="...",
    model=Spec(...),
    tools=(...),
    output_schema=None,
    limits=RunLimits(...),
)
```

The caller selects the model explicitly through the existing `Spec`/`resolve_spec` machinery. The harness does not route by task. Existing primary/fallback selection may be represented by a model-caller dependency, but fallback remains outside runner semantics.

### 8.2 Tool definition

A tool separates model-visible description from harness-only policy:

```python
Tool(
    name: str,
    description: str,
    input_schema: JsonSchema,
    output_schema: JsonSchema | None,
    execute: Callable,
    side_effect: SideEffect,      # none | local | external
    approval: ApprovalMode,       # never | always | policy
    timeout_seconds: float | None,
)
```

Tool names are unique within an agent. Input is always a JSON object. Output is normalized to `ToolResult`, whose status is `success`, `warning`, or `error`, with a short summary, structured content, artifact references, and safe recovery guidance where relevant.

The callable is excluded from serialized state. Checkpoints refer to tools by stable name plus definition fingerprint; the application reconstructs the executable registry on resume.

### 8.3 Tool execution

The runner delegates execution to a `ToolExecutor` protocol. The default local executor validates input, checks policy, invokes the callable, validates output, sanitizes exceptions, and returns an outcome. This seam enables future sandbox, remote, MCP, or durable executors without altering the loop.

Phase 1 executes multiple calls sequentially in response order. The data model preserves a list and stable IDs so a later async executor may safely add bounded concurrency. Parallel execution is not safe merely because a model emitted calls together; policy and side-effect independence must permit it.

## 9. Runner state machine

```text
CREATED
   │ start
   ▼
RUNNING ── final output ─────────────► COMPLETED
   │
   ├── tool calls ─► execute/observe ─┘ (next turn)
   ├── approval required ────────────► PAUSED
   ├── refusal ──────────────────────► REFUSED
   ├── limit or deadline reached ────► STOPPED
   ├── cancellation ─────────────────► CANCELLED
   └── unrecoverable failure ────────► FAILED
```

`PAUSED` is terminal today: there is no resume, and a paused log ends on an
unanswered tool call, which `history=` rejects. A caller proceeds by re-running
the task with an `Approver`. Resuming in place waits on the §13 conditions.

Terminal states are immutable. Every transition emits an event. `RunResult` always includes status, stop reason, accumulated usage, final or partial state, and events or a trace reference.

### 9.1 Loop algorithm

1. Validate agent definition and model capabilities.
2. Initialize run ID, state, budgets, context, and event sink.
3. Before each turn, check cancellation, wall time, turn budget, and token budget.
4. Ask `ContextPolicy` for provider-neutral messages.
5. Invoke the injected emissary model caller.
6. Record usage and normalized decision.
7. On final output, validate the output contract and complete.
8. On refusal, stop as `REFUSED`.
9. On tool calls, resolve and validate every call before executing any call from that batch.
10. Check policy/approval for each call.
11. Execute permitted calls, append observations, and update budgets.
12. Repeat until terminal.

Validating a complete batch before execution prevents a malformed second call from being discovered only after the first has changed state.

### 9.2 Limits

Finite defaults:

- Maximum model turns.
- Maximum total tool calls.
- Maximum consecutive tool errors.
- Optional wall-clock duration.
- Optional aggregate input/output tokens.

Cost limits require a caller-supplied pricing policy because model prices are mutable and provider-specific. The core records usage but does not ship guessed price tables.

## 10. Error and retry taxonomy

```text
EmissaryError
  ProviderError             existing provider/API failure
  CapabilityError           requested operation unsupported
  ModelBehaviorError        malformed or contradictory model result
  AgentDefinitionError      invalid static configuration
  ToolResolutionError       unknown/duplicate tool
  ToolValidationError       invalid input or output schema
  ToolExecutionError        harness/executor could not execute
  RunLimitExceeded          bounded stop represented in RunResult
```

Tool-domain failures such as “record not found” are normally `ToolResult(status="error")` observations, not Python exceptions. Harness failures such as an executor violating its contract terminate the run.

Provider fallback happens inside an injected `ModelCaller` built from existing emissary selection policy. It can retry an unavailable model turn before any tool effect occurs. The runner never retries a whole prior turn after tools have executed.

## 11. Context management

Conversation state lives in the event log, not in the runner (ADR-0011). The
model-visible surface is a pure projection, `derive_messages`, folded over the
log before each turn. Two invariants hold: **model-visible means logged**, and
**omission is an event**.

`ContextPolicy.plan` receives the projected surface and returns `ContextOp`s —
each a logged replacement of a range, with an empty replacement meaning a drop.
Ops are destructive: the log records them and later projections inherit them, so
a summarising policy pays its cost once rather than every turn. A policy cannot
execute tools, call models, or inject unlogged messages.

The default policy is deliberately transparent:

1. Keep instructions and the initial task.
2. Preserve tool-call/result pairs atomically.
3. Keep recent turns verbatim.
4. Replace oversized tool content with its precomputed summary and artifact references.
5. Stop with `CONTEXT_LIMIT` if the request still cannot fit.

Model-generated summarization is not part of the default because it creates hidden cost, recursion, and another source of semantic loss. A summarizing policy may be supplied explicitly later and must produce trace events.

## 12. Events and observability

Events are immutable, ordered, and carry run ID, sequence number, timestamp, and parent run ID where applicable.

Event kinds, with the message-bearing ones marked — those carry the content the
projection reconstructs, so their payloads are part of the durable contract:

- `run_started`
- `user_message` **(projects)**
- `model_call_started` / `model_call_completed` **(projects)** / `model_call_failed`
- `approval_resolved`
- `tool_call_rejected`
- `tool_call_started` / `tool_call_completed` **(projects)**
- `context_compacted` **(projects)**
- `run_completed` / `run_stopped`

Lifecycle kinds and message-bearing kinds are kept disjoint so the fold is a
total match over a closed set, and so a future injected turn is an additive
event rather than a schema change.

An `EventSink` may collect in memory, log locally, or export elsewhere. The default must not upload data. Sensitive model/tool content is optional; metadata, hashes, usage, timings, and statuses remain observable with content redaction.

The event stream is the canonical evaluation trajectory. Logs are a presentation of events, not the source of truth.

## 13. Pause, approval, and persistence

Phase 1 supports synchronous approval callbacks. The state model nevertheless represents `PAUSED` and pending calls so callback logic does not become baked into the runner.

Durable resume is added only after:

- `RunState` has a versioned serialization format.
- Tools have definition fingerprints.
- Completed call IDs and results are checkpointed.
- Resume rejects incompatible agent/tool definitions.
- State-changing retry semantics are explicit.

Checkpoint boundaries are after a model decision and after each tool completion. A local store can provide at-least-once recovery; exactly-once effects require idempotent tools or an external durable execution system. Temporal/LangGraph integrations belong in adapters.

## 14. Workflow and multi-agent composition

Deterministic workflows should remain ordinary Python composition around `run()` until repeated needs justify combinators. Likely later primitives are sequential steps, deterministic routing, bounded parallel map, and evaluator/optimizer loops. The first evaluator/optimizer loop is `emissary.improve` (ADR-0029): one bounded round around a consumer's own eval command, producing a branch for review.

Delegation is a nested run:

- Parent allocates a child model, tools, context, and sub-budget.
- Child emits events linked by parent run ID.
- Child returns a normal structured tool result to the parent.
- Parent retains authority over approval and total budget.

This single primitive can support manager/worker patterns without introducing a separate multi-agent runtime.

## 15. Security model

1. Tool authority is supplied by the application and scoped per run.
2. Prompt text cannot grant permissions.
3. Tool arguments and outputs are schema validated.
4. Side-effect metadata drives approval and retry policy.
5. Tool timeouts and output-size limits protect the runner.
6. Errors and events are sanitized before reaching model context or external sinks.
7. Cancellation is checked before model calls and tool execution.
8. High-risk tools should execute behind isolated `ToolExecutor` implementations.
9. Tool result content is untrusted context and remains distinguishable from system instructions.

## 16. Testing and evaluation architecture

### Unit and state-machine tests

A `ScriptedModelCaller` returns predefined normalized decisions. Tests assert exact transitions, events, calls, context, budgets, and terminal outcomes. Every branch in the state machine is reachable without an SDK or network.

### Wire contract tests

Mocked SDK payloads prove each adapter maps provider messages to the same normalized decisions, including multiple calls, text completion, refusal, malformed arguments, missing IDs, usage, and unsupported capabilities.

### Compatibility tests

Existing `call_tool`, `call_choice`, provider, and fallback tests remain unchanged. Consumer contract fixtures should be added before shared adapter refactors.

### Trajectory evaluations

Versioned scenarios include an initial state, task, tools, deterministic grader, allowed/forbidden effects, and efficiency bounds. Report final-state success plus tool selection, arguments, order, errors, turns, tokens, and cost. Model version and harness version are independent dimensions.

## 17. Proposed decision ledger

D1–D8, D13, D14 and D9a are accepted as ADR-0001–0008, 0010, 0009 and 0025;
the rest remain proposed.

| ID | Decision | Alternative rejected | Validation |
|---|---|---|---|
| D1 | Emissary is the only model-call boundary | Direct SDK use in runner | Static import test; fake caller runner tests |
| D2 | Add `call_model`; preserve specialized APIs | Generalize `call_tool` into a union | Existing 47 tests and consumer checks stay green |
| D3 | Use discriminated model decisions | Optional fields/raw dicts | Exhaustive state-machine tests; type checking later |
| D4 | Capability negotiation fails loud | Assume wire compatibility or silently degrade | Unsupported-operation tests for every adapter |
| D5 | Start with synchronous bounded runner | Async-first or graph-first | Complete two real consumer journeys without framework dependencies |
| D6 | Tool executor is a protocol seam | Runner invokes arbitrary callables directly | Local executor and fake executor pass same contract suite |
| D7 | Sequential multi-call execution initially | Drop extra calls or parallelize blindly | Preserve order/IDs; benchmark before async concurrency |
| D8 | Events are canonical trajectory | Ad-hoc logs | Reconstruct test run outcome from events |
| D9 | Transparent context policy first | Automatic model summarization/memory | Context-boundary tests; add sophistication only after eval failure |
| D9a | Memory is a harness capability; storage is the application's (ADR-0025) | Memory per application, or emissary-owned databases | Store-protocol tests with in-memory fakes; import test keeps harness free of memory |
| D10 | Serializable state before storage | Couple core to SQLite/Temporal/LangGraph | Round-trip state tests without persistence dependency |
| D11 | Nested run is future delegation primitive | Specialized multi-agent framework | Demonstrate manager/worker scenario with parent-child events and budgets |
| D12 | Pricing is injected, usage is core | Mutable built-in price catalog | Usage limits work with no network/current pricing data |
| D13 | Use a standards-compliant JSON Schema validator at trust boundaries | Handwritten partial validation | Official schema conformance tests; dependency/security review before adoption |
| D14 | Enforce SOLID through module ownership and dependency rules | Large framework classes or abstraction-per-type ceremony | Architecture import tests, protocol contract tests, and per-PR responsibility review |

## 18. Decision validation gate

A proposed decision becomes accepted only when:

1. Its consumer need is named.
2. At least one simpler alternative is evaluated.
3. Its invariant can be tested mechanically.
4. Its rollback or compatibility path is documented.
5. It does not move provider behavior above the emissary model boundary.
6. It avoids infrastructure that no current milestone requires.

The first implementation milestone may begin after D1–D8 are approved. D9–D12 can remain proposed until their phases begin.
