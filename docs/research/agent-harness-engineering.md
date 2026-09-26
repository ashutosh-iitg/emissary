# Agent Harness Engineering: Research and Direction for Emissary

*Research date: 2026-08-14 | Scope: goals, architecture patterns, runtime design, and an incremental direction for this repository*

## Executive summary

An agent harness is the deterministic runtime around a model that turns repeated model calls into a controlled computation. The model proposes what to do; the harness owns what may happen, executes actions, records observations, updates context, decides whether to continue, and enforces stop conditions. Its quality is therefore governed at least as much by the action and observation interfaces as by the model or prompt.

For emissary, the right next step is a small, provider-agnostic, in-process loop—not a graph framework, multi-agent platform, memory database, or distributed workflow engine. The first useful architecture is:

```text
Agent definition (instructions, tools, output contract)
                        │
                        ▼
Runner ── model turn ──► Decision
  ▲                      ├── tool calls ─► policy/approval ─► executor
  │                      ├── final output
  └──── observations ◄───└── refusal/error
            │
            ├── context policy
            ├── budgets and stop conditions
            └── event/trace sink
```

The key design move is to split the current provider-specific response handling from loop orchestration. Wire adapters should normalize a model turn into a provider-neutral decision; the runner should know nothing about SDK response objects; tools should be ordinary typed capabilities behind a narrow executor contract.

## 1. What a harness is

Anthropic distinguishes workflows—where code prescribes the path—from agents—where the model chooses its path and tool use. It recommends beginning with the simplest composable pattern because autonomy trades latency and cost for flexibility. A harness can support both: deterministic workflow code can invoke a bounded agent loop as one step, while the loop permits model-directed action selection within explicit limits. [Anthropic: Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)

The minimal agent loop is:

1. Build the model-visible context from instructions, task state, tools, and prior observations.
2. Ask the model for a decision.
3. If final output is produced, validate and finish.
4. If tool calls are produced, validate and execute them, append observations, and repeat.
5. Stop on completion, pause, cancellation, policy violation, or a budget limit.

OpenAI's runner uses this same basic state machine: model output either completes, hands off, or emits tool calls that are executed and fed into another turn; `max_turns` is an explicit termination guard. [OpenAI Agents SDK: Running agents](https://openai.github.io/openai-agents-python/running_agents/)

The harness—not the model—must own:

- Tool registration, schemas, invocation, concurrency, timeout, and cancellation.
- Authorization and approval before side effects.
- Run state, identifiers, budgets, and stop conditions.
- Conversion of tool outcomes into useful model observations.
- Context selection, truncation, compaction, and artifact references.
- Retry classification and idempotency policy.
- Events, traces, usage, replay data, and evaluation hooks.

## 2. Core engineering goals

### Controllability

The model receives a bounded action space rather than ambient application authority. High-risk actions should be separate, narrowly typed tools so they can have stricter approval and retry policies. Input/output guardrails are not enough for a multi-step agent: checks around each tool invocation are required because that is where state-changing effects occur. [OpenAI Agents SDK: Guardrails](https://openai.github.io/openai-agents-python/guardrails/)

### Legibility

Every run should explain itself as an ordered event stream: model turn, proposed action, approval, tool result, context update, completion, or stop. OpenAI's tracing records model generations, tool calls, handoffs, guardrails, and custom events; this is a good indication of the minimum useful observability surface even when traces remain local. [OpenAI Agents SDK: Tracing](https://openai.github.io/openai-agents-python/tracing/)

### Recoverability

Failures need separate semantics:

- A model/provider availability failure may retry or switch provider.
- Invalid model behavior should normally fail the turn, not silently shop for a better answer.
- A tool-domain error should become an observation the model can potentially correct.
- A harness/invariant failure should stop the run.
- A process failure requires a checkpoint if resumption is promised.

MCP makes a similar distinction: tool execution errors belong in tool results so a model can self-correct, while protocol failures remain protocol errors. It also supports output schemas so clients can validate structured observations. [MCP tool specification](https://modelcontextprotocol.io/specification/2025-06-18/server/tools)

### Boundedness

Every run needs limits independent of model cooperation: model turns, wall time, token usage, tool calls, consecutive failures, and optionally monetary cost. Limits should produce a typed stop reason and preserve the partial trajectory rather than merely throwing an opaque timeout.

### Portability

Agent logic should depend on provider-neutral messages, decisions, usage, and errors. Provider extensions may improve execution but must not leak into the runner. This aligns closely with emissary's existing “two wires, one normalized result” design.

### Evaluability

Final-answer success is necessary but insufficient. The SWE-agent work showed that changing the agent-computer interface materially changes task completion, while trajectory-aware evaluation exposes wrong tool selection, arguments, and ordering that a final score hides. [SWE-agent paper](https://papers.neurips.cc/paper_files/paper/2024/file/5a7c947568c1b1328ccc5230172e1e7c-Paper-Conference.pdf), [TRAJECT-Bench](https://arxiv.org/abs/2510.04550)

Useful harness metrics are:

- Task completion rate and pass@k.
- Completion rate by model, tool set, and harness version.
- Turns, tool calls, retries, tokens, latency, and cost per successful run.
- Invalid-action and tool-error rates.
- Approval/policy interruption rates.
- Trajectory correctness where a reference path exists.
- Time-horizon curves rather than one aggregate score for tasks of very different complexity. METR defines a task-completion time horizon as the human task duration at which an agent reaches a specified reliability. [METR: Task-completion time horizons](https://metr.org/time-horizons/)

## 3. The four interfaces that determine harness quality

### Action interface

Tool names must be distinct, descriptions operational, and input schemas narrow. Overlapping catch-all tools force the model to infer distinctions that the system itself has failed to define. The SWE-agent paper calls this an agent-computer interface and demonstrates that interface design can materially improve software-agent performance without changing the underlying model. [SWE-agent paper](https://arxiv.org/abs/2405.15793)

A tool definition needs more than a callable:

```python
Tool(
    name="read_file",
    description="Read UTF-8 text from one workspace-relative file.",
    input_schema=...,
    output_schema=...,
    execute=read_file,
    side_effect="read",
    approval="never",
    timeout_seconds=10,
)
```

The side-effect and approval metadata are harness policy, not model hints.

### Observation interface

Raw stdout, SDK exceptions, and enormous documents are poor observations. A useful result has a deterministic envelope such as:

```python
ToolResult(
    status="success",       # success | warning | error
    summary="Read 82 lines",
    content={...},
    artifacts=("src/app.py",),
    retryable=False,
)
```

Results should contain enough recovery information for the next decision, while artifacts and large payloads should be referenced rather than repeatedly inlined.

### Context interface

Context engineering is a repeated selection problem, not just system-prompt writing. Long loops continually create candidate context—messages, tool outputs, plans, files, and summaries—and the harness must decide what remains model-visible. Anthropic recommends treating context as finite and curating it every inference turn. [Anthropic: Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)

The first version should expose a `ContextPolicy` seam but ship only a transparent default: retain instructions and recent events, replace oversized tool results with summaries plus artifact references, and fail before exceeding a configured budget. Automatic semantic memory can come later, once evaluations show that simple truncation/compaction is inadequate.

### Control interface

Control flow should be represented explicitly:

```text
RUNNING → COMPLETED
        → PAUSED_APPROVAL → RUNNING
        → STOPPED_LIMIT
        → CANCELLED
        → FAILED_MODEL
        → FAILED_TOOL
        → FAILED_HARNESS
```

A loop should never encode completion as “the while-loop happened to end.” A typed result with a stop reason makes callers, tests, and operators agree on what happened.

## 4. Architecture patterns and when to use them

### Direct bounded loop

Best first abstraction for one agent selecting among tools. It has the smallest state surface and is easy to test from recorded model turns. This should be emissary's initial target.

### Deterministic workflow around agentic steps

Use sequential, routing, parallel, evaluator/optimizer, or orchestrator/worker patterns when the business path is known. Code owns the graph; models perform judgment within nodes. This is easier to audit than giving the model authority over the whole process and follows Anthropic's progression from simple workflows to agents. [Anthropic: Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)

### Graph/state-machine runtime

Useful when an application needs branching, cycles, parallel nodes, resumable human input, and inspectable state transitions. LangGraph identifies durable execution, persistence, streaming, and human-in-the-loop as orchestration concerns. Its interrupt model also exposes an important rule: resumed nodes may re-execute, so effects around pause points must be idempotent. [LangGraph overview](https://langchain-ai.github.io/langgraph/index.html), [LangGraph interrupts](https://langchain-ai.github.io/langgraph/concepts/breakpoints/)

Emissary should not build a graph DSL initially. A small runner can later become a node inside one, or a graph package can depend on the runner.

### Durable workflow engine

Needed when runs must survive process restarts, span long waits, or guarantee progress around external effects. Temporal's central separation is useful even without adopting Temporal: deterministic workflows orchestrate while activities perform non-deterministic LLM calls and external I/O. Recorded activity results make replay possible without repeating non-deterministic work. [Temporal AI reference architecture](https://go.temporal.io/platform-hub/ai-engineering/ai-reference-architecture)

This belongs behind an optional persistence/runtime adapter, not in emissary's first loop. In-process checkpoints cannot honestly promise exactly-once effects; tool idempotency keys and an external durable executor are needed for that class of guarantee.

### Multi-agent delegation

Manager/worker and handoff patterns are useful only when specialization, context isolation, or independent parallel work demonstrably improves outcomes. They multiply prompts, state, routing failure modes, and evaluation difficulty. Model-independent primitives—nested runs, delegation events, and scoped budgets—should precede any `MultiAgent` abstraction.

## 5. Proposed system design for emissary

### Preserve the existing boundary

Today emissary is a provider-normalized structured-call library. The harness should build above it, but agent loops require a more general model turn than the current forced `call_tool` contract.

Recommended layers:

```text
emissary/provider.py       Provider registry and resolved Spec
emissary/wire/             SDK translation only
emissary/model.py          Provider-neutral model turn API
emissary/tools.py          Tool definition, validation, executor protocol
emissary/agent.py          Agent definition and limits
emissary/runner.py         Bounded plan/act/observe loop
emissary/events.py         Typed run event stream and sinks
emissary/state.py          Serializable run state and stop reasons
```

The normalized model-turn result should be a discriminated union, not optional fields:

```python
ModelTurn = FinalOutput | ToolCalls | Refusal
```

Each tool call needs a stable call ID, name, validated arguments, and provider provenance. Parallel calls should be representable even if the first runner executes sequentially.

### Core contracts

```python
@dataclass(frozen=True)
class Agent:
    instructions: str
    tools: tuple[Tool, ...]
    output_schema: dict | None = None

@dataclass(frozen=True)
class RunLimits:
    max_turns: int = 12
    max_tool_calls: int = 40
    max_seconds: float | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None

class ToolExecutor(Protocol):
    def execute(self, call: ToolCall, context: RunContext) -> ToolResult: ...

class EventSink(Protocol):
    def emit(self, event: RunEvent) -> None: ...
```

The runner should accept dependencies explicitly. It should not discover tools globally, load settings frameworks, create storage, or choose an approval UI.

### One-turn algorithm

```text
1. Check cancellation and budgets.
2. Ask ContextPolicy for model-visible input.
3. Call the normalized model-turn API.
4. Record usage and emit ModelTurnCompleted.
5. FinalOutput: validate output schema and complete.
6. Refusal: stop with a typed reason.
7. ToolCalls:
   a. resolve each exact tool name;
   b. validate arguments;
   c. run policy/approval;
   d. execute with timeout and call ID;
   e. validate/normalize ToolResult;
   f. append observations and emit events;
   g. continue.
```

Retries should be layered:

- Provider call retry/fallback remains emissary's existing availability policy.
- Tool retries are tool-specific and off by default for state-changing operations.
- A whole model turn should not replay completed effects unless call IDs are idempotent.
- Harness invariant failures never retry automatically.

### Streaming and async

The current package is synchronous. Do not make the first loop async merely because agents may eventually run concurrently. First make the state machine and event protocol independent of transport. An async runner can then mirror the sync API when there is a real caller requiring streamed model output or concurrent tools.

### Persistence seam

Make `RunState` serializable early, but do not claim durable resume initially. A future `CheckpointStore` can save state only at safe boundaries—after a model turn and after each completed tool result. A checkpoint must include:

- Harness/schema version.
- Run and parent-run IDs.
- Agent definition reference or fingerprint.
- Conversation/observation state.
- Budgets consumed and pending tool calls.
- Completed tool call IDs and results.
- Current status and approval request.

Code/tool-version compatibility must be checked on resume. Durable engines such as Temporal or graph runtimes can later adapt this state model rather than forcing their storage semantics into the core.

## 6. Security and safety properties

- Treat model output as untrusted input; validate every tool name and argument.
- Keep authorization in the executor/policy layer, never in prompt instructions alone.
- Separate read-only and state-changing tools and require approval by policy metadata.
- Give runs least-privilege credentials and workspace scope.
- Place timeouts and output-size limits around tools.
- Sanitize tool errors and traces so secrets are not returned to the model or logs.
- Record approvals and rejected actions as events.
- Require idempotency keys for retryable state changes.
- Make cancellation cooperative and check it between calls.
- Default traces to excluding sensitive content while retaining metadata and hashes.

## 7. Testing strategy

The harness can be tested without network access, consistent with this repository's existing rule.

### Deterministic runner tests

Use a scripted fake model that returns a sequence of `ModelTurn` values. Assert exact events, context updates, tool invocations, usage, and stop reasons. Cover:

- Direct final output.
- One and multiple tool turns.
- Unknown tool and invalid arguments.
- Tool domain error followed by model recovery.
- Refusal and malformed model behavior.
- Every budget boundary and cancellation.
- Approval pause/reject/resume.
- Provider fallback without replaying completed tools.
- Context compaction and oversized observations.

### Tool contract tests

Generate invalid values from each schema, verify output schema enforcement, timeout behavior, error sanitization, and side-effect policy. Tool fixtures should make state changes inspectable so duplicate execution is detectable.

### Trajectory evaluations

Maintain small task environments with deterministic end-state graders. Store the complete event trajectory and compare:

- Final state correctness.
- Required/forbidden actions.
- Argument correctness and dependency order.
- Efficiency budgets.

Pin model and harness versions separately. A model upgrade and a tool-description change are different experimental variables.

## 8. Recommended roadmap

### Phase 0: Write the contract

Create an ADR defining ownership boundaries, normalized `ModelTurn`, tool/result schemas, run states, stop reasons, and explicit non-goals. Confirm requirements against the two current consumers before changing the public API.

### Phase 1: Minimal bounded runner

- Add provider-neutral `ModelTurn` with final output or tool calls.
- Add `Tool`, `ToolCall`, and `ToolResult`.
- Add a synchronous `run()` with max turns/tool calls and typed outcomes.
- Add an in-memory event collector.
- Test entirely with scripted model turns and mocked wire clients.

Success criterion: a caller can define two local tools and complete a multi-turn loop with a fully inspectable trajectory, while invalid actions and exhausted budgets fail predictably.

### Phase 2: Policy and context

- Approval decisions and resumable in-memory pause state.
- Tool timeouts, cancellation, and side-effect metadata.
- Pluggable context policy with simple compaction/artifact references.
- Usage/cost aggregation and redacted tracing.

### Phase 3: Workflow composition

- Deterministic sequential, routing, and parallel combinators around bounded runs.
- Nested-run/delegation primitive with parent/child budgets and events.
- Add multi-agent conveniences only after evaluations justify them.

### Phase 4: Durable adapters

- Versioned checkpoint contract.
- SQLite reference store for local recovery if demanded by a consumer.
- Optional Temporal/LangGraph integration rather than an in-house distributed scheduler.

## 9. Explicit non-goals for the first release

- A graph DSL.
- Autonomous background workers or a hosted service.
- Vector memory or retrieval framework.
- Multi-agent teams/handoffs.
- MCP client/server implementation.
- Browser, shell, or filesystem tools bundled with broad authority.
- Exactly-once tool execution.
- Automatic retries of arbitrary state-changing tools.
- Provider-specific agent semantics in the runner.

## 10. Decisions to resolve before implementation

1. Should the normalized turn support plain final text, structured final output, or both? Existing consumers favor structured results, but general agent loops often need text completion.
2. Does Phase 1 support multiple tool calls from one model turn, and if so, execute sequentially while preserving future parallel semantics?
3. Should tools be sync-only initially, or is there already an async consumer requirement?
4. What is the minimum approval contract: callback-only, or serializable pause/resume from the start?
5. Is emissary intended to stay a small library embedded by `stria` and `doom`, or become a separately deployed runtime? The proposed design assumes the former.
6. Which maintained JSON Schema implementation should enforce tool and structured-output contracts? Handwritten partial validation is not acceptable, but the dependency should be selected only after checking Python 3.13 support, supported draft, package health, and error quality.

## Conclusion

The strategic opportunity is not to compete immediately with full agent frameworks. It is to extend emissary's existing strength—small, explicit, provider-neutral contracts—one level upward into a trustworthy loop kernel. A good first harness makes model-directed execution bounded, inspectable, testable, and composable. Graphs, durable workers, memory, and multi-agent orchestration should be adapters or later packages built only when concrete consumers require them.
