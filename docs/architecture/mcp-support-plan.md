# MCP support implementation plan

Date: 2026-10-01. Status: implemented (phases 1–5); see ADR-0030 and ADR-0031. Sections 1–2 describe the code as it was before this work.

Deviations from the plan as written:

- With no authorizer configured the machine applies the compatibility policy without an effect or events, so existing logs and the golden replay fixture are unchanged. Runs with tool sources record the delegation on `tools_prepared`.
- Strict-schema conversion is not implemented: prepared MCP tools tell the OpenAI-compatible wire not to force strict mode.
- `StreamableHTTP.auth` is an `httpx2.Auth` (the SDK's HTTP stack), not a bespoke interface.
- Explicit refresh/invalidations for shared preparation are not built; a shared `PreparedRun` is frozen for its lifetime.

Emissary is the harness/runtime. `Agent` is the application's immutable configuration for that runtime. A direct tool is registered as a callable; it can perform computation, query a database, or call an external API. MCP describes a different discovery and invocation mechanism, not a different class of application capability.

## 1. Findings from the current implementation

| Area | Current behavior | Relevant source |
| --- | --- | --- |
| Configuration | `Agent.tools` is a tuple of `Tool`; construction performs no discovery. | `src/emissary/harness/agent.py` |
| Tool contract | `Tool` combines definition, callable, approval, API classification, and retry policy. Schema dictionaries remain mutable despite the frozen dataclass. | `harness/tools.py` |
| Registry | Validates API classification, rejects duplicate names, resolves by name, exposes `ToolDefinition`. | `harness/tools.py:174` |
| Preparation | No separate preparation stage. The machine builds `ToolRegistry(agent.tools)` when first driven. | `harness/machine.py:69` |
| Loop | Both drivers use the same generator state machine. Whole batches are resolved/schema-validated before execution; execution/approval are sequential. Results are logged and projected into the next model turn. | `harness/machine.py`, `harness/runner.py` |
| Async | `arun` awaits an executor return value. `LocalToolExecutor.execute` invokes the tool synchronously, so an async callable becomes an unawaited coroutine inside a result. The executor protocol describes sync returns only. | `harness/runner.py:231`, `harness/tools.py:146` |
| Policy | `approval_for` returns ALLOW immediately for `approval="never"`. There is no independent authorization hook or principal/tenant context. | `harness/policy.py:22` |
| Execution context | `ToolContext` contains only run ID and attempt. Idempotent direct tools receive an injected keyword argument. | `harness/tools.py:37` |
| Provider conversion | OpenAI-compatible uses `function.parameters` and may set `strict=True`; Anthropic uses `input_schema`; Gemini uses `parameters_json_schema`. They pass through input schemas. | `llm/wire/openai_compatible.py:220`, `anthropic.py:160`, `gemini.py:225` |
| Capability checks | `llm/model.py` rejects providers lacking chat/tool-calling support. Not every Emissary provider is an agent-loop provider. | `llm/model.py:52` |
| Observations | `ToolResult` is serialized as JSON into `ToolMessage.content`. Native image/audio tool observations are not supported. Returning `ToolResult` currently bypasses the local executor's output-schema validation. | `harness/tools.py:164`, `harness/projection.py:212` |
| Reliability | Limits, cancellation, retries, circuits, event projections, persistence and deterministic replay already exist. Approval PAUSE is terminal today, not durable resume. | `harness/runner.py`, `machine.py`, `eval/replay.py`, architecture documentation |
| MCP | No client dependency, transport, discovery, server configuration, or invocation implementation. | `pyproject.toml`, package inventory |

## 2. Proposed architecture

```text
Agent: direct tools + tool sources
                |
        async preparation
        connect / discover / filter / snapshot / bind
                |
        one prepared registry
                |
        existing agent_machine
        model -> resolve -> validate batch
              -> authorize -> approve
              -> authorize current attempt -> execute
                |
        routing executor
          /             \
 direct executor     MCP executor
 callable/API        SDK client -> server -> API/DB
          \             /
             ToolResult -> events -> next model turn
```

Use the official Python MCP SDK behind an optional `emissary[mcp]` extra. Pin a tested supported SDK major/minor range when implementation begins; keep SDK classes and transport details inside `emissary/mcp/`. The neutral harness must not import MCP or HTTP SDKs. Official SDK guidance covers connection lifecycle, `list_tools`, `call_tool`, and stdio/HTTP clients: [MCP client documentation](https://py.sdk.modelcontextprotocol.io/client/).

Keep one state machine and sequential multi-call execution. Preparation is outside the machine; authorization is an additional effect, allowing async application policy without introducing I/O into the machine.

Do not implement a gateway, signed grants, new hard business limits, tool search, parallel execution, or a universal plugin framework for the initial release. Clients own business rules and backend enforcement.

## 3. Proposed client configuration

Illustrative API names; all additions below are proposed:

```python
from emissary import Agent, arun
from emissary.mcp import MCPToolset, Stdio, StreamableHTTP, MCPToolPolicy

claims = MCPToolset(
    namespace="claims",
    transport=StreamableHTTP(
        url="https://claims.example.com/mcp",
        auth=claims_auth,  # trusted transport collaborator, not a token string in a schema
    ),
    include_tools=("lookup_claim", "list_documents"),
    default_policy=MCPToolPolicy(
        api_scope="external", approval="never", max_attempts=1,
    ),
)
files = MCPToolset(
    namespace="files",
    transport=Stdio(command="python", args=("files_server.py",)),
    include_tools=("read_document",),
    default_policy=MCPToolPolicy(api_scope="internal", approval="never"),
)
agent = Agent(
    name="claims_assistant",
    instructions="Review the claim using the available capabilities.",
    tools=(calculate_total, lookup_customer),  # existing Tool instances
    toolsets=(claims, files),
)
result = await arun(
    agent, "Review claim 123", caller=caller,
    authorizer=claims_policy,
    authorization_context=AuthorizationContext(
        principal_id="user-42", tenant_id="tenant-7",
    ),
    approver=approver,
)
```

Direct-only clients omit `toolsets`; MCP-only clients omit `tools`; mixed clients supply both. Direct tools retain their existing explicit API classification. MCP access is classified by client configuration, not inferred from transport: localhost can reach external services, and remote endpoints can belong to an internal API.

`arun` automatically prepares and closes sources for one run. Add `async with prepare(agent) as prepared:` plus `arun(..., prepared=prepared)` for multiple runs within one owned connection lifetime. Prepared objects are tied to an event loop, configuration snapshot, and authentication identity; do not reuse them across tenants/identities. Each run receives separate authorization context and decisions.

`run` remains supported for direct synchronous tools. Initially reject async sources/tools with a clear message directing the caller to `arun`; do not silently create a new event loop per call. A whole-run sync convenience facade can follow separately.

## 4. Smallest interface changes and ownership

| Interface/module | Planned change |
| --- | --- |
| `Agent` | Append `toolsets: tuple[ToolSource, ...] = ()` after existing fields to preserve positional compatibility. No MCP imports. |
| New `harness/sources.py` | Narrow `ToolSource` protocol: `open()` returns an async context manager yielding a `PreparedToolSource`, whose `list_tools()` returns executable `Tool` bindings plus neutral origin metadata. No model calls. |
| New `harness/preparation.py` | Own `PreparedRun`, connection stack, snapshots, merged registry, source metadata and executor routes. Preparation returns a complete catalog or fails; no partially usable required source. |
| `Tool`, `ToolRegistry` | Preserve public constructor/resolve behavior. MCP bindings have real async invocation callables, not dummy executors. Copy definitions during preparation. Store origin separately in prepared bindings: namespace, original name, endpoint identity, contract fingerprint. |
| `ToolExecutor` | Preserve sync protocol; add `AsyncToolExecutor` typing with awaitable results. Existing custom executors remain usable for direct tools. |
| New async direct executor | Await async callable output before normalization. Factor validation/normalization helpers with the sync executor. Blocking sync direct tools need an explicitly selected thread/off-process executor; automatic offloading would change thread-affinity/cancellation behavior. |
| New routing executor | Route by prepared binding, not name parsing. Delegate direct tools to caller-supplied executor or bundled direct executor; MCP tools to bundled MCP executor. In mixed runs, existing `executor=` customizes direct tools only; document this. |
| `run`, `arun` | Add authorizer/context arguments; `arun` adds optional prepared handle and manages preparation when absent. |
| `agent_machine` | Accept prepared registry, authorizer effects/context, and emit policy decisions. Retain legacy direct registry fallback for standalone machine tests. No MCP/session logic. |
| `effects.py` | Add `AuthorizeTool`; extend execution context with a bound authorization decision. Keep `ExecuteTool`/executor signatures intact. |
| `policy.py` | Add authorization request/context/decision types and explicit compatibility policy; preserve existing approval callback. |
| `state.py` / events / replay | Add authorization-denied/error stop reasons and safe policy/preparation metadata. Add recorded policy replay without live credentials or servers. |
| `llm/decision.py`, wire helpers | Add an appended optional per-definition strict-schema preference if needed; derive provider schema separately from original validation schema. Preserve defaults for existing direct tools. |
| `mcp/` | Source configuration, SDK connection adapters, discovery, invocation, result normalization. SDK imports are opt-in. |

## 5. Discovery, naming and schema conversion

For each configured source: open transport/client, negotiate the supported protocol, confirm tools capability, fetch all discovery pages, apply include/exclude filters, validate definitions, apply client policy overrides, then bind invocation to the same source identity. Detect repeating cursors and failed pages; discard partial catalogs. Do not automatically inject server instructions into system instructions.

Require a unique client-supplied namespace. Preserve direct names. Generate MCP aliases using `namespace__original_name` when within portable provider constraints. Sanitize unsupported characters and use a truncated prefix plus a stable hash of the namespace/original-name pair when normalization or length limits require it. Validate the final alias against the whole registry; fail on any remaining collision and allow explicit client aliases. Never resolve collisions by server discovery order or incremental suffixes. Sort each source's exposed tools by alias; preserve configured source order and direct-tool order for a stable catalog.

Keep an explicit mapping `alias -> source identity, original tool name, contract fingerprint, invocation binding`. The executor never splits an alias to select a server. A schema fingerprint alone is not a guarantee of unchanged server implementation; backend versioning/enforcement remains outside this plan.

Freeze the catalog for each run. Start without cross-run discovery caching. With shared preparation, provide explicit refresh between runs; invalidate on relevant SDK notifications, respecting negotiated protocol support. Refresh must not replace bindings in an active run. Later caching must key by authorization identity, source configuration, protocol and catalog version; use detached copies.

MCP discovery returns JSON Schema definitions; Emissary already has neutral tool definitions and three function-call wire adapters. Preserve the original schema for executor validation; derive model-facing schemas in wire helpers. Support object input schemas, local `$defs`/references and declared dialects deliberately; reject unsupported cases with the source/tool identity. Do not fetch arbitrary remote schema references during preparation or execution.

For strict OpenAI-compatible wires, enable strict mode only when a semantics-preserving supported conversion is possible. Otherwise use non-strict function calling where supported, or fail with a capability error. Do not silently turn optional fields into required fields or alter argument semantics. Apply the same compatibility path on model fallback; injected opaque model callers own their provider compatibility contract.

The MCP tools specification defines schemas, result forms and untrusted annotations: [tools specification](https://modelcontextprotocol.io/specification/2026-07-28/server/tools). Server annotations may inform client-selected policy but must not automatically grant approval exemptions or retries.

## 6. Independent authorization and exact-request binding

Add a trusted `AuthorizationContext` supplied by the application (principal, tenant, nonsecret application references). Model arguments cannot override it. Application authorizers may be synchronous or asynchronous; asynchronous authorizers are awaited by `arun` through `AuthorizeTool`. They receive tool origin, original/alias identity, definition fingerprint, immutable argument snapshot, run/call/attempt identity and trusted context. They can inspect authoritative backend state through their own collaborator.

Sequence:

1. Resolve and snapshot every proposed call; schema-validate the entire batch before effects.
2. Authorize each call in the batch before executing any of that batch. Denial stops the run with no tool effects from that batch.
3. For each call, obtain approval through the existing callback when configured. `approval="never"` never skips authorization.
4. Immediately before every execution attempt, including the first and retries, reauthorize the exact request against current trusted context. Rejection here stops later execution but cannot undo earlier calls in the batch; batches are not transactions.
5. Dispatch only if the final decision matches the immutable request, source binding and execution identity. Record the result normally.

Use an internal `InvocationRequest` containing owned canonical JSON bytes rather than relying on frozen dataclasses containing mutable dictionaries. Canonicalization must be documented, reject non-finite/non-JSON values, and be stable for the supported Python runtime. Build a digest over an unambiguous structured envelope including source identity, tool fingerprint, alias/original name, arguments, principal/tenant, run and logical call identity. The decision additionally binds attempt, expiry if present and policy version. Give callbacks detached views; neither approval nor authorization mutation may change the snapshot subsequently dispatched. Reconstruct executor arguments from that snapshot and verify the decision immediately before delegation.

Trusted execution metadata (including idempotency keys) stays separate from model arguments. Preserve direct tools' current keyword injection as a documented compatibility adapter; bind its effective injected values before authorization and prevent model overrides. MCP must not blindly receive an `idempotency_key` argument it has not declared. Initial MCP default is one attempt; retries require client-established idempotency and a supported server convention. No automatic retry after uncertain side effects.

When no authorizer is supplied, invoke an explicit `AllowRegisteredTools` compatibility policy: the application has delegated its registered catalog. This preserves existing behavior, does not claim resource authorization, and is recorded as such. A configured policy error fails closed. Authorization and human approval are separate; approval cannot override DENY.

OAuth/API credentials are resolved only by transport configuration or trusted direct-tool implementations. Do not store raw tokens in schemas, model messages, grants, fingerprints, events, errors or configuration reprs. Discovery may require authentication before any tool call; the execution check does not delay all credential acquisition until after approval. MCP and backend permissions remain enforced server-side: [MCP authorization specification](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization).

The built-in dispatcher guarantees use of the approved request; arbitrary injected executors and direct callables remain trusted extension code. In-process policy cannot contain code that can independently read secrets or make network calls. Signed remote grants are a later optional adapter requiring verifier cooperation, audience restrictions, replay handling and backend enforcement. Exact arguments do not resolve backend state races: contextual constraints must also be checked atomically by the API/DB executing the operation.

## 7. Lifecycle, invocation and results

Use an owned async connection stack with reverse-order cleanup, including when a later server fails discovery. Default sources are required; optional sources may be omitted only when explicitly configured, with an observable preparation diagnostic. Reject attempts to use absent tools; do not reroute them to another server.

Support stdio subprocess parameters (command, argument vector, working directory, explicitly supplied environment) and Streamable HTTP endpoint/auth collaborators. Avoid shell command interpolation and full environment inheritance. Connection ownership means SDK lifecycle ownership, not a requirement that every protocol version has a server-side session. Use SDK protocol negotiation, cancellation and current/legacy HTTP behavior rather than implementing JSON-RPC manually. SSE is outside the initial acceptance scope.

Allocate run identity and overall deadline before automatic preparation. Include preparation in run duration; use separate configured connect/discovery/cleanup timeouts with bounded cleanup even after cancellation. Record run-start/preparation events before discovery, and classify preparation failure/cancellation as terminal outcomes. Refactor the machine's initial events to accept a prepared prefix so sequence numbers, projection and replay remain canonical. Do not duplicate `run_started`. Explicit preparation outside a run has its own diagnostic/error surface and deadline.

After preparation, invocation sends only original tool name and authorized structured arguments to the bound SDK client. Preserve the provider's call ID for model observations; transport request IDs are separate. Keep transient connection recovery distinct from replaying a tool call. Do not advertise sampling/elicitation or other server-requested capabilities until Emissary has explicit budgeted handlers; unsupported input-required responses fail clearly rather than recursively invoking a model.

Normalize MCP responses into `ToolResult(content={"text": [...], "structured": ...})`, using SDK-to-JSON conversion and sanitized summaries. Map tool-reported errors to model-visible error results. Map protocol/transport failures to safe execution errors; classify timeouts explicitly, with retries conservative. Validate successful structured content against the declared output schema before wrapping it in `ToolResult`—the existing local early return would otherwise bypass validation. Error results need not satisfy a success-output schema.

First release supports text, JSON and explicit resource references. Embedded text may be preserved; image/audio/binary blocks require an explicitly configured artifact sink or a clear unsupported-content result. Never silently drop them or stuff base64 into model text. Resource references do not imply authorization to fetch them. Keep existing result byte limits and error/circuit behavior; larger results require the client's artifact strategy. Never claim cancellation rolled back a backend action.

## 8. Implementation phases and acceptance gates

| Phase | Files/work | Acceptance gate |
| --- | --- | --- |
| 1: neutral contracts and async direct execution | `sources.py`, `preparation.py`, `tools.py`, runner typing; append `Agent.toolsets` | Direct sync behavior remains compatible; async callable results are awaited; source snapshots and deterministic routing tested without MCP installed. |
| 2: common authorization | `policy.py`, `effects.py`, `machine.py`, `runner.py`, `state.py`, events/projection/replay | DENY blocks direct and fake-source calls even with approval NEVER; mutated views cannot change dispatch; every retry is reauthorized; policy exceptions stop safely. |
| 3: MCP adapters | `mcp/config.py`, `client.py`, `tools.py`, optional dependency; automatic preparation integration | Real stdio and loopback Streamable HTTP servers discover/invoke/close; authentication remains outside model data; cancellation and partial preparation clean up. |
| 4: schema/results compatibility | neutral definition preference, wire schema helper, MCP normalization | Original validation contracts preserved; all tool-calling wires handle supported schemas; incompatible schemas fail explicitly; text/structured/error/resource results tested. |
| 5: complete workflow and documentation | public exports, examples, README, proposed ADRs, architecture/import tests, replay fixtures | Direct-only, MCP-only and mixed multi-hop runs use the same machine; existing tests/fixtures remain readable; base import works without optional SDK. |

Phases 1–5 together are the first MCP release. Connection reuse can land with preparation; cache optimizations, native multimodal tool results, full OAuth UI/token-store management, signed grants and synchronous MCP convenience are follow-ups, not prerequisites. Auth collaborator injection is required in phase 3.

Do not silently rewrite golden fixtures to absorb new events. Define compatibility for legacy logs and a recorded authorization replay path, and migrate event schemas only if serialization semantics actually require it.

## 9. Tests

The principal integration test uses a scripted model caller and actual fixture MCP servers, so failures isolate harness behavior from model variation:

1. Expose a direct `lookup_customer` tool whose fixture calls an HTTP API.
2. Model selects it and receives a customer ID.
3. Model calls `claims__lookup_claim` over Streamable HTTP using that ID.
4. Model calls `files__read_document` over stdio using the document ID returned by claims.
5. Model calls direct `calculate_total` using the retrieved values, then returns a final answer.
6. Assert call arguments, server identities, preserved call IDs, each subsequent model context, policy-before-execution order, event trajectory, terminal status, and closed connections/processes.

Additional contract tests:

- Direct-only sync/async and MCP-only/mixed runs; unchanged custom direct executor behavior; same machine for all modes.
- Two servers exposing the same original name; direct/MCP collisions, explicit aliases, normalization collisions, long names, changing discovery order.
- All discovery pages, repeated cursors, absent tools capability, partial failure, required versus optional source, frozen active catalogs, identity-separated reuse.
- Input/output schema failures, dialect/local-reference handling, unavailable remote refs, strict-schema incompatibility, provider fallback, providers lacking tool calling.
- All-or-no-effects schema/authorization rejection for a batch; no claim of atomicity after execution starts.
- Authorization always runs with approval NEVER; tenant/principal spoofing in arguments rejected; policy errors fail closed; approval cannot override DENY.
- Mutation of original arguments, nested dictionaries, approval views and schemas after preparation; source binding or digest mismatch; expiry before dispatch; policy changes between retries.
- Stable logical idempotency key for direct retries; no unexpected MCP argument injection; no retry of uncertain writes.
- Credential sent to the fixture transport/API but absent from caller inputs, errors, reprs, serialization and events; separate identities do not share authorization/cache data.
- Successful text/structured content, tool `isError`, protocol error, timeout, malformed/nonserializable result, output mismatch, oversized result, explicit unsupported binary content.
- Cancellation/deadline during connect, discovery, authorization and invocation; cleanup after later-source failure; no leaked subprocess; no duplicate terminal/start events.
- Pure machine authorization effects, both drivers' exhaustive dispatch, deterministic replay and persisted legacy-fixture compatibility without live MCP connections.
- Architecture tests: no MCP imports in harness/LLM layers, no HTTP SDK outside approved adapters, optional extra absent on base install. Extend the existing HTTP import rule narrowly for MCP adapters.

Run targeted tests after each phase, then the existing full pytest suite and repository lint/format checks for the completed feature. Fixture transports run locally without real service credentials; live-service validation is optional application integration work.

## 10. Design alternatives

- A separate MCP runner would duplicate policy, retries and event projection; rejected.
- Making every direct tool an MCP server adds deployment/protocol work to ordinary callables; rejected.
- Only adding an MCP executor leaves discovery, registration and ownership to every client; insufficient.
- Injecting async wrappers into the current local executor does not work because coroutine outputs are not awaited; requires the phase 1 change.
- Mandatory gateway/signed grant infrastructure cannot enforce arbitrary third-party servers and unnecessarily expands the first release; keep an adapter path.

The first release is complete when direct tools and both MCP transports share discovery preparation, one registry, independent authorization and approval, one execution machine, and verifiable request/result histories.
