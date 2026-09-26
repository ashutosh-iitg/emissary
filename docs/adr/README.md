# Architecture Decision Records

| ADR | Title | Status | Date |
|---|---|---|---|
| [0001](0001-one-model-boundary.md) | Emissary is the sole model boundary | accepted | 2026-08-14 |
| [0002](0002-add-call-model.md) | Add call_model and preserve specialized APIs | accepted | 2026-08-14 |
| [0003](0003-discriminated-model-decisions.md) | Use discriminated model decisions | accepted | 2026-08-14 |
| [0004](0004-explicit-capabilities.md) | Fail explicitly on unsupported capabilities | accepted | 2026-08-14 |
| [0005](0005-bounded-sync-runner-first.md) | Build a bounded synchronous runner first | accepted | 2026-08-14 |
| [0006](0006-tool-executor-protocol.md) | Isolate tool execution behind a protocol | accepted | 2026-08-14 |
| [0007](0007-sequential-tool-execution.md) | Execute tool-call batches sequentially first | accepted | 2026-08-14 |
| [0008](0008-events-are-canonical.md) | Use events as the canonical trajectory | accepted | 2026-08-14 |
| [0009](0009-solid-dependency-rules.md) | Enforce SOLID through dependency rules | accepted | 2026-08-14 |
| [0010](0010-json-schema-validation.md) | Use Draft 2020-12 JSON Schema at tool boundaries | accepted | 2026-08-14 |
| [0011](0011-event-log-derived-messages.md) | Derive model-visible messages from the event log | accepted | 2026-08-15 |
| [0012](0012-idempotency-keys-for-tool-retries.md) | Idempotency keys gate state-changing tool retries | accepted | 2026-08-15 |
| [0013](0013-orthogonal-tool-outcomes.md) | Report orthogonal tool outcomes independently | accepted | 2026-08-15 |
| [0014](0014-per-tool-circuit-breaking.md) | Per-tool circuit breaking | accepted | 2026-08-15 |
| [0015](0015-deterministic-replay-tests.md) | Deterministic replay tests from recorded event logs | accepted | 2026-08-15 |
| [0016](0016-structured-prompt-assembly.md) | Structured prompt assembly | superseded by 0017 | 2026-08-15 |
| [0017](0017-request-assembly.md) | Assemble requests, not system prompts | accepted | 2026-08-15 |
| [0018](0018-reasoning-is-two-concepts.md) | Reasoning output is two separate concepts | accepted | 2026-08-15 |
| [0019](0019-thinking-control.md) | Provider-neutral thinking control, gated on capability | accepted | 2026-08-15 |
| [0020](0020-when-a-provider-earns-a-native-wire.md) | When a provider earns a native wire | accepted | 2026-08-15 |
| [0021](0021-credentials-are-a-strategy.md) | Credentials are a strategy, not an env var name | accepted | 2026-08-15 |
| [0022](0022-streaming-is-an-observation-channel.md) | Streaming is an observation channel, not a second result type | accepted | 2026-08-15 |
| [0023](0023-async-at-the-model-boundary.md) | Async at the model boundary, not (yet) in the runner | accepted | 2026-08-16 |
| [0024](0024-sans-io-runner-core.md) | One loop, two drivers — a sans-I/O runner core | accepted | 2026-08-16 |
| [0025](0025-memory-is-a-harness-capability.md) | Memory is a harness capability; storage is the application's | accepted | 2026-09-27 |
| [0026](0026-embeddings-and-ocr.md) | Embeddings and OCR are calls, not conversations | accepted | 2026-09-27 |
| [0027](0027-jev-decision-wire.md) | Jev gets a native wire, behind `call_choice` | accepted | 2026-09-27 |

