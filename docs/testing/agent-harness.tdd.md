# Agent Harness TDD Evidence

**Date:** 2026-08-14  
**Source:** `plans/agent-harness-blueprint.md`; journeys were refined from the approved architecture and ADRs.

## Implemented guarantees

| Area | RED evidence | GREEN evidence | Guarantee |
|---|---|---|---|
| Neutral model contracts | `ModuleNotFoundError: emissary.decision` | `7 passed` in `tests/test_model_types.py` | Messages, capabilities, decisions, usage, and provenance have provider-neutral validated shapes. |
| Wire normalization | Six `AttributeError` failures for missing wire `call_model` | `21 passed` across general and legacy wire tests | Anthropic and OpenAI-compatible responses normalize to the same final/tool/refusal contract. |
| Public caller/fallback | `ModuleNotFoundError: emissary.model` | `tests/test_model.py` passes | One credential-gated caller dispatches by wire and fallback occurs only for retryable availability failure. |
| Tool boundary | `ModuleNotFoundError: emissary.tools` | `4 passed` in `tests/test_tools.py` | Complete batches can be schema-validated before effects; output and exception boundaries are normalized. |
| Bounded runner | `ModuleNotFoundError: emissary.agent` | `5 passed` in `tests/test_runner.py` | Tool loops complete, recover from tool errors, reject invalid batches without partial effects, and stop on typed limits/refusal. |
| Policy/context | `ModuleNotFoundError: emissary.policy` | `3 passed` in `tests/test_policy_context.py` | Sensitive tools require external authority and context trimming preserves call/result pairs. |
| Evaluation | `ModuleNotFoundError: emissary.evaluation` | `2 passed` in `tests/test_evaluation.py` | Repeated runs report pass rate/usage and trajectories support required/forbidden event grading. |
| Persistence | `ModuleNotFoundError: emissary.persistence` | `1 passed` in `tests/test_persistence.py` | Complete run records round-trip through a versioned SQLite store. |
| Dependency architecture | Test introduced with public-surface integration | `2 passed` in `tests/test_architecture.py` | Provider SDK imports stay in wire adapters and harness core does not depend on providers or wires. |

## Final verification

```text
uv run pytest -q
82 passed in 2.27s

uv run ruff check .
All checks passed!

uv run black --target-version py313 src tests
37 files left unchanged after formatting

git diff --check
passed with no output
```

## Coverage and known gaps

The repository does not configure a coverage plugin, so no percentage is claimed. Every new module has direct tests and the full pre-existing suite remains green. No test reaches a live model endpoint.

Intentional architecture gates remain:

- The SQLite adapter archives versioned run records; it is not durable execution and cannot resume a pending side effect.
- Async/parallel execution requires a concurrency ADR and effect-independence policy.
- Workflow and multi-agent helpers are not promoted until at least two consumers demonstrate a shared pattern; nested runs can currently be composed with ordinary Python and bounded `run()` calls.
- Automatic model-based context summarization is not included; context selection is deterministic and network-free.

These are omitted guarantees, not silently skipped tests.

---

# Reliability & Prompt Assembly (ADRs 0011–0017)

**Date:** 2026-08-15 | **Source:** the approved six-stage implementation plan.

| Stage | RED evidence | GREEN evidence | Guarantee |
|---|---|---|---|
| 0011 event log | `ImportError: cannot import name 'ContextOp'` | `15 passed` in `tests/test_projection.py` | The log is the only source of conversation state; messages are a pure projection; every omission is a logged op. |
| 0013 orthogonal outcomes | `TypeError: unexpected keyword argument 'timed_out'` | `tests/test_tools.py` | `status` stays a pure severity axis; timeout and retryability are independent fields. |
| 0012 idempotency | `ImportError: cannot import name 'ToolContext'` | `tests/test_tools.py`, `tests/test_runner.py` | Retries require a declared-idempotent tool and deliver a key stable across attempts. |
| 0014 circuit breaking | `TypeError: unexpected keyword argument 'max_tool_failures'` | `tests/test_runner.py` | One flaky tool opens its own circuit instead of consuming the run's budget. |
| 0015 replay | `ImportError: emissary.eval.replay` | `4 passed` in `tests/test_replay.py` | A recorded run replays to an identical trajectory, survives persistence, and fails loudly when it diverges. |
| 0017 request assembly | `ImportError: emissary.llm.prompt` | `4 passed` in `tests/test_prompt.py` | A request is one value with a stable fingerprint; the two call shapes cannot be mixed. |

## Final verification

```text
uv run pytest -q                    120 passed
uv run ruff check .                 All checks passed!
uv run black --check src tests      clean
git diff --check                    clean

stria:  manage.py test intake       129 passed (2 skipped)
doom:   pytest -q                    19 passed
```

## Deviations from the plan, and why

- **The executor still re-validates input.** The plan folded in a "double-validation
  cleanup". `test_local_executor_validates_input_before_effects` exists to pin that
  no effect runs on invalid input, and `ToolExecutor` is a public protocol that
  cannot assume its caller validated. Dropping the check traded a safety property
  for a micro-optimisation, so it was kept.
- **`Agent.instructions` stays a `str`.** ADR-0017 said it would accept a `Prompt`;
  see the amendment recorded in that ADR.
- **stria's prompt hashes change.** Removing the `{document}`/`{criteria}`
  placeholders alters `LAW_LAYER_PROMPT_HASH` and `RELATIONSHIPS_PROMPT_HASH`.
  That is the audit trail working: the prompt genuinely changed, and drafts made
  before and after are now distinguishable.

## Known gaps

Assistant text accompanying *tool calls* is still discarded — in the wire adapter,
not the runner. The event payload accommodates the fix additively; it was left
out of scope deliberately.
