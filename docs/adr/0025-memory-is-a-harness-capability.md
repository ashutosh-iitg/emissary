# ADR-0025: Memory Is a Harness Capability; Storage Is the Application's

**Date:** 2026-09-27  
**Status:** accepted  
**Deciders:** Project maintainers  
**Supersedes:** the "vector memory, retrieval" non-goal in `docs/architecture/agent-harness.md` §3 and decision D9

## Context

The first stable harness ruled memory out: every run started from one task string, and
anything an application wanted the model to remember it had to paste in itself. A new
consumer, Tavi (a learning companion for children), needs an agent that continues a
conversation across requests, remembers the learner across sessions, and adapts how it
teaches. Built inside each application, that logic would be rewritten per consumer and
would drift. Putting it in emissary needs one rule kept: emissary stays a library with no
state.

The research this rests on separates three independent questions (CoALA, Sumers et al.
2023; MemGPT; Generative Agents; LangGraph's memory guidance):

- **How long it lasts.** Immediate context (one model call), working memory (a session),
  and long-term memory (across sessions).
- **What it holds.** Semantic memory is facts, episodic memory is events, and procedural
  memory is how to act.
- **What is the source of truth.** Scores, mastery, curricula and due dates are
  structured application data. A vector index can help find memories, but it is never
  the record.

## Decision

**Emissary owns the contracts, the tools and the policies. The application owns
storage.**

- **Working memory.**
  - `run`/`arun` accept `history`: earlier messages, recorded as one `history_loaded`
    event, so replay reproduces exactly what the model saw.
  - History must open with a user message and end on a plain assistant reply.
  - A `Scratchpad` (session notes) is reached through the `take_note` tool and shown to
    the model at the start of each run.
- **Long-term memory** is three record types behind small store protocols (`FactStore`,
  `EpisodeStore`, `ProcedureStore`):
  - A `Fact` is `explicit` (the user said it) or `inferred`. An inferred fact cannot exist
    without episode ids as evidence. It carries no confidence score, because
    self-reported confidence is uncalibrated.
  - An `Episode` is a bounded notable event, with refs to the application's own records.
  - A `Procedure` is a user-specific strategy that becomes `active` only when enough
    separate episodes support it. Global rules are not procedures: they are the agent's
    instructions, versioned in application code and never written at runtime.
- **Vector stores.** Any `VectorStore` is wrapped by `vector_search_tool`. The
  application declares the filter schema (input), and the match shape is fixed (output).
  Both are validated like any tool, so a filter the application didn't declare never
  reaches the store.
- **Write path, split by type.**
  - Explicit facts and scratch notes are written in the response path through tools.
  - Episodes, inferred facts and strategies are written after the run:
    `consolidate`/`aconsolidate` asks the model the judgment questions, and
    `apply_consolidation` decides in code what is stored.
  - Nothing is inferred without an episode, an inference never retracts what the user
    said, and a strategy is promoted only at `promote_at` supporting episodes (3 by
    default, a placeholder until real consolidations are reviewed).
- **Staleness.** An inferred fact carries `last_supported_at`. Consolidation can
  reaffirm it (the new episode is added as evidence and the time renewed) or contradict
  it, and contradiction wins when both happen. `load_recall(inferred_max_age=...)` leaves
  out inferred facts nothing has supported within the window, without deleting them.
  Explicit facts never go stale. The window is the application's choice.
- **Read path.** `load_recall` reads a compact profile, and `with_memory` returns a new
  `Agent` with it appended after the agent's own instructions:
  - Inferred facts are labelled as such.
  - Candidate strategies are withheld.
  - Anything larger is fetched on demand (`recall_episodes`, vector search).
- **Dependency direction.** `memory` depends on `harness` and `llm`, and neither of them
  depends on `memory`. An architecture test enforces this.

## Alternatives Considered

- **Memory inside each application.** No change to emissary, but every consumer
  re-derives the same evidence and promotion rules and gets them subtly different.
  Rejected at the user's direction.
- **Render earlier turns into the task string.** No harness change, but it loses message
  structure, tool-call pairing and reasoning state, and the persisted history mixes what
  was said with what was recalled. Rejected.
- **Emissary ships database-backed stores.** Convenient, but emissary would then hold
  state and choose a database for every consumer. Rejected: stores are protocols.
- **Every memory written by tools in the response path.** The simplest option, but it
  adds latency to every reply and lets one ambiguous turn become a permanent trait.
  Rejected.
- **The agent edits its own procedures.** Rejected: a single conversation could rewrite
  teaching or safety behaviour. Strategies need separate, evidenced support, and global
  rules are unreachable.

## Consequences

### Positive

- One implementation of memory policy for every consumer, tested without a network or
  a database.
- Memory writes appear in the run's event log like any other tool call.
- Stores are scoped by the application, so no tool can name another user's memory.

### Negative

- Memory in the system prompt changes it whenever facts or notes change, which reduces
  prompt-cache hits compared with a static system prompt. It was chosen so that the
  persisted history holds only what was said.
- Store protocols are synchronous. An async application (Django under ASGI) wraps its
  stores, or supplies an executor that returns awaitables, which `arun` already awaits.

### Risks

- Strategy names are how evidence accumulates. A model that renames the same strategy
  splits its support. The consolidation prompt asks it to reuse listed names.
- `RecentHistory` keeps the first message of the surface. With history seeded, that is
  the oldest historical message, not the new task. It's harmless for short sessions and
  should be revisited if long seeded histories get trimmed.
