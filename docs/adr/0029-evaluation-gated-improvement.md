# ADR-0029: Evaluation-Gated Improvement of Projects Built on emissary

**Date:** 2026-09-28  
**Status:** accepted  
**Deciders:** Project maintainers  
**Amends:** architecture §3 (no built-in filesystem tools), for `emissary.improve` only

## Context

A project built on emissary — stria is the named consumer — should be able to
use the harness to improve itself: its prompts, tool descriptions, tool code and
control flow. Two systems were read in full before deciding:

- **AVO** (arXiv 2603.24517) replaces mutation with a coding agent,
  `Vary(P_t) = Agent(P_t, K, f)`, and commits a version only if it "passes
  correctness checks and matches or improves" the best so far (§3.2). Its `f` is
  low-noise (10 timed runs and a deterministic correctness check, §4.1), and it
  has **no held-out set**: the same benchmark drives evolution and reporting.
- **Prime Agent** (arXiv 2608.23552) refines prompts, memories, skills and
  subagent specs at runtime, automatically every 25 turns by default, with no
  outcome gate. In one Factorio trace refinement saved an exploit of the metric
  as a reusable skill (§3.5); the authors prescribe least-privilege interfaces,
  independent validation and auditable rollback. No ablation isolates
  refinement, and no intervals are reported (Table 1, §5).

## Decision

`emissary.improve.improve()` runs **one bounded round** over a consumer's git
repository. The model proposes; code decides.

1. **The improver is an emissary `Agent`** with four tools confined to a
   throwaway worktree: `list_files`, `read_file`, `search`, `write_file`.
   Writes are allowed only under consumer-declared `editable` globs. There is no
   shell tool.
2. **The evaluator is the consumer's command**, run by code in its own process
   (a candidate's modules cannot be reloaded into this one). It reads the split
   from `EMISSARY_EVAL_SPLIT` and writes scores and failing runs to the file in
   `EMISSARY_EVAL_OUTPUT` (`write_eval_json`).
3. **`protected` paths are invisible**, not merely read-only: they hold graders
   and holdout scenarios. A candidate whose diff touches one is rejected.
4. **The gate is `eval.compare`**: accept iff total successes rise by at least
   `margin` and no scenario the baseline passed on every attempt falls to mostly
   failing. The improver sees search-split failures (rendered from events) and
   earlier rejected attempts; the first candidate to pass is scored on the
   **holdout exactly once**.
5. **An accepted candidate becomes a branch** for human review, with a report
   putting the improver's claimed effect beside the measured one. Nothing is
   written to the working branch or at runtime (consistent with ADR-0025).

## Alternatives Considered

- **A single `call_tool` returning a patch.** Simpler, no new tools, but it
  cannot read code it was not shown and does not scale past a few files; AVO's
  central claim is that single-turn generation is the limitation.
- **AVO's rule as written (≥ best on the visible suite).** Sound for a
  low-noise `f`; with ~10 LLM scenarios at a few attempts it accepts noise and,
  without a holdout, overfits the suite.
- **A statistical test.** With a handful of scenarios and attempts it has no
  power, and it would add a dependency.
- **A stagnation supervisor or candidate archive (AVO §3.3).** Neither was
  ablated; both serve week-long continuous runs. A library round has nothing to
  stall, and git branches already are the archive.
- **Runtime refinement (Prime Agent).** Rejected: changes that no evaluation
  checked would reach users.
- **Giving the improver the evaluator (AVO's agent calls `f`).** Rejected: an
  improver that can run or read its grader can optimise the grader.

## Consequences

### Positive

- Any project built on emissary can improve itself against its own evals without
  a human writing each change, and every change arrives as a reviewable branch.
- `eval.compare` and `write_eval_json` are useful on their own for A/B checks.

### Negative

- `improve` ships filesystem tools and runs subprocesses. It stays opt-in:
  nothing else in emissary imports it (`tests/test_architecture.py`).
- Rounds are expensive: each attempt re-runs the consumer's search suite.

### Risks

- **The eval command runs model-written code.** A worktree is not a security
  sandbox; run `improve` inside a container or CI job. emissary provides no
  isolation.
- **`protected` must cover every grader and scenario file.** A consumer who
  leaves one editable reopens the exploit path; `improve` refuses an empty
  `protected` but cannot check completeness.
- **The gate trusts the grader.** A weak grader (e.g. "any number appears")
  admits changes that game it; the human review is the backstop.
