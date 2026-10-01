"""One bounded round of evaluation-gated improvement over a consumer's repository.

The model proposes; code decides (Rule 5). An emissary agent — the improver —
edits a throwaway git worktree through the tools in `workspace`. It never runs
the evaluator and never sees the holdout: the consumer's eval command is run by
this module, and only its search-split failures are shown to the improver.

A candidate must beat the baseline on the search split (`compare`), and the
first one that does is scored on the holdout exactly once. Scoring the holdout
again for a second candidate would be shopping for one that happens to pass —
the same reason emissary falls back only once. An accepted candidate becomes a
branch for a human to review; nothing here touches the working branch.

The eval command runs code a model wrote. A worktree is not a security sandbox:
run `improve` inside a container or CI job.
"""

import logging
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..eval.evaluation import ScenarioScore, compare
from ..harness.agent import Agent, RunLimits
from ..harness.execution.runner import run
from ..harness.state import RunStatus
from ..harness.tooling.tools import Tool, ToolResult
from ..llm.model import ModelCaller
from .contract import OUTPUT_ENV, SPLIT_ENV, EvalResult, read_eval_json
from .digest import failure_digest
from .workspace import Workspace

logger = logging.getLogger(__name__)

IMPROVER_LIMITS = RunLimits(
    max_turns=40,
    max_model_attempts=40,
    max_tool_calls=80,
    max_tool_attempts=80,
    max_duration_seconds=1800.0,
)

INSTRUCTIONS = """\
You improve an AI agent project by editing its source code in a git worktree.

You are shown runs that failed the project's evaluation. Find the cause in the \
code — prompts, tool descriptions, tool implementations, control flow — and fix \
it with the smallest general change. You cannot see or run the evaluation, and \
you must not special-case the inputs you are shown: the change is scored on \
scenarios you have not seen, and a human reviews the diff.

Use list_files, read_file and search to understand the code before writing. \
Only files marked (editable) can be changed. When you are done, call `submit` \
exactly once with a summary of the change and the effect you expect it to have."""


class EvalCommandFailed(RuntimeError):
    """The consumer's eval command failed on the unchanged code.

    Without a baseline there is nothing to compare a candidate with, so this
    ends the round rather than being recorded as a rejected attempt.
    """


@dataclass(frozen=True)
class Attempt:
    number: int
    summary: str
    expected_effect: str
    changed: tuple[str, ...]
    diff: str
    search: tuple[ScenarioScore, ...] = ()
    rejected: tuple[str, ...] = ()


@dataclass(frozen=True)
class ImprovementDecision:
    accepted: bool
    base: str
    branch: str | None
    baseline_search: tuple[ScenarioScore, ...]
    attempts: tuple[Attempt, ...]
    baseline_holdout: tuple[ScenarioScore, ...] = ()
    candidate_holdout: tuple[ScenarioScore, ...] = ()
    reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def report(self) -> str:
        return render_report(self)


def improve(
    repo: str | Path,
    *,
    caller: ModelCaller,
    eval_command: Sequence[str],
    editable: Sequence[str],
    protected: Sequence[str],
    proposals: int = 3,
    margin: int = 1,
    eval_timeout_seconds: float = 1800.0,
    limits: RunLimits = IMPROVER_LIMITS,
    max_digest_chars: int = 20_000,
) -> ImprovementDecision:
    """Run one round and return the decision; `decision.report` is for the reviewer.

    `protected` must cover every file the eval command reads — graders and
    scenarios for both splits. They are hidden from the improver and any change
    to them rejects the candidate.
    """
    if proposals <= 0:
        raise ValueError("proposals must be positive")
    if not protected:
        raise ValueError("protected must name the eval files; an editable grader can be gamed")
    root = Path(repo).resolve()
    if _git(root, "status", "--porcelain"):
        raise ValueError(f"{root} has uncommitted changes; commit or stash them first")
    base = _git(root, "rev-parse", "HEAD")

    with tempfile.TemporaryDirectory(prefix="emissary-improve-") as scratch:
        worktrees = _Worktrees(root, Path(scratch))
        try:
            return _round(
                root,
                base,
                worktrees,
                caller=caller,
                eval_command=eval_command,
                editable=editable,
                protected=protected,
                proposals=proposals,
                margin=margin,
                eval_timeout_seconds=eval_timeout_seconds,
                limits=limits,
                max_digest_chars=max_digest_chars,
            )
        finally:
            worktrees.remove_all()


def _round(
    root: Path,
    base: str,
    worktrees: "_Worktrees",
    *,
    caller: ModelCaller,
    eval_command: Sequence[str],
    editable: Sequence[str],
    protected: Sequence[str],
    proposals: int,
    margin: int,
    eval_timeout_seconds: float,
    limits: RunLimits,
    max_digest_chars: int,
) -> ImprovementDecision:
    def evaluate_at(commit: str, split: str) -> EvalResult:
        return _run_eval(worktrees.add(commit), eval_command, split, eval_timeout_seconds)

    try:
        baseline = evaluate_at(base, "search")
    except _EvalError as exc:
        raise EvalCommandFailed(f"on the unchanged code: {exc}") from None
    digest = failure_digest(baseline.failures, max_chars=max_digest_chars)

    attempts: list[Attempt] = []
    for number in range(1, proposals + 1):
        tree = worktrees.add(base)
        workspace = Workspace(tree, editable=editable, protected=protected)
        attempt, commit = _propose(number, tree, workspace, caller, limits, digest, attempts)
        if commit is not None:
            try:
                candidate = evaluate_at(commit, "search")
            except _EvalError as exc:
                attempt = _reject(attempt, f"eval command failed on the candidate: {exc}")
            else:
                comparison = compare(baseline.scores, candidate.scores, margin=margin)
                attempt = _with_search(attempt, candidate.scores, comparison.reasons)
        attempts.append(attempt)
        if attempt.rejected:
            logger.warning("improve: attempt %d rejected: %s", number, "; ".join(attempt.rejected))
            continue
        assert commit is not None  # an attempt without a commit is always rejected
        return _holdout(root, base, commit, attempts, baseline, evaluate_at, margin)

    logger.warning("improve: no candidate beat the baseline in %d attempts", proposals)
    return ImprovementDecision(
        False,
        base,
        None,
        baseline.scores,
        tuple(attempts),
        reasons=("no candidate beat the baseline on the search split",),
    )


def _holdout(
    root: Path,
    base: str,
    commit: str,
    attempts: list[Attempt],
    baseline: EvalResult,
    evaluate_at: Callable[[str, str], EvalResult],
    margin: int,
) -> ImprovementDecision:
    try:
        before = evaluate_at(base, "holdout")
    except _EvalError as exc:
        raise EvalCommandFailed(f"on the unchanged code: {exc}") from None
    try:
        after = evaluate_at(commit, "holdout").scores
        reasons = compare(before.scores, after, margin=margin).reasons
    except _EvalError as exc:
        after, reasons = (), (f"eval command failed on the candidate: {exc}",)

    branch = None
    if reasons:
        logger.warning("improve: candidate rejected on the holdout: %s", "; ".join(reasons))
    else:
        branch = f"emissary/improve/{base[:12]}-{len(attempts)}"
        _git(root, "branch", branch, commit)
    return ImprovementDecision(
        not reasons,
        base,
        branch,
        baseline.scores,
        tuple(attempts),
        before.scores,
        after,
        tuple(reasons),
    )


def _propose(
    number: int,
    tree: Path,
    workspace: Workspace,
    caller: ModelCaller,
    limits: RunLimits,
    digest: str,
    prior: Sequence[Attempt],
) -> tuple[Attempt, str | None]:
    submitted: dict[str, str] = {}

    def submit(summary: str, expected_effect: str) -> ToolResult:
        submitted.update(summary=summary, expected_effect=expected_effect)
        return ToolResult("success", "submitted; end your turn")

    submit_tool = Tool(
        "submit",
        "Finish: summarise the change and the effect you expect it to have.",
        {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "minLength": 1},
                "expected_effect": {"type": "string", "minLength": 1},
            },
            "required": ["summary", "expected_effect"],
            "additionalProperties": False,
        },
        submit,
        api_scope="none",
    )
    agent = Agent("improver", INSTRUCTIONS, (*workspace.tools(), submit_tool), limits)
    result = run(agent, _task(digest, prior), caller=caller)

    _git(tree, "add", "-A")
    changed = tuple(_git(tree, "diff", "--cached", "--name-only").splitlines())
    attempt = Attempt(
        number,
        submitted.get("summary", "(the improver did not submit a summary)"),
        submitted.get("expected_effect", "(not stated)"),
        changed,
        _git(tree, "diff", "--cached"),
    )
    if result.status is not RunStatus.COMPLETED:
        return _reject(attempt, f"improver stopped: {result.stop_reason.value}"), None
    if not changed:
        return _reject(attempt, "the improver changed nothing"), None
    # The tools already refuse these writes; checked again on the diff itself
    # because this is the property the whole loop rests on.
    out_of_scope = [path for path in changed if not workspace.is_editable(path)]
    if out_of_scope:
        return _reject(attempt, f"changed files outside the editable scope: {out_of_scope}"), None
    _git(
        tree,
        "-c",
        "user.name=emissary",
        "-c",
        "user.email=emissary@localhost",
        "commit",
        "--quiet",
        "--no-verify",
        "-m",
        f"emissary improve: {attempt.summary}",
    )
    return attempt, _git(tree, "rev-parse", "HEAD")


def _task(digest: str, prior: Sequence[Attempt]) -> str:
    sections = ["# Failing runs\n\n" + digest]
    if prior:
        history = "\n\n".join(
            f"## Attempt {a.number}: {a.summary}\n"
            f"changed: {', '.join(a.changed) or 'nothing'}\n"
            f"rejected: {'; '.join(a.rejected)}"
            for a in prior
        )
        sections.append(
            "# Earlier attempts this round (rejected; each started from the same code)\n\n"
            + history
        )
    return "\n\n".join(sections)


def _reject(attempt: Attempt, reason: str) -> Attempt:
    return _with_search(attempt, attempt.search, (*attempt.rejected, reason))


def _with_search(
    attempt: Attempt, search: Sequence[ScenarioScore], rejected: Sequence[str]
) -> Attempt:
    return Attempt(
        attempt.number,
        attempt.summary,
        attempt.expected_effect,
        attempt.changed,
        attempt.diff,
        tuple(search),
        tuple(rejected),
    )


class _EvalError(Exception):
    pass


def _run_eval(tree: Path, command: Sequence[str], split: str, timeout: float) -> EvalResult:
    output = tree.parent / f"{tree.name}.{split}.json"
    env = {**os.environ, SPLIT_ENV: split, OUTPUT_ENV: str(output)}
    try:
        completed = subprocess.run(
            list(command),
            cwd=tree,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise _EvalError(f"timed out after {timeout}s") from None
    if completed.returncode != 0:
        tail = completed.stderr.strip().splitlines()[-5:]
        raise _EvalError(f"exit {completed.returncode}: {' | '.join(tail)}")
    try:
        return read_eval_json(output.read_text())
    except (OSError, ValueError) as exc:
        raise _EvalError(str(exc)) from None


class _Worktrees:
    def __init__(self, root: Path, scratch: Path):
        self.root = root
        self.scratch = scratch
        self.paths: list[Path] = []

    def add(self, commit: str) -> Path:
        path = self.scratch / f"tree-{len(self.paths)}"
        _git(self.root, "worktree", "add", "--quiet", "--detach", str(path), commit)
        self.paths.append(path)
        return path

    def remove_all(self) -> None:
        for path in self.paths:
            shutil.rmtree(path, ignore_errors=True)
        _git(self.root, "worktree", "prune")


def _git(cwd: Path, *args: str) -> str:
    completed = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


def render_report(decision: ImprovementDecision) -> str:
    verdict = f"accepted as branch `{decision.branch}`" if decision.accepted else "rejected"
    lines = [
        "# emissary improve report",
        "",
        f"- base: `{decision.base}`",
        f"- verdict: {verdict}",
    ]
    lines += [f"- reason: {reason}" for reason in decision.reasons]
    before = {s.name: s for s in decision.baseline_search}
    for attempt in decision.attempts:
        lines += [
            "",
            f"## Attempt {attempt.number}: {attempt.summary}",
            "",
            f"**Expected effect (the improver's claim):** {attempt.expected_effect}",
        ]
        if attempt.rejected:
            lines += ["", *(f"- rejected: {reason}" for reason in attempt.rejected)]
        if attempt.search:
            lines += ["", "| search scenario | before | after |", "|---|---|---|"]
            lines += [
                f"| {s.name} | {before[s.name].successes}/{s.attempts} "
                f"| {s.successes}/{s.attempts} |"
                for s in attempt.search
            ]
        lines += ["", "```diff", attempt.diff or "(no changes)", "```"]
    if decision.candidate_holdout:
        held = {s.name: s for s in decision.baseline_holdout}
        lines += [
            "",
            "## Holdout (scored once)",
            "",
            "| scenario | before | after |",
            "|---|---|---|",
        ]
        lines += [
            f"| {s.name} | {held[s.name].successes}/{s.attempts} | {s.successes}/{s.attempts} |"
            for s in decision.candidate_holdout
        ]
    return "\n".join(lines) + "\n"


__all__ = [
    "IMPROVER_LIMITS",
    "Attempt",
    "EvalCommandFailed",
    "ImprovementDecision",
    "improve",
    "render_report",
]
