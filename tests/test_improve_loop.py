import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pytest

from emissary.harness.events import RunEvent
from emissary.harness.state import RunResult, RunStatus, StopReason
from emissary.improve.digest import failure_digest
from emissary.improve.loop import improve
from emissary.llm.decision import FinalOutput, ModelResult, ToolCall, ToolCalls, Usage

# Scores each scenario by whether the prompt holds its keyword. The holdout
# failure carries a marker the improver must never see.
GRADER = """\
import json, os
from pathlib import Path

split = os.environ["EMISSARY_EVAL_SPLIT"]
prompt = Path("agent/prompt.txt").read_text()
with open(os.environ["IMPROVE_TEST_LOG"], "a") as log:
    log.write(split + "\\n")
scenarios = {"search": {"alpha": "fixed", "beta": "fixed"}, "holdout": {"holdout-only": "general"}}

def failed_run(name):
    text = "HOLDOUT-SECRET task" if split == "holdout" else f"task for {name}"
    events = [
        {"run_id": "r", "sequence": 1, "kind": "user_message",
         "data": {"content": [{"text": text, "cache": False}]},
         "occurred_at": "2026-01-01T00:00:00+00:00"},
    ]
    return {"schema_version": 2, "run_id": "r", "status": "completed", "stop_reason": "completed",
            "output": {"text": "wrong", "value": None},
            "usage": {"input_tokens": 1, "output_tokens": 1, "cached_input_tokens": 0},
            "events": events}

result = {"schema": 1, "scenarios": []}
for name, keyword in scenarios[split].items():
    passed = keyword in prompt
    result["scenarios"].append({"name": name, "attempts": 2, "successes": 2 if passed else 0,
                                "failures": [] if passed else [failed_run(name)] * 2})
Path(os.environ["EMISSARY_EVAL_OUTPUT"]).write_text(json.dumps(result))
"""


@dataclass
class ScriptedImprover:
    """Plays one improver run per script; records everything it was shown."""

    scripts: list
    seen: list = field(default_factory=list)
    _current: list = field(default_factory=list)

    def __call__(self, *, system, messages, tools=(), settings=None):
        self.seen.append(system + "\n" + repr(messages))
        if not self._current:
            self._current = list(self.scripts.pop(0))
        return ModelResult(self._current.pop(0), "fake", "scripted", Usage(1, 1))


def writes(path: str, content: str) -> list:
    return [
        ToolCalls((ToolCall("w", "write_file", {"path": path, "content": content}),)),
        ToolCalls(
            (
                ToolCall(
                    "s", "submit", {"summary": f"edit {path}", "expected_effect": "alpha passes"}
                ),
            )
        ),
        FinalOutput(text="done"),
    ]


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "project"
    (root / "agent").mkdir(parents=True)
    (root / "evals").mkdir()
    (root / "agent" / "prompt.txt").write_text("be helpful\n")
    (root / "evals" / "grader.py").write_text(GRADER)
    git(root, "init", "--quiet", "--initial-branch=main")
    git(root, "add", "-A")
    git(root, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "--quiet", "-m", "init")
    monkeypatch.setenv("IMPROVE_TEST_LOG", str(tmp_path / "evals.log"))
    return root


def run_improve(repo: Path, improver: ScriptedImprover, **overrides):
    options = {
        "caller": improver,
        "eval_command": [sys.executable, "evals/grader.py"],
        "editable": ["agent/**"],
        "protected": ["evals/**"],
        "proposals": 1,
    }
    return improve(repo, **(options | overrides))


def holdout_runs(repo: Path) -> int:
    return (repo.parent / "evals.log").read_text().splitlines().count("holdout")


def test_an_accepted_change_becomes_a_branch_and_never_touches_main(repo):
    main_before = git(repo, "rev-parse", "main")

    decision = run_improve(repo, ScriptedImprover([writes("agent/prompt.txt", "fixed general\n")]))

    assert decision.accepted
    assert git(repo, "show", f"{decision.branch}:agent/prompt.txt") == "fixed general"
    assert git(repo, "rev-parse", "main") == main_before
    assert (repo / "agent" / "prompt.txt").read_text() == "be helpful\n"
    # Worktrees are scratch; only the repository's own checkout remains.
    assert len(git(repo, "worktree", "list").splitlines()) == 1


def test_the_report_puts_the_claimed_effect_beside_the_measured_one(repo):
    decision = run_improve(repo, ScriptedImprover([writes("agent/prompt.txt", "fixed general\n")]))

    report = decision.report
    assert decision.base in report
    assert "alpha passes" in report
    assert "| alpha | 0/2 | 2/2 |" in report
    assert "| holdout-only | 0/2 | 2/2 |" in report


def test_an_improver_cannot_pass_by_rewriting_the_grader(repo):
    # The DGM / Prime §3.5 failure: the cheapest way to raise a score is to
    # change what measures it. The write is refused, so nothing changes.
    improver = ScriptedImprover([writes("evals/grader.py", "print('all pass')\n")])

    decision = run_improve(repo, improver)

    assert not decision.accepted
    assert decision.branch is None
    assert "changed nothing" in decision.attempts[0].rejected[0]
    assert git(repo, "branch", "--list", "emissary/*") == ""


def test_the_holdout_never_reaches_the_improver(repo):
    improver = ScriptedImprover([writes("agent/prompt.txt", "fixed general\n")])

    run_improve(repo, improver)

    shown = "\n".join(improver.seen)
    assert "task for alpha" in shown
    assert "HOLDOUT-SECRET" not in shown
    assert "holdout-only" not in shown


def test_the_holdout_is_scored_once_even_when_it_rejects(repo):
    # "fixed" wins the search split but not the holdout. A second proposal
    # scored on the holdout would be shopping for one that happens to pass.
    improver = ScriptedImprover(
        [writes("agent/prompt.txt", "fixed\n"), writes("agent/prompt.txt", "fixed general\n")]
    )

    decision = run_improve(repo, improver, proposals=2)

    assert not decision.accepted
    assert len(decision.attempts) == 1
    assert holdout_runs(repo) == 2  # the baseline and the one candidate
    assert git(repo, "branch", "--list", "emissary/*") == ""


def test_a_rejected_attempt_is_shown_to_the_next_one(repo):
    improver = ScriptedImprover(
        [writes("agent/prompt.txt", "still wrong\n"), writes("agent/prompt.txt", "fixed general\n")]
    )

    decision = run_improve(repo, improver, proposals=2)

    assert decision.accepted
    assert "Attempt 1" in improver.seen[-1]
    assert "gain" in decision.attempts[0].rejected[0]


def test_uncommitted_changes_are_refused(repo):
    (repo / "agent" / "prompt.txt").write_text("edited\n")

    with pytest.raises(ValueError, match="uncommitted"):
        run_improve(repo, ScriptedImprover([]))


def test_the_digest_carries_what_a_transcript_drops():
    at = datetime(2026, 1, 1, tzinfo=UTC)
    events = (
        RunEvent(
            "r", 1, "user_message", {"content": [{"text": "convert 3 km", "cache": False}]}, at
        ),
        RunEvent(
            "r",
            2,
            "tool_call_completed",
            {
                "call_id": "c",
                "tool": "convert",
                "status": "error",
                "result": {"summary": "bad unit"},
            },
            at,
        ),
    )
    run = RunResult("r", RunStatus.STOPPED, StopReason.MAX_TOOL_ERRORS, None, Usage(1, 1), events)

    digest = failure_digest({"km": (run,)}, max_chars=10_000)

    assert "convert 3 km" in digest
    assert "tool convert error: bad unit" in digest
    assert "max_tool_errors" in digest
    assert len(failure_digest({"km": (run,)}, max_chars=60)) <= 60
