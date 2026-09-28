import json
from datetime import UTC, datetime

import pytest

from emissary.eval.evaluation import EvaluationReport, EvaluationScenario, ScenarioScore, evaluate
from emissary.harness.events import RunEvent
from emissary.harness.state import RunResult, RunStatus, StopReason
from emissary.improve.contract import read_eval_json, write_eval_json
from emissary.llm.decision import FinalOutput, Usage


def completed(answer: str) -> RunResult:
    events = (RunEvent("run", 1, "run_started", {"agent": "a"}, datetime(2026, 1, 1, tzinfo=UTC)),)
    return RunResult(
        f"run-{answer}",
        RunStatus.COMPLETED,
        StopReason.COMPLETED,
        FinalOutput(answer),
        Usage(1, 1),
        events,
    )


def test_a_cleanly_completed_wrong_answer_is_kept_as_evidence():
    # The commonest failure is a confident wrong answer; if only crashed runs
    # were kept, the improver would never see it.
    outcomes = [completed("42"), completed("41")]
    report = evaluate(
        EvaluationScenario("answer", lambda: outcomes.pop(0), lambda r: r.output.text == "42"),
        attempts=2,
    )

    result = read_eval_json(write_eval_json([report]))

    assert result.scores == (ScenarioScore("answer", 2, 1),)
    assert [run.output.text for run in result.failures["answer"]] == ["41"]


def test_a_report_without_per_run_verdicts_is_refused():
    # Guessing which runs failed would hand the improver passing runs as failures.
    report = EvaluationReport("answer", 1, 0, (completed("41"),))

    with pytest.raises(ValueError, match="per-run verdicts"):
        write_eval_json([report])


def test_a_malformed_result_fails_loudly():
    with pytest.raises(ValueError, match="invalid result"):
        read_eval_json(json.dumps({"schema": 1, "scenarios": [{"name": "a", "attempts": 1}]}))
    with pytest.raises(ValueError, match="invalid result"):
        read_eval_json("not json")
