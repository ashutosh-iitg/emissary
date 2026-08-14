from emissary.decision import FinalOutput, Usage
from emissary.evaluation import EvaluationScenario, EventGrader, evaluate
from emissary.events import RunEvent
from emissary.state import RunResult, RunStatus, StopReason


def result(*, kinds=("run_started", "run_completed"), completed=True):
    events = tuple(RunEvent("run", index, kind, {}, None) for index, kind in enumerate(kinds, 1))
    return RunResult(
        "run",
        RunStatus.COMPLETED if completed else RunStatus.FAILED,
        StopReason.COMPLETED if completed else StopReason.MODEL_ERROR,
        FinalOutput(text="done") if completed else None,
        Usage(4, 2),
        (),
        events,
    )


def test_event_grader_checks_required_and_forbidden_trajectory_events():
    grader = EventGrader(required=("run_completed",), forbidden=("tool_call_rejected",))

    assert grader(result())
    assert not grader(result(kinds=("run_started", "tool_call_rejected")))


def test_evaluation_reports_success_and_efficiency_over_repeated_attempts():
    outcomes = [result(), result(completed=False), result()]
    scenario = EvaluationScenario(
        "bounded", lambda: outcomes.pop(0), lambda run: run.status is RunStatus.COMPLETED
    )

    report = evaluate(scenario, attempts=3)

    assert report.successes == 2
    assert report.pass_rate == 2 / 3
    assert report.average_input_tokens == 4
    assert len(report.runs) == 3
