import pytest

from emissary.eval.evaluation import (
    EvaluationScenario,
    EventGrader,
    ScenarioScore,
    compare,
    evaluate,
)
from emissary.harness.events import RunEvent
from emissary.harness.state import RunResult, RunStatus, StopReason
from emissary.llm.decision import FinalOutput, Usage


def result(*, kinds=("run_started", "run_completed"), completed=True):
    events = tuple(RunEvent("run", index, kind, {}, None) for index, kind in enumerate(kinds, 1))
    return RunResult(
        "run",
        RunStatus.COMPLETED if completed else RunStatus.FAILED,
        StopReason.COMPLETED if completed else StopReason.MODEL_ERROR,
        FinalOutput(text="done") if completed else None,
        Usage(4, 2),
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


def scores(**passed):
    return [ScenarioScore(name, 3, successes) for name, successes in passed.items()]


def test_a_gain_elsewhere_cannot_pay_for_breaking_a_reliable_scenario():
    # Net +1, but `reliable` went from always passing to mostly failing: an
    # improvement loop that accepted this would trade a working task for noise.
    comparison = compare(scores(reliable=3, weak=0), scores(reliable=1, weak=3), margin=1)

    assert not comparison.accepted
    assert comparison.gain == 1
    assert any("reliable" in reason for reason in comparison.reasons)


def test_a_single_flip_on_a_reliable_scenario_is_treated_as_noise():
    comparison = compare(scores(reliable=3, weak=0), scores(reliable=2, weak=2), margin=1)

    assert comparison.accepted
    assert comparison.gain == 1


def test_a_candidate_must_clear_the_margin_to_be_accepted():
    assert not compare(scores(a=1, b=1), scores(a=2, b=1), margin=2).accepted
    assert compare(scores(a=1, b=1), scores(a=2, b=2), margin=2).accepted


def test_suites_that_do_not_match_cannot_be_compared():
    # Dropping a scenario from the candidate would otherwise hide its regression.
    with pytest.raises(ValueError, match="suites differ"):
        compare(scores(a=1, b=1), scores(a=3))
    with pytest.raises(ValueError, match="attempts"):
        compare(scores(a=1), [ScenarioScore("a", 5, 5)])
