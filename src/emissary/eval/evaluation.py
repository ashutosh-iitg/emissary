"""Deterministic evaluation over complete run results and trajectories."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ..harness.state import RunResult


@dataclass(frozen=True)
class EvaluationScenario:
    name: str
    execute: Callable[[], RunResult]
    grade: Callable[[RunResult], bool]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("scenario name must not be empty")


@dataclass(frozen=True)
class EventGrader:
    required: tuple[str, ...] = ()
    forbidden: tuple[str, ...] = ()

    def __call__(self, result: RunResult) -> bool:
        kinds = {event.kind for event in result.events}
        return all(kind in kinds for kind in self.required) and not any(
            kind in kinds for kind in self.forbidden
        )


@dataclass(frozen=True)
class ScenarioScore:
    """How often one scenario passed: all `compare` needs, and all a subprocess can report."""

    name: str
    attempts: int
    successes: int

    def __post_init__(self) -> None:
        if self.attempts <= 0:
            raise ValueError("attempts must be positive")
        if not 0 <= self.successes <= self.attempts:
            raise ValueError("successes must be between 0 and attempts")


@dataclass(frozen=True)
class EvaluationReport:
    scenario: str
    attempts: int
    successes: int
    runs: tuple[RunResult, ...]

    @property
    def pass_rate(self) -> float:
        return self.successes / self.attempts

    @property
    def score(self) -> ScenarioScore:
        return ScenarioScore(self.scenario, self.attempts, self.successes)

    @property
    def average_input_tokens(self) -> float:
        return sum(run.usage.input_tokens for run in self.runs) / self.attempts

    @property
    def average_output_tokens(self) -> float:
        return sum(run.usage.output_tokens for run in self.runs) / self.attempts


def evaluate(scenario: EvaluationScenario, *, attempts: int = 1) -> EvaluationReport:
    if attempts <= 0:
        raise ValueError("attempts must be positive")
    runs = tuple(scenario.execute() for _ in range(attempts))
    successes = sum(scenario.grade(result) for result in runs)
    return EvaluationReport(scenario.name, attempts, successes, runs)


@dataclass(frozen=True)
class Comparison:
    accepted: bool
    gain: int
    reasons: tuple[str, ...]


def compare(
    baseline: Sequence[ScenarioScore], candidate: Sequence[ScenarioScore], *, margin: int = 1
) -> Comparison:
    """Accept a candidate only if it passes `margin` more attempts in total and
    breaks nothing the baseline reliably did.

    Counts, not rates or a significance test: with a handful of scenarios and
    attempts, a test has no power and a rate hides how few trials there were.
    The guard applies only to scenarios the baseline passed on every attempt,
    because a single flip elsewhere is indistinguishable from noise — while a
    reliable scenario falling to mostly failing is a real regression that a
    gain on another scenario must not be allowed to pay for.
    """
    if margin < 0:
        raise ValueError("margin must be non-negative")
    before = {score.name: score for score in baseline}
    after = {score.name: score for score in candidate}
    if len(before) != len(baseline) or len(after) != len(candidate):
        raise ValueError("scenario names must be unique")
    if before.keys() != after.keys():
        raise ValueError(
            f"suites differ: baseline has {sorted(before)}, candidate has {sorted(after)}"
        )
    for name, score in before.items():
        if score.attempts != after[name].attempts:
            raise ValueError(f"scenario {name!r} ran a different number of attempts")

    gain = sum(score.successes for score in candidate) - sum(score.successes for score in baseline)
    reasons = [
        f"{name}: passed {score.attempts}/{score.attempts} before, "
        f"{after[name].successes}/{score.attempts} after"
        for name, score in before.items()
        if score.successes == score.attempts and 2 * after[name].successes < score.attempts
    ]
    if gain < margin:
        reasons.append(f"gain {gain} is below the margin {margin}")
    return Comparison(not reasons, gain, tuple(reasons))


__all__ = [
    "Comparison",
    "EvaluationReport",
    "EvaluationScenario",
    "EventGrader",
    "ScenarioScore",
    "compare",
    "evaluate",
]
