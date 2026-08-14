"""Deterministic evaluation over complete run results and trajectories."""

from collections.abc import Callable
from dataclasses import dataclass

from .state import RunResult


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
class EvaluationReport:
    scenario: str
    attempts: int
    successes: int
    runs: tuple[RunResult, ...]

    @property
    def pass_rate(self) -> float:
        return self.successes / self.attempts

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


__all__ = ["EvaluationReport", "EvaluationScenario", "EventGrader", "evaluate"]
