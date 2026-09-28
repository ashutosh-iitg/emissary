"""What a consumer's eval command hands back to `improve`.

The command runs in its own process because the code it evaluates is the code
being changed: a candidate's modules cannot be reloaded cleanly into this one.
It learns which split to run from `SPLIT_ENV` and writes its result to the file
named by `OUTPUT_ENV` — a file rather than stdout, so that anything the agent
under evaluation prints cannot corrupt the result.
"""

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from jsonschema import Draft202012Validator, ValidationError

from ..eval.evaluation import EvaluationReport, ScenarioScore
from ..harness.state import RunResult
from ..storage.persistence import deserialize_run, serialize_run

SPLIT_ENV = "EMISSARY_EVAL_SPLIT"
OUTPUT_ENV = "EMISSARY_EVAL_OUTPUT"
SCHEMA_VERSION = 1

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "schema": {"const": SCHEMA_VERSION},
        "scenarios": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "minLength": 1},
                    "attempts": {"type": "integer", "minimum": 1},
                    "successes": {"type": "integer", "minimum": 0},
                    "failures": {"type": "array", "items": {"type": "object"}},
                },
                "required": ["name", "attempts", "successes", "failures"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["schema", "scenarios"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class EvalResult:
    scores: tuple[ScenarioScore, ...]
    failures: Mapping[str, tuple[RunResult, ...]] = field(default_factory=dict)


def write_eval_json(reports: Iterable[EvaluationReport]) -> str:
    """Render `evaluate()` reports in the form `improve` reads.

    Every run the grader failed is kept as evidence — including runs that
    completed, since a wrong answer delivered cleanly is the commonest failure.
    """
    return json.dumps(
        {
            "schema": SCHEMA_VERSION,
            "scenarios": [
                {
                    "name": report.scenario,
                    "attempts": report.attempts,
                    "successes": report.successes,
                    "failures": [
                        json.loads(serialize_run(run))
                        for run, passed in zip(report.runs, _verdicts(report), strict=True)
                        if not passed
                    ],
                }
                for report in reports
            ],
        }
    )


def _verdicts(report: EvaluationReport) -> tuple[bool, ...]:
    if len(report.passed) != len(report.runs):
        raise ValueError(
            f"report {report.scenario!r} has no per-run verdicts; build it with evaluate()"
        )
    return report.passed


def read_eval_json(text: str) -> EvalResult:
    try:
        payload = json.loads(text)
        Draft202012Validator(_SCHEMA).validate(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"eval command wrote an invalid result: {exc}") from exc
    scenarios = payload["scenarios"]
    return EvalResult(
        scores=tuple(
            ScenarioScore(item["name"], item["attempts"], item["successes"]) for item in scenarios
        ),
        failures={
            item["name"]: tuple(deserialize_run(json.dumps(run)) for run in item["failures"])
            for item in scenarios
            if item["failures"]
        },
    )


__all__ = [
    "OUTPUT_ENV",
    "SCHEMA_VERSION",
    "SPLIT_ENV",
    "EvalResult",
    "read_eval_json",
    "write_eval_json",
]
