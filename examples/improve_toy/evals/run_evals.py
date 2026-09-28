"""The toy project's eval command: the fixed `f` that `improve` scores against.

Protected in `run_improve.py`, so the improver can neither read nor change it.
Run from the project root as `python -m evals.run_evals`.
"""

import os
import re
from pathlib import Path

from agent.converter import AGENT

import emissary
from emissary.improve import OUTPUT_ENV, SPLIT_ENV, write_eval_json

SPLITS = {
    "search": {
        "miles-to-km": ("How many kilometres is 5 miles?", 8.04672),
        "feet-to-metres": ("Convert 100 feet to metres.", 30.48),
        "km-to-miles": ("What is 3 km in miles?", 1.864114),
    },
    "holdout": {
        "marathon": ("A marathon is 26.2 miles. How many kilometres is that?", 42.164813),
        "miles-to-feet": ("How many feet are in 2 miles?", 10560.0),
    },
}


def within(expected: float):
    def grade(result: emissary.RunResult) -> bool:
        text = result.output.text if result.output and result.output.text else ""
        numbers = [float(n.replace(",", "")) for n in re.findall(r"-?[\d,]*\.?\d+", text)]
        return any(abs(n - expected) <= abs(expected) * 1e-3 for n in numbers)

    return grade


def main() -> None:
    caller = emissary.SpecModelCaller(emissary.parse_spec(os.environ["IMPROVE_TOY_SPEC"]))
    reports = [
        emissary.evaluate(
            emissary.EvaluationScenario(
                name, lambda task=task: emissary.run(AGENT, task, caller=caller), within(expected)
            ),
            attempts=3,
        )
        for name, (task, expected) in SPLITS[os.environ[SPLIT_ENV]].items()
    ]
    Path(os.environ[OUTPUT_ENV]).write_text(write_eval_json(reports))


if __name__ == "__main__":
    main()
