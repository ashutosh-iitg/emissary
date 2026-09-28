"""Run one `improve` round on the toy converter. Live and opt-in: it calls a model.

    IMPROVE_TOY_SPEC=anthropic:<model-id> uv run python examples/improve_toy/run_improve.py

The toy is copied into a fresh git repository first, as a consumer project
would be, so the branch `improve` creates lands there rather than in emissary.
Writes `improve-report.md` to the current directory.
"""

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import emissary
from emissary.improve import improve

HERE = Path(__file__).parent


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    caller = emissary.SpecModelCaller(emissary.parse_spec(os.environ["IMPROVE_TOY_SPEC"]))
    project = Path(tempfile.mkdtemp(prefix="improve-toy-"))
    for part in ("agent", "evals"):
        shutil.copytree(HERE / part, project / part)
    for args in (
        ["init", "--quiet", "--initial-branch=main"],
        ["add", "-A"],
        ["-c", "user.name=toy", "-c", "user.email=toy@localhost", "commit", "--quiet", "-m", "toy"],
    ):
        subprocess.run(["git", *args], cwd=project, check=True)

    decision = improve(
        project,
        caller=caller,
        eval_command=[sys.executable, "-m", "evals.run_evals"],
        editable=["agent/**"],
        protected=["evals/**"],
    )
    Path("improve-report.md").write_text(decision.report)
    verdict = (
        f"accepted: branch {decision.branch} in {project}" if decision.accepted else "rejected"
    )
    print(f"{verdict}; report in improve-report.md")


if __name__ == "__main__":
    main()
