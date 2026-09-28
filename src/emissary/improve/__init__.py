"""Evaluation-gated improvement of an agent project built on emissary (ADR-0029)."""

from .contract import OUTPUT_ENV, SPLIT_ENV, EvalResult, read_eval_json, write_eval_json
from .digest import failure_digest
from .loop import (
    IMPROVER_LIMITS,
    Attempt,
    EvalCommandFailed,
    ImprovementDecision,
    improve,
    render_report,
)
from .workspace import Workspace

__all__ = [
    "IMPROVER_LIMITS",
    "OUTPUT_ENV",
    "SPLIT_ENV",
    "Attempt",
    "EvalCommandFailed",
    "EvalResult",
    "ImprovementDecision",
    "Workspace",
    "failure_digest",
    "improve",
    "read_eval_json",
    "render_report",
    "write_eval_json",
]
