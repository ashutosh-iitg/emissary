"""Terminal run outcomes, stop reasons, and accumulated state."""

from dataclasses import dataclass
from enum import Enum

from ..llm.decision import FinalOutput, Usage
from ..llm.messages import Message
from .events import RunEvent
from .projection import derive_messages


class RunStatus(str, Enum):
    COMPLETED = "completed"
    REFUSED = "refused"
    STOPPED = "stopped"
    PAUSED = "paused"
    CANCELLED = "cancelled"
    FAILED = "failed"


class StopReason(str, Enum):
    COMPLETED = "completed"
    REFUSAL = "refusal"
    MAX_TURNS = "max_turns"
    MAX_TOOL_CALLS = "max_tool_calls"
    MAX_TOOL_ERRORS = "max_tool_errors"
    TOKEN_LIMIT = "token_limit"
    INVALID_TOOL = "invalid_tool"
    MODEL_ERROR = "model_error"
    APPROVAL_REQUIRED = "approval_required"
    APPROVAL_REJECTED = "approval_rejected"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class RunResult:
    run_id: str
    status: RunStatus
    stop_reason: StopReason
    output: FinalOutput | None
    usage: Usage
    events: tuple[RunEvent, ...]

    @property
    def messages(self) -> tuple[Message, ...]:
        """The final model-visible surface, re-derived from the log on access.

        Not stored, so it cannot disagree with ``events``. Under a trimming
        policy this is the surface the model last saw, not the whole history —
        that is ``derive_messages(events, apply_ops=False)``. Bind it to a local
        before iterating; every access re-folds the log.
        """
        return derive_messages(self.events)


__all__ = ["RunResult", "RunStatus", "StopReason"]
