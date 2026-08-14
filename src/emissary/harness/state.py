"""Terminal run outcomes, stop reasons, and accumulated state."""

from dataclasses import dataclass
from enum import Enum

from ..llm.decision import FinalOutput, Usage
from ..llm.messages import Message
from .events import RunEvent


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
    messages: tuple[Message, ...]
    events: tuple[RunEvent, ...]


__all__ = ["RunResult", "RunStatus", "StopReason"]
