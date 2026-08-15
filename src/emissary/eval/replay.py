"""Re-run a recorded trajectory through the state machine (ADR-0015).

Scripted tests assert the branches an author thought of. A recorded log is a
whole real run, so replaying one catches changes to control flow or projection
that every scripted test would still pass. Both substitutes are deterministic
and network-free.
"""

from ..harness.projection import model_result_from_data, tool_result_from_data
from ..harness.state import RunResult
from ..harness.tools import LocalToolExecutor, Tool, ToolContext, ToolResult
from ..llm.decision import ModelResult, ModelSettings, ToolCall, ToolDefinition
from ..llm.messages import Message


class ReplayExhausted(AssertionError):
    """The replayed run asked for something the recording does not contain.

    An assertion rather than a domain error: it means the code under test
    diverged from the recorded trajectory, which is the failure replay exists
    to detect.
    """


class ReplayModelCaller:
    """Hand back the model turns the recorded run actually received, in order."""

    def __init__(self, recorded: RunResult):
        self._turns = [
            model_result_from_data(event.data)
            for event in recorded.events
            if event.kind == "model_call_completed"
        ]
        self._served = 0

    def __call__(
        self,
        *,
        system: str,
        messages: tuple[Message, ...],
        tools: tuple[ToolDefinition, ...] = (),
        settings: ModelSettings | None = None,
    ) -> ModelResult:
        if self._served >= len(self._turns):
            raise ReplayExhausted(
                f"the run asked for turn {self._served + 1}; "
                f"the recording holds {len(self._turns)}"
            )
        turn = self._turns[self._served]
        self._served += 1
        return turn


class ReplayToolExecutor:
    """Return the outcome each recorded call produced, keyed by call id.

    Validation is delegated to the real local executor so a recorded run that
    ended in `INVALID_TOOL` replays down the same path.
    """

    def __init__(self, recorded: RunResult):
        self._outcomes = {
            event.data["call_id"]: tool_result_from_data(event.data["result"])
            for event in recorded.events
            if event.kind == "tool_call_completed"
        }
        self._validator = LocalToolExecutor()

    def validate(self, call: ToolCall, tool: Tool) -> ToolResult | None:
        return self._validator.validate(call, tool)

    def execute(self, call: ToolCall, tool: Tool, context: ToolContext) -> ToolResult:
        try:
            return self._outcomes[call.id]
        except KeyError:
            raise ReplayExhausted(f"no recorded outcome for call {call.id!r}") from None


def trajectory(result: RunResult) -> list[tuple[int, str]]:
    """The comparable shape of a run: ordered kinds, without the run id or clock."""
    return [(event.sequence, event.kind) for event in result.events]


__all__ = [
    "ReplayExhausted",
    "ReplayModelCaller",
    "ReplayToolExecutor",
    "trajectory",
]
