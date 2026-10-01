"""Re-run a recorded trajectory through the state machine (ADR-0015).

Scripted tests assert the branches an author thought of. A recorded log is a
whole real run, so replaying one catches changes to control flow or projection
that every scripted test would still pass. Both substitutes are deterministic
and network-free.
"""

from ..harness.conversation.projection import model_result_from_data, tool_result_from_data
from ..harness.policy import AuthorizationDecision, InvocationRequest
from ..harness.state import RunResult
from ..harness.tooling.tools import LocalToolExecutor, Tool, ToolContext, ToolResult
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

    `serves_every_tool` tells a routing executor to send source-bound tools here
    too, so a recorded run that used tool sources replays with no live server.
    """

    serves_every_tool = True

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


class RecordedAuthorizer:
    """Answer each authorization question the way the recorded run was answered.

    Keyed by call and attempt and consumed in order, because one attempt is
    asked twice (batch, then just before execution). No live policy, credential
    or server is needed, and a divergence in what is asked fails loudly.
    """

    def __init__(self, recorded: RunResult):
        self._answers: dict[tuple[str, int], list[dict]] = {}
        for event in recorded.events:
            if event.kind == "authorization_resolved":
                key = (event.data["call_id"], event.data["attempt"])
                self._answers.setdefault(key, []).append(event.data)

    def __call__(self, request: InvocationRequest) -> AuthorizationDecision:
        queue = self._answers.get((request.call_id, request.attempt))
        if not queue:
            raise ReplayExhausted(
                f"no recorded authorization for call {request.call_id!r} attempt {request.attempt}"
            )
        recorded = queue.pop(0)
        if recorded["allowed"]:
            return request.allow(recorded["reason"], policy_version=recorded["policy_version"])
        return request.deny(recorded["reason"], policy_version=recorded["policy_version"])


def trajectory(result: RunResult) -> list[tuple[int, str]]:
    """The comparable shape of a run: ordered kinds, without the run id or clock."""
    return [(event.sequence, event.kind) for event in result.events]


__all__ = [
    "RecordedAuthorizer",
    "ReplayExhausted",
    "ReplayModelCaller",
    "ReplayToolExecutor",
    "trajectory",
]
