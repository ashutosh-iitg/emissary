"""Bounded synchronous plan/act/observe state machine."""

import json
import uuid
from dataclasses import asdict

from ..llm.decision import FinalOutput, Refusal, ToolCalls, Usage
from ..llm.errors import ProviderError
from ..llm.messages import AssistantMessage, Message, TextBlock, ToolMessage, UserMessage
from ..llm.model import ModelCaller
from .agent import Agent
from .context import CompleteHistory, ContextPolicy
from .events import EventSink, InMemoryEventSink, RunEvent, new_event
from .policy import ApprovalDecision, Approver, approval_for
from .state import RunResult, RunStatus, StopReason
from .tools import LocalToolExecutor, ToolExecutor, ToolRegistry


def run(
    agent: Agent,
    task: str,
    *,
    caller: ModelCaller,
    executor: ToolExecutor | None = None,
    event_sink: EventSink | None = None,
    context_policy: ContextPolicy | None = None,
    approver: Approver | None = None,
) -> RunResult:
    """Run one agent until a typed terminal outcome is reached."""
    if not task:
        raise ValueError("task must not be empty")

    run_id = uuid.uuid4().hex
    active_executor = executor or LocalToolExecutor()
    sink = event_sink or InMemoryEventSink()
    context = context_policy or CompleteHistory()
    registry = ToolRegistry(agent.tools)
    messages: tuple[Message, ...] = (UserMessage((TextBlock(task),)),)
    events: list[RunEvent] = []
    usage = Usage(0, 0)
    tool_count = 0
    consecutive_errors = 0

    def emit(kind: str, **data) -> None:
        event = new_event(run_id, len(events) + 1, kind, **data)
        events.append(event)
        sink.emit(event)

    def finish(
        status: RunStatus, reason: StopReason, output: FinalOutput | None = None
    ) -> RunResult:
        kind = "run_completed" if status is RunStatus.COMPLETED else "run_stopped"
        emit(kind, status=status.value, reason=reason.value)
        return RunResult(run_id, status, reason, output, usage, messages, tuple(events))

    emit("run_started", agent=agent.name)
    for turn in range(agent.limits.max_turns):
        emit("model_call_started", turn=turn + 1)
        try:
            model_result = caller(
                system=agent.instructions,
                messages=context.select(messages),
                tools=registry.definitions,
                settings=agent.model_settings,
            )
        except ProviderError as exc:
            emit("model_call_failed", retryable=exc.retryable)
            return finish(RunStatus.FAILED, StopReason.MODEL_ERROR)

        usage = Usage(
            usage.input_tokens + model_result.usage.input_tokens,
            usage.output_tokens + model_result.usage.output_tokens,
            usage.cached_input_tokens + model_result.usage.cached_input_tokens,
        )
        emit("model_call_completed", decision=type(model_result.decision).__name__)
        if (
            agent.limits.max_input_tokens is not None
            and usage.input_tokens > agent.limits.max_input_tokens
        ) or (
            agent.limits.max_output_tokens is not None
            and usage.output_tokens > agent.limits.max_output_tokens
        ):
            return finish(RunStatus.STOPPED, StopReason.TOKEN_LIMIT)

        decision = model_result.decision
        if isinstance(decision, FinalOutput):
            return finish(RunStatus.COMPLETED, StopReason.COMPLETED, decision)
        if isinstance(decision, Refusal):
            return finish(RunStatus.REFUSED, StopReason.REFUSAL)
        if not isinstance(decision, ToolCalls):
            return finish(RunStatus.FAILED, StopReason.MODEL_ERROR)

        if tool_count + len(decision.calls) > agent.limits.max_tool_calls:
            return finish(RunStatus.STOPPED, StopReason.MAX_TOOL_CALLS)

        resolved = []
        for call in decision.calls:
            try:
                tool = registry.resolve(call.name)
            except KeyError:
                emit("tool_call_rejected", call_id=call.id, reason="unknown_tool")
                return finish(RunStatus.FAILED, StopReason.INVALID_TOOL)
            invalid = active_executor.validate(call, tool)
            if invalid is not None:
                emit("tool_call_rejected", call_id=call.id, reason=invalid.summary)
                return finish(RunStatus.FAILED, StopReason.INVALID_TOOL)
            resolved.append((call, tool))

        messages += (AssistantMessage(tool_calls=decision.calls),)
        for call, tool in resolved:
            approval = approval_for(call, tool, approver)
            emit("approval_resolved", call_id=call.id, decision=approval.value)
            if approval is ApprovalDecision.PAUSE:
                return finish(RunStatus.PAUSED, StopReason.APPROVAL_REQUIRED)
            if approval is ApprovalDecision.REJECT:
                return finish(RunStatus.STOPPED, StopReason.APPROVAL_REJECTED)
            emit("tool_call_started", call_id=call.id, tool=call.name)
            outcome = active_executor.execute(call, tool)
            tool_count += 1
            consecutive_errors = consecutive_errors + 1 if outcome.status == "error" else 0
            emit("tool_call_completed", call_id=call.id, status=outcome.status)
            messages += (
                ToolMessage(
                    call_id=call.id,
                    tool_name=call.name,
                    content=json.dumps(asdict(outcome), sort_keys=True),
                ),
            )
            if consecutive_errors >= agent.limits.max_consecutive_tool_errors:
                return finish(RunStatus.STOPPED, StopReason.MAX_TOOL_ERRORS)

    return finish(RunStatus.STOPPED, StopReason.MAX_TURNS)


__all__ = ["run"]
