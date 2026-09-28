"""The bounded plan/act/observe loop, as a generator of effects (ADR-0024).

Every policy the harness has lives here: turn and token limits,
batch-validate-before-execute ordering, per-tool circuit breaking, idempotent
retries, approval resolution, and the six terminal conditions. None of it is
I/O. The three things that are — the model call, admission checking, and tool
execution — are `yield`ed as effects for a driver to perform.

That is the entire reason this module exists. Written as a plain loop it would
have to be copied for `arun`, and the copy would drift on the next policy
change. Here `run` and `arun` are drivers, and a new limit is written once.

The body is a line-for-line translation of the synchronous loop it replaces:
each former call site is now a `yield`, and nothing else moved.
"""

import json
import logging
from collections.abc import Generator
from dataclasses import asdict
from typing import Any

from ..llm.decision import FinalOutput, Refusal, ToolCalls, Usage
from ..llm.errors import ProviderError
from ..llm.messages import Message, TextBlock, UserMessage
from .agent import Agent, ModelAttemptLimitExceeded, RunCancelled, RunTimedOut
from .context import CompleteHistory, ContextPolicy
from .effects import CallModel, Effect, ExecuteTool, ValidateTool, WaitRetry
from .events import EventSink, NullEventSink, RunEvent, new_event
from .policy import ApprovalDecision, Approver, approval_for
from .projection import (
    context_op_data,
    derive_messages,
    message_to_data,
    model_result_data,
    tool_result_data,
    user_message_data,
    validate_history,
)
from .state import RunResult, RunStatus, StopReason
from .tools import ToolContext, ToolRegistry, ToolResult

logger = logging.getLogger(__name__)


def agent_machine(
    agent: Agent,
    task: str,
    *,
    run_id: str,
    event_sink: EventSink | None = None,
    context_policy: ContextPolicy | None = None,
    approver: Approver | None = None,
    history: tuple[Message, ...] = (),
) -> Generator[Effect, Any, RunResult]:
    """Drive one agent to a typed terminal outcome, yielding work to be done.

    `run_id` is a parameter rather than generated here: a pure core has no
    business inventing identity, and a fixed id makes a driven run's log stable
    enough to compare.
    """
    if not task:
        raise ValueError("task must not be empty")
    if history:
        validate_history(history)

    sink = event_sink if event_sink is not None else NullEventSink()
    context = context_policy or CompleteHistory()
    registry = ToolRegistry(agent.tools)
    events: list[RunEvent] = []
    usage = Usage(0, 0)
    tool_count = 0
    tool_attempts = 0
    internal_api_attempts = 0
    external_api_attempts = 0
    consecutive_errors = 0
    tool_failures: dict[str, int] = {}
    open_circuits: set[str] = set()

    def emit(kind: str, **data) -> None:
        event = new_event(run_id, len(events) + 1, kind, **data)
        events.append(event)
        try:
            sink.emit(event)
        except Exception:
            logger.exception("agent event sink failed for %s", kind)

    def finish(
        status: RunStatus, reason: StopReason, output: FinalOutput | None = None
    ) -> RunResult:
        kind = "run_completed" if status is RunStatus.COMPLETED else "run_stopped"
        emit(kind, status=status.value, reason=reason.value)
        return RunResult(run_id, status, reason, output, usage, tuple(events))

    def interrupted(exc: RunCancelled) -> RunResult:
        if isinstance(exc, RunTimedOut):
            return finish(RunStatus.STOPPED, StopReason.MAX_DURATION)
        return finish(RunStatus.CANCELLED, StopReason.CANCELLED)

    emit("run_started", agent=agent.name)
    if history:
        emit("history_loaded", messages=[message_to_data(message) for message in history])
    emit("user_message", **user_message_data(UserMessage((TextBlock(task),))))

    for turn in range(agent.limits.max_turns):
        surface = derive_messages(events)
        ops = context.plan(surface)
        if ops:
            for op in ops:
                emit("context_compacted", **context_op_data(op))
            # Re-fold rather than apply locally: the surface the model sees comes
            # from exactly one code path, so machine and projection cannot drift.
            surface = derive_messages(events)

        request_bytes = len(
            json.dumps(
                {
                    "system": agent.instructions,
                    "messages": [message_to_data(message) for message in surface],
                    "tools": [asdict(definition) for definition in registry.definitions],
                },
                ensure_ascii=False,
            ).encode("utf-8")
        )
        if request_bytes > agent.limits.max_model_input_bytes:
            return finish(RunStatus.STOPPED, StopReason.MODEL_INPUT_LIMIT)

        emit("model_call_started", turn=turn + 1)
        try:
            # A driver reports provider failure by throwing in, so this reads
            # exactly as the direct call it replaces.
            model_result = yield CallModel(
                system=agent.instructions,
                messages=surface,
                tools=registry.definitions,
                settings=agent.model_settings,
            )
        except ProviderError as exc:
            emit("model_call_failed", retryable=exc.retryable)
            return finish(RunStatus.FAILED, StopReason.MODEL_ERROR)
        except ModelAttemptLimitExceeded:
            return finish(RunStatus.STOPPED, StopReason.MAX_MODEL_ATTEMPTS)
        except RunCancelled as exc:
            return interrupted(exc)

        usage = Usage(
            usage.input_tokens + model_result.usage.input_tokens,
            usage.output_tokens + model_result.usage.output_tokens,
            usage.cached_input_tokens + model_result.usage.cached_input_tokens,
        )
        emit("model_call_completed", **model_result_data(model_result))
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
            try:
                invalid = yield ValidateTool(call, tool)
            except RunCancelled as exc:
                return interrupted(exc)
            if invalid is not None:
                emit("tool_call_rejected", call_id=call.id, reason=invalid.summary)
                return finish(RunStatus.FAILED, StopReason.INVALID_TOOL)
            resolved.append((call, tool))

        for call, tool in resolved:
            approval = approval_for(call, tool, approver)
            emit("approval_resolved", call_id=call.id, decision=approval.value)
            if approval is ApprovalDecision.PAUSE:
                return finish(RunStatus.PAUSED, StopReason.APPROVAL_REQUIRED)
            if approval is ApprovalDecision.REJECT:
                return finish(RunStatus.STOPPED, StopReason.APPROVAL_REJECTED)
            if call.name in open_circuits:
                outcome = ToolResult(
                    "error", f"{call.name} is unavailable after repeated failures in this run"
                )
            else:
                emit("tool_call_started", call_id=call.id, tool=call.name)
                api_scope = tool.effective_api_scope
                for attempt in range(1, tool.max_attempts + 1):
                    if tool_attempts >= agent.limits.max_tool_attempts:
                        return finish(RunStatus.STOPPED, StopReason.MAX_TOOL_ATTEMPTS)
                    if (
                        api_scope == "internal"
                        and internal_api_attempts >= agent.limits.max_internal_api_attempts
                    ):
                        return finish(RunStatus.STOPPED, StopReason.MAX_INTERNAL_API_ATTEMPTS)
                    if (
                        api_scope == "external"
                        and external_api_attempts >= agent.limits.max_external_api_attempts
                    ):
                        return finish(RunStatus.STOPPED, StopReason.MAX_EXTERNAL_API_ATTEMPTS)
                    tool_attempts += 1
                    if api_scope == "internal":
                        internal_api_attempts += 1
                    elif api_scope == "external":
                        external_api_attempts += 1
                    try:
                        outcome = yield ExecuteTool(call, tool, ToolContext(run_id, attempt))
                    except RunCancelled as exc:
                        return interrupted(exc)
                    # `max_attempts > 1` already implies idempotent (Tool rejects
                    # otherwise), so a retryable failure is safe to repeat here.
                    if not outcome.retryable or attempt == tool.max_attempts:
                        break
                    emit(
                        "tool_call_retried",
                        call_id=call.id,
                        attempt=attempt,
                        reason=outcome.summary,
                    )
                    try:
                        yield WaitRetry(
                            min(tool.retry_backoff_seconds * 2 ** min(attempt - 1, 16), 30.0)
                        )
                    except RunCancelled as exc:
                        return interrupted(exc)
                if outcome.status == "error":
                    tool_failures[call.name] = tool_failures.get(call.name, 0) + 1
                    if tool_failures[call.name] >= agent.limits.max_tool_failures:
                        open_circuits.add(call.name)
                        emit(
                            "tool_circuit_opened",
                            tool=call.name,
                            failures=tool_failures[call.name],
                        )
                else:
                    tool_failures[call.name] = 0
            tool_count += 1
            try:
                result_bytes = len(json.dumps(asdict(outcome), ensure_ascii=False).encode("utf-8"))
            except (TypeError, ValueError):
                result_bytes = agent.limits.max_tool_result_bytes + 1
            if result_bytes > agent.limits.max_tool_result_bytes:
                outcome = ToolResult("error", f"{call.name} result exceeded run byte limit")
                emit("tool_call_completed", **tool_result_data(call, outcome))
                return finish(RunStatus.STOPPED, StopReason.TOOL_RESULT_LIMIT)
            consecutive_errors = consecutive_errors + 1 if outcome.status == "error" else 0
            emit("tool_call_completed", **tool_result_data(call, outcome))
            if consecutive_errors >= agent.limits.max_consecutive_tool_errors:
                return finish(RunStatus.STOPPED, StopReason.MAX_TOOL_ERRORS)

    return finish(RunStatus.STOPPED, StopReason.MAX_TURNS)


__all__ = ["agent_machine"]
