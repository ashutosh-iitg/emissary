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
from collections.abc import Generator, Sequence
from dataclasses import asdict
from typing import Any

from ...llm.decision import FinalOutput, Refusal, ToolCalls, Usage
from ...llm.errors import ProviderError
from ...llm.messages import Message, TextBlock, UserMessage
from ..agent import Agent, ModelAttemptLimitExceeded, RunCancelled, RunTimedOut
from ..conversation.context import CompleteHistory, ContextPolicy
from ..conversation.events import EventSink, NullEventSink, RunEvent, new_event
from ..conversation.projection import (
    context_op_data,
    derive_messages,
    message_to_data,
    model_result_data,
    tool_result_data,
    user_message_data,
    validate_history,
)
from ..policy import (
    ApprovalDecision,
    Approver,
    AuthorizationContext,
    AuthorizationDecision,
    AuthorizationMismatch,
    AuthorizerFailed,
    InvocationRequest,
    approval_for,
    build_request,
    decision_data,
    verify_decision,
)
from ..state import RunResult, RunStatus, StopReason
from ..tooling.preparation import PreparedRun, direct_prepared
from ..tooling.tools import ToolContext, ToolResult
from .effects import AuthorizeTool, CallModel, Effect, ExecuteTool, ValidateTool, WaitRetry

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
    prepared: PreparedRun | None = None,
    authorize: bool = False,
    authorization_context: AuthorizationContext | None = None,
    prefix_events: Sequence[RunEvent] = (),
) -> Generator[Effect, Any, RunResult]:
    """Drive one agent to a typed terminal outcome, yielding work to be done.

    `run_id` is a parameter rather than generated here: a pure core has no
    business inventing identity, and a fixed id makes a driven run's log stable
    enough to compare.

    `prepared` is the finished tool catalog; without it the agent's direct tools
    are used. With `authorize`, every call is put to the application's policy as
    an `AuthorizeTool` effect — for the whole batch before anything runs, and
    again before each attempt. `prefix_events` are events the driver already
    recorded (run start, preparation) so the log stays one canonical sequence.
    """
    if not task:
        raise ValueError("task must not be empty")
    if history:
        validate_history(history)

    sink = event_sink if event_sink is not None else NullEventSink()
    context = context_policy or CompleteHistory()
    prepared = prepared if prepared is not None else direct_prepared(agent)
    registry = prepared.registry
    principal = authorization_context or AuthorizationContext("unspecified")
    events: list[RunEvent] = list(prefix_events)
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

    def authorize_request(
        request: InvocationRequest, stage: str
    ) -> Generator[Effect, Any, tuple[RunResult | None, AuthorizationDecision | None]]:
        """Fail closed: a policy that errors, mismatches or expires is a stop, never an allow."""
        try:
            decision = yield AuthorizeTool(request)
        except AuthorizerFailed:
            emit("authorization_failed", call_id=request.call_id, stage=stage, reason="error")
            return finish(RunStatus.FAILED, StopReason.AUTHORIZATION_ERROR), None
        except RunCancelled as exc:
            return interrupted(exc), None
        try:
            allowed = verify_decision(request, decision)
        except AuthorizationMismatch:
            emit("authorization_failed", call_id=request.call_id, stage=stage, reason="mismatch")
            return finish(RunStatus.FAILED, StopReason.AUTHORIZATION_ERROR), None
        emit("authorization_resolved", stage=stage, **decision_data(request, decision))
        if not allowed:
            return finish(RunStatus.STOPPED, StopReason.AUTHORIZATION_DENIED), None
        return None, decision

    if not prefix_events:
        emit("run_started", agent=agent.name)
    if prepared.has_sources:
        emit(
            "tools_prepared",
            authorization="policy" if authorize else "allow_registered_tools",
            **prepared.catalog_data(),
        )
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
                request = build_request(
                    run_id=run_id,
                    call=call,
                    tool=tool,
                    origin=prepared.origin_of(call.name),
                    context=principal,
                )
            except (TypeError, ValueError):
                emit("tool_call_rejected", call_id=call.id, reason="arguments_not_json")
                return finish(RunStatus.FAILED, StopReason.INVALID_TOOL)
            try:
                # The detached copy is what every callback sees; `request` is
                # what runs, so no callback can change the dispatched arguments.
                invalid = yield ValidateTool(request.detached_call(), tool)
            except RunCancelled as exc:
                return interrupted(exc)
            if invalid is not None:
                emit("tool_call_rejected", call_id=call.id, reason=invalid.summary)
                return finish(RunStatus.FAILED, StopReason.INVALID_TOOL)
            resolved.append((call, tool, request))

        if authorize:
            for _call, _tool, request in resolved:
                stopped, _decision = yield from authorize_request(request, "batch")
                if stopped is not None:
                    return stopped

        for call, tool, request in resolved:
            approval = approval_for(request.detached_call(), tool, approver)
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
                    attempt_request = request.for_attempt(attempt)
                    attempt_decision = None
                    if authorize:
                        stopped, attempt_decision = yield from authorize_request(
                            attempt_request, "attempt"
                        )
                        if stopped is not None:
                            return stopped
                    tool_attempts += 1
                    if api_scope == "internal":
                        internal_api_attempts += 1
                    elif api_scope == "external":
                        external_api_attempts += 1
                    try:
                        outcome = yield ExecuteTool(
                            call,
                            tool,
                            ToolContext(run_id, attempt, attempt_decision),
                            attempt_request,
                        )
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
