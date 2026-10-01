"""Two drivers over one loop (ADR-0024).

Neither of these holds policy. `machine.agent_machine` decides everything and
yields the three things it cannot do itself; these perform them and send the
outcomes back. `run` and `arun` differ by one `await`, which is the property
that keeps a new limit or terminal condition from having to be written twice.
"""

import asyncio
import contextlib
import inspect
import logging
import threading
import time
import uuid
from dataclasses import dataclass, replace
from typing import Any

from ...llm.decision import Usage
from ...llm.errors import ProviderError
from ...llm.messages import Message
from ...llm.model import (
    AsyncFallbackModelCaller,
    AsyncModelCaller,
    AsyncSpecModelCaller,
    FallbackModelCaller,
    ModelCaller,
    SpecModelCaller,
)
from ..agent import Agent, ModelAttemptLimitExceeded, RunCancelled, RunTimedOut
from ..conversation.context import ContextPolicy
from ..conversation.events import EventSink, NullEventSink, RunEvent, new_event
from ..policy import (
    Approver,
    AuthorizationContext,
    AuthorizationMismatch,
    Authorizer,
    AuthorizerFailed,
    verify_decision,
)
from ..state import RunResult, RunStatus, StopReason
from ..tooling.preparation import (
    PreparationError,
    PreparedRun,
    RoutingToolExecutor,
    direct_prepared,
    prepare,
)
from ..tooling.tools import AsyncLocalToolExecutor, LocalToolExecutor, ToolExecutor, ToolResult
from .effects import AuthorizeTool, CallModel, Effect, ExecuteTool, ValidateTool, WaitRetry
from .machine import agent_machine

logger = logging.getLogger(__name__)


def run(
    agent: Agent,
    task: str,
    *,
    caller: ModelCaller,
    executor: ToolExecutor | None = None,
    event_sink: EventSink | None = None,
    context_policy: ContextPolicy | None = None,
    approver: Approver | None = None,
    history: tuple[Message, ...] = (),
    cancel_event: threading.Event | None = None,
    authorizer: Authorizer | None = None,
    authorization_context: AuthorizationContext | None = None,
) -> RunResult:
    """Run one agent until a typed terminal outcome is reached.

    Direct synchronous tools only: tool sources and coroutine tools need `arun`,
    and a hidden event loop per call would be a worse surprise than this error.
    """
    _require_synchronous(agent)
    _require_context_with_authorizer(authorizer, authorization_context)
    machine = agent_machine(
        agent,
        task,
        run_id=uuid.uuid4().hex,
        event_sink=event_sink,
        context_policy=context_policy,
        approver=approver,
        history=history,
        authorize=authorizer is not None,
        authorization_context=authorization_context,
    )
    active = executor or LocalToolExecutor()
    outcome: Any = None
    failure: ProviderError | ModelAttemptLimitExceeded | RunCancelled | AuthorizerFailed | None = (
        None
    )
    deadline = time.monotonic() + agent.limits.max_duration_seconds

    def check_stop() -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise RunCancelled()
        if time.monotonic() >= deadline:
            raise RunTimedOut()

    if cancel_event is not None and isinstance(caller, FallbackModelCaller):

        def retry_sleep(seconds: float) -> None:
            if cancel_event.wait(min(seconds, max(0.0, deadline - time.monotonic()))):
                raise RunCancelled()
            check_stop()

        caller = replace(caller, retry_sleep=retry_sleep)
    caller = _budget_caller(caller, agent.limits.max_model_attempts, check_stop=check_stop)

    try:
        while True:
            try:
                effect = machine.throw(failure) if failure is not None else machine.send(outcome)
            except StopIteration as stop:
                return stop.value
            failure, outcome = None, None
            try:
                # Checked before dispatch, like every count budget: the deadline
                # stops new work and never discards a result already paid for.
                check_stop()
                outcome = _perform(effect, caller, active, deadline, cancel_event, authorizer)
            except (
                ProviderError,
                ModelAttemptLimitExceeded,
                RunCancelled,
                AuthorizerFailed,
            ) as exc:
                failure = exc
    finally:
        machine.close()


async def arun(
    agent: Agent,
    task: str,
    *,
    caller: AsyncModelCaller,
    executor: ToolExecutor | None = None,
    event_sink: EventSink | None = None,
    context_policy: ContextPolicy | None = None,
    approver: Approver | None = None,
    history: tuple[Message, ...] = (),
    authorizer: Authorizer | None = None,
    authorization_context: AuthorizationContext | None = None,
    prepared: PreparedRun | None = None,
) -> RunResult:
    """`run` on an event loop — the same machine, awaiting each effect.

    The executor may be synchronous: most tools are local computation, and
    async at the model boundary should not force every one of them to be a
    coroutine. Tool callables may be coroutine functions.

    Tool sources are opened inside the run, on its deadline, and closed when it
    ends. Pass `prepared` (from `prepare(agent)`) to share connections across
    runs; the caller then owns that lifetime and must not share it across
    tenants or authentication identities.
    """
    _require_context_with_authorizer(authorizer, authorization_context)
    run_id = uuid.uuid4().hex
    deadline = time.monotonic() + agent.limits.max_duration_seconds
    drive = _ArunInputs(
        agent, task, caller, executor, event_sink, context_policy, approver, history,
        authorizer, authorization_context, run_id, deadline,
    )  # fmt: skip
    if prepared is not None:
        prepared.require_for(agent, authorization_context)
    if prepared is not None or not agent.toolsets:
        return await _drive(drive, prepared or direct_prepared(agent), ())

    sink = event_sink if event_sink is not None else NullEventSink()
    log: list[RunEvent] = []

    def record(kind: str, **data: Any) -> None:
        event = new_event(run_id, len(log) + 1, kind, **data)
        log.append(event)
        try:
            sink.emit(event)
        except Exception:
            logger.exception("agent event sink failed for %s", kind)

    def stopped(status: RunStatus, reason: StopReason) -> RunResult:
        record("run_stopped", status=status.value, reason=reason.value)
        return RunResult(run_id, status, reason, None, Usage(0, 0), tuple(log))

    record("run_started", agent=agent.name)
    record("tool_preparation_started", sources=[source.namespace for source in agent.toolsets])
    async with contextlib.AsyncExitStack() as stack:
        try:
            async with asyncio.timeout(max(0.0, deadline - time.monotonic())) as timer:
                opened = await stack.enter_async_context(prepare(agent))
        except PreparationError as exc:
            record("tool_preparation_failed", reason=str(exc))
            return stopped(RunStatus.FAILED, StopReason.PREPARATION_FAILED)
        except TimeoutError:
            if not timer.expired():
                raise
            record("tool_preparation_failed", reason="deadline exceeded")
            return stopped(RunStatus.STOPPED, StopReason.MAX_DURATION)
        except asyncio.CancelledError:
            stopped(RunStatus.CANCELLED, StopReason.CANCELLED)
            raise
        return await _drive(drive, opened, tuple(log))


@dataclass(frozen=True)
class _ArunInputs:
    agent: Agent
    task: str
    caller: AsyncModelCaller
    executor: ToolExecutor | None
    event_sink: EventSink | None
    context_policy: ContextPolicy | None
    approver: Approver | None
    history: tuple[Message, ...]
    authorizer: Authorizer | None
    authorization_context: AuthorizationContext | None
    run_id: str
    deadline: float


async def _drive(
    inputs: "_ArunInputs", prepared: PreparedRun, prefix_events: tuple[RunEvent, ...]
) -> RunResult:
    agent, deadline = inputs.agent, inputs.deadline
    machine = agent_machine(
        agent,
        inputs.task,
        run_id=inputs.run_id,
        event_sink=inputs.event_sink,
        context_policy=inputs.context_policy,
        approver=inputs.approver,
        history=inputs.history,
        prepared=prepared,
        authorize=inputs.authorizer is not None,
        authorization_context=inputs.authorization_context,
        prefix_events=prefix_events,
    )
    if prepared.has_sources:
        active = RoutingToolExecutor(prepared, inputs.executor)
    else:
        active = inputs.executor or AsyncLocalToolExecutor()
    outcome: Any = None
    failure: ProviderError | ModelAttemptLimitExceeded | RunTimedOut | AuthorizerFailed | None = (
        None
    )
    caller = _budget_caller(inputs.caller, agent.limits.max_model_attempts, asynchronous=True)

    try:
        while True:
            try:
                effect = machine.throw(failure) if failure is not None else machine.send(outcome)
            except StopIteration as stop:
                return stop.value
            failure, outcome = None, None
            timer = asyncio.timeout(deadline - time.monotonic())
            try:
                if time.monotonic() >= deadline:
                    raise RunTimedOut()
                async with timer:
                    outcome = await _aperform(effect, caller, active, deadline, inputs.authorizer)
            except (
                ProviderError,
                ModelAttemptLimitExceeded,
                RunTimedOut,
                AuthorizerFailed,
            ) as exc:
                failure = exc
            except TimeoutError:
                # Only our own deadline is a run stop; a TimeoutError raised by
                # the caller or a tool is theirs to report.
                if timer.expired():
                    failure = RunTimedOut()
                elif isinstance(effect, AuthorizeTool):
                    failure = AuthorizerFailed()
                else:
                    raise
    except asyncio.CancelledError:
        # A cancelled task cannot return a RunResult, but its log still ends on
        # a terminal event before the cancellation carries on.
        with contextlib.suppress(StopIteration):
            machine.throw(RunCancelled())
        raise
    finally:
        machine.close()


def _require_synchronous(agent: Agent) -> None:
    if agent.toolsets:
        raise ValueError(f"agent {agent.name!r} has toolsets, which need arun()")
    for tool in agent.tools:
        if inspect.iscoroutinefunction(tool.execute):
            raise ValueError(f"tool {tool.name!r} is a coroutine function and needs arun()")


def _require_context_with_authorizer(
    authorizer: Authorizer | None, context: AuthorizationContext | None
) -> None:
    if authorizer is not None and context is None:
        raise ValueError("an authorizer requires an authorization_context")


def _budget_caller(caller, max_attempts: int, *, asynchronous: bool = False, check_stop=None):
    """Count built-in physical attempts; count an opaque caller once per invocation."""
    attempts = 0

    def admit() -> None:
        nonlocal attempts
        if check_stop is not None:
            check_stop()
        if attempts >= max_attempts:
            raise ModelAttemptLimitExceeded()
        attempts += 1

    if isinstance(
        caller,
        (SpecModelCaller, AsyncSpecModelCaller, FallbackModelCaller, AsyncFallbackModelCaller),
    ):
        existing = caller.before_attempt

        def combined() -> None:
            admit()
            if existing is not None:
                existing()

        return replace(caller, before_attempt=combined)
    if asynchronous:

        async def async_caller(**kwargs):
            admit()
            return await caller(**kwargs)

        return async_caller

    def sync_caller(**kwargs):
        admit()
        return caller(**kwargs)

    return sync_caller


def _perform(
    effect: Effect,
    caller: ModelCaller,
    executor: ToolExecutor,
    deadline: float,
    cancel_event: threading.Event | None = None,
    authorizer: Authorizer | None = None,
) -> Any:
    """Exhaustive over the effect union (ADR-0024). An unknown effect raises:
    a silent fallthrough would perform it as whichever branch came last."""
    if isinstance(effect, CallModel):
        return caller(
            system=effect.system,
            messages=effect.messages,
            tools=effect.tools,
            settings=effect.settings,
        )
    if isinstance(effect, ValidateTool):
        return executor.validate(effect.call, effect.tool)
    if isinstance(effect, WaitRetry):
        delay = min(effect.seconds, max(0.0, deadline - time.monotonic()))
        if cancel_event is None:
            time.sleep(delay)
        elif cancel_event.wait(delay):
            raise RunCancelled()
        return None
    if isinstance(effect, AuthorizeTool):
        return _authorize(effect, authorizer)
    if isinstance(effect, ExecuteTool):
        return _execute(effect, executor)
    raise TypeError(f"unhandled effect {type(effect).__name__}")


def _authorize(effect: AuthorizeTool, authorizer: Authorizer | None) -> Any:
    """Any failure of the policy is `AuthorizerFailed`: policy code is an
    extension boundary, and its exceptions must not choose the run's outcome."""
    if authorizer is None:
        raise AuthorizerFailed()
    try:
        return authorizer(effect.request)
    except Exception as exc:
        logger.exception("authorizer failed for call %s", effect.request.call_id)
        raise AuthorizerFailed() from exc


def _execute(effect: ExecuteTool, executor: ToolExecutor) -> Any:
    """Dispatch what was authorized, not what the model message still holds.

    The decision is checked again here, immediately before delegation: it may
    have expired since the machine accepted it.
    """
    request = effect.request
    if request is None:
        return executor.execute(effect.call, effect.tool, effect.context)
    if effect.context.authorization is not None:
        try:
            verify_decision(request, effect.context.authorization)
        except AuthorizationMismatch:
            return ToolResult(
                "error", f"{effect.tool.name} was not dispatched: authorization lapsed"
            )
    return executor.execute(request.detached_call(), effect.tool, effect.context)


async def _aperform(
    effect: Effect,
    caller: AsyncModelCaller,
    executor: ToolExecutor,
    deadline: float,
    authorizer: Authorizer | None = None,
) -> Any:
    """The same dispatch, awaiting whatever turns out to be awaitable.

    Reusing `_perform` is deliberate: were the dispatch written twice, an
    effect added to one and not the other would hang instead of failing.
    """
    if isinstance(effect, WaitRetry):
        await asyncio.sleep(min(effect.seconds, max(0.0, deadline - time.monotonic())))
        return None
    outcome = _perform(effect, caller, executor, deadline, None, authorizer)
    if not inspect.isawaitable(outcome):
        return outcome
    if not isinstance(effect, AuthorizeTool):
        return await outcome
    try:
        return await outcome
    except TimeoutError:
        raise  # the run's own deadline; `_drive` tells it apart from the policy's
    except Exception as exc:
        logger.exception("authorizer failed for call %s", effect.request.call_id)
        raise AuthorizerFailed() from exc


__all__ = ["arun", "run"]
