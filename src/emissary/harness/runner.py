"""Two drivers over one loop (ADR-0024).

Neither of these holds policy. `machine.agent_machine` decides everything and
yields the three things it cannot do itself; these perform them and send the
outcomes back. `run` and `arun` differ by one `await`, which is the property
that keeps a new limit or terminal condition from having to be written twice.
"""

import asyncio
import contextlib
import inspect
import threading
import time
import uuid
from dataclasses import replace
from typing import Any

from ..llm.errors import ProviderError
from ..llm.messages import Message
from ..llm.model import (
    AsyncFallbackModelCaller,
    AsyncModelCaller,
    AsyncSpecModelCaller,
    FallbackModelCaller,
    ModelCaller,
    SpecModelCaller,
)
from .agent import Agent, ModelAttemptLimitExceeded, RunCancelled, RunTimedOut
from .context import ContextPolicy
from .effects import CallModel, Effect, ExecuteTool, ValidateTool, WaitRetry
from .events import EventSink
from .machine import agent_machine
from .policy import Approver
from .state import RunResult
from .tools import LocalToolExecutor, ToolExecutor


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
) -> RunResult:
    """Run one agent until a typed terminal outcome is reached."""
    machine = agent_machine(
        agent,
        task,
        run_id=uuid.uuid4().hex,
        event_sink=event_sink,
        context_policy=context_policy,
        approver=approver,
        history=history,
    )
    active = executor or LocalToolExecutor()
    outcome: Any = None
    failure: ProviderError | ModelAttemptLimitExceeded | RunCancelled | None = None
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
                outcome = _perform(effect, caller, active, deadline, cancel_event)
            except (ProviderError, ModelAttemptLimitExceeded, RunCancelled) as exc:
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
) -> RunResult:
    """`run` on an event loop — the same machine, awaiting each effect.

    The executor may be synchronous: most tools are local computation, and
    async at the model boundary should not force every one of them to be a
    coroutine.
    """
    machine = agent_machine(
        agent,
        task,
        run_id=uuid.uuid4().hex,
        event_sink=event_sink,
        context_policy=context_policy,
        approver=approver,
        history=history,
    )
    active = executor or LocalToolExecutor()
    outcome: Any = None
    failure: ProviderError | ModelAttemptLimitExceeded | RunTimedOut | None = None
    deadline = time.monotonic() + agent.limits.max_duration_seconds
    caller = _budget_caller(caller, agent.limits.max_model_attempts, asynchronous=True)

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
                    outcome = await _aperform(effect, caller, active, deadline)
            except (ProviderError, ModelAttemptLimitExceeded, RunTimedOut) as exc:
                failure = exc
            except TimeoutError:
                # Only our own deadline is a run stop; a TimeoutError raised by
                # the caller or a tool is theirs to report.
                if not timer.expired():
                    raise
                failure = RunTimedOut()
    except asyncio.CancelledError:
        # A cancelled task cannot return a RunResult, but its log still ends on
        # a terminal event before the cancellation carries on.
        with contextlib.suppress(StopIteration):
            machine.throw(RunCancelled())
        raise
    finally:
        machine.close()


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
    if isinstance(effect, ExecuteTool):
        return executor.execute(effect.call, effect.tool, effect.context)
    raise TypeError(f"unhandled effect {type(effect).__name__}")


async def _aperform(
    effect: Effect, caller: AsyncModelCaller, executor: ToolExecutor, deadline: float
) -> Any:
    """The same dispatch, awaiting whatever turns out to be awaitable.

    Reusing `_perform` is deliberate: were the dispatch written twice, an
    effect added to one and not the other would hang instead of failing.
    """
    if isinstance(effect, WaitRetry):
        await asyncio.sleep(min(effect.seconds, max(0.0, deadline - time.monotonic())))
        return None
    outcome = _perform(effect, caller, executor, deadline)
    return await outcome if inspect.isawaitable(outcome) else outcome


__all__ = ["arun", "run"]
