"""Two drivers over one loop (ADR-0024).

Neither of these holds policy. `machine.agent_machine` decides everything and
yields the three things it cannot do itself; these perform them and send the
outcomes back. `run` and `arun` differ by one `await`, which is the property
that keeps a new limit or terminal condition from having to be written twice.
"""

import inspect
import uuid
from typing import Any

from ..llm.errors import ProviderError
from ..llm.model import AsyncModelCaller, ModelCaller
from .agent import Agent
from .context import ContextPolicy
from .effects import CallModel, Effect, ValidateTool
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
) -> RunResult:
    """Run one agent until a typed terminal outcome is reached."""
    machine = agent_machine(
        agent,
        task,
        run_id=uuid.uuid4().hex,
        event_sink=event_sink,
        context_policy=context_policy,
        approver=approver,
    )
    active = executor or LocalToolExecutor()
    outcome: Any = None
    failure: ProviderError | None = None

    while True:
        try:
            effect = machine.throw(failure) if failure is not None else machine.send(outcome)
        except StopIteration as stop:
            return stop.value
        failure, outcome = None, None
        try:
            outcome = _perform(effect, caller, active)
        except ProviderError as exc:
            # Reported by throwing in, so the machine's handling reads as the
            # `try/except` around a direct call that it replaced.
            failure = exc


async def arun(
    agent: Agent,
    task: str,
    *,
    caller: AsyncModelCaller,
    executor: ToolExecutor | None = None,
    event_sink: EventSink | None = None,
    context_policy: ContextPolicy | None = None,
    approver: Approver | None = None,
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
    )
    active = executor or LocalToolExecutor()
    outcome: Any = None
    failure: ProviderError | None = None

    while True:
        try:
            effect = machine.throw(failure) if failure is not None else machine.send(outcome)
        except StopIteration as stop:
            return stop.value
        failure, outcome = None, None
        try:
            outcome = await _aperform(effect, caller, active)
        except ProviderError as exc:
            failure = exc


def _perform(effect: Effect, caller: ModelCaller, executor: ToolExecutor) -> Any:
    """Exhaustive over the effect union; a missing branch would hang the
    machine rather than raise, so the union is kept small (ADR-0024)."""
    if isinstance(effect, CallModel):
        return caller(
            system=effect.system,
            messages=effect.messages,
            tools=effect.tools,
            settings=effect.settings,
        )
    if isinstance(effect, ValidateTool):
        return executor.validate(effect.call, effect.tool)
    return executor.execute(effect.call, effect.tool, effect.context)


async def _aperform(effect: Effect, caller: AsyncModelCaller, executor: ToolExecutor) -> Any:
    """The same dispatch, awaiting whatever turns out to be awaitable.

    Reusing `_perform` is deliberate: were the dispatch written twice, an
    effect added to one and not the other would hang instead of failing.
    """
    outcome = _perform(effect, caller, executor)
    return await outcome if inspect.isawaitable(outcome) else outcome


__all__ = ["arun", "run"]
