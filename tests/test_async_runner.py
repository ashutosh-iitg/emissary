"""One loop, two drivers (ADR-0024).

`tests/test_runner.py` pins the loop's behaviour and is deliberately unmodified
by the refactor — if the machine drifts, it fails. This file pins the two
properties that only exist once the loop is a machine: that `arun` reaches the
same trajectory as `run`, and that the machine itself performs no I/O at all.
"""

import asyncio
from dataclasses import dataclass, field

from emissary.harness.agent import Agent, RunLimits
from emissary.harness.effects import CallModel, ExecuteTool, ValidateTool
from emissary.harness.events import InMemoryEventSink
from emissary.harness.machine import agent_machine
from emissary.harness.runner import arun, run
from emissary.harness.state import RunStatus, StopReason
from emissary.harness.tools import LocalToolExecutor, Tool, ToolResult
from emissary.llm.decision import FinalOutput, ModelResult, ToolCall, ToolCalls, Usage
from emissary.llm.errors import ProviderError

ADD = Tool("add", "Add.", {"type": "object"}, lambda a, b: {"sum": a + b})


@dataclass
class ScriptedCaller:
    decisions: list
    messages_seen: list = field(default_factory=list)

    def __call__(self, *, system, messages, tools=(), settings=None):
        self.messages_seen.append(messages)
        return ModelResult(self.decisions.pop(0), "fake", "scripted", Usage(2, 1))


@dataclass
class AsyncScriptedCaller:
    decisions: list
    messages_seen: list = field(default_factory=list)

    async def __call__(self, *, system, messages, tools=(), settings=None):
        self.messages_seen.append(messages)
        await asyncio.sleep(0)
        return ModelResult(self.decisions.pop(0), "fake", "scripted", Usage(2, 1))


class AsyncExecutor:
    """Validates locally, executes with an await — the shape a remote one has."""

    def __init__(self) -> None:
        self._local = LocalToolExecutor()

    def validate(self, call, tool):
        return self._local.validate(call, tool)

    async def execute(self, call, tool, context):
        await asyncio.sleep(0)
        return self._local.execute(call, tool, context)


def _script():
    return [ToolCalls((ToolCall("one", "add", {"a": 2, "b": 3}),)), FinalOutput(text="5")]


def _agent():
    return Agent("calculator", "Use tools.", tools=(ADD,), limits=RunLimits())


def _kinds(sink):
    return [event.kind for event in sink.events]


async def test_arun_reaches_the_same_trajectory_as_run():
    """The whole point of the refactor: not merely the same answer, but the
    same sequence of recorded facts."""
    sync_sink, async_sink = InMemoryEventSink(), InMemoryEventSink()

    sync_result = run(
        _agent(),
        "add",
        caller=ScriptedCaller(_script()),
        executor=LocalToolExecutor(),
        event_sink=sync_sink,
    )
    async_result = await arun(
        _agent(),
        "add",
        caller=AsyncScriptedCaller(_script()),
        executor=AsyncExecutor(),
        event_sink=async_sink,
    )

    assert _kinds(async_sink) == _kinds(sync_sink)
    assert async_result.status is sync_result.status is RunStatus.COMPLETED
    assert async_result.output == sync_result.output == FinalOutput(text="5")
    assert async_result.usage == sync_result.usage == Usage(4, 2)
    assert async_result.messages == sync_result.messages


async def test_arun_accepts_a_synchronous_executor():
    """Async at the model boundary should not force every tool to be async —
    most tools are local computation."""
    result = await arun(
        _agent(),
        "add",
        caller=AsyncScriptedCaller(_script()),
        executor=LocalToolExecutor(),
    )

    assert result.status is RunStatus.COMPLETED


async def test_arun_classifies_a_provider_failure_like_run_does():
    """Errors reach the machine by `throw`; losing that would turn a recorded
    model_call_failed into an unhandled exception."""

    class Failing:
        async def __call__(self, **kwargs):
            raise ProviderError("overloaded", retryable=True)

    sink = InMemoryEventSink()
    result = await arun(_agent(), "add", caller=Failing(), event_sink=sink)

    assert result.status is RunStatus.FAILED
    assert result.stop_reason is StopReason.MODEL_ERROR
    assert "model_call_failed" in _kinds(sink)


async def test_two_runs_interleave_on_one_thread():
    """`asyncio.to_thread` would have capped this at the thread pool; the
    point of an async driver is that it does not (ADR-0024)."""
    order: list[str] = []

    def caller_for(label):
        async def call(*, system, messages, tools=(), settings=None):
            order.append(f"{label}-start")
            await asyncio.sleep(0.01)
            order.append(f"{label}-end")
            return ModelResult(FinalOutput(text=label), "fake", "scripted", Usage(1, 1))

        return call

    await asyncio.gather(
        arun(_agent(), "a", caller=caller_for("a")),
        arun(_agent(), "b", caller=caller_for("b")),
    )

    # Both started before either finished — genuine interleaving, not sequence.
    assert order.index("b-start") < order.index("a-end")


# --- The machine itself ----------------------------------------------------


def test_the_machine_performs_no_io_at_all():
    """Driven by hand with no caller and no executor. If any I/O were left in
    the loop, this test could not be written."""
    machine = agent_machine(_agent(), "add", run_id="fixed", event_sink=InMemoryEventSink())
    decisions = _script()
    effects: list = []
    outcome = None

    try:
        while True:
            effect = machine.send(outcome)
            effects.append(effect)
            if isinstance(effect, CallModel):
                outcome = ModelResult(decisions.pop(0), "fake", "scripted", Usage(2, 1))
            elif isinstance(effect, ValidateTool):
                outcome = None
            elif isinstance(effect, ExecuteTool):
                outcome = ToolResult("success", "added", {"sum": 5})
    except StopIteration as stop:
        result = stop.value

    assert [type(effect) for effect in effects] == [
        CallModel,
        ValidateTool,
        ExecuteTool,
        CallModel,
    ]
    assert result.status is RunStatus.COMPLETED
    assert result.run_id == "fixed"


def test_the_machine_validates_a_whole_batch_before_executing_any_of_it():
    """The ordering ADR-0007 depends on. A driver cannot restore it if the
    machine yields validate and execute interleaved."""
    calls = (ToolCall("one", "add", {"a": 1, "b": 1}), ToolCall("two", "add", {"a": 2, "b": 2}))
    machine = agent_machine(_agent(), "add", run_id="fixed", event_sink=InMemoryEventSink())
    kinds: list[str] = []
    outcome = None

    # One turn only: a later turn's validations would otherwise be compared
    # against this turn's executions, which says nothing about the ordering.
    machine.send(None)
    outcome = ModelResult(ToolCalls(calls), "fake", "scripted", Usage(1, 1))
    for _ in range(len(calls) * 2):
        effect = machine.send(outcome)
        kinds.append(type(effect).__name__)
        outcome = None if isinstance(effect, ValidateTool) else ToolResult("success", "ok", {})

    assert kinds == ["ValidateTool", "ValidateTool", "ExecuteTool", "ExecuteTool"]


def test_the_machine_is_deterministic_given_a_run_id():
    """A fixed id makes the log stable apart from timestamps, which is what
    lets two trajectories be compared at all."""

    def drive():
        machine = agent_machine(_agent(), "add", run_id="fixed", event_sink=InMemoryEventSink())
        outcome = None
        try:
            while True:
                effect = machine.send(outcome)
                outcome = (
                    ModelResult(FinalOutput(text="5"), "fake", "scripted", Usage(2, 1))
                    if isinstance(effect, CallModel)
                    else None
                )
        except StopIteration as stop:
            return stop.value

    first, second = drive(), drive()
    assert [event.kind for event in first.events] == [event.kind for event in second.events]
    assert [event.data for event in first.events] == [event.data for event in second.events]


def test_a_driver_that_stops_early_does_not_corrupt_the_log():
    """Closing the generator mid-run must not leave a half-written event; the
    machine holds its own state and simply stops."""
    sink = InMemoryEventSink()
    machine = agent_machine(_agent(), "add", run_id="fixed", event_sink=sink)
    machine.send(None)
    machine.close()

    assert _kinds(sink) == ["run_started", "user_message", "model_call_started"]


def test_the_effect_union_stays_small_enough_for_thin_drivers():
    """A union that grows past a handful is the signal this shape has stopped
    paying for itself (ADR-0024). Both drivers must stay exhaustive over it."""
    from emissary.harness import effects

    assert set(effects.__all__) == {"CallModel", "Effect", "ExecuteTool", "ValidateTool"}
