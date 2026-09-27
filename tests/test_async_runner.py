"""One loop, two drivers (ADR-0024).

`tests/test_runner.py` pins the loop's behaviour and is deliberately unmodified
by the refactor — if the machine drifts, it fails. This file pins the two
properties that only exist once the loop is a machine: that `arun` reaches the
same trajectory as `run`, and that the machine itself performs no I/O at all.
"""

import asyncio
import threading
import time
from dataclasses import dataclass, field

import pytest

from emissary.harness.agent import Agent, RunLimits
from emissary.harness.effects import CallModel, ExecuteTool, ValidateTool
from emissary.harness.events import InMemoryEventSink
from emissary.harness.machine import agent_machine
from emissary.harness.runner import arun, run
from emissary.harness.state import RunStatus, StopReason
from emissary.harness.tools import LocalToolExecutor, Tool, ToolResult
from emissary.llm.decision import FinalOutput, ModelResult, ToolCall, ToolCalls, Usage
from emissary.llm.errors import ProviderError
from emissary.llm.model import AsyncFallbackModelCaller, FallbackModelCaller
from emissary.llm.provider import parse_spec

ADD = Tool("add", "Add.", {"type": "object"}, lambda a, b: {"sum": a + b}, api_scope="none")


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


async def test_async_run_deadline_cancels_stalled_model_call():
    cancelled = asyncio.Event()

    async def stalled(**kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    agent = Agent("stalled", "Wait.", limits=RunLimits(max_duration_seconds=0.01))
    with pytest.raises(TimeoutError):
        await arun(agent, "go", caller=stalled)
    assert cancelled.is_set()


def test_tool_retry_attempt_budget_stops_run():
    tool = Tool(
        "fail",
        "Fail.",
        {"type": "object"},
        lambda **kwargs: ToolResult("error", "retry", retryable=True),
        idempotent=True,
        max_attempts=10,
        api_scope="none",
    )
    agent = Agent("bounded", "Use tools.", tools=(tool,), limits=RunLimits(max_tool_attempts=2))
    caller = ScriptedCaller([ToolCalls((ToolCall("one", "fail", {}),))])
    result = run(agent, "go", caller=caller)
    assert result.stop_reason is StopReason.MAX_TOOL_ATTEMPTS
    assert len([event for event in result.events if event.kind == "tool_call_retried"]) == 2


def test_sync_run_checks_deadline_after_blocking_effect_returns():
    def slow(**kwargs):
        time.sleep(0.02)
        return ModelResult(FinalOutput(text="late"), "fake", "scripted", Usage(1, 1))

    agent = Agent("slow", "Wait.", limits=RunLimits(max_duration_seconds=0.001))
    with pytest.raises(TimeoutError):
        run(agent, "go", caller=slow)


@pytest.mark.parametrize(
    ("scope", "side_effect", "limit_name", "reason"),
    [
        ("internal", "none", "max_internal_api_attempts", StopReason.MAX_INTERNAL_API_ATTEMPTS),
        (None, "external", "max_external_api_attempts", StopReason.MAX_EXTERNAL_API_ATTEMPTS),
    ],
)
def test_api_attempt_budget_counts_before_dispatch(scope, side_effect, limit_name, reason):
    attempts = []

    def invoke():
        attempts.append(1)
        return ToolResult("success", "ok")

    tool = Tool("api", "API.", {"type": "object"}, invoke, api_scope=scope, side_effect=side_effect)
    limits = RunLimits(**{limit_name: 1})
    agent = Agent("bounded", "Use tools.", tools=(tool,), limits=limits)
    caller = ScriptedCaller([ToolCalls((ToolCall("one", "api", {}), ToolCall("two", "api", {})))])

    result = run(agent, "go", caller=caller)

    assert result.stop_reason is reason
    assert len(attempts) == 1


def test_tool_retry_uses_exponential_backoff(monkeypatch):
    delays = []
    monkeypatch.setattr("emissary.harness.runner.time.sleep", delays.append)
    attempts = []

    def invoke(*, idempotency_key):
        attempts.append(idempotency_key)
        return (
            ToolResult("error", "retry", retryable=True)
            if len(attempts) < 3
            else ToolResult("success", "ok")
        )

    tool = Tool(
        "api",
        "API.",
        {"type": "object"},
        invoke,
        idempotent=True,
        max_attempts=3,
        api_scope="external",
    )
    agent = Agent("bounded", "Use tools.", tools=(tool,))
    caller = ScriptedCaller([ToolCalls((ToolCall("one", "api", {}),)), FinalOutput(text="done")])

    assert run(agent, "go", caller=caller).status is RunStatus.COMPLETED
    assert delays == [0.25, 0.5]
    assert len(set(attempts)) == 1


def test_physical_model_attempt_budget_stops_fallback_before_extra_request(monkeypatch):
    attempts = []

    def failed_request(spec, **kwargs):
        attempts.append(spec.name)
        raise ProviderError("unavailable", retryable=True)

    monkeypatch.setattr("emissary.llm.model.call_model", failed_request)
    monkeypatch.setattr("emissary.llm.retry.RETRY_DELAYS", (0.0, 0.0, 0.0))
    caller = FallbackModelCaller(parse_spec("anthropic"), parse_spec("kimi"))
    agent = Agent("bounded", "Use model.", limits=RunLimits(max_model_attempts=2))

    result = run(agent, "go", caller=caller)

    assert result.stop_reason is StopReason.MAX_MODEL_ATTEMPTS
    assert attempts == ["anthropic", "anthropic"]


def test_sync_cancellation_prevents_model_retry(monkeypatch):
    cancel_event = threading.Event()
    attempts = []

    def failed_request(spec, **kwargs):
        attempts.append(spec.name)
        cancel_event.set()
        raise ProviderError("unavailable", retryable=True)

    monkeypatch.setattr("emissary.llm.model.call_model", failed_request)
    monkeypatch.setattr("emissary.llm.retry.RETRY_DELAYS", (10.0, 30.0, 60.0))
    caller = FallbackModelCaller(parse_spec("anthropic"))

    result = run(_agent(), "go", caller=caller, cancel_event=cancel_event)

    assert result.status is RunStatus.CANCELLED
    assert result.stop_reason is StopReason.CANCELLED
    assert attempts == ["anthropic"]


async def test_async_cancellation_stops_pending_model_request():
    started = asyncio.Event()
    closed = asyncio.Event()

    async def request(spec, **kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            closed.set()

    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr("emissary.llm.model.acall_model", request)
        caller = AsyncFallbackModelCaller(parse_spec("anthropic"))
        task = asyncio.create_task(arun(_agent(), "go", caller=caller))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert closed.is_set()


def test_sdk_clients_do_not_add_hidden_retries(monkeypatch):
    from unittest.mock import MagicMock

    from emissary.llm.wire import anthropic, openai_compatible

    openai_client = MagicMock()
    anthropic_client = MagicMock()
    monkeypatch.setattr("openai.OpenAI", openai_client)
    monkeypatch.setattr("anthropic.Anthropic", anthropic_client)

    openai_compatible._client(parse_spec("kimi"))
    anthropic._create(__import__("anthropic"), {})

    assert openai_client.call_args.kwargs["max_retries"] == 0
    assert openai_client.call_args.kwargs["timeout"] == 30.0
    assert anthropic_client.call_args.kwargs["max_retries"] == 0
    assert anthropic_client.call_args.kwargs["timeout"] == 30.0


async def test_cancelled_gemini_stream_closes_its_generator(monkeypatch):
    from unittest.mock import MagicMock

    from emissary.llm.wire import gemini

    started = asyncio.Event()
    closed = asyncio.Event()

    async def chunks():
        try:
            started.set()
            await asyncio.Event().wait()
            yield MagicMock()
        finally:
            closed.set()

    client = MagicMock()

    async def open_stream(**kwargs):
        return chunks()

    client.aio.models.generate_content_stream = open_stream
    monkeypatch.setattr(gemini, "_client", lambda spec: client)

    class Sink:
        async def on_text(self, delta):
            pass

        async def on_thinking(self, delta):
            pass

    task = asyncio.create_task(gemini._astream(parse_spec("gemini"), {}, Sink()))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert closed.is_set()


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

    assert set(effects.__all__) == {
        "CallModel",
        "Effect",
        "ExecuteTool",
        "ValidateTool",
        "WaitRetry",
    }


def test_model_input_limit_rejects_before_a_billable_request():
    caller = ScriptedCaller([FinalOutput(text="unused")])
    agent = Agent("bounded", "Use tools.", limits=RunLimits(max_model_input_bytes=1))

    result = run(agent, "large task", caller=caller)

    assert result.stop_reason is StopReason.MODEL_INPUT_LIMIT
    assert caller.messages_seen == []


def test_oversized_tool_result_never_enters_the_next_prompt():
    tool = Tool("large", "Large.", {"type": "object"}, lambda: "x" * 1000, api_scope="none")
    caller = ScriptedCaller([ToolCalls((ToolCall("one", "large", {}),))])
    agent = Agent(
        "bounded", "Use tools.", tools=(tool,), limits=RunLimits(max_tool_result_bytes=100)
    )

    result = run(agent, "go", caller=caller)

    assert result.stop_reason is StopReason.TOOL_RESULT_LIMIT
    assert len(caller.messages_seen) == 1
    completed = [event for event in result.events if event.kind == "tool_call_completed"]
    assert len(completed) == 1
    assert "x" * 100 not in str(completed[0].data)


def test_failing_event_sink_cannot_erase_a_completed_tool_effect():
    effects = []
    tool = Tool(
        "write", "Write.", {"type": "object"}, lambda: effects.append(1) or "ok", api_scope="none"
    )
    caller = ScriptedCaller([ToolCalls((ToolCall("one", "write", {}),)), FinalOutput(text="done")])

    class BrokenSink:
        def emit(self, event):
            if event.kind == "tool_call_completed":
                raise RuntimeError("observer broke")

    result = run(Agent("a", "i", tools=(tool,)), "go", caller=caller, event_sink=BrokenSink())

    assert result.status is RunStatus.COMPLETED
    assert effects == [1]
    assert "tool_call_completed" in [event.kind for event in result.events]


def test_read_only_tools_must_classify_api_access():
    tool = Tool("lookup", "Lookup.", {"type": "object"}, dict)
    with pytest.raises(ValueError, match="api_scope"):
        run(Agent("a", "i", tools=(tool,)), "go", caller=ScriptedCaller([]))
