from dataclasses import dataclass, field

from emissary.harness.agent import Agent, RunLimits
from emissary.harness.events import InMemoryEventSink
from emissary.harness.runner import run
from emissary.harness.state import RunStatus, StopReason
from emissary.harness.tools import LocalToolExecutor, Tool, ToolResult
from emissary.llm.decision import FinalOutput, ModelResult, Refusal, ToolCall, ToolCalls, Usage
from emissary.llm.messages import AssistantMessage, TextBlock, ToolMessage, UserMessage


@dataclass
class ScriptedCaller:
    decisions: list
    messages_seen: list = field(default_factory=list)

    def __call__(self, *, system, messages, tools=(), settings=None):
        self.messages_seen.append(messages)
        decision = self.decisions.pop(0)
        return ModelResult(decision, "fake", "scripted", Usage(2, 1))


def test_runner_completes_after_a_tool_observation():
    caller = ScriptedCaller(
        [ToolCalls((ToolCall("one", "add", {"a": 2, "b": 3}),)), FinalOutput(text="5")]
    )
    agent = Agent(
        "calculator",
        "Use tools.",
        tools=(Tool("add", "Add.", {"type": "object"}, lambda a, b: {"sum": a + b}),),
    )
    sink = InMemoryEventSink()

    result = run(agent, "add", caller=caller, executor=LocalToolExecutor(), event_sink=sink)

    assert result.status is RunStatus.COMPLETED
    assert result.output == FinalOutput(text="5")
    assert result.usage == Usage(4, 2)
    assert any(isinstance(message, ToolMessage) for message in caller.messages_seen[1])
    assert [event.kind for event in sink.events][-1] == "run_completed"


def test_invalid_tool_batch_has_no_effects():
    effects = []
    caller = ScriptedCaller(
        [
            ToolCalls(
                (
                    ToolCall("one", "write", {"value": 1}),
                    ToolCall("two", "missing", {}),
                )
            )
        ]
    )
    agent = Agent(
        "writer",
        "Write.",
        tools=(
            Tool(
                "write",
                "Write.",
                {"type": "object"},
                lambda **kwargs: effects.append(kwargs),
            ),
        ),
    )

    result = run(agent, "write", caller=caller)

    assert result.status is RunStatus.FAILED
    assert result.stop_reason is StopReason.INVALID_TOOL
    assert effects == []


def test_tool_errors_are_observations_the_model_can_recover_from():
    caller = ScriptedCaller(
        [ToolCalls((ToolCall("one", "lookup", {}),)), FinalOutput(text="recovered")]
    )
    agent = Agent(
        "lookup",
        "Look up.",
        tools=(Tool("lookup", "Lookup.", {"type": "object"}, lambda: 1 / 0),),
    )

    result = run(agent, "lookup", caller=caller)

    assert result.status is RunStatus.COMPLETED
    assert '"status": "error"' in caller.messages_seen[1][-1].content


def test_turn_and_tool_limits_are_terminal_typed_results():
    call = ToolCalls((ToolCall("one", "noop", {}),))
    tool = Tool("noop", "Noop.", {"type": "object"}, dict)

    turn_limited = run(
        Agent("a", "i", tools=(tool,), limits=RunLimits(max_turns=1)),
        "task",
        caller=ScriptedCaller([call, FinalOutput(text="late")]),
    )
    assert turn_limited.stop_reason is StopReason.MAX_TURNS

    tool_limited = run(
        Agent("a", "i", tools=(tool,), limits=RunLimits(max_tool_calls=0)),
        "task",
        caller=ScriptedCaller([call]),
    )
    assert tool_limited.stop_reason is StopReason.MAX_TOOL_CALLS


def test_an_idempotent_tool_retries_under_the_same_key():
    """ADR-0012's named risk: the downstream must be able to recognise the
    second delivery as the same logical operation."""
    keys = []

    def flaky(*, idempotency_key):
        keys.append(idempotency_key)
        if len(keys) == 1:
            return ToolResult("error", "connection reset", retryable=True)
        return ToolResult("success", "fetched")

    tool = Tool("fetch", "Fetch.", {"type": "object"}, flaky, idempotent=True, max_attempts=2)
    caller = ScriptedCaller([ToolCalls((ToolCall("one", "fetch", {}),)), FinalOutput(text="done")])

    result = run(Agent("a", "i", tools=(tool,)), "task", caller=caller)

    assert result.status is RunStatus.COMPLETED
    assert len(keys) == 2 and keys[0] == keys[1]
    assert [event.kind for event in result.events].count("tool_call_retried") == 1


def test_a_tool_that_has_not_declared_idempotency_runs_exactly_once():
    attempts = []

    def unsafe():
        attempts.append(1)
        return ToolResult("error", "connection reset", retryable=True)

    tool = Tool("charge", "Charge.", {"type": "object"}, unsafe)
    caller = ScriptedCaller([ToolCalls((ToolCall("one", "charge", {}),)), FinalOutput(text="x")])

    run(Agent("a", "i", tools=(tool,)), "task", caller=caller)

    assert attempts == [1]


def test_a_flaky_tool_is_isolated_rather_than_consuming_the_whole_run():
    """Interleaved successes keep resetting the global consecutive-error count,
    so without a per-tool circuit the run would keep calling a dead tool until
    it ran out of turns."""
    executed = []

    def flaky():
        executed.append(1)
        return ToolResult("error", "upstream down")

    tools = (
        Tool("flaky", "Flaky.", {"type": "object"}, flaky),
        Tool("ok", "Fine.", {"type": "object"}, dict),
    )
    decisions = []
    for n in range(3):
        decisions.append(ToolCalls((ToolCall(f"f{n}", "flaky", {}),)))
        decisions.append(ToolCalls((ToolCall(f"o{n}", "ok", {}),)))
    decisions.append(FinalOutput(text="done"))

    result = run(
        Agent("a", "i", tools=tools, limits=RunLimits(max_tool_failures=2)),
        "task",
        caller=ScriptedCaller(decisions),
    )

    assert result.status is RunStatus.COMPLETED
    assert executed == [1, 1]  # the third call never reached the tool
    opened = [event for event in result.events if event.kind == "tool_circuit_opened"]
    assert [event.data["tool"] for event in opened] == ["flaky"]
    blocked = next(
        message
        for message in result.messages
        if isinstance(message, ToolMessage) and message.call_id == "f2"
    )
    assert "unavailable" in blocked.content


def test_the_first_model_call_sees_exactly_the_task():
    """Invariant 1 of ADR-0011 at the boundary: nothing reaches the model that
    was not logged first."""
    caller = ScriptedCaller([FinalOutput(text="done")])

    run(Agent("a", "i"), "the task", caller=caller)

    assert caller.messages_seen[0] == (UserMessage((TextBlock("the task"),)),)


def test_a_rejected_batch_still_records_what_the_model_proposed():
    """The assistant message projects from the model decision, before batch
    validation — so calls that never ran are still in the surface. That is
    factually what happened, and what a resumed run needs."""
    calls = (ToolCall("one", "write", {}), ToolCall("two", "missing", {}))
    tool = Tool("write", "Write.", {"type": "object"}, dict)

    result = run(
        Agent("writer", "Write.", tools=(tool,)),
        "write",
        caller=ScriptedCaller([ToolCalls(calls)]),
    )

    assert result.stop_reason is StopReason.INVALID_TOOL
    assert result.messages[-1] == AssistantMessage(tool_calls=calls)


def test_refusal_is_not_reported_as_success():
    result = run(Agent("a", "i"), "task", caller=ScriptedCaller([Refusal("policy")]))

    assert result.status is RunStatus.REFUSED
    assert result.stop_reason is StopReason.REFUSAL
