from dataclasses import dataclass, field

from emissary.harness.agent import Agent, RunLimits
from emissary.harness.events import InMemoryEventSink
from emissary.harness.runner import run
from emissary.harness.state import RunStatus, StopReason
from emissary.harness.tools import LocalToolExecutor, Tool
from emissary.llm.decision import FinalOutput, ModelResult, Refusal, ToolCall, ToolCalls, Usage
from emissary.llm.messages import ToolMessage


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


def test_refusal_is_not_reported_as_success():
    result = run(Agent("a", "i"), "task", caller=ScriptedCaller([Refusal("policy")]))

    assert result.status is RunStatus.REFUSED
    assert result.stop_reason is StopReason.REFUSAL
