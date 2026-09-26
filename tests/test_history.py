import pytest

from emissary.harness.agent import Agent
from emissary.harness.events import InMemoryEventSink
from emissary.harness.projection import derive_messages
from emissary.harness.runner import arun, run
from emissary.harness.state import RunStatus
from emissary.llm.decision import FinalOutput, ModelResult, ToolCall, Usage
from emissary.llm.messages import AssistantMessage, TextBlock, ToolMessage, UserMessage

AGENT = Agent("tutor", "Teach gently.")

EARLIER = (
    UserMessage((TextBlock("Tell me about the moon."),)),
    AssistantMessage(text="It seems to change shape because we see it lit from different sides."),
)


class ScriptedCaller:
    def __init__(self, *decisions):
        self.decisions = list(decisions)
        self.messages_seen = []

    def __call__(self, *, system, messages, tools=(), settings=None):
        self.messages_seen.append(messages)
        return ModelResult(self.decisions.pop(0), "fake", "scripted", Usage(1, 1))


class AsyncScriptedCaller(ScriptedCaller):
    async def __call__(self, *, system, messages, tools=(), settings=None):
        return super().__call__(system=system, messages=messages, tools=tools, settings=settings)


def test_a_seeded_conversation_reaches_the_model_before_the_new_task():
    caller = ScriptedCaller(FinalOutput(text="Yes, the same moon!"))

    run(AGENT, "Is it the same moon every night?", caller=caller, history=EARLIER)

    first_call = caller.messages_seen[0]
    assert first_call[:2] == EARLIER
    assert first_call[2] == UserMessage((TextBlock("Is it the same moon every night?"),))


def test_history_is_part_of_the_log_so_replay_reproduces_what_the_model_saw():
    caller = ScriptedCaller(FinalOutput(text="Yes."))
    sink = InMemoryEventSink()

    run(AGENT, "Same moon?", caller=caller, history=EARLIER, event_sink=sink)

    assert derive_messages(sink.events)[:2] == EARLIER


async def test_async_runs_accept_history_too():
    caller = AsyncScriptedCaller(FinalOutput(text="Yes."))

    result = await arun(AGENT, "Same moon?", caller=caller, history=EARLIER)

    assert result.status is RunStatus.COMPLETED
    assert caller.messages_seen[0][:2] == EARLIER


def test_history_that_stops_mid_tool_exchange_is_rejected():
    """A provider refuses a transcript whose tool call never got its result."""
    unfinished = (
        UserMessage((TextBlock("Count the stars."),)),
        AssistantMessage(tool_calls=(ToolCall("c1", "count", {}),)),
    )

    with pytest.raises(ValueError, match="complete exchange"):
        run(AGENT, "Next?", caller=ScriptedCaller(), history=unfinished)


def test_a_tool_call_left_unanswered_earlier_in_history_is_rejected():
    abandoned = (
        UserMessage((TextBlock("Count the stars."),)),
        AssistantMessage(tool_calls=(ToolCall("c1", "count", {}),)),
        AssistantMessage(text="Never mind, let's sing instead."),
    )

    with pytest.raises(ValueError, match="never answered"):
        run(AGENT, "Next?", caller=ScriptedCaller(), history=abandoned)


def test_history_must_open_with_the_user():
    with pytest.raises(ValueError, match="user message"):
        run(AGENT, "Next?", caller=ScriptedCaller(), history=EARLIER[1:])


def test_a_tool_result_without_its_call_is_rejected():
    orphaned = (
        UserMessage((TextBlock("Hi"),)),
        ToolMessage("missing", "count", "{}"),
        AssistantMessage(text="Hello"),
    )

    with pytest.raises(ValueError, match="orphaned"):
        run(AGENT, "Next?", caller=ScriptedCaller(), history=orphaned)


def test_a_run_without_history_logs_no_history_event():
    sink = InMemoryEventSink()

    run(AGENT, "Hi", caller=ScriptedCaller(FinalOutput(text="Hello")), event_sink=sink)

    assert "history_loaded" not in [event.kind for event in sink.events]
