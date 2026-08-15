"""The event log is the only source of conversation state (ADR-0011).

Every assertion here pins an exact projected surface. A projection bug changes
what the model sees without failing any behavioural test, so these are golden.
"""

import json

import pytest

from emissary.harness.context import ContextOp
from emissary.harness.events import new_event
from emissary.harness.projection import (
    context_op_data,
    derive_messages,
    message_from_data,
    message_to_data,
    model_result_data,
    tool_result_data,
    user_message_data,
)
from emissary.harness.tools import ToolResult
from emissary.llm.decision import (
    FinalOutput,
    ModelResult,
    Refusal,
    ToolCall,
    ToolCalls,
    Usage,
)
from emissary.llm.messages import AssistantMessage, TextBlock, ToolMessage, UserMessage

CALL = ToolCall("one", "add", {"a": 2, "b": 3})
OUTCOME = ToolResult("success", "add completed", {"sum": 5})


def event(sequence, kind, **data):
    return new_event("run", sequence, kind, **data)


def user_event(sequence=1, text="add 2 and 3"):
    return event(sequence, "user_message", **user_message_data(UserMessage((TextBlock(text),))))


def model_event(sequence, decision):
    result = ModelResult(decision, "fake", "fake-model", Usage(2, 1), finish_reason="stop")
    return event(sequence, "model_call_completed", **model_result_data(result))


def tool_event(sequence, call=CALL, outcome=OUTCOME):
    return event(sequence, "tool_call_completed", **tool_result_data(call, outcome))


def test_an_empty_log_projects_no_messages():
    assert derive_messages(()) == ()


def test_lifecycle_events_contribute_nothing_to_the_surface():
    events = (
        event(1, "run_started", agent="a"),
        user_event(2),
        event(3, "model_call_started", turn=1),
        event(4, "approval_resolved", call_id="one", decision="allow"),
        event(5, "tool_call_started", call_id="one", tool="add"),
    )

    assert derive_messages(events) == (UserMessage((TextBlock("add 2 and 3"),)),)


def test_a_full_tool_turn_projects_every_message_including_the_final_text():
    """The trailing AssistantMessage is the bug this ADR fixes: the runner
    previously recorded no assistant text at all."""
    events = (
        user_event(1),
        model_event(2, ToolCalls((CALL,))),
        tool_event(3),
        model_event(4, FinalOutput(text="5")),
    )

    assert derive_messages(events) == (
        UserMessage((TextBlock("add 2 and 3"),)),
        AssistantMessage(tool_calls=(CALL,)),
        ToolMessage(
            call_id="one",
            tool_name="add",
            content=json.dumps(
                {
                    "artifacts": [],
                    "content": {"sum": 5},
                    "retryable": False,
                    "status": "success",
                    "summary": "add completed",
                    "timed_out": False,
                },
                sort_keys=True,
            ),
        ),
        AssistantMessage(text="5"),
    )


def test_a_refusal_projects_nothing_because_its_reason_is_harness_authored():
    events = (user_event(1), model_event(2, Refusal("the model declined this request")))

    assert derive_messages(events) == (UserMessage((TextBlock("add 2 and 3"),)),)


def test_structured_only_output_projects_no_assistant_message():
    """`value` is structured output; AssistantMessage cannot hold it without
    inventing a rendering."""
    events = (user_event(1), model_event(2, FinalOutput(value={"sum": 5})))

    assert derive_messages(events) == (UserMessage((TextBlock("add 2 and 3"),)),)


def test_a_compaction_replaces_its_range_and_leaves_later_events_alone():
    op = ContextOp(1, 3, "recent_history:2")
    events = (
        user_event(1),
        model_event(2, ToolCalls((CALL,))),
        tool_event(3),
        event(4, "context_compacted", **context_op_data(op)),
        model_event(5, FinalOutput(text="5")),
    )

    assert derive_messages(events) == (
        UserMessage((TextBlock("add 2 and 3"),)),
        AssistantMessage(text="5"),
    )


def test_a_compaction_can_substitute_a_summary_for_the_range_it_replaces():
    summary = UserMessage((TextBlock("[earlier turns summarised: computed 2 + 3]"),))
    op = ContextOp(1, 3, "summarised", replacement=(summary,))
    events = (
        user_event(1),
        model_event(2, ToolCalls((CALL,))),
        tool_event(3),
        event(4, "context_compacted", **context_op_data(op)),
    )

    assert derive_messages(events) == (UserMessage((TextBlock("add 2 and 3"),)), summary)


def test_pre_compaction_history_stays_recoverable_for_audit():
    op = ContextOp(1, 3, "recent_history:2")
    events = (
        user_event(1),
        model_event(2, ToolCalls((CALL,))),
        tool_event(3),
        event(4, "context_compacted", **context_op_data(op)),
    )

    assert len(derive_messages(events, apply_ops=False)) == 3
    assert len(derive_messages(events)) == 1


def test_an_out_of_range_compaction_fails_loud():
    events = (
        user_event(1),
        event(2, "context_compacted", **context_op_data(ContextOp(1, 9, "bogus"))),
    )

    with pytest.raises(ValueError, match="range"):
        derive_messages(events)


def test_a_compaction_that_orphans_a_tool_result_fails_loud():
    """Blueprint Step 8's exit criterion, checked once for every policy rather
    than trusted to each one."""
    events = (
        user_event(1),
        model_event(2, ToolCalls((CALL,))),
        tool_event(3),
        event(4, "context_compacted", **context_op_data(ContextOp(1, 2, "drops the call"))),
    )

    with pytest.raises(ValueError, match="orphan"):
        derive_messages(events)


@pytest.mark.parametrize(
    "message",
    [
        UserMessage((TextBlock("plain"), TextBlock("cached", cache=True))),
        AssistantMessage(text="thinking"),
        AssistantMessage(tool_calls=(CALL,)),
        ToolMessage("one", "add", "result"),
    ],
)
def test_every_message_variant_round_trips_through_the_codec(message):
    assert message_from_data(message_to_data(message)) == message


def test_tool_results_survive_a_json_round_trip_byte_identically():
    """A tuple in ToolResult.content renders identically but is a different
    object after persistence; normalising at emit keeps projections equal."""
    outcome = ToolResult("success", "listed", {"items": ("a", "b")})
    data = tool_result_data(CALL, outcome)

    assert data == json.loads(json.dumps(data))
