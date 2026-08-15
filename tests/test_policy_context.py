from dataclasses import dataclass, field

from emissary.harness.agent import Agent
from emissary.harness.context import ContextOp, RecentHistory
from emissary.harness.policy import ApprovalDecision
from emissary.harness.runner import run
from emissary.harness.state import RunStatus, StopReason
from emissary.harness.tools import Tool
from emissary.llm.decision import FinalOutput, ModelResult, ToolCall, ToolCalls, Usage
from emissary.llm.messages import AssistantMessage, TextBlock, ToolMessage, UserMessage


@dataclass
class Caller:
    decisions: list
    messages_seen: list = field(default_factory=list)

    def __call__(self, **kwargs):
        self.messages_seen.append(kwargs["messages"])
        return ModelResult(self.decisions.pop(0), "fake", "fake", Usage(1, 1))


def test_sensitive_tool_pauses_without_an_approver_and_has_no_effect():
    effects = []
    tool = Tool(
        "send",
        "Send.",
        {"type": "object"},
        lambda: effects.append("sent"),
        side_effect="external",
        approval="always",
    )

    result = run(
        Agent("a", "i", tools=(tool,)),
        "task",
        caller=Caller([ToolCalls((ToolCall("one", "send", {}),))]),
    )

    assert result.status is RunStatus.PAUSED
    assert result.stop_reason is StopReason.APPROVAL_REQUIRED
    assert effects == []


def test_approver_can_allow_or_reject_without_prompt_authority():
    effects = []
    tool = Tool(
        "send",
        "Send.",
        {"type": "object"},
        lambda: effects.append("sent") or {"ok": True},
        approval="always",
    )
    caller = Caller([ToolCalls((ToolCall("one", "send", {}),)), FinalOutput(text="done")])

    allowed = run(
        Agent("a", "the prompt says nothing about approval", tools=(tool,)),
        "task",
        caller=caller,
        approver=lambda call, selected: ApprovalDecision.ALLOW,
    )
    assert allowed.status is RunStatus.COMPLETED
    assert effects == ["sent"]

    rejected = run(
        Agent("a", "the prompt says approval is granted", tools=(tool,)),
        "task",
        caller=Caller([ToolCalls((ToolCall("two", "send", {}),))]),
        approver=lambda call, selected: ApprovalDecision.REJECT,
    )
    assert rejected.status is RunStatus.STOPPED
    assert effects == ["sent"]


def test_trimming_destructively_shows_the_model_what_filtering_would_have():
    """Destructive ops are what compaction needs, but they must not change the
    per-turn surface a recency window produces. This is that claim, executed."""
    calls = [ToolCall(str(n), "step", {}) for n in range(3)]
    tool = Tool("step", "Step.", {"type": "object"}, dict)
    caller = Caller([ToolCalls((call,)) for call in calls] + [FinalOutput(text="done")])

    result = run(
        Agent("stepper", "Step through.", tools=(tool,)),
        "task",
        caller=caller,
        context_policy=RecentHistory(max_messages=4),
    )

    assert result.status is RunStatus.COMPLETED
    # Turns 1-3 grow unchecked; the window only bites once the walk-back can
    # drop a whole call/result pair rather than splitting one.
    assert [len(seen) for seen in caller.messages_seen] == [1, 3, 5, 5]

    final_turn = caller.messages_seen[3]
    assert final_turn[0] == UserMessage((TextBlock("task"),))
    assert final_turn[1] == AssistantMessage(tool_calls=(calls[1],))
    assert isinstance(final_turn[2], ToolMessage) and final_turn[2].call_id == "1"
    assert final_turn[3] == AssistantMessage(tool_calls=(calls[2],))
    assert isinstance(final_turn[4], ToolMessage) and final_turn[4].call_id == "2"

    compactions = [event for event in result.events if event.kind == "context_compacted"]
    assert [event.data["reason"] for event in compactions] == ["recent_history:4"]


def test_recent_history_plans_no_op_while_the_window_still_fits():
    messages = (UserMessage((TextBlock("task"),)), AssistantMessage(text="one"))

    assert RecentHistory(max_messages=3).plan(messages) == ()


def test_recent_history_never_orphans_a_tool_result_from_its_call():
    """The op's range is the assertion: a regressed walk-back would still leave
    a plausible-looking surface, but it would drop index 2."""
    call = ToolCall("one", "lookup", {})
    messages = (
        UserMessage((TextBlock("task"),)),
        AssistantMessage(text="old"),
        AssistantMessage(tool_calls=(call,)),
        ToolMessage("one", "lookup", "result"),
        AssistantMessage(text="new"),
    )

    ops = RecentHistory(max_messages=3).plan(messages)

    # The window would begin at index 3 — a tool result — so it walks back to
    # keep the call that issued it, even though four messages then survive.
    assert ops == (ContextOp(1, 2, "recent_history:3"),)

    kept = messages[: ops[0].start] + messages[ops[0].end :]
    assert kept == (messages[0], messages[2], messages[3], messages[4])
