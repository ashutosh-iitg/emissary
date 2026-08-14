from dataclasses import dataclass

from emissary.harness.agent import Agent
from emissary.harness.context import RecentHistory
from emissary.harness.policy import ApprovalDecision
from emissary.harness.runner import run
from emissary.harness.state import RunStatus, StopReason
from emissary.harness.tools import Tool
from emissary.llm.decision import FinalOutput, ModelResult, ToolCall, ToolCalls, Usage
from emissary.llm.messages import AssistantMessage, TextBlock, ToolMessage, UserMessage


@dataclass
class Caller:
    decisions: list

    def __call__(self, **kwargs):
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


def test_recent_history_never_orphans_a_tool_result_from_its_call():
    call = ToolCall("one", "lookup", {})
    messages = (
        UserMessage((TextBlock("task"),)),
        AssistantMessage(text="old"),
        AssistantMessage(tool_calls=(call,)),
        ToolMessage("one", "lookup", "result"),
        AssistantMessage(text="new"),
    )

    selected = RecentHistory(max_messages=3).select(messages)

    assert selected[0] == messages[0]
    assert messages[2] in selected
    assert messages[3] in selected
