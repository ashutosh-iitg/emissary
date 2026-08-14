"""Approval decisions kept outside prompts and tool implementations."""

from enum import Enum
from typing import Protocol

from ..llm.decision import ToolCall
from .tools import Tool


class ApprovalDecision(str, Enum):
    ALLOW = "allow"
    REJECT = "reject"
    PAUSE = "pause"


class Approver(Protocol):
    def __call__(self, call: ToolCall, tool: Tool) -> ApprovalDecision: ...


def approval_for(call: ToolCall, tool: Tool, approver: Approver | None) -> ApprovalDecision:
    if tool.approval == "never":
        return ApprovalDecision.ALLOW
    if approver is None:
        return ApprovalDecision.PAUSE
    return approver(call, tool)


__all__ = ["ApprovalDecision", "Approver"]
