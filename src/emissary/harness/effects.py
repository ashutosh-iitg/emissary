"""What the loop needs done, described rather than performed (ADR-0024).

The runner's policy is not I/O; three things inside it are. Naming those three
as values lets one loop serve a synchronous and an asynchronous driver, instead
of the policy being copied once per concurrency model.

Each effect is a request. The driver performs it and sends the outcome back:

| effect         | outcome the driver sends back |
|----------------|-------------------------------|
| `CallModel`    | `ModelResult`, or a `ProviderError` thrown in |
| `ValidateTool` | `ToolResult` when the call is rejected, else `None` |
| `ExecuteTool`  | `ToolResult` |

Deliberately not effects: event emission, which is synchronous and so costs an
async driver nothing, and approval, which already has a designed asynchronous
path in `PAUSE`. Both are recorded in ADR-0024 as additive if that changes.
"""

from dataclasses import dataclass

from ..llm.decision import ModelSettings, ToolCall, ToolDefinition
from ..llm.messages import Message
from .tools import Tool, ToolContext


@dataclass(frozen=True)
class CallModel:
    """One model turn. The machine has already applied any context operations,
    so `messages` is exactly the surface the model should see."""

    system: str
    messages: tuple[Message, ...]
    tools: tuple[ToolDefinition, ...]
    settings: ModelSettings | None


@dataclass(frozen=True)
class ValidateTool:
    """Admission check for one call, yielded for the whole batch before any
    `ExecuteTool` — a driver cannot restore that ordering if it is lost."""

    call: ToolCall
    tool: Tool


@dataclass(frozen=True)
class ExecuteTool:
    """One attempt at one call. `context` carries the attempt number and the
    idempotency key, so a retry is distinguishable from a first try."""

    call: ToolCall
    tool: Tool
    context: ToolContext


Effect = CallModel | ValidateTool | ExecuteTool

__all__ = ["CallModel", "Effect", "ExecuteTool", "ValidateTool"]
