"""What the loop needs done, described rather than performed (ADR-0024).

The runner's policy is not I/O; four things inside it are. Naming them as
values lets one loop serve a synchronous and an asynchronous driver, instead
of the policy being copied once per concurrency model.

Each effect is a request. The driver performs it and sends the outcome back,
or throws in `RunCancelled` / `RunTimedOut` instead of performing it:

| effect         | outcome the driver sends back |
|----------------|-------------------------------|
| `CallModel`    | `ModelResult`, or a `ProviderError` thrown in |
| `ValidateTool` | `ToolResult` when the call is rejected, else `None` |
| `AuthorizeTool`| `AuthorizationDecision`, or `AuthorizerFailed` thrown in |
| `ExecuteTool`  | `ToolResult` |
| `WaitRetry`    | `None`, once the delay has elapsed |

Deliberately not effects: event emission, which is synchronous and so costs an
async driver nothing, and approval, which already has a designed asynchronous
path in `PAUSE`. Both are recorded in ADR-0024 as additive if that changes.
"""

from dataclasses import dataclass

from ...llm.decision import ModelSettings, ToolCall, ToolDefinition
from ...llm.messages import Message
from ..policy import InvocationRequest
from ..tooling.tools import Tool, ToolContext


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
class AuthorizeTool:
    """Ask the application's policy about one exact request.

    Yielded for a whole batch before any execution, and again immediately
    before every attempt. The outcome is an `AuthorizationDecision`; a policy
    that raises is thrown in as `AuthorizerFailed`.
    """

    request: InvocationRequest


@dataclass(frozen=True)
class ExecuteTool:
    """One attempt at one call. `context` carries the attempt number and the
    idempotency key, so a retry is distinguishable from a first try.

    `request` is the authorized snapshot: a driver executes what it describes,
    not `call`, whose arguments a callback may have mutated.
    """

    call: ToolCall
    tool: Tool
    context: ToolContext
    request: InvocationRequest | None = None


@dataclass(frozen=True)
class WaitRetry:
    """Delay before another attempt at an idempotent tool call."""

    seconds: float


Effect = CallModel | ValidateTool | AuthorizeTool | ExecuteTool | WaitRetry

__all__ = ["AuthorizeTool", "CallModel", "Effect", "ExecuteTool", "ValidateTool", "WaitRetry"]
