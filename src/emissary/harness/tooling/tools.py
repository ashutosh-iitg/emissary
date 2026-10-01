"""Tool contracts and the default local execution boundary."""

import hashlib
import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

from jsonschema import Draft202012Validator, SchemaError, ValidationError
from jsonschema.validators import validator_for

from ...llm.decision import ToolCall, ToolDefinition

if TYPE_CHECKING:
    from ..policy import AuthorizationDecision


@dataclass(frozen=True)
class ToolResult:
    """`status` is the severity axis and only that.

    Facts that are not severities get their own field, because they can co-occur
    with any status: a call can time out and still have succeeded partially, and
    retryability is a property of the failure, not of how bad it was (ADR-0013).
    """

    status: Literal["success", "warning", "error"]
    summary: str
    content: Any = None
    artifacts: tuple[str, ...] = ()
    timed_out: bool = False
    retryable: bool = False

    def __post_init__(self) -> None:
        if self.status not in ("success", "warning", "error"):
            raise ValueError("invalid tool result status")
        if not self.summary:
            raise ValueError("tool result summary must not be empty")


@dataclass(frozen=True)
class ToolContext:
    """What one attempt of one call knows about the run it belongs to."""

    run_id: str
    attempt: int = 1
    authorization: "AuthorizationDecision | None" = None

    def __post_init__(self) -> None:
        if self.attempt < 1:
            raise ValueError("attempt must be positive")


def idempotency_key(run_id: str, call_id: str) -> str:
    """Identical across every attempt of one call — that is the whole point.

    Deriving it from the run and call rather than the attempt is what lets a
    downstream recognise the retry as the same logical operation (ADR-0012).
    """
    return f"{run_id}:{call_id}"


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    execute: Callable[..., Any]
    output_schema: dict[str, Any] | None = None
    side_effect: Literal["none", "local", "external"] = "none"
    approval: Literal["never", "always", "policy"] = "never"
    idempotent: bool = False
    max_attempts: int = 1
    api_scope: Literal["none", "internal", "external"] | None = None
    retry_backoff_seconds: float = 0.25
    strict_schema: bool | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("tool name must not be empty")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if self.max_attempts > 1 and not self.idempotent:
            raise ValueError(f"tool {self.name!r} cannot retry without declaring itself idempotent")
        if self.api_scope not in (None, "none", "internal", "external"):
            raise ValueError(f"tool {self.name!r} has invalid api_scope")
        if self.side_effect == "external" and self.api_scope == "none":
            raise ValueError(f"tool {self.name!r} cannot classify external effects as no API")
        if not 0 <= self.retry_backoff_seconds < float("inf"):
            raise ValueError(
                f"tool {self.name!r} retry_backoff_seconds must be finite and non-negative"
            )
        try:
            validator_for(self.input_schema, Draft202012Validator).check_schema(self.input_schema)
            if self.output_schema is not None:
                validator_for(self.output_schema, Draft202012Validator).check_schema(
                    self.output_schema
                )
        except SchemaError as exc:
            raise ValueError(f"invalid JSON schema for tool {self.name!r}: {exc.message}") from exc

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            self.name,
            self.description,
            self.input_schema,
            output_schema=self.output_schema,
            strict=self.strict_schema,
        )

    @property
    def effective_api_scope(self) -> Literal["none", "internal", "external"]:
        """Infer from a declared side effect or require explicit read-only classification."""
        if self.api_scope is not None:
            return self.api_scope
        if self.side_effect == "external":
            return "external"
        if self.side_effect == "local":
            return "none"
        raise ValueError(f"tool {self.name!r} must declare api_scope when side_effect is none")

    @property
    def fingerprint(self) -> str:
        contract = {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "side_effect": self.side_effect,
            "approval": self.approval,
            "api_scope": self.api_scope,
        }
        encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


class ToolExecutor(Protocol):
    def validate(self, call: ToolCall, tool: Tool) -> ToolResult | None: ...

    def execute(self, call: ToolCall, tool: Tool, context: ToolContext) -> ToolResult: ...


class LocalToolExecutor:
    """Validate and invoke in-process tools without exposing exceptions.

    One attempt per call. Whether to attempt again is run policy and lives in
    the runner, which owns budgets and the event log.
    """

    def validate(self, call: ToolCall, tool: Tool) -> ToolResult | None:
        try:
            schema_validator(tool.input_schema).validate(call.arguments)
        except ValidationError as exc:
            return ToolResult("error", f"invalid input for {tool.name}: {exc.message}")
        # A schema that cannot be evaluated (e.g. an unresolvable reference) must
        # end as a controlled rejection, not an exception out of the driver.
        except Exception:  # noqa: BLE001
            return ToolResult("error", f"schema for {tool.name} could not be evaluated")
        return None

    def execute(self, call: ToolCall, tool: Tool, context: ToolContext) -> ToolResult:
        # Re-validated even though the runner validates whole batches first:
        # this is a trust boundary, and it must hold for any caller.
        invalid = self.validate(call, tool)
        if invalid is not None:
            return invalid

        arguments = direct_arguments(call, tool, context)

        try:
            output = tool.execute(**arguments)
        # Tool code is an untrusted extension boundary. Its exception types are
        # unknowable, and none may escape into model context with secret details.
        except Exception:  # noqa: BLE001
            return ToolResult("error", f"{tool.name} failed")

        return normalize_output(tool, output)


def schema_validator(schema: dict[str, Any]):
    """Validate in the dialect the schema declares; 2020-12 when it declares none."""
    return validator_for(schema, Draft202012Validator)(schema)


def direct_arguments(call: ToolCall, tool: Tool, context: ToolContext) -> dict[str, Any]:
    """Model arguments plus the trusted idempotency key a direct tool declared it wants."""
    arguments = dict(call.arguments)
    if tool.idempotent:
        arguments["idempotency_key"] = idempotency_key(context.run_id, call.id)
    return arguments


def normalize_output(tool: Tool, output: Any) -> ToolResult:
    if isinstance(output, ToolResult):
        return output
    if tool.output_schema is not None:
        try:
            schema_validator(tool.output_schema).validate(output)
        except ValidationError as exc:
            return ToolResult("error", f"invalid output from {tool.name}: {exc.message}")
        except Exception:  # noqa: BLE001
            return ToolResult("error", f"output schema for {tool.name} could not be evaluated")
    return ToolResult("success", f"{tool.name} completed", output)


class AsyncToolExecutor(Protocol):
    def validate(self, call: ToolCall, tool: Tool) -> ToolResult | None: ...

    async def execute(self, call: ToolCall, tool: Tool, context: ToolContext) -> ToolResult: ...


class AsyncLocalToolExecutor:
    """`LocalToolExecutor` for callables that may be coroutine functions.

    A synchronous callable runs inline on the event loop: moving it to a thread
    would change thread affinity and cancellation semantics, so blocking tools
    need an executor the caller chose deliberately.

    `inject_idempotency_key` is off for prepared source bindings, which must
    never receive an argument their remote schema did not declare.
    """

    def __init__(self, *, inject_idempotency_key: bool = True) -> None:
        self._local = LocalToolExecutor()
        self._inject = inject_idempotency_key

    def validate(self, call: ToolCall, tool: Tool) -> ToolResult | None:
        return self._local.validate(call, tool)

    async def execute(self, call: ToolCall, tool: Tool, context: ToolContext) -> ToolResult:
        invalid = self.validate(call, tool)
        if invalid is not None:
            return invalid

        arguments = direct_arguments(call, tool, context) if self._inject else dict(call.arguments)
        try:
            output = tool.execute(**arguments)
            if inspect.isawaitable(output):
                output = await output
        except Exception:  # noqa: BLE001 - same untrusted boundary as the sync executor
            return ToolResult("error", f"{tool.name} failed")
        return normalize_output(tool, output)


class ToolRegistry:
    def __init__(self, tools: tuple[Tool, ...]):
        # Raises for a read-only tool with no declared API access, so the gap
        # fails registration rather than the first attempt mid-run.
        for tool in tools:
            _scope = tool.effective_api_scope
        by_name = {tool.name: tool for tool in tools}
        if len(by_name) != len(tools):
            raise ValueError("duplicate tool names are not allowed")
        self._by_name = by_name
        self._tools = tools

    def resolve(self, name: str) -> Tool:
        try:
            return self._by_name[name]
        except KeyError as exc:
            raise KeyError(f"unknown tool {name!r}") from exc

    @property
    def definitions(self) -> tuple[ToolDefinition, ...]:
        return tuple(tool.definition for tool in self._tools)


__all__ = [
    "AsyncLocalToolExecutor",
    "AsyncToolExecutor",
    "LocalToolExecutor",
    "Tool",
    "ToolContext",
    "ToolExecutor",
    "ToolRegistry",
    "ToolResult",
    "idempotency_key",
    "schema_validator",
]
