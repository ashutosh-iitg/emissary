"""Tool contracts and the default local execution boundary."""

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from jsonschema import Draft202012Validator, SchemaError, ValidationError

from .decision import ToolCall, ToolDefinition


@dataclass(frozen=True)
class ToolResult:
    status: Literal["success", "warning", "error"]
    summary: str
    content: Any = None
    artifacts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in ("success", "warning", "error"):
            raise ValueError("invalid tool result status")
        if not self.summary:
            raise ValueError("tool result summary must not be empty")


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    execute: Callable[..., Any]
    output_schema: dict[str, Any] | None = None
    side_effect: Literal["none", "local", "external"] = "none"
    approval: Literal["never", "always", "policy"] = "never"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("tool name must not be empty")
        try:
            Draft202012Validator.check_schema(self.input_schema)
            if self.output_schema is not None:
                Draft202012Validator.check_schema(self.output_schema)
        except SchemaError as exc:
            raise ValueError(f"invalid JSON schema for tool {self.name!r}: {exc.message}") from exc

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            self.name, self.description, self.input_schema, output_schema=self.output_schema
        )

    @property
    def fingerprint(self) -> str:
        contract = {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "side_effect": self.side_effect,
            "approval": self.approval,
        }
        encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


class ToolExecutor(Protocol):
    def validate(self, call: ToolCall, tool: Tool) -> ToolResult | None: ...

    def execute(self, call: ToolCall, tool: Tool) -> ToolResult: ...


class LocalToolExecutor:
    """Validate and invoke in-process tools without exposing exceptions."""

    def validate(self, call: ToolCall, tool: Tool) -> ToolResult | None:
        try:
            Draft202012Validator(tool.input_schema).validate(call.arguments)
        except ValidationError as exc:
            return ToolResult("error", f"invalid input for {tool.name}: {exc.message}")
        return None

    def execute(self, call: ToolCall, tool: Tool) -> ToolResult:
        invalid = self.validate(call, tool)
        if invalid is not None:
            return invalid

        try:
            output = tool.execute(**call.arguments)
        # Tool code is an untrusted extension boundary. Its exception types are
        # unknowable, and none may escape into model context with secret details.
        except Exception:  # noqa: BLE001
            return ToolResult("error", f"{tool.name} failed")

        if isinstance(output, ToolResult):
            return output
        if tool.output_schema is not None:
            try:
                Draft202012Validator(tool.output_schema).validate(output)
            except ValidationError as exc:
                return ToolResult("error", f"invalid output from {tool.name}: {exc.message}")
        return ToolResult("success", f"{tool.name} completed", output)


class ToolRegistry:
    def __init__(self, tools: tuple[Tool, ...]):
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


__all__ = ["LocalToolExecutor", "Tool", "ToolExecutor", "ToolRegistry", "ToolResult"]
