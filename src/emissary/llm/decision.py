"""Provider-neutral model capabilities, decisions, provenance, and usage."""

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("tool name must not be empty")
        if self.input_schema.get("type") != "object":
            raise ValueError("tool input_schema must be an object schema")


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("tool call id must not be empty")
        if not self.name:
            raise ValueError("tool call name must not be empty")
        if not isinstance(self.arguments, dict):
            raise TypeError("tool call arguments must be an object")


@dataclass(frozen=True)
class FinalOutput:
    text: str | None = None
    value: Any | None = None

    def __post_init__(self) -> None:
        if self.text is None and self.value is None:
            raise ValueError("final output needs text or value")


@dataclass(frozen=True)
class ToolCalls:
    calls: tuple[ToolCall, ...]

    def __post_init__(self) -> None:
        if not self.calls:
            raise ValueError("tool decision needs at least one call")
        ids = [call.id for call in self.calls]
        if len(ids) != len(set(ids)):
            raise ValueError("tool call ids must be unique")


@dataclass(frozen=True)
class Refusal:
    reason: str

    def __post_init__(self) -> None:
        if not self.reason:
            raise ValueError("refusal reason must not be empty")


ModelDecision = FinalOutput | ToolCalls | Refusal


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0

    def __post_init__(self) -> None:
        if min(self.input_tokens, self.output_tokens, self.cached_input_tokens) < 0:
            raise ValueError("token counts must be non-negative")

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class ModelCapabilities:
    tool_calling: bool = False
    parallel_tool_calls: bool = False
    structured_output: bool = False
    logprobs: bool = False


@dataclass(frozen=True)
class ModelSettings:
    max_output_tokens: int | None = None
    tool_choice: Literal["auto", "required", "none"] = "auto"

    def __post_init__(self) -> None:
        if self.max_output_tokens is not None and self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if self.tool_choice not in ("auto", "required", "none"):
            raise ValueError("tool_choice must be 'auto', 'required', or 'none'")


@dataclass(frozen=True)
class ModelResult:
    decision: ModelDecision
    provider: str
    model: str
    usage: Usage
    finish_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.provider:
            raise ValueError("provider must not be empty")
        if not self.model:
            raise ValueError("model must not be empty")


__all__ = [
    "FinalOutput",
    "ModelCapabilities",
    "ModelDecision",
    "ModelResult",
    "ModelSettings",
    "Refusal",
    "ToolCall",
    "ToolCalls",
    "ToolDefinition",
    "Usage",
]
