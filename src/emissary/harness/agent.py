"""Static agent configuration and finite run budgets."""

from dataclasses import dataclass, field

from ..llm.decision import ModelSettings
from .tools import Tool


@dataclass(frozen=True)
class RunLimits:
    max_turns: int = 12
    max_tool_calls: int = 40
    max_consecutive_tool_errors: int = 3
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.max_turns <= 0:
            raise ValueError("max_turns must be positive")
        if self.max_tool_calls < 0:
            raise ValueError("max_tool_calls must be non-negative")
        if self.max_consecutive_tool_errors <= 0:
            raise ValueError("max_consecutive_tool_errors must be positive")


@dataclass(frozen=True)
class Agent:
    name: str
    instructions: str
    tools: tuple[Tool, ...] = ()
    limits: RunLimits = field(default_factory=RunLimits)
    model_settings: ModelSettings = field(default_factory=ModelSettings)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("agent name must not be empty")
        if not self.instructions:
            raise ValueError("agent instructions must not be empty")


__all__ = ["Agent", "RunLimits"]
