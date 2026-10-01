"""Static agent configuration and finite run budgets."""

from dataclasses import dataclass, field

from ..llm.decision import ModelSettings
from .tooling.sources import ToolSource
from .tooling.tools import Tool


class ModelAttemptLimitExceeded(Exception):
    """The run cannot admit another physical model request."""


class RunCancelled(Exception):
    """The owner cancelled a synchronous run."""


class RunTimedOut(RunCancelled):
    """The clock cancelled the run: `max_duration_seconds` elapsed."""


@dataclass(frozen=True)
class RunLimits:
    max_turns: int = 12
    max_model_attempts: int = 12
    max_tool_calls: int = 40
    max_tool_attempts: int = 40
    max_internal_api_attempts: int = 20
    max_external_api_attempts: int = 5
    max_duration_seconds: float = 300.0
    max_model_input_bytes: int = 524_288
    max_tool_result_bytes: int = 65_536
    max_consecutive_tool_errors: int = 3
    # Per tool, so one flaky tool cannot spend the whole run's error budget —
    # and cannot hide behind other tools' successes resetting the global count.
    max_tool_failures: int = 3
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.max_turns <= 0:
            raise ValueError("max_turns must be positive")
        if self.max_model_attempts <= 0:
            raise ValueError("max_model_attempts must be positive")
        if self.max_tool_calls < 0:
            raise ValueError("max_tool_calls must be non-negative")
        if self.max_tool_attempts < 0:
            raise ValueError("max_tool_attempts must be non-negative")
        if self.max_internal_api_attempts < 0:
            raise ValueError("max_internal_api_attempts must be non-negative")
        if self.max_external_api_attempts < 0:
            raise ValueError("max_external_api_attempts must be non-negative")
        if not 0 < self.max_duration_seconds < float("inf"):
            raise ValueError("max_duration_seconds must be finite and positive")
        if self.max_model_input_bytes <= 0:
            raise ValueError("max_model_input_bytes must be positive")
        if self.max_tool_result_bytes <= 0:
            raise ValueError("max_tool_result_bytes must be positive")
        if self.max_consecutive_tool_errors <= 0:
            raise ValueError("max_consecutive_tool_errors must be positive")
        if self.max_tool_failures <= 0:
            raise ValueError("max_tool_failures must be positive")


@dataclass(frozen=True)
class Agent:
    name: str
    instructions: str
    tools: tuple[Tool, ...] = ()
    limits: RunLimits = field(default_factory=RunLimits)
    model_settings: ModelSettings = field(default_factory=ModelSettings)
    toolsets: tuple[ToolSource, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("agent name must not be empty")
        if not self.instructions:
            raise ValueError("agent instructions must not be empty")


__all__ = ["Agent", "RunLimits"]
