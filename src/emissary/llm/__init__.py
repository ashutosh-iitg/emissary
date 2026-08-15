"""Provider-neutral LLM calls, contracts, selection, and adapters."""

from .calls import acall_choice, acall_tool, call_choice, call_tool
from .decision import (
    FinalOutput,
    ModelCapabilities,
    ModelResult,
    ModelSettings,
    ReasoningState,
    Refusal,
    ToolCall,
    ToolCalls,
    ToolDefinition,
    Usage,
)
from .errors import CapabilityError, ProviderError
from .messages import AssistantMessage, Message, TextBlock, ToolMessage, UserMessage
from .model import (
    AsyncFallbackModelCaller,
    AsyncModelCaller,
    AsyncSpecModelCaller,
    FallbackModelCaller,
    ModelCaller,
    SpecModelCaller,
    acall_model,
    call_model,
)
from .prompt import Prompt
from .provider import PROVIDERS, Provider, Spec, key_present, parse_spec
from .result import CallResult, ChoiceResult
from .selection import call_tool_with_fallback, resolve_spec
from .streaming import AsyncStreamSink, StreamSink

__all__ = [
    "PROVIDERS",
    "AssistantMessage",
    "AsyncFallbackModelCaller",
    "AsyncModelCaller",
    "AsyncSpecModelCaller",
    "AsyncStreamSink",
    "CallResult",
    "CapabilityError",
    "ChoiceResult",
    "FallbackModelCaller",
    "FinalOutput",
    "Message",
    "ModelCaller",
    "ModelCapabilities",
    "ModelResult",
    "ModelSettings",
    "Prompt",
    "Provider",
    "ProviderError",
    "ReasoningState",
    "Refusal",
    "Spec",
    "SpecModelCaller",
    "StreamSink",
    "TextBlock",
    "ToolCall",
    "ToolCalls",
    "ToolDefinition",
    "ToolMessage",
    "Usage",
    "UserMessage",
    "acall_choice",
    "acall_model",
    "acall_tool",
    "call_choice",
    "call_model",
    "call_tool",
    "call_tool_with_fallback",
    "key_present",
    "parse_spec",
    "resolve_spec",
]
