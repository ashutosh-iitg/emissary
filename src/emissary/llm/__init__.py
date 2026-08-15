"""Provider-neutral LLM calls, contracts, selection, and adapters."""

from .calls import call_choice, call_tool
from .decision import (
    FinalOutput,
    ModelCapabilities,
    ModelResult,
    ModelSettings,
    Refusal,
    ToolCall,
    ToolCalls,
    ToolDefinition,
    Usage,
)
from .errors import CapabilityError, ProviderError
from .messages import AssistantMessage, Message, TextBlock, ToolMessage, UserMessage
from .model import FallbackModelCaller, ModelCaller, SpecModelCaller, call_model
from .prompt import Prompt
from .provider import PROVIDERS, Provider, Spec, key_present, parse_spec
from .result import CallResult, ChoiceResult
from .selection import call_tool_with_fallback, resolve_spec

__all__ = [
    "PROVIDERS",
    "AssistantMessage",
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
    "Refusal",
    "Spec",
    "SpecModelCaller",
    "TextBlock",
    "ToolCall",
    "ToolCalls",
    "ToolDefinition",
    "ToolMessage",
    "Usage",
    "UserMessage",
    "call_choice",
    "call_model",
    "call_tool",
    "call_tool_with_fallback",
    "key_present",
    "parse_spec",
    "resolve_spec",
]
