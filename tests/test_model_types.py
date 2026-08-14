"""Provider-neutral model contracts used by every wire and future runner."""

from dataclasses import asdict

import pytest

from emissary.decision import (
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
from emissary.messages import AssistantMessage, TextBlock, ToolMessage, UserMessage


def test_text_blocks_preserve_cache_intent_without_provider_shapes():
    block = TextBlock("source document", cache=True)

    assert asdict(block) == {"text": "source document", "cache": True}


def test_each_message_variant_enforces_its_own_valid_shape():
    call = ToolCall(id="call-1", name="lookup", arguments={"term": "agent"})

    assert UserMessage((TextBlock("find it"),)).content[0].text == "find it"
    assert AssistantMessage(tool_calls=(call,)).tool_calls == (call,)
    assert ToolMessage(call_id="call-1", tool_name="lookup", content="found").content == "found"

    with pytest.raises(ValueError, match="content"):
        UserMessage(())
    with pytest.raises(ValueError, match="text or tool calls"):
        AssistantMessage()
    with pytest.raises(ValueError, match="call_id"):
        ToolMessage(call_id="", tool_name="lookup", content="found")


def test_tool_definitions_and_calls_are_json_object_contracts():
    definition = ToolDefinition(
        name="lookup",
        description="Look up one term.",
        input_schema={"type": "object", "properties": {"term": {"type": "string"}}},
    )
    call = ToolCall(id="call-1", name="lookup", arguments={"term": "agent"})

    assert definition.input_schema["type"] == "object"
    assert call.arguments == {"term": "agent"}

    with pytest.raises(ValueError, match="name"):
        ToolDefinition(name="", description="d", input_schema={"type": "object"})
    with pytest.raises(ValueError, match="object schema"):
        ToolDefinition(name="lookup", description="d", input_schema={"type": "array"})
    with pytest.raises(TypeError, match="arguments"):
        ToolCall(id="call-1", name="lookup", arguments=["not", "an", "object"])


def test_decisions_are_disjoint_and_validate_variant_invariants():
    call = ToolCall(id="call-1", name="lookup", arguments={})

    assert FinalOutput(text="done").text == "done"
    assert FinalOutput(value={"answer": 1}).value == {"answer": 1}
    assert ToolCalls((call,)).calls == (call,)
    assert Refusal("policy").reason == "policy"

    with pytest.raises(ValueError, match="text or value"):
        FinalOutput()
    with pytest.raises(ValueError, match="at least one"):
        ToolCalls(())
    with pytest.raises(ValueError, match="unique"):
        ToolCalls((call, call))
    with pytest.raises(ValueError, match="reason"):
        Refusal("")


def test_usage_rejects_negative_counts_and_reports_a_total():
    usage = Usage(input_tokens=10, output_tokens=4, cached_input_tokens=3)

    assert usage.total_tokens == 14
    with pytest.raises(ValueError, match="non-negative"):
        Usage(input_tokens=-1, output_tokens=0)


def test_capabilities_are_conservative_and_settings_are_provider_neutral():
    assert ModelCapabilities() == ModelCapabilities(
        tool_calling=False,
        parallel_tool_calls=False,
        structured_output=False,
        logprobs=False,
    )

    settings = ModelSettings(max_output_tokens=100, tool_choice="required")
    assert settings.max_output_tokens == 100
    with pytest.raises(ValueError, match="max_output_tokens"):
        ModelSettings(max_output_tokens=0)
    with pytest.raises(ValueError, match="tool_choice"):
        ModelSettings(tool_choice="sometimes")


def test_model_result_combines_one_decision_with_provenance_and_usage():
    result = ModelResult(
        decision=FinalOutput(text="done"),
        provider="vllm",
        model="qwen",
        usage=Usage(input_tokens=10, output_tokens=1),
        finish_reason="stop",
    )

    assert result.decision == FinalOutput(text="done")
    assert result.usage.total_tokens == 11
    with pytest.raises(ValueError, match="provider"):
        ModelResult(
            decision=FinalOutput(text="done"),
            provider="",
            model="qwen",
            usage=Usage(0, 0),
        )
