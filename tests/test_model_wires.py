"""Both SDK wires normalize conversational turns to the same contract."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from emissary import ProviderError, parse_spec
from emissary.decision import FinalOutput, ModelSettings, Refusal, ToolCalls, ToolDefinition
from emissary.messages import AssistantMessage, TextBlock, ToolMessage, UserMessage
from emissary.wire import anthropic_wire, openai_wire

TOOLS = (
    ToolDefinition(
        name="lookup",
        description="Look up a term.",
        input_schema={"type": "object", "properties": {"term": {"type": "string"}}},
    ),
)
MESSAGES = (UserMessage((TextBlock("find agent"),)),)


def test_openai_wire_normalizes_final_text_and_usage():
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="done", tool_calls=None), finish_reason="stop"
            )
        ],
        model="gpt-5",
        usage=SimpleNamespace(prompt_tokens=7, completion_tokens=2, prompt_tokens_details=None),
    )
    client = MagicMock()
    client.chat.completions.create.return_value = response

    with patch("openai.OpenAI", return_value=client):
        result = openai_wire.call_model(parse_spec("openai:gpt-5"), system="s", messages=MESSAGES)

    assert result.decision == FinalOutput(text="done")
    assert result.usage.total_tokens == 9


def test_openai_wire_normalizes_all_tool_calls_and_preserves_ids():
    calls = [
        SimpleNamespace(
            id="one", function=SimpleNamespace(name="lookup", arguments=json.dumps({"term": "a"}))
        ),
        SimpleNamespace(
            id="two", function=SimpleNamespace(name="lookup", arguments=json.dumps({"term": "b"}))
        ),
    ]
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=None, tool_calls=calls), finish_reason="tool_calls"
            )
        ],
        model="qwen",
        usage=SimpleNamespace(prompt_tokens=4, completion_tokens=3, prompt_tokens_details=None),
    )
    client = MagicMock()
    client.chat.completions.create.return_value = response

    with patch("openai.OpenAI", return_value=client) as ctor:
        result = openai_wire.call_model(
            parse_spec("vllm:qwen"), system="s", messages=MESSAGES, tools=TOOLS
        )
        sent = ctor.return_value.chat.completions.create.call_args.kwargs

    assert isinstance(result.decision, ToolCalls)
    assert [call.id for call in result.decision.calls] == ["one", "two"]
    assert sent["tools"][0]["function"]["name"] == "lookup"


def test_openai_wire_translates_assistant_and_tool_history():
    prior = (
        UserMessage((TextBlock("find it"),)),
        AssistantMessage(text="checking"),
        ToolMessage(call_id="one", tool_name="lookup", content="found"),
    )
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="done", tool_calls=None), finish_reason="stop"
            )
        ],
        model="qwen",
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, prompt_tokens_details=None),
    )
    client = MagicMock()
    client.chat.completions.create.return_value = response

    with patch("openai.OpenAI", return_value=client) as ctor:
        openai_wire.call_model(parse_spec("vllm:qwen"), system="s", messages=prior)
        sent = ctor.return_value.chat.completions.create.call_args.kwargs["messages"]

    assert [message["role"] for message in sent] == ["system", "user", "assistant", "tool"]
    assert sent[-1]["tool_call_id"] == "one"


def test_anthropic_wire_normalizes_final_text_tool_calls_and_refusal():
    usage = SimpleNamespace(input_tokens=5, output_tokens=2, cache_read_input_tokens=1)
    client = MagicMock()
    client.messages.create.side_effect = [
        SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text="done")],
            model="claude",
            usage=usage,
        ),
        SimpleNamespace(
            stop_reason="tool_use",
            content=[
                SimpleNamespace(type="tool_use", id="one", name="lookup", input={"term": "a"})
            ],
            model="claude",
            usage=usage,
        ),
        SimpleNamespace(stop_reason="refusal", content=[], model="claude", usage=usage),
    ]

    with patch("anthropic.Anthropic", return_value=client):
        final = anthropic_wire.call_model(parse_spec("anthropic"), system="s", messages=MESSAGES)
        tools = anthropic_wire.call_model(
            parse_spec("anthropic"), system="s", messages=MESSAGES, tools=TOOLS
        )
        refusal = anthropic_wire.call_model(parse_spec("anthropic"), system="s", messages=MESSAGES)

    assert final.decision == FinalOutput(text="done")
    assert isinstance(tools.decision, ToolCalls)
    assert tools.decision.calls[0].id == "one"
    assert isinstance(refusal.decision, Refusal)


@pytest.mark.parametrize("wire", [openai_wire, anthropic_wire])
def test_empty_or_contradictory_responses_fail_as_model_behavior(wire):
    usage = SimpleNamespace(
        input_tokens=0,
        output_tokens=0,
        cache_read_input_tokens=0,
        prompt_tokens=0,
        completion_tokens=0,
        prompt_tokens_details=None,
    )
    if wire is openai_wire:
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=None, tool_calls=None), finish_reason="stop"
                )
            ],
            model="qwen",
            usage=usage,
        )
        target = "openai.OpenAI"
        spec = parse_spec("vllm:qwen")
        client = MagicMock()
        client.chat.completions.create.return_value = response
    else:
        response = SimpleNamespace(stop_reason="end_turn", content=[], model="claude", usage=usage)
        target = "anthropic.Anthropic"
        spec = parse_spec("anthropic")
        client = MagicMock()
        client.messages.create.return_value = response

    with patch(target, return_value=client), pytest.raises(ProviderError, match="usable decision"):
        wire.call_model(spec, system="s", messages=MESSAGES, settings=ModelSettings())
