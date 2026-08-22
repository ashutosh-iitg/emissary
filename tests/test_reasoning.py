"""Reasoning is two things: text we may log, and state we must echo back.

The second is a correctness obligation. Anthropic rejects tampered thinking
signatures, Gemini 3+ rejects a tool follow-up without `thought_signature`, and
DeepSeek/Kimi return 400 on turn two when `reasoning_content` is missing from
the assistant history. Every test here fails as a provider 400 in production,
not as a cosmetic difference.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from emissary import CapabilityError, parse_spec
from emissary.llm.decision import ModelSettings, ReasoningState, ToolDefinition
from emissary.llm.messages import AssistantMessage, TextBlock, UserMessage
from emissary.llm.wire import anthropic, openai_compatible

MESSAGES = (UserMessage((TextBlock("find agent"),)),)
TOOLS = (
    ToolDefinition(
        name="lookup",
        description="Look up a term.",
        input_schema={"type": "object", "properties": {"term": {"type": "string"}}},
    ),
)


def _openai_response(*, content, tool_calls=None, reasoning=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    if reasoning is not None:
        message.reasoning_content = reasoning
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        model="deepseek-v4-pro",
        usage=SimpleNamespace(prompt_tokens=4, completion_tokens=3, prompt_tokens_details=None),
    )


def test_openai_wire_captures_reasoning_content_as_text_and_replayable_state():
    """DeepSeek and Kimi volunteer `reasoning_content`. It is both the readable
    trace and the value their API demands back on the next turn."""
    client = MagicMock()
    client.chat.completions.create.return_value = _openai_response(
        content="six", reasoning="Let me count the letters."
    )

    with patch("openai.OpenAI", return_value=client):
        result = openai_compatible.call_model(parse_spec("deepseek"), system="s", messages=MESSAGES)

    assert result.thinking == "Let me count the letters."
    assert result.reasoning == ReasoningState(
        wire="openai", blocks=({"reasoning_content": "Let me count the letters."},)
    )


def test_openai_wire_sends_reasoning_content_back_on_the_next_turn():
    """The live bug: rebuilding history dropped `reasoning_content`, so turn two
    returned `400 reasoning_content in thinking mode must be passed back`."""
    prior = (
        UserMessage((TextBlock("count them"),)),
        AssistantMessage(
            text="six",
            reasoning=ReasoningState(wire="openai", blocks=({"reasoning_content": "counting"},)),
        ),
    )
    client = MagicMock()
    client.chat.completions.create.return_value = _openai_response(content="done")

    with patch("openai.OpenAI", return_value=client) as ctor:
        openai_compatible.call_model(parse_spec("deepseek"), system="s", messages=prior)
        sent = ctor.return_value.chat.completions.create.call_args.kwargs["messages"]

    assistant = next(message for message in sent if message["role"] == "assistant")
    assert assistant["reasoning_content"] == "counting"


def test_reasoning_state_from_another_wire_is_never_forwarded():
    """A fallback from Anthropic to an OpenAI-compatible provider must drop the
    thinking blocks, not forward a payload the second provider cannot parse."""
    prior = (
        UserMessage((TextBlock("count them"),)),
        AssistantMessage(
            text="six",
            reasoning=ReasoningState(
                wire="anthropic", blocks=({"type": "thinking", "signature": "abc"},)
            ),
        ),
    )
    client = MagicMock()
    client.chat.completions.create.return_value = _openai_response(content="done")

    with patch("openai.OpenAI", return_value=client) as ctor:
        openai_compatible.call_model(parse_spec("deepseek"), system="s", messages=prior)
        sent = ctor.return_value.chat.completions.create.call_args.kwargs["messages"]

    assistant = next(message for message in sent if message["role"] == "assistant")
    assert "reasoning_content" not in assistant


def test_deepseek_dialect_translates_thinking_into_extra_body():
    client = MagicMock()
    client.chat.completions.create.return_value = _openai_response(content="ok")

    with patch("openai.OpenAI", return_value=client) as ctor:
        openai_compatible.call_model(
            parse_spec("deepseek"),
            system="s",
            messages=MESSAGES,
            settings=ModelSettings(thinking="off"),
        )
        sent = ctor.return_value.chat.completions.create.call_args.kwargs

    assert sent["extra_body"]["thinking"] == {"type": "disabled"}


def test_default_thinking_sends_no_parameter_at_all():
    """Existing consumers must keep issuing byte-identical requests."""
    client = MagicMock()
    client.chat.completions.create.return_value = _openai_response(content="ok")

    with patch("openai.OpenAI", return_value=client) as ctor:
        openai_compatible.call_model(parse_spec("deepseek"), system="s", messages=MESSAGES)
        sent = ctor.return_value.chat.completions.create.call_args.kwargs

    assert "extra_body" not in sent
    assert "reasoning_effort" not in sent


def test_asking_a_provider_to_stop_thinking_that_cannot_fails_loud():
    """Kimi K3 always reasons. Silently ignoring `off` would bill a caller who
    explicitly asked not to be billed."""
    with pytest.raises(CapabilityError, match="thinking"):
        openai_compatible.call_model(
            parse_spec("kimi"),
            system="s",
            messages=MESSAGES,
            settings=ModelSettings(thinking="off"),
        )


def test_anthropic_wire_captures_thinking_blocks_and_replays_them_verbatim():
    """The signature is opaque and the API rejects tampering, so the blocks must
    survive the round trip byte-for-byte."""
    blocks = [
        SimpleNamespace(type="thinking", thinking="counting the letters", signature="sig-1"),
        SimpleNamespace(type="text", text="six"),
    ]
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(
        stop_reason="end_turn",
        content=blocks,
        model="claude",
        usage=SimpleNamespace(input_tokens=5, output_tokens=2, cache_read_input_tokens=0),
    )

    with patch("anthropic.Anthropic", return_value=client):
        result = anthropic.call_model(parse_spec("anthropic"), system="s", messages=MESSAGES)

    assert result.thinking == "counting the letters"
    assert result.reasoning == ReasoningState(
        wire="anthropic",
        blocks=({"type": "thinking", "thinking": "counting the letters", "signature": "sig-1"},),
    )

    replayed = (
        UserMessage((TextBlock("count them"),)),
        AssistantMessage(text="six", reasoning=result.reasoning),
    )
    with patch("anthropic.Anthropic", return_value=client) as ctor:
        anthropic.call_model(parse_spec("anthropic"), system="s", messages=replayed)
        sent = ctor.return_value.messages.create.call_args.kwargs["messages"]

    assistant = next(message for message in sent if message["role"] == "assistant")
    assert assistant["content"][0] == {
        "type": "thinking",
        "thinking": "counting the letters",
        "signature": "sig-1",
    }


def test_anthropic_visible_thinking_requests_a_summary():
    client = MagicMock()
    client.messages.create.return_value = SimpleNamespace(
        stop_reason="end_turn",
        content=[SimpleNamespace(type="text", text="six")],
        model="claude",
        usage=SimpleNamespace(input_tokens=5, output_tokens=2, cache_read_input_tokens=0),
    )

    with patch("anthropic.Anthropic", return_value=client) as ctor:
        anthropic.call_model(
            parse_spec("anthropic"),
            system="s",
            messages=MESSAGES,
            settings=ModelSettings(thinking="visible"),
        )
        sent = ctor.return_value.messages.create.call_args.kwargs

    assert sent["thinking"] == {"type": "adaptive", "display": "summarized"}


def test_reasoning_state_survives_the_event_log():
    """The harness re-sends assistant turns from the projection, so state that
    does not round-trip through an event is state the next request will lack."""
    from emissary.harness.projection import model_result_data, model_result_from_data
    from emissary.llm.decision import FinalOutput, ModelResult, Usage

    original = ModelResult(
        decision=FinalOutput(text="six"),
        provider="deepseek",
        model="deepseek-v4-pro",
        usage=Usage(1, 1),
        thinking="counting",
        reasoning=ReasoningState(wire="openai", blocks=({"reasoning_content": "counting"},)),
    )
    data = model_result_data(original)

    assert json.loads(json.dumps(data)) == data, "event payloads must be JSON-native"
    assert model_result_from_data(data) == original


def test_tool_turns_carry_reasoning_state_into_the_projected_message():
    """Gemini and DeepSeek both reject the *second* turn of a tool loop when the
    assistant message lacks its reasoning."""
    from emissary.harness.events import new_event
    from emissary.harness.projection import derive_messages, model_result_data, user_message_data
    from emissary.llm.decision import ModelResult, ToolCall, ToolCalls, Usage

    reasoning = ReasoningState(wire="openai", blocks=({"reasoning_content": "picking a tool"},))
    events = [
        new_event("run", 1, "user_message", **user_message_data(MESSAGES[0])),
        new_event(
            "run",
            2,
            "model_call_completed",
            **model_result_data(
                ModelResult(
                    decision=ToolCalls((ToolCall("one", "lookup", {"term": "a"}),), text="looking"),
                    provider="deepseek",
                    model="deepseek-v4-pro",
                    usage=Usage(1, 1),
                    reasoning=reasoning,
                )
            ),
        ),
    ]

    messages = derive_messages(events)
    assistant = messages[-1]
    assert isinstance(assistant, AssistantMessage)
    assert assistant.reasoning == reasoning


OPENROUTER = "openrouter:nvidia/nemotron-3-ultra-550b-a55b:free"


def _openrouter_response(*, content, reasoning=None, reasoning_details=None):
    """OpenRouter puts the trace in `reasoning` and the replayable payload in
    `reasoning_details` — neither under the name DeepSeek and Kimi use."""
    message = SimpleNamespace(content=content, tool_calls=None)
    if reasoning is not None:
        message.reasoning = reasoning
    if reasoning_details is not None:
        message.reasoning_details = reasoning_details
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        model="nvidia/nemotron-3-ultra-550b-a55b:free",
        usage=SimpleNamespace(prompt_tokens=31, completion_tokens=73, prompt_tokens_details=None),
    )


DETAILS = [
    {"type": "reasoning.text", "text": "The word is s-t-r-a-w-b-e-r-r-y.", "index": 0},
]


def test_openrouter_reasoning_is_read_from_its_own_field_names():
    """Reading only `reasoning_content` would leave `thinking` empty for every
    OpenRouter call — no error, just a reasoning model whose reasoning the
    harness never records."""
    client = MagicMock()
    client.chat.completions.create.return_value = _openrouter_response(
        content="three", reasoning="The word is s-t-r-a-w-b-e-r-r-y.", reasoning_details=DETAILS
    )

    with patch("openai.OpenAI", return_value=client):
        result = openai_compatible.call_model(parse_spec(OPENROUTER), system="s", messages=MESSAGES)

    assert result.thinking == "The word is s-t-r-a-w-b-e-r-r-y."
    assert result.reasoning == ReasoningState(
        wire="openrouter", blocks=({"reasoning_details": DETAILS},)
    )


def test_openrouter_replays_reasoning_details_unmodified():
    """The array is what lets the model continue the reasoning it already did.
    Flattening it to the trace text would be accepted by the API and quietly
    restart the reasoning — a cost and quality regression with no error."""
    prior = (
        UserMessage((TextBlock("count them"),)),
        AssistantMessage(
            text="three",
            reasoning=ReasoningState(wire="openrouter", blocks=({"reasoning_details": DETAILS},)),
        ),
    )
    client = MagicMock()
    client.chat.completions.create.return_value = _openrouter_response(content="still three")

    with patch("openai.OpenAI", return_value=client) as ctor:
        openai_compatible.call_model(parse_spec(OPENROUTER), system="s", messages=prior)
        sent = ctor.return_value.chat.completions.create.call_args.kwargs["messages"]

    assistant = next(message for message in sent if message["role"] == "assistant")
    assert assistant["reasoning_details"] == DETAILS


def test_reasoning_is_not_forwarded_between_providers_sharing_this_wire():
    """Same wire, incompatible payloads: DeepSeek has no use for OpenRouter's
    `reasoning_details` array, and OpenRouter cannot resume from DeepSeek's
    `reasoning_content` string. A fallback between the two must drop it rather
    than hand the second provider a field the first one's format."""
    from_openrouter = AssistantMessage(
        text="three",
        reasoning=ReasoningState(wire="openrouter", blocks=({"reasoning_details": DETAILS},)),
    )
    from_deepseek = AssistantMessage(
        text="three",
        reasoning=ReasoningState(wire="openai", blocks=({"reasoning_content": "counting"},)),
    )
    client = MagicMock()
    client.chat.completions.create.return_value = _openai_response(content="done")

    for spec, message, leaked in (
        ("deepseek", from_openrouter, "reasoning_details"),
        (OPENROUTER, from_deepseek, "reasoning_content"),
    ):
        with patch("openai.OpenAI", return_value=client) as ctor:
            openai_compatible.call_model(
                parse_spec(spec),
                system="s",
                messages=(UserMessage((TextBlock("count them"),)), message),
            )
            sent = ctor.return_value.chat.completions.create.call_args.kwargs["messages"]

        assistant = next(item for item in sent if item["role"] == "assistant")
        assert leaked not in assistant


@pytest.mark.parametrize(
    ("setting", "expected"),
    [
        ("off", {"enabled": False}),
        # `on` still reasons and is still billed; `exclude` only withholds the
        # trace, which is the disclosure half of the neutral setting.
        ("on", {"enabled": True, "exclude": True}),
        ("visible", {"enabled": True, "exclude": False}),
    ],
)
def test_openrouter_dialect_translates_every_thinking_setting(setting, expected):
    client = MagicMock()
    client.chat.completions.create.return_value = _openrouter_response(content="ok")

    with patch("openai.OpenAI", return_value=client) as ctor:
        openai_compatible.call_model(
            parse_spec(OPENROUTER),
            system="s",
            messages=MESSAGES,
            settings=ModelSettings(thinking=setting),
        )
        sent = ctor.return_value.chat.completions.create.call_args.kwargs

    assert sent["extra_body"]["reasoning"] == expected
