"""The native Gemini wire, and the registry and credentials that admit it.

Gemini earns a native adapter because the OpenAI-compatibility layer drops
`thought_signature`, and Gemini 3+ rejects a multi-turn tool call without it
(ADR-0020). The signature tests below are the reason this module exists.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from emissary import CapabilityError, ProviderError, key_present, parse_spec
from emissary.llm.credentials import ApiKey, GoogleADC, Unauthenticated
from emissary.llm.decision import ModelSettings, ToolCalls, ToolDefinition
from emissary.llm.messages import AssistantMessage, TextBlock, UserMessage
from emissary.llm.wire import WIRES, gemini

MESSAGES = (UserMessage((TextBlock("find agent"),)),)
TOOLS = (
    ToolDefinition(
        name="lookup",
        description="Look up a term.",
        input_schema={"type": "object", "properties": {"term": {"type": "string"}}},
    ),
)


def _part(**kwargs):
    base = {"text": None, "thought": None, "thought_signature": None, "function_call": None}
    return SimpleNamespace(**{**base, **kwargs})


def _response(*parts, finish="STOP"):
    return SimpleNamespace(
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=list(parts), role="model"), finish_reason=finish
            )
        ],
        usage_metadata=SimpleNamespace(
            prompt_token_count=5, candidates_token_count=2, cached_content_token_count=1
        ),
        model_version="gemini-3.6-flash",
    )


def _client(response):
    client = MagicMock()
    client.models.generate_content.return_value = response
    return client


def test_gemini_wire_normalizes_final_text_and_usage():
    with patch("google.genai.Client", return_value=_client(_response(_part(text="done")))):
        result = gemini.call_model(parse_spec("gemini"), system="s", messages=MESSAGES)

    assert result.decision.text == "done"
    assert result.usage.total_tokens == 7
    assert result.usage.cached_input_tokens == 1


def test_gemini_wire_normalizes_function_calls_and_keeps_ids():
    call = SimpleNamespace(id="call-1", name="lookup", args={"term": "a"})
    response = _response(_part(text="looking"), _part(function_call=call))

    with patch("google.genai.Client", return_value=_client(response)):
        result = gemini.call_model(parse_spec("gemini"), system="s", messages=MESSAGES, tools=TOOLS)

    assert isinstance(result.decision, ToolCalls)
    assert result.decision.calls[0].id == "call-1"
    assert result.decision.calls[0].arguments == {"term": "a"}
    assert result.decision.text == "looking"


def test_gemini_wire_synthesizes_an_id_when_the_api_omits_one():
    """The Developer API may return a function call with no id, but the harness
    matches every tool result to its call by id — a blank one loses the pairing."""
    call = SimpleNamespace(id=None, name="lookup", args={"term": "a"})

    with patch("google.genai.Client", return_value=_client(_response(_part(function_call=call)))):
        result = gemini.call_model(parse_spec("gemini"), system="s", messages=MESSAGES, tools=TOOLS)

    assert result.decision.calls[0].id


def test_gemini_captures_thought_signatures_and_replays_them_verbatim():
    """Gemini 3+ returns 400 on a tool follow-up whose parts lost their
    `thought_signature`. This is the whole justification for the native wire."""
    call = SimpleNamespace(id="call-1", name="lookup", args={"term": "a"})
    response = _response(
        _part(text="thinking it through", thought=True, thought_signature=b"sig-bytes"),
        _part(function_call=call, thought_signature=b"call-sig"),
    )

    with patch("google.genai.Client", return_value=_client(response)):
        result = gemini.call_model(parse_spec("gemini"), system="s", messages=MESSAGES, tools=TOOLS)

    assert result.thinking == "thinking it through"
    assert result.reasoning is not None
    assert result.reasoning.wire == "gemini"

    replayed = (
        UserMessage((TextBlock("count"),)),
        AssistantMessage(text="x", reasoning=result.reasoning),
    )
    with patch("google.genai.Client", return_value=_client(_response(_part(text="ok")))) as ctor:
        gemini.call_model(parse_spec("gemini"), system="s", messages=replayed)
        contents = ctor.return_value.models.generate_content.call_args.kwargs["contents"]

    model_turn = next(item for item in contents if item["role"] == "model")
    signatures = [part.get("thought_signature") for part in model_turn["parts"]]
    assert "c2lnLWJ5dGVz" in signatures, "signature must survive as sent, not be dropped"


def test_gemini_thinking_dialect_maps_visible_to_include_thoughts():
    with patch("google.genai.Client", return_value=_client(_response(_part(text="ok")))) as ctor:
        gemini.call_model(
            parse_spec("gemini"),
            system="s",
            messages=MESSAGES,
            settings=ModelSettings(thinking="visible"),
        )
        config = ctor.return_value.models.generate_content.call_args.kwargs["config"]

    assert config["thinking_config"] == {"include_thoughts": True}


def test_gemini_refusal_is_a_refusal_not_an_empty_answer():
    """A safety stop returns a 200 with no usable parts. Reporting that as an
    empty completion would let a caller treat a block as a real answer."""
    with patch("google.genai.Client", return_value=_client(_response(finish="PROHIBITED_CONTENT"))):
        result = gemini.call_model(parse_spec("gemini"), system="s", messages=MESSAGES)

    assert result.decision.__class__.__name__ == "Refusal"


def test_vertex_builds_an_enterprise_client_with_project_and_location(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj-1")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")

    with patch("google.genai.Client", return_value=_client(_response(_part(text="ok")))) as ctor:
        gemini.call_model(parse_spec("vertex:gemini-3-pro"), system="s", messages=MESSAGES)

    kwargs = ctor.call_args.kwargs
    assert kwargs["enterprise"] is True
    assert kwargs["project"] == "proj-1"
    assert kwargs["location"] == "us-central1"


def test_gemini_developer_api_does_not_use_enterprise():
    with patch("google.genai.Client", return_value=_client(_response(_part(text="ok")))) as ctor:
        gemini.call_model(parse_spec("gemini"), system="s", messages=MESSAGES)

    assert not ctor.call_args.kwargs.get("enterprise")


def test_every_provider_names_a_registered_wire():
    """The registry replaced an if/else chain; an unregistered wire used to be
    an unreachable branch and is now a startup-visible gap."""
    from emissary.llm.provider import PROVIDERS

    assert {provider.wire for provider in PROVIDERS.values()} <= set(WIRES)


def test_scoring_is_refused_by_capability_not_by_wire_name():
    """Anthropic and Gemini both lack logprobs. Gating on the capability rather
    than on one wire's name is what keeps the third wire from being forgotten."""
    from emissary.llm import call_choice

    for name in ("anthropic", "gemini"):
        with pytest.raises(ProviderError, match="logprobs"):
            call_choice(parse_spec(name), labels=["A", "B"], system="s", blocks=())


def test_thinking_control_is_refused_where_the_provider_cannot_express_it():
    from emissary.llm import call_model

    with pytest.raises(CapabilityError, match="thinking"):
        call_model(
            parse_spec("vllm:qwen"),
            system="s",
            messages=MESSAGES,
            settings=ModelSettings(thinking="visible"),
        )


def test_api_key_credential_reports_presence_and_value(monkeypatch):
    credential = ApiKey("SOME_KEY")
    monkeypatch.delenv("SOME_KEY", raising=False)
    assert credential.available() is False
    assert credential.token() is None

    monkeypatch.setenv("SOME_KEY", "abc")
    assert credential.available() is True
    assert credential.token() == "abc"
    assert "SOME_KEY" in credential.describe()


def test_unauthenticated_credential_is_always_available(monkeypatch):
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    assert Unauthenticated("VLLM_API_KEY").available() is True
    assert key_present(parse_spec("vllm:qwen")) is True


def test_google_adc_needs_a_project_and_never_raises(monkeypatch):
    """`key_present` must answer without a network call and without blowing up
    when GCP libraries are absent — selection calls it before every request."""
    credential = GoogleADC("GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION")
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    assert credential.available() is False

    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj-1")
    with patch("google.auth.default", side_effect=Exception("no ADC on this box")):
        assert credential.available() is False
