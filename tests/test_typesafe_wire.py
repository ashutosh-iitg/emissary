"""TypeSafe's Jev behind `call_choice` — real httpx parsing over a mock transport."""

import json

import httpx
import pytest

from emissary import (
    CapabilityError,
    ProviderError,
    acall_choice,
    call_choice,
    call_model,
    call_tool,
    parse_spec,
)
from emissary.llm.messages import TextBlock, UserMessage

JEV = parse_spec("typesafe")
LABELS = ["SAFE", "FLAG"]
SYSTEM = "Is this message safe for a young child?"


def _answer(probabilities):
    return {
        "model": "jev-1.13.0",
        "answers": {"choice": {"type": "choice", "choice": "SAFE", "probabilities": probabilities}},
        "usage": {"input_tokens": 342},
    }


def _replying(body, status=200):
    return lambda request: httpx.Response(status, json=body)


def _serve(monkeypatch, handler):
    """Route every httpx client the wire opens through `handler`, recording requests."""
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    transport = httpx.MockTransport(record)
    real_client, real_async_client = httpx.Client, httpx.AsyncClient
    monkeypatch.setattr(httpx, "Client", lambda **kw: real_client(transport=transport, **kw))
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kw: real_async_client(transport=transport, **kw)
    )
    return seen


def _choose(labels=LABELS):
    return call_choice(JEV, system=SYSTEM, blocks=(TextBlock("x"),), labels=labels)


@pytest.fixture(autouse=True)
def typesafe_key(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "sk-test")


def test_the_prompt_becomes_jevs_instructions_state_and_criteria(monkeypatch):
    seen = _serve(monkeypatch, _replying(_answer({"SAFE": 0.9, "FLAG": 0.1})))

    call_choice(JEV, system=SYSTEM, blocks=(TextBlock("hello"), TextBlock("there")), labels=LABELS)

    (request,) = seen
    assert str(request.url) == "https://api.typesafe.ai/v1/systemone"
    assert request.headers["authorization"] == "Bearer sk-test"
    assert json.loads(request.content) == {
        "model": "jev-latest",
        "state": "hello\n\nthere",
        "questions": {
            "choice": {
                "type": "choice",
                "instructions": SYSTEM,
                "criteria": {"SAFE": "SAFE", "FLAG": "FLAG"},
            }
        },
    }


def test_jevs_calibrated_probabilities_are_the_result(monkeypatch):
    _serve(monkeypatch, _replying(_answer({"SAFE": 0.87, "FLAG": 0.13})))

    out = _choose()

    assert out.probabilities == {"SAFE": 0.87, "FLAG": 0.13}
    assert out.label == "SAFE"
    assert (out.provider, out.model) == ("typesafe", "jev-1.13.0")
    assert (out.input_tokens, out.output_tokens) == (342, 0)


def test_labels_need_not_differ_by_first_token(monkeypatch):
    """That rule exists for logprob scoring, which sees one token. Jev scores whole
    labels, so holding it to the rule would refuse label sets it answers correctly."""
    _serve(monkeypatch, _replying(_answer({"FLAG_VIOLENCE": 0.2, "FLAG_SELF_HARM": 0.8})))

    out = _choose(["FLAG_VIOLENCE", "FLAG_SELF_HARM"])

    assert out.label == "FLAG_SELF_HARM"


def test_an_answer_over_different_labels_is_not_retryable(monkeypatch):
    """A threshold is only meaningful against the choice the caller posed, and asking
    again would answer the same."""
    _serve(monkeypatch, _replying(_answer({"SAFE": 1.0})))

    with pytest.raises(ProviderError, match="labels") as caught:
        _choose()

    assert not caught.value.retryable


@pytest.mark.parametrize(
    ("status", "retryable"), [(529, True), (429, True), (422, False), (401, False)]
)
def test_status_codes_follow_the_shared_retry_policy(monkeypatch, status, retryable):
    _serve(monkeypatch, _replying({"detail": "nope"}, status=status))

    with pytest.raises(ProviderError, match=str(status)) as caught:
        _choose()

    assert caught.value.retryable is retryable


def test_an_unreachable_endpoint_is_retryable(monkeypatch):
    def refuse(request):
        raise httpx.ConnectError("refused", request=request)

    _serve(monkeypatch, refuse)

    with pytest.raises(ProviderError, match="could not reach") as caught:
        _choose()

    assert caught.value.retryable


def test_a_body_that_is_not_json_is_not_retryable(monkeypatch):
    _serve(monkeypatch, lambda request: httpx.Response(200, text="<html>"))

    with pytest.raises(ProviderError, match="JSON") as caught:
        _choose()

    assert not caught.value.retryable


def test_duplicate_labels_are_refused_before_any_request(monkeypatch):
    seen = _serve(monkeypatch, _replying({}))

    with pytest.raises(ProviderError, match="unique"):
        _choose(["SAFE", "SAFE"])

    assert seen == []


def test_a_missing_key_fails_before_any_request(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY")
    seen = _serve(monkeypatch, _replying({}))

    with pytest.raises(ProviderError, match="TYPESAFE_API_KEY"):
        _choose()

    assert seen == []


def test_jev_is_not_a_conversational_or_tool_model():
    """It answers typed questions and nothing else; admitting it to either call shape
    would fail against an endpoint that does not exist."""
    with pytest.raises(CapabilityError):
        call_model(JEV, system="s", messages=(UserMessage((TextBlock("hi"),)),))
    with pytest.raises(CapabilityError):
        call_tool(JEV, system="s", blocks=(TextBlock("x"),), tool={"name": "t"})


async def test_acall_choice_matches_call_choice(monkeypatch):
    _serve(monkeypatch, _replying(_answer({"SAFE": 0.3, "FLAG": 0.7})))

    out = await acall_choice(JEV, system=SYSTEM, blocks=(TextBlock("x"),), labels=LABELS)

    assert out.label == "FLAG"
