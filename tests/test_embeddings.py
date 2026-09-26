"""`embed` — Jina and self-hosted OpenAI-compatible endpoints, mocked at the SDK client."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest

from emissary import CapabilityError, ProviderError, aembed, embed, parse_spec

JINA = "jina:jina-embeddings-v5-text-small"


def _response(*vectors_by_index: tuple[int, list[float]]):
    return SimpleNamespace(
        data=[SimpleNamespace(index=index, embedding=vector) for index, vector in vectors_by_index],
        model="jina-embeddings-v5-text-small",
        usage=SimpleNamespace(prompt_tokens=7, total_tokens=7),
    )


def _mock_client(response=None, side_effect=None, *, is_async=False):
    client = MagicMock()
    if is_async:
        client.embeddings.create = AsyncMock()
    create = client.embeddings.create
    if side_effect is not None:
        create.side_effect = side_effect
    else:
        create.return_value = response
    return patch("openai.AsyncOpenAI" if is_async else "openai.OpenAI", return_value=client)


@pytest.fixture
def jina_key(monkeypatch):
    monkeypatch.setenv("JINA_API_KEY", "k")


def test_vectors_come_back_in_input_order_even_when_the_server_reorders(jina_key):
    """Callers zip vectors with their own documents; the API only promises `index`."""
    response = _response((1, [0.0, 1.0]), (0, [1.0, 0.0]))
    with _mock_client(response):
        out = embed(parse_spec(JINA), ["first", "second"])

    assert out.vectors == ((1.0, 0.0), (0.0, 1.0))
    assert out.provider == "jina"
    assert out.input_tokens == 7


def test_floats_are_requested_explicitly(jina_key):
    """The SDK otherwise asks for base64 and decodes it itself — a parameter Jina
    names differently, so the default request would rest on an unverified coincidence."""
    with _mock_client(_response((0, [1.0]))) as ctor:
        embed(parse_spec(JINA), ["x"])

    assert ctor.return_value.embeddings.create.call_args.kwargs["encoding_format"] == "float"


@pytest.mark.parametrize(
    ("input_type", "task"), [("query", "retrieval.query"), ("document", "retrieval.passage")]
)
def test_jina_receives_the_retrieval_task_for_the_side_being_embedded(jina_key, input_type, task):
    """Jina embeds queries and passages asymmetrically; embedding both with the same
    task silently degrades every search built on the index."""
    with _mock_client(_response((0, [1.0]))) as ctor:
        embed(parse_spec(JINA), ["x"], input_type=input_type)

    sent = ctor.return_value.embeddings.create.call_args.kwargs
    assert sent["extra_body"] == {"task": task}


def test_no_task_is_sent_when_the_caller_names_no_input_type(jina_key):
    with _mock_client(_response((0, [1.0]))) as ctor:
        embed(parse_spec(JINA), ["x"])

    assert "extra_body" not in ctor.return_value.embeddings.create.call_args.kwargs


def test_self_hosted_endpoint_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("EMBEDDINGS_BASE_URL", "http://embedder:8080/v1")
    with _mock_client(_response((0, [1.0]))) as ctor:
        embed(parse_spec("embeddings:bge-m3"), ["x"])

    assert ctor.call_args.kwargs["base_url"] == "http://embedder:8080/v1"


def test_self_hosted_endpoint_without_a_base_url_fails_before_any_request(monkeypatch):
    """With no URL the SDK would silently default to api.openai.com and send the
    documents to a vendor the operator never chose."""
    monkeypatch.delenv("EMBEDDINGS_BASE_URL", raising=False)
    with _mock_client() as ctor, pytest.raises(ProviderError, match="EMBEDDINGS_BASE_URL"):
        embed(parse_spec("embeddings:bge-m3"), ["x"])

    ctor.assert_not_called()


def test_an_input_type_the_provider_cannot_express_is_refused(monkeypatch):
    """Dropping it would hand symmetric vectors to a caller that asked for
    query-side ones — a quality loss nobody would ever trace."""
    monkeypatch.setenv("EMBEDDINGS_BASE_URL", "http://embedder:8080/v1")
    with _mock_client() as ctor, pytest.raises(CapabilityError, match="input_type"):
        embed(parse_spec("embeddings:bge-m3"), ["x"], input_type="query")

    ctor.assert_not_called()


def test_a_chat_only_provider_cannot_embed(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    with pytest.raises(CapabilityError, match="embeddings"):
        embed(parse_spec("anthropic"), ["x"])


def test_nothing_to_embed_is_a_caller_error(jina_key):
    with _mock_client() as ctor, pytest.raises(ProviderError, match="at least one"):
        embed(parse_spec(JINA), [])

    ctor.assert_not_called()


def test_a_missing_key_fails_before_any_request(monkeypatch):
    monkeypatch.delenv("JINA_API_KEY", raising=False)
    with _mock_client() as ctor, pytest.raises(ProviderError, match="JINA_API_KEY"):
        embed(parse_spec(JINA), ["x"])

    ctor.assert_not_called()


def test_a_vector_count_mismatch_is_not_retryable(jina_key):
    """A short answer cannot be matched back to its documents; asking again is not
    recovery, and guessing the alignment would index the wrong text."""
    with _mock_client(_response((0, [1.0]))), pytest.raises(ProviderError) as caught:
        embed(parse_spec(JINA), ["a", "b"])

    assert not caught.value.retryable


def test_rate_limiting_is_retryable(jina_key):
    request = httpx.Request("POST", "https://api.jina.ai/v1/embeddings")
    error = openai.RateLimitError(
        "slow down", response=httpx.Response(429, request=request), body=None
    )
    with _mock_client(side_effect=error), pytest.raises(ProviderError) as caught:
        embed(parse_spec(JINA), ["x"])

    assert caught.value.retryable


async def test_aembed_matches_embed(jina_key):
    with _mock_client(_response((0, [0.5, 0.5])), is_async=True) as ctor:
        out = await aembed(parse_spec(JINA), ["x"], input_type="document")

    assert out.vectors == ((0.5, 0.5),)
    assert ctor.return_value.embeddings.create.call_args.kwargs["extra_body"] == {
        "task": "retrieval.passage"
    }


def test_hosted_openai_embeds_with_floats_and_no_retrieval_task(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    with _mock_client(_response((0, [1.0]))) as ctor:
        out = embed(parse_spec("openai:text-embedding-3-small"), ["x"])

    sent = ctor.return_value.embeddings.create.call_args.kwargs
    assert sent["encoding_format"] == "float"
    assert "extra_body" not in sent
    assert out.provider == "openai"


def test_hosted_openai_refuses_an_input_type(monkeypatch):
    """OpenAI's embeddings are symmetric; there is no query side to ask for."""
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    with _mock_client() as ctor, pytest.raises(CapabilityError, match="input_type"):
        embed(parse_spec("openai:text-embedding-3-small"), ["x"], input_type="query")

    ctor.assert_not_called()


def test_voyage_request_goes_over_the_real_sdk_as_voyage_expects(monkeypatch):
    """Voyage rejects `encoding_format: "float"` (only null or base64), and reports
    usage as `total_tokens`. Run through the real SDK so the JSON body itself is
    checked, not just the kwargs handed to it."""
    monkeypatch.setenv("VOYAGE_API_KEY", "k")
    bodies = []

    def voyage(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"object": "embedding", "embedding": [0.25, 0.5], "index": 0}],
                "model": "voyage-4",
                "usage": {"total_tokens": 3},
            },
        )

    real_openai = openai.OpenAI
    monkeypatch.setattr(
        openai,
        "OpenAI",
        lambda **kw: real_openai(
            **kw, http_client=httpx.Client(transport=httpx.MockTransport(voyage))
        ),
    )

    out = embed(parse_spec("voyage:voyage-4"), ["x"], input_type="document")

    assert bodies == [
        {"input": ["x"], "model": "voyage-4", "encoding_format": None, "input_type": "document"}
    ]
    assert out.vectors == ((0.25, 0.5),)
    assert out.input_tokens == 3
