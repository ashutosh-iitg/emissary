"""`embed` on the native Gemini wire — Gemini and Vertex, mocked at the SDK client."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from emissary import ProviderError, aembed, embed, parse_spec

pytest.importorskip("google.genai")


def _response(*vectors, token_counts=None):
    counts = token_counts or [None] * len(vectors)
    return SimpleNamespace(
        embeddings=[
            SimpleNamespace(
                values=list(vector),
                statistics=SimpleNamespace(token_count=count) if count is not None else None,
            )
            for vector, count in zip(vectors, counts, strict=True)
        ]
    )


def _mock_client(response=None, side_effect=None):
    client = MagicMock()
    client.aio.models.embed_content = AsyncMock(return_value=response)
    if side_effect is not None:
        client.models.embed_content.side_effect = side_effect
    else:
        client.models.embed_content.return_value = response
    return patch("google.genai.Client", return_value=client)


def _sent_texts(ctor) -> list[str]:
    contents = ctor.return_value.models.embed_content.call_args.kwargs["contents"]
    return [content.parts[0].text for content in contents]


@pytest.fixture(autouse=True)
def gemini_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")


def test_each_text_is_its_own_content_so_each_gets_its_own_vector():
    """Given bare strings, gemini-embedding-2 fuses them into ONE vector — the
    caller would index several documents under a single blended embedding."""
    with _mock_client(_response([1.0], [2.0])) as ctor:
        out = embed(parse_spec("gemini:gemini-embedding-001"), ["a", "b"])

    assert _sent_texts(ctor) == ["a", "b"]
    assert out.vectors == ((1.0,), (2.0,))
    assert out.provider == "gemini"


@pytest.mark.parametrize(
    ("input_type", "task_type"), [("query", "RETRIEVAL_QUERY"), ("document", "RETRIEVAL_DOCUMENT")]
)
def test_embedding_001_takes_the_retrieval_side_as_task_type(input_type, task_type):
    with _mock_client(_response([1.0])) as ctor:
        embed(parse_spec("gemini:gemini-embedding-001"), ["a"], input_type=input_type)

    config = ctor.return_value.models.embed_content.call_args.kwargs["config"]
    assert config.task_type == task_type
    assert _sent_texts(ctor) == ["a"]


@pytest.mark.parametrize(
    ("input_type", "text"),
    [("query", "task: search result | query: a"), ("document", "title: none | text: a")],
)
def test_embedding_2_takes_the_retrieval_side_as_its_documented_prefix(input_type, text):
    """gemini-embedding-2 does not accept `task_type`; Google's documented way to
    embed asymmetrically is these prefixes, and without them queries and documents
    land in the same symmetric space."""
    with _mock_client(_response([1.0])) as ctor:
        embed(parse_spec("gemini:gemini-embedding-2"), ["a"], input_type=input_type)

    config = ctor.return_value.models.embed_content.call_args.kwargs["config"]
    assert config is None or config.task_type is None
    assert _sent_texts(ctor) == [text]


def test_no_input_type_sends_the_text_untouched():
    with _mock_client(_response([1.0])) as ctor:
        embed(parse_spec("gemini:gemini-embedding-2"), ["a"])

    assert _sent_texts(ctor) == ["a"]


def test_a_fused_answer_is_not_retryable():
    """Fewer vectors than texts means the inputs were aggregated; no alignment
    back to the documents exists."""
    with _mock_client(_response([1.0])), pytest.raises(ProviderError) as caught:
        embed(parse_spec("gemini:gemini-embedding-2"), ["a", "b"])

    assert not caught.value.retryable


def test_token_counts_are_summed_where_the_api_reports_them():
    with _mock_client(_response([1.0], [2.0], token_counts=[3, 4])):
        out = embed(parse_spec("gemini:gemini-embedding-001"), ["a", "b"])

    assert out.input_tokens == 7


def test_api_errors_follow_the_shared_retry_policy():
    from google.genai import errors

    error = errors.APIError(429, {"error": {"message": "slow down"}})
    with _mock_client(side_effect=error), pytest.raises(ProviderError) as caught:
        embed(parse_spec("gemini:gemini-embedding-001"), ["a"])

    assert caught.value.retryable


def test_vertex_embeds_through_the_same_wire(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")
    monkeypatch.setattr("google.auth.default", lambda: (object(), "p"))
    with _mock_client(_response([1.0])) as ctor:
        out = embed(parse_spec("vertex:gemini-embedding-001"), ["a"], input_type="query")

    assert out.provider == "vertex"
    assert ctor.call_args.kwargs["enterprise"] is True


async def test_aembed_matches_embed():
    with _mock_client(_response([0.5])) as ctor:
        out = await aembed(parse_spec("gemini:gemini-embedding-2"), ["a"], input_type="query")

    assert out.vectors == ((0.5,),)
    contents = ctor.return_value.aio.models.embed_content.call_args.kwargs["contents"]
    assert contents[0].parts[0].text == "task: search result | query: a"
