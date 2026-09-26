"""`ocr` — Jina's jina-ocr-v1 over the OpenAI-compatible wire, mocked at the SDK client."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from emissary import CapabilityError, ProviderError, aocr, call_model, ocr, parse_spec
from emissary.llm.messages import TextBlock, UserMessage

OCR = "jina:jina-ocr-v1"
PAGE = "https://example.com/page.png"


def _response(content="# Title\n\n| a | b |", finish_reason="stop"):
    message = SimpleNamespace(content=content)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
        model="jina-ocr-v1",
        usage=SimpleNamespace(prompt_tokens=900, completion_tokens=40),
    )


def _mock_client(response=None, *, is_async=False):
    client = MagicMock()
    if is_async:
        client.chat.completions.create = AsyncMock()
    client.chat.completions.create.return_value = response
    return patch("openai.AsyncOpenAI" if is_async else "openai.OpenAI", return_value=client)


@pytest.fixture(autouse=True)
def jina_key(monkeypatch):
    monkeypatch.setenv("JINA_API_KEY", "k")


def test_the_page_is_sent_as_an_image_part_and_markdown_comes_back():
    with _mock_client(_response()) as ctor:
        out = ocr(parse_spec(OCR), image_url=PAGE)

    sent = ctor.return_value.chat.completions.create.call_args.kwargs
    assert sent["model"] == "jina-ocr-v1"
    assert sent["messages"] == [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": PAGE}}]}
    ]
    assert out.markdown == "# Title\n\n| a | b |"
    assert (out.input_tokens, out.output_tokens) == (900, 40)


def test_a_base64_data_uri_is_accepted():
    with _mock_client(_response()):
        ocr(parse_spec(OCR), image_url="data:image/png;base64,iVBORw0KGgo=")


def test_a_local_path_is_refused_before_any_request():
    """The API only fetches http(s) URLs or reads data URIs; a path would reach it as
    a URL it cannot resolve and come back as a vague remote error."""
    with _mock_client() as ctor, pytest.raises(ProviderError, match="data:image"):
        ocr(parse_spec(OCR), image_url="/tmp/page.png")

    ctor.assert_not_called()


def test_a_truncated_transcription_is_refused():
    """A page cut off at the token limit reads as a complete page; a caller indexing
    it would never learn that the rest of the document is missing."""
    with (
        _mock_client(_response(finish_reason="length")),
        pytest.raises(ProviderError, match="truncated") as caught,
    ):
        ocr(parse_spec(OCR), image_url=PAGE)

    assert not caught.value.retryable


def test_an_empty_transcription_is_refused():
    with _mock_client(_response(content="")), pytest.raises(ProviderError, match="no text"):
        ocr(parse_spec(OCR), image_url=PAGE)


def test_a_provider_without_ocr_is_refused(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    with pytest.raises(CapabilityError, match="OCR"):
        ocr(parse_spec("openai:gpt-5"), image_url=PAGE)


def test_jina_cannot_be_driven_as_a_conversational_model():
    """Its chat endpoint serves only the OCR model and takes no tools; admitting it
    would fail on the remote side with an error that names neither problem."""
    with pytest.raises(CapabilityError, match="conversation"):
        call_model(parse_spec(OCR), system="s", messages=(UserMessage((TextBlock("hi"),)),))


async def test_aocr_matches_ocr():
    with _mock_client(_response(), is_async=True):
        out = await aocr(parse_spec(OCR), image_url=PAGE)

    assert out.provider == "jina"
    assert out.markdown.startswith("# Title")
