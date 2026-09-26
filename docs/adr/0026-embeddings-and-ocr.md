# ADR-0026: Embeddings and OCR Are Calls, Not Conversations

**Date:** 2026-09-27  
**Status:** accepted  
**Deciders:** Project maintainers  
**Amends:** ADR-0025 — `VectorStore` no longer stands for "emissary has no embedding call"

## Context

Tavi needs to embed text for its pgvector memory and to read page images (worksheets,
scanned books) as text. Emissary had neither. `VectorStore`'s docstring said embedding
was the store's concern because "emissary has no embedding call, and pinning one here
would choose a model for every application".

Two backends are wanted:

- **Jina**, hosted. `POST https://api.jina.ai/v1/embeddings` takes OpenAI's embeddings
  request plus a `task` extension (`retrieval.query`, `retrieval.passage`, …) that embeds
  queries and passages asymmetrically. `jina-ocr-v1` reads a page image into Markdown
  over `POST /v1/chat/completions`, which serves only that model and takes no tools.
  Both were verified against `api.jina.ai/openapi.json` on 2026-09-27.
- **A self-deployed OpenAI-compatible embeddings server** (vLLM, TEI, Infinity, Ollama),
  at an address only the deployment knows.
- **The hosted chat vendors' own embeddings**, where they exist:
  - OpenAI's `/v1/embeddings` is symmetric, with no query or document side.
  - Gemini and Vertex use `embedContent`. `gemini-embedding-001` takes a `task_type`.
    `gemini-embedding-2` refuses that field and documents text prefixes instead
    (`task: search result | query: …` and `title: none | text: …`). Given bare
    strings, it also fuses them into one vector for the whole list.
  - Anthropic has no embeddings API and points to Voyage AI. Voyage's
    `/v1/embeddings` is OpenAI-shaped. It takes `input_type` directly, accepts only
    `null` or `"base64"` for `encoding_format`, and reports usage as `total_tokens`.
    Verified against docs.voyageai.com and ai.google.dev on 2026-09-27.

## Decision

- **Two calls, `embed`/`aembed` and `ocr`/`aocr`, next to `call_tool` and `call_choice`**
  in `llm/calls.py`. Each returns its own result type (`EmbeddingResult`, `OcrResult`),
  for the reason `ChoiceResult` is not a `CallResult` union.
- **Both ride the existing OpenAI-compatible wire.** Jina's embeddings and OCR endpoints
  are OpenAI-shaped, and so is every common self-hosted server. Under ADR-0020 nothing
  here earns a new adapter.
- **Who embeds.** `openai`, `gemini` and `vertex` gain the capability. Gemini's default
  model is a chat model, so embedding there needs a model named, as in
  `gemini:gemini-embedding-2`. Three providers are new:
  - `voyage`: `VOYAGE_API_KEY`, on the OpenAI-compatible wire.
  - `jina`: `JINA_API_KEY`. It has no default model, because it serves both tasks and a
    default can fit only one of them.
  - `embeddings`: `EMBEDDINGS_BASE_URL`, with an optional `EMBEDDINGS_API_KEY`. It has no
    fallback address. Without a base URL the OpenAI SDK would send the documents to
    api.openai.com, so the call is refused before any request.
- **Capabilities gate every call** (ADR-0004). `ModelCapabilities` gains:
  - `embeddings` and `ocr`, gating the new calls;
  - `chat`, which defaults to true. `call_model` and `call_tool` refuse a provider that
    cannot hold a conversation, rather than letting the remote endpoint fail with an
    error that names neither problem.
- **`input_type` is neutral: `"query"` or `"document"`.** `Provider.embedding_dialect`
  names how each API spells it, and the wire maps it:
  - Jina: `task`.
  - Voyage: `input_type`.
  - Gemini: `task_type`, or the documented prefix for `gemini-embedding-2`.
  - Where the dialect is `"none"` (OpenAI, self-hosted), `input_type` is refused, not
    dropped. A caller that asked for query-side vectors and silently got symmetric ones
    would have a worse index and no way to trace why.
- **One `Content` per text on the Gemini wire.** A fused answer would store several
  documents under one blended vector.
- **Embeddings never fall back.** Vectors from two models do not share a space. A
  fallback embedding would sit in the index looking valid and match nothing. The calls
  therefore make one attempt and carry `ProviderError.retryable` like every other call.
  Retrying on the same model is the caller's choice.
- **Order is restored from `index`, and a count mismatch is not retryable.** Guessing the
  alignment would store one document under another's vector.
- **A truncated or empty transcription is an error.** A page cut off at the token limit
  reads as a whole page, and nothing downstream could tell.
- **`VectorStore` is unchanged.** Embedding the query stays the store's concern. A store
  may now call `embed` to do it, but emissary still does not pick the model.

## Alternatives Considered

- **Embed inside `vector_search_tool`.** Emissary would embed the query before calling
  the store. Rejected: it binds every run to one embedding model through the harness,
  and it has to match whichever model indexed the store, which only the application
  knows.
- **A dedicated Jina wire.** Rejected under ADR-0020. The OpenAI-compatible request
  already expresses everything, and `task` rides in `extra_body` as DeepSeek's
  `thinking` does.
- **Reuse `vllm` for self-hosted embeddings.** It works when the embedder and the chat
  model share one server. Rejected as the only option: a deployment often runs them
  apart, and one `VLLM_BASE_URL` cannot point at both.
- **Silently ignore `input_type` where unsupported.** Rejected for the reason in the
  decision above.

## Consequences

### Positive

- Tavi gets embeddings and OCR through the same credential gating, error policy and
  tests as every other call.
- Providers are admitted by capability, so a future embedding vendor is a table entry.

### Negative

- `openai_compatible.py` grows by about 130 lines and `gemini.py` by about 70.
- Choosing between `task_type` and prefixes on Gemini depends on the model name
  (`PREFIXED_MODELS`). A future model that also refuses `task_type` needs adding there.
- A blank page makes `ocr` raise rather than return `""`. Callers that feed separator
  sheets must catch it.

### Risks

- The OCR request sends the image alone, with no instruction. Jina documents the model
  as single-purpose. This is unverified against a live key.
- Voyage and the Gemini prefix path are tested against the documented shapes. The
  Voyage body goes through the real OpenAI SDK over a mock transport. Neither has been
  checked against a live key.
- `encoding_format="float"` is sent explicitly because the SDK's default base64 path
  relies on a parameter Jina names `embedding_type`. If Jina begins rejecting unknown
  fields, this is where it will show.
