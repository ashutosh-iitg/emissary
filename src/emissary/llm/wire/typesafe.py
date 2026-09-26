"""TypeSafe's System One endpoint — Jev, a decision model (ADR-0027).

Jev does not converse. It reads a state, answers typed questions about it, and
returns probabilities it was trained to calibrate. So this wire serves exactly
one call, `call_choice`, posed as a single `choice` question: the system text
is the question, the blocks are the state, and each label is an option.

Plain HTTP rather than an SDK: TypeSafe publishes none, and one POST does not
justify a dependency.
"""

from typing import Any

import httpx

from ..errors import ProviderError, retryable_status
from ..messages import TextBlock
from ..provider import Spec
from ..result import ChoiceResult

QUESTION = "choice"
"""The key the one question is posed under; Jev echoes it back in `answers`."""

MAX_OPTIONS = 255
"""TypeSafe's documented ceiling on options per choice question."""

TIMEOUT_SECONDS = 30.0
"""Jev answers in well under a second; this bounds a stalled connection, not a
slow model, and is far below the retry ladder's first 10s-plus wait."""

ERROR_BODY_CHARS = 500


def call_choice(
    spec: Spec, *, system: str, blocks: tuple[TextBlock, ...], labels: list[str]
) -> ChoiceResult:
    """Jev's calibrated distribution over `labels`, as the caller posed them."""
    _validate_labels(spec, labels)
    try:
        with httpx.Client(timeout=TIMEOUT_SECONDS) as client:
            response = client.post(
                _url(spec), json=_request(spec, system, blocks, labels), headers=_headers(spec)
            )
    except httpx.TransportError as exc:
        raise ProviderError(f"{spec}: could not reach the API ({exc})", retryable=True) from exc

    return _normalize(spec, response, labels)


async def acall_choice(
    spec: Spec, *, system: str, blocks: tuple[TextBlock, ...], labels: list[str]
) -> ChoiceResult:
    """`call_choice` on the async client."""
    _validate_labels(spec, labels)
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
            response = await client.post(
                _url(spec), json=_request(spec, system, blocks, labels), headers=_headers(spec)
            )
    except httpx.TransportError as exc:
        raise ProviderError(f"{spec}: could not reach the API ({exc})", retryable=True) from exc

    return _normalize(spec, response, labels)


def _validate_labels(spec: Spec, labels: list[str]) -> None:
    # No first-token rule, unlike the logprob wire: Jev scores whole labels.
    if not labels or not all(labels):
        raise ProviderError(f"{spec}: call_choice needs at least one non-empty label")
    if len(set(labels)) != len(labels):
        raise ProviderError(f"{spec}: labels must be unique")
    if len(labels) > MAX_OPTIONS:
        raise ProviderError(f"{spec}: Jev takes at most {MAX_OPTIONS} labels")


def _url(spec: Spec) -> str:
    return f"{spec.provider.resolved_base_url()}/systemone"


def _headers(spec: Spec) -> dict[str, str]:
    return {"Authorization": f"Bearer {spec.provider.credential.token()}"}


def _request(
    spec: Spec, system: str, blocks: tuple[TextBlock, ...], labels: list[str]
) -> dict[str, Any]:
    # Each label is its own description: `call_choice` carries no richer one, and
    # the question text is where a caller explains what the labels mean.
    return {
        "model": spec.model,
        "state": "\n\n".join(block.text for block in blocks),
        "questions": {
            QUESTION: {
                "type": "choice",
                "instructions": system,
                "criteria": {label: label for label in labels},
            }
        },
    }


def _normalize(spec: Spec, response: httpx.Response, labels: list[str]) -> ChoiceResult:
    if response.is_error:
        raise ProviderError(
            f"{spec}: {response.status_code} {response.text[:ERROR_BODY_CHARS]}",
            retryable=retryable_status(response.status_code),
        )
    try:
        body = response.json()
    except ValueError as exc:
        raise ProviderError(f"{spec}: response was not JSON") from exc

    probabilities = (body.get("answers") or {}).get(QUESTION, {}).get("probabilities") or {}
    if set(probabilities) != set(labels):
        # Not retryable: a distribution over other options cannot be thresholded
        # against the choice the caller posed, and asking again answers the same.
        raise ProviderError(
            f"{spec}: answer covers {sorted(probabilities)}, not the labels {sorted(labels)}"
        )
    usage = body.get("usage") or {}
    return ChoiceResult(
        probabilities={label: float(probabilities[label]) for label in labels},
        provider=spec.name,
        model=body.get("model") or spec.model,
        input_tokens=usage.get("input_tokens", 0),
        # Jev bills no output tokens and may omit the field.
        output_tokens=usage.get("output_tokens", 0),
        cached_input_tokens=0,
    )
