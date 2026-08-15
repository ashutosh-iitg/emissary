"""Building one request: system text plus ordered content blocks (ADR-0017).

Not a template engine and not a section registry. Consumers' system prompts are
static strings; their structure lives in the blocks — conventionally a cached
document followed by a per-call instruction. This type makes that shape explicit
so a document cannot be mistaken for a template hole, and gives the whole
request a stable identity.
"""

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .messages import TextBlock


@dataclass(frozen=True)
class Prompt:
    system: str
    blocks: tuple[TextBlock, ...] = ()

    def __post_init__(self) -> None:
        if not self.system:
            raise ValueError("prompt system text must not be empty")

    @property
    def fingerprint(self) -> str:
        """Identity of exactly what will be sent.

        Covers the system text and every block's text and cache flag, so two
        prompts share a fingerprint only if the model would see the same
        request. Suitable as an audit identity on whatever the call produced.
        """
        contract = {
            "system": self.system,
            "blocks": [{"text": block.text, "cache": block.cache} for block in self.blocks],
        }
        encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def as_block(block: TextBlock | dict[str, Any]) -> TextBlock:
    """Accept the legacy `{"text": ..., "cache": ...}` dict at call boundaries."""
    if isinstance(block, TextBlock):
        return block
    return TextBlock(block["text"], block.get("cache", False))


def build_prompt(
    prompt: Prompt | None,
    system: str | None,
    blocks: tuple[TextBlock | dict[str, Any], ...],
) -> Prompt:
    """Normalise the two accepted call shapes into one, rejecting a mix.

    Accepting both silently would let a caller believe blocks were sent when
    only the prompt's own were.
    """
    if prompt is not None:
        if system is not None or blocks:
            raise ValueError("pass either prompt= or system=/blocks=, not both")
        return prompt
    if system is None:
        raise ValueError("a call needs prompt= or system=")
    return Prompt(system, tuple(as_block(block) for block in blocks))


__all__ = ["Prompt", "as_block", "build_prompt"]
