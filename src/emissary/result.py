from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CallResult:
    """A strict-tool payload, plus what produced it and what it cost."""

    payload: dict[str, Any]
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
