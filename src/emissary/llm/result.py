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


@dataclass(frozen=True)
class ChoiceResult:
    """A probability over a fixed label set, read from the model's own logprobs.

    A separate type rather than a `payload` union on `CallResult`: a caller
    that wants a score and a caller that wants tool arguments want different
    things, and folding both into one optional-field type is what made
    `payload: dict | str` unsound for consumers. The five provenance and usage
    fields are repeated because they are data, not logic — the alternative
    costs more than it saves.
    """

    probabilities: dict[str, float]
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int

    @property
    def label(self) -> str:
        """The most probable label."""
        return max(self.probabilities, key=lambda name: self.probabilities[name])

    def probability(self, label: str) -> float:
        """The probability assigned to one label, 0.0 if it drew no mass."""
        return self.probabilities.get(label, 0.0)
