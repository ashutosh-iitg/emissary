from dataclasses import dataclass


@dataclass(frozen=True)
class CallResult:
    """What a call produced, plus what produced it and what it cost.

    `payload` is a `dict` for `call_tool` (the tool's arguments) and a `str`
    for `call_text` (the model's text). One result shape for both call kinds
    keeps token/cost accounting uniform regardless of which entry point a
    caller used.
    """

    payload: dict | str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
