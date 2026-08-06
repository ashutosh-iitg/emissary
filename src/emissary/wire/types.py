"""The vocabulary both wire adapters speak.

Neutral ground: it lives here rather than in either adapter so neither
depends on the other for the shape of a call.
"""

from typing import Any

Block = dict[str, Any]
"""`{"text": str, "cache": bool}`. `cache=True` marks the ephemeral prompt-cache
breakpoint — put it on content a caller will resend across many calls (e.g. a
document a tool-calling loop re-sends per section). Honoured on the Anthropic
wire; the OpenAI-compatible wire has no equivalent and concatenates instead."""
