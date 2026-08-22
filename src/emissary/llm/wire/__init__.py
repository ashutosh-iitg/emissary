"""Provider SDK translation adapters, and the table that selects between them.

Three wire formats serving eight providers (ADR-0020). The registry replaces
an `if provider.wire == "anthropic"` chain that grew a branch per wire; adding
one is now a table entry, and `test_gemini_wire` asserts every provider names
a wire that exists.

Not every wire serves every call — Gemini has no logprobs, so `call_choice` is
unreachable there. That precondition is enforced by `ModelCapabilities` before
dispatch (ADR-0004), which is why the adapters do not all carry the same
methods.
"""

from types import ModuleType

from . import anthropic, gemini, openai_compatible

WIRES: dict[str, ModuleType] = {
    "anthropic": anthropic,
    "openai": openai_compatible,
    "gemini": gemini,
}

__all__ = ["WIRES", "anthropic", "gemini", "openai_compatible"]
