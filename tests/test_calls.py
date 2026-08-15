"""`calls.py` dispatches by wire and refuses to call out with no credential."""

import pytest

from emissary import ProviderError, parse_spec
from emissary.llm.calls import call_tool
from emissary.llm.messages import TextBlock

TOOL = {"name": "record", "description": "d", "input_schema": {"type": "object"}}


def test_a_missing_key_is_refused_before_any_network_call(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(ProviderError, match="ANTHROPIC_API_KEY"):
        call_tool(parse_spec("anthropic"), system="s", blocks=(TextBlock("d"),), tool=TOOL)


def test_vllm_needs_no_key_to_pass_the_gate(monkeypatch):
    """Reaching the wire adapter itself is as far as this test goes — no
    server is running, so the real connection failure is expected."""
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)

    with pytest.raises(ProviderError, match="could not reach the API"):
        call_tool(parse_spec("vllm:qwen3-8b"), system="s", blocks=(TextBlock("d"),), tool=TOOL)
