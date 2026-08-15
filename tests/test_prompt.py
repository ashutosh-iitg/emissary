"""Requests are assembled as one value with a stable identity (ADR-0017)."""

import pytest

from emissary.llm.messages import TextBlock
from emissary.llm.prompt import Prompt, build_prompt

DOCUMENT = TextBlock("the whole statute", cache=True)
INSTRUCTION = TextBlock("extract the obligations")


def test_a_prompt_carries_the_document_as_a_block_not_a_template_hole():
    """The shape both consumers converged on, now expressible as one value."""
    prompt = Prompt("You extract obligations.", (DOCUMENT, INSTRUCTION))

    assert prompt.blocks[0].cache is True
    assert prompt.blocks[1].cache is False
    with pytest.raises(ValueError, match="system"):
        Prompt("")


def test_the_fingerprint_covers_everything_the_model_will_see():
    """It is an audit identity, so anything that changes the request must
    change it — including a cache flag, which changes the wire payload."""
    base = Prompt("system", (DOCUMENT, INSTRUCTION))

    assert base.fingerprint == Prompt("system", (DOCUMENT, INSTRUCTION)).fingerprint
    assert base.fingerprint != Prompt("other", (DOCUMENT, INSTRUCTION)).fingerprint
    assert base.fingerprint != Prompt("system", (INSTRUCTION, DOCUMENT)).fingerprint
    assert base.fingerprint != Prompt("system", (TextBlock(DOCUMENT.text), INSTRUCTION)).fingerprint


def test_the_legacy_dict_shape_still_builds_a_prompt():
    built = build_prompt(None, "system", ({"text": "doc", "cache": True}, {"text": "instr"}))

    assert built == Prompt("system", (TextBlock("doc", cache=True), TextBlock("instr")))


def test_mixing_the_two_call_shapes_is_rejected_rather_than_silently_merged():
    """Accepting both would let a caller believe blocks were sent when only the
    prompt's own were."""
    prompt = Prompt("system", (INSTRUCTION,))

    with pytest.raises(ValueError, match="not both"):
        build_prompt(prompt, None, ({"text": "ignored"},))
    with pytest.raises(ValueError, match="not both"):
        build_prompt(prompt, "system", ())
    with pytest.raises(ValueError, match="needs"):
        build_prompt(None, None, ())
