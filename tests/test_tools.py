from unittest.mock import MagicMock

import pytest

from emissary.decision import ToolCall
from emissary.tools import LocalToolExecutor, Tool, ToolRegistry, ToolResult


def test_registry_rejects_duplicate_names_and_exposes_model_definitions():
    first = Tool("lookup", "Look up.", {"type": "object"}, dict)
    second = Tool("lookup", "Different.", {"type": "object"}, dict)

    with pytest.raises(ValueError, match="duplicate"):
        ToolRegistry((first, second))

    registry = ToolRegistry((first,))
    assert registry.resolve("lookup") is first
    assert registry.definitions[0].name == "lookup"


def test_local_executor_validates_input_before_effects():
    effect = MagicMock(return_value={"value": 1})
    tool = Tool(
        "lookup",
        "Look up.",
        {
            "type": "object",
            "properties": {"term": {"type": "string"}},
            "required": ["term"],
            "additionalProperties": False,
        },
        effect,
    )

    result = LocalToolExecutor().execute(ToolCall("one", "lookup", {"term": 7}), tool)

    assert result.status == "error"
    assert "invalid" in result.summary
    effect.assert_not_called()


def test_local_executor_normalizes_success_and_validates_output():
    tool = Tool(
        "lookup",
        "Look up.",
        {"type": "object"},
        lambda: {"value": "wrong"},
        output_schema={
            "type": "object",
            "properties": {"value": {"type": "integer"}},
            "required": ["value"],
        },
    )

    result = LocalToolExecutor().execute(ToolCall("one", "lookup", {}), tool)

    assert result.status == "error"
    assert "invalid output" in result.summary

    successful = Tool("lookup", "Look up.", {"type": "object"}, lambda: {"value": 1})
    assert LocalToolExecutor().execute(ToolCall("two", "lookup", {}), successful) == ToolResult(
        status="success", summary="lookup completed", content={"value": 1}
    )


def test_local_executor_sanitizes_unexpected_exceptions():
    def fail():
        raise RuntimeError("secret-token")

    tool = Tool("lookup", "Look up.", {"type": "object"}, fail)
    result = LocalToolExecutor().execute(ToolCall("one", "lookup", {}), tool)

    assert result.status == "error"
    assert result.summary == "lookup failed"
    assert "secret-token" not in result.summary
