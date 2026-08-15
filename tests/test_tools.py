from unittest.mock import MagicMock

import pytest

from emissary.harness.tools import (
    LocalToolExecutor,
    Tool,
    ToolContext,
    ToolRegistry,
    ToolResult,
)
from emissary.llm.decision import ToolCall


def test_retries_cannot_be_configured_without_declaring_idempotency():
    """The safety property is structural, not advisory: there is no way to ask
    for a retry on a tool that has not said duplicates are safe."""
    with pytest.raises(ValueError, match="idempotent"):
        Tool("send", "Send.", {"type": "object"}, dict, max_attempts=3)


def test_only_idempotent_tools_are_handed_a_key_and_it_survives_attempts():
    seen = []

    def record(**kwargs):
        seen.append(kwargs.get("idempotency_key"))
        return {"ok": True}

    plain = Tool("plain", "Plain.", {"type": "object"}, record)
    safe = Tool("safe", "Safe.", {"type": "object"}, record, idempotent=True, max_attempts=2)
    executor = LocalToolExecutor()

    executor.execute(ToolCall("one", "plain", {}), plain, ToolContext("run", attempt=1))
    executor.execute(ToolCall("two", "safe", {}), safe, ToolContext("run", attempt=1))
    executor.execute(ToolCall("two", "safe", {}), safe, ToolContext("run", attempt=2))

    assert seen[0] is None
    assert seen[1] == seen[2] and seen[1] is not None


def test_outcome_dimensions_are_reported_independently_of_severity():
    """A timeout is not a severity. Folding it into `status` would make a write
    that timed out mid-flight indistinguishable from one rejected outright —
    the difference that decides whether a retry is safe."""
    timed_out = ToolResult("error", "fetch timed out", timed_out=True, retryable=True)
    degraded = ToolResult("warning", "partial page returned", timed_out=True)

    assert (timed_out.status, timed_out.timed_out, timed_out.retryable) == ("error", True, True)
    assert (degraded.status, degraded.timed_out, degraded.retryable) == ("warning", True, False)
    assert ToolResult("success", "ok").timed_out is False


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

    result = LocalToolExecutor().execute(
        ToolCall("one", "lookup", {"term": 7}), tool, ToolContext("run")
    )

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

    result = LocalToolExecutor().execute(ToolCall("one", "lookup", {}), tool, ToolContext("run"))

    assert result.status == "error"
    assert "invalid output" in result.summary

    successful = Tool("lookup", "Look up.", {"type": "object"}, lambda: {"value": 1})
    assert LocalToolExecutor().execute(
        ToolCall("two", "lookup", {}), successful, ToolContext("run")
    ) == ToolResult(status="success", summary="lookup completed", content={"value": 1})


def test_local_executor_sanitizes_unexpected_exceptions():
    def fail():
        raise RuntimeError("secret-token")

    tool = Tool("lookup", "Look up.", {"type": "object"}, fail)
    result = LocalToolExecutor().execute(ToolCall("one", "lookup", {}), tool, ToolContext("run"))

    assert result.status == "error"
    assert result.summary == "lookup failed"
    assert "secret-token" not in result.summary
