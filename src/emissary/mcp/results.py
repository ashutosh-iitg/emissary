"""MCP `CallToolResult` -> Emissary `ToolResult` (mcp-support-plan §7).

First release carries text, JSON and explicit resource references. Binary
blocks are never dropped silently and never inlined as base64: they are listed
as unsupported with their size, and the result is downgraded to a warning.
"""

from typing import Any

from jsonschema import ValidationError

from ..harness.tooling.tools import ToolResult, schema_validator


def normalize_result(tool_name: str, result: Any, output_schema: dict[str, Any] | None):
    text: list[str] = []
    resources: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    for block in result.content:
        _sort_block(block, text, resources, unsupported)

    content: dict[str, Any] = {"text": text, "structured": result.structured_content}
    if resources:
        content["resources"] = resources
    if unsupported:
        content["unsupported"] = unsupported

    if result.is_error:
        return ToolResult("error", f"{tool_name} reported an error", content)

    if output_schema is not None and result.structured_content is not None:
        try:
            schema_validator(output_schema).validate(result.structured_content)
        except ValidationError as exc:
            return ToolResult("error", f"invalid output from {tool_name}: {exc.message}")
        except Exception:  # noqa: BLE001 - e.g. an unresolvable reference
            return ToolResult("error", f"output schema for {tool_name} could not be evaluated")

    if unsupported:
        return ToolResult(
            "warning",
            f"{tool_name} completed; {len(unsupported)} unsupported block(s) omitted",
            content,
        )
    return ToolResult("success", f"{tool_name} completed", content)


def _sort_block(block: Any, text: list, resources: list, unsupported: list) -> None:
    kind = getattr(block, "type", None)
    if kind == "text":
        text.append(block.text)
    elif kind == "resource_link":
        resources.append({"uri": str(block.uri), "name": block.name, "mime_type": block.mime_type})
    elif kind == "resource":
        embedded = block.resource
        body = getattr(embedded, "text", None)
        if body is not None:
            text.append(body)
        else:
            unsupported.append(_unsupported("resource", embedded.mime_type, embedded.blob))
    elif kind in ("image", "audio"):
        unsupported.append(_unsupported(kind, block.mime_type, block.data))
    else:
        unsupported.append({"type": str(kind), "mime_type": None, "encoded_bytes": 0})


def _unsupported(kind: str, mime_type: str | None, payload: str) -> dict[str, Any]:
    return {"type": kind, "mime_type": mime_type, "encoded_bytes": len(payload or "")}


__all__ = ["normalize_result"]
