"""Deliberate schema acceptance for discovered tools (mcp-support-plan §5).

Remote schemas are untrusted input. They are checked for the dialects and
reference forms Emissary's validator actually handles; anything else is
rejected naming the tool, rather than half-validated.
"""

from typing import Any

SUPPORTED_DIALECTS = frozenset(
    {
        "https://json-schema.org/draft/2020-12/schema",
        "http://json-schema.org/draft-07/schema",
    }
)


class UnsupportedSchema(ValueError):
    """A discovered schema this client cannot validate faithfully."""


def check_schema(schema: Any, *, where: str) -> None:
    if not isinstance(schema, dict):
        raise UnsupportedSchema(f"{where}: schema must be an object")
    dialect = schema.get("$schema")
    if dialect is not None and str(dialect).rstrip("#") not in SUPPORTED_DIALECTS:
        raise UnsupportedSchema(f"{where}: unsupported $schema {dialect!r}")
    _reject_remote_references(schema, where)
    _reject_unresolved_references(schema, schema, where)


def _reject_remote_references(node: Any, where: str) -> None:
    """Only local `#...` references resolve; nothing is ever fetched, because a
    schema that makes the client call out is an SSRF vector."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key in ("$ref", "$dynamicRef") and not (
                isinstance(value, str) and value.startswith("#")
            ):
                raise UnsupportedSchema(f"{where}: non-local {key} {value!r}")
            _reject_remote_references(value, where)
    elif isinstance(node, list):
        for item in node:
            _reject_remote_references(item, where)


def _reject_unresolved_references(root: Any, node: Any, where: str) -> None:
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and not _resolves(root, ref):
            raise UnsupportedSchema(f"{where}: unresolved $ref {ref!r}")
        for value in node.values():
            _reject_unresolved_references(root, value, where)
    elif isinstance(node, list):
        for item in node:
            _reject_unresolved_references(root, item, where)


def _resolves(root: Any, ref: str) -> bool:
    """Only JSON-pointer fragments; a named anchor (`#foo`) is not supported."""
    if ref == "#":
        return True
    if not ref.startswith("#/"):
        return False
    target = root
    for part in ref[2:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(target, dict) and part in target:
            target = target[part]
        elif isinstance(target, list) and part.isdigit() and int(part) < len(target):
            target = target[int(part)]
        else:
            return False
    return True


__all__ = ["UnsupportedSchema", "check_schema"]
