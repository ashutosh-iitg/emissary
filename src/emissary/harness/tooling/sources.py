"""Where tools come from besides the `Agent.tools` tuple (docs/architecture/mcp-support-plan.md).

A source is configuration until opened. Opening may connect, authenticate and
discover, so it is async and happens in `preparation`, never inside the
machine. The harness knows nothing about any concrete protocol: an MCP source
is just one implementation of these two protocols.
"""

from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Protocol

from .tools import Tool


@dataclass(frozen=True)
class ToolOrigin:
    """Where a prepared tool really lives. Kept beside the `Tool`, never in it,
    so authorization can name the true target without parsing an alias."""

    source: str
    original_name: str
    endpoint: str
    fingerprint: str


@dataclass(frozen=True)
class ToolBinding:
    """An executable tool plus the identity of what it is bound to."""

    tool: Tool
    origin: ToolOrigin


class PreparedToolSource(Protocol):
    async def list_tools(self) -> tuple[ToolBinding, ...]:
        """The complete catalog, or an exception. A partial catalog is never returned."""
        ...


class ToolSource(Protocol):
    namespace: str
    required: bool

    def open(self) -> AbstractAsyncContextManager[PreparedToolSource]: ...


__all__ = ["PreparedToolSource", "ToolBinding", "ToolOrigin", "ToolSource"]
