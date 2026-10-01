"""MCP tool sources for the agent harness. Requires `pip install emissary[mcp]`.

The harness never imports this package; it only sees the `ToolSource` protocol.
"""

from .config import MCPToolPolicy, MCPToolset, Stdio, StreamableHTTP

__all__ = ["MCPToolPolicy", "MCPToolset", "Stdio", "StreamableHTTP"]
