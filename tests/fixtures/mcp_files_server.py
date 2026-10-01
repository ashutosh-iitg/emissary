"""Stdio MCP fixture server. Writes its pid so tests can prove it was reaped."""

import os
from pathlib import Path

from mcp.server.mcpserver import MCPServer

server = MCPServer("files")
Path(os.environ["FILES_PID_FILE"]).write_text(str(os.getpid()))

DOCUMENTS = {"doc-9": "Water damage, kitchen. Estimate 1200."}


@server.tool()
def read_document(document_id: str) -> dict:
    """Read a document by id."""
    return {"document_id": document_id, "body": DOCUMENTS[document_id]}


@server.tool()
def blob() -> str:
    """Never called by the model; present so discovery sees more than one tool."""
    return "x"


if __name__ == "__main__":
    server.run("stdio")
