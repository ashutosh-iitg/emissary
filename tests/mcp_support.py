"""Helpers shared by the MCP tests: a loopback Streamable HTTP server and a scripted model."""

import asyncio
import socket
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import httpx2
import uvicorn
from mcp.server.mcpserver import MCPServer
from pydantic import BaseModel

from emissary.llm.decision import ModelResult, Usage

FILES_SERVER = Path(__file__).parent / "fixtures" / "mcp_files_server.py"
TOKEN = "s3cr3t-claims-token"


class BearerAuth(httpx2.Auth):
    """The trusted transport collaborator: credentials live here and nowhere else."""

    def auth_flow(self, request):
        request.headers["Authorization"] = f"Bearer {TOKEN}"
        yield request


class Claim(BaseModel):
    claim_id: str
    document_id: str
    amount: int


def build_claims_server(seen_arguments: list) -> MCPServer:
    server = MCPServer("claims")

    @server.tool()
    def lookup_claim(customer_id: str) -> Claim:
        """Find the open claim for a customer."""
        seen_arguments.append(("lookup_claim", {"customer_id": customer_id}))
        return Claim(claim_id="cl-1", document_id="doc-9", amount=1200)

    @server.tool()
    def list_documents() -> dict:
        """List documents."""
        return {"documents": ["doc-9"]}

    return server


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class _RequireBearer:
    """ASGI guard: the server side of the auth collaborator."""

    def __init__(self, app, rejected: list):
        self.app, self.rejected = app, rejected

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            headers = dict(scope["headers"])
            if headers.get(b"authorization") != f"Bearer {TOKEN}".encode():
                self.rejected.append(scope["path"])
                await send({"type": "http.response.start", "status": 401, "headers": []})
                await send({"type": "http.response.body", "body": b""})
                return
        await self.app(scope, receive, send)


@asynccontextmanager
async def serve_http(server: MCPServer, rejected: list | None = None):
    port = _free_port()
    app = _RequireBearer(server.streamable_http_app(), rejected if rejected is not None else [])
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", lifespan="on")
    http = uvicorn.Server(config)
    task = asyncio.create_task(http.serve())
    while not http.started:
        await asyncio.sleep(0.01)
    try:
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        http.should_exit = True
        await task


@dataclass
class ScriptedModel:
    decisions: list
    contexts: list = field(default_factory=list)

    async def __call__(self, *, system, messages, tools=(), settings=None):
        self.contexts.append(messages)
        return ModelResult(self.decisions.pop(0), "fake", "scripted", Usage(1, 1))
