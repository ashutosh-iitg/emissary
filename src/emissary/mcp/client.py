"""The only module that talks to the MCP SDK's client (ADR-0009 style boundary).

SDK classes and transport detail stay here: the harness sees `ToolBinding`s.
Protocol negotiation, cancellation and HTTP behaviour belong to the SDK; this
code does not hand-roll JSON-RPC.
"""

import asyncio
import copy
import hashlib
import json
import logging
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

import httpx2
from mcp import Client, StdioServerParameters
from mcp.client.streamable_http import streamable_http_client

from ..harness.tooling.sources import ToolBinding, ToolOrigin
from ..harness.tooling.tools import Tool, ToolResult
from .config import MCPToolset, Stdio, StreamableHTTP
from .naming import default_alias, validate_alias
from .results import normalize_result
from .schema import check_schema

logger = logging.getLogger(__name__)


@asynccontextmanager
async def open_toolset(toolset: MCPToolset):
    async with AsyncExitStack() as stack:
        transport = await _transport(toolset, stack)
        # cache=None: no cross-run discovery caching, so one run's catalog can
        # never be served to another identity.
        client = Client(transport, cache=None)
        async with asyncio.timeout(toolset.connect_timeout_seconds):
            await stack.enter_async_context(client)
        yield _Connection(toolset, client)


async def _transport(toolset: MCPToolset, stack: AsyncExitStack):
    spec = toolset.transport
    if isinstance(spec, Stdio):
        return StdioServerParameters(
            command=spec.command,
            args=list(spec.args),
            cwd=spec.cwd,
            env=dict(spec.env) or None,
        )
    assert isinstance(spec, StreamableHTTP)
    http = await stack.enter_async_context(
        httpx2.AsyncClient(auth=spec.auth, timeout=httpx2.Timeout(30.0, read=300.0))
    )
    return streamable_http_client(spec.url, http_client=http)


class _Connection:
    def __init__(self, toolset: MCPToolset, client: Client) -> None:
        self._toolset = toolset
        self._client = client

    async def list_tools(self) -> tuple[ToolBinding, ...]:
        if self._client.server_capabilities.tools is None:
            raise RuntimeError(f"server {self._toolset.namespace!r} does not offer tools")
        discovered = await self._discover()
        bindings = tuple(self._bind(tool) for tool in discovered if self._selected(tool.name))
        aliases = [binding.tool.name for binding in bindings]
        if len(set(aliases)) != len(aliases):
            raise ValueError(f"source {self._toolset.namespace!r} produced duplicate aliases")
        return bindings

    async def _discover(self) -> list[Any]:
        """Every page, or an exception: a partial catalog is never returned."""
        tools: list[Any] = []
        cursors: set[str] = set()
        cursor: str | None = None
        for _ in range(self._toolset.max_discovery_pages):
            page = await self._client.list_tools(cursor=cursor)
            tools.extend(page.tools)
            cursor = page.next_cursor
            if cursor is None:
                return tools
            if cursor in cursors:
                raise RuntimeError("tool discovery repeated a cursor")
            cursors.add(cursor)
        raise RuntimeError("tool discovery exceeded max_discovery_pages")

    def _selected(self, name: str) -> bool:
        include = self._toolset.include_tools
        return (include is None or name in include) and name not in self._toolset.exclude_tools

    def _bind(self, tool: Any) -> ToolBinding:
        toolset = self._toolset
        where = f"{toolset.namespace}/{tool.name}"
        check_schema(tool.input_schema, where=where)
        if tool.output_schema is not None:
            check_schema(tool.output_schema, where=where)
        policy = toolset.policy_for(tool.name)
        explicit = dict(toolset.aliases).get(tool.name)
        alias = (
            validate_alias(explicit) if explicit else default_alias(toolset.namespace, tool.name)
        )
        input_schema = copy.deepcopy(tool.input_schema)
        output_schema = copy.deepcopy(tool.output_schema)
        fingerprint = _contract_fingerprint(
            tool.name, tool.description, input_schema, output_schema
        )
        return ToolBinding(
            Tool(
                name=alias,
                description=tool.description or "",
                input_schema=input_schema,
                output_schema=output_schema,
                execute=self._invoker(tool.name, alias, output_schema),
                side_effect=policy.side_effect,
                approval=policy.approval,
                idempotent=policy.idempotent,
                max_attempts=policy.max_attempts,
                api_scope=policy.api_scope,
                # Remote schemas are not guaranteed to survive strict-mode
                # rewriting with their meaning intact, so the wire is told not to.
                strict_schema=False,
            ),
            ToolOrigin(toolset.namespace, tool.name, toolset.transport.endpoint, fingerprint),
        )

    def _invoker(self, original_name: str, alias: str, output_schema: dict[str, Any] | None):
        client, timeout = self._client, self._toolset.call_timeout_seconds

        async def invoke(**arguments: Any) -> ToolResult:
            # Only the original name and the authorized arguments cross the wire.
            try:
                async with asyncio.timeout(timeout):
                    result = await client.call_tool(original_name, arguments)
            except TimeoutError:
                return ToolResult("error", f"{alias} timed out", timed_out=True)
            except Exception:
                logger.exception("MCP call to %s failed", alias)
                return ToolResult("error", f"{alias} failed")
            return normalize_result(alias, result, output_schema)

        return invoke


def _contract_fingerprint(name: str, description: str | None, *schemas: Any) -> str:
    contract = {"name": name, "description": description, "schemas": schemas}
    return hashlib.sha256(json.dumps(contract, sort_keys=True, default=str).encode()).hexdigest()
