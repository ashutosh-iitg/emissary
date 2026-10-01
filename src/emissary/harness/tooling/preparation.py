"""Async preparation: connect, discover, snapshot, bind (mcp-support-plan §4).

Preparation is outside the machine on purpose. The machine stays free of I/O;
what it receives is a finished `PreparedRun`, a frozen catalog with the identity
of every tool's true target. Either the whole catalog is returned or
preparation fails — a required source is never half available.
"""

import asyncio
import copy
import logging
from collections.abc import AsyncIterator, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

from ...llm.decision import ToolCall
from ..agent import Agent
from ..policy import AuthorizationContext
from .sources import ToolOrigin
from .tools import (
    AsyncLocalToolExecutor,
    AsyncToolExecutor,
    Tool,
    ToolContext,
    ToolExecutor,
    ToolRegistry,
    ToolResult,
)

logger = logging.getLogger(__name__)

CLEANUP_TIMEOUT_SECONDS = 10.0


class PreparationError(Exception):
    """A tool source could not be prepared. Messages never carry transport detail."""


@dataclass(frozen=True)
class PreparedRun:
    registry: ToolRegistry
    origins: Mapping[str, ToolOrigin]
    omitted: tuple[str, ...] = ()
    has_sources: bool = False
    tools: tuple[Tool, ...] = ()
    toolsets: tuple[Any, ...] = ()
    identity: tuple[str, str | None] | None = None

    def require_for(self, agent: Agent, context: AuthorizationContext | None) -> None:
        """A catalog is only valid for the configuration and identity it was built for.

        Otherwise a catalog prepared for a privileged agent could be handed to
        an agent that never declared those tools, and run with its authority.
        """
        if self.tools != agent.tools or self.toolsets != agent.toolsets:
            raise ValueError(f"prepared run was not built for agent {agent.name!r}")
        if self.identity is not None:
            actual = (context.principal_id, context.tenant_id) if context else None
            if actual != self.identity:
                raise ValueError("prepared run is bound to a different principal or tenant")

    def origin_of(self, tool_name: str) -> ToolOrigin | None:
        return self.origins.get(tool_name)

    def catalog_data(self) -> dict[str, Any]:
        return {
            "tools": [
                {
                    "name": name,
                    "source": origin.source,
                    "original_name": origin.original_name,
                    "endpoint": origin.endpoint,
                    "fingerprint": origin.fingerprint,
                }
                for name, origin in self.origins.items()
            ],
            "omitted": list(self.omitted),
        }


def direct_prepared(agent: Agent) -> PreparedRun:
    """Preparation for an agent with nothing to connect to."""
    if agent.toolsets:
        raise ValueError(
            f"agent {agent.name!r} has toolsets; open them with prepare() or run it with arun()"
        )
    return PreparedRun(ToolRegistry(agent.tools), MappingProxyType({}), tools=agent.tools)


@asynccontextmanager
async def prepare(
    agent: Agent, *, context: AuthorizationContext | None = None
) -> AsyncIterator[PreparedRun]:
    """Open every source, freeze the catalog, and close everything on exit.

    Use directly (`async with prepare(agent) as prepared`) to share one
    connection lifetime across several `arun(..., prepared=prepared)` calls.
    Pass the `context` the sources authenticate as: the prepared run is then
    bound to that principal and tenant, and `arun` refuses any other.
    """
    _reject_duplicate_namespaces(agent)
    async with AsyncExitStack() as guard:
        stack = AsyncExitStack()
        guard.push_async_callback(_close_bounded, stack)
        tools = list(agent.tools)
        origins: dict[str, ToolOrigin] = {}
        omitted: list[str] = []
        for source in agent.toolsets:
            bindings = await _open_source(source, stack, omitted)
            for binding in sorted(bindings, key=lambda b: b.tool.name):
                name = binding.tool.name
                if name in origins or any(name == t.name for t in tools):
                    raise PreparationError(
                        f"tool name {name!r} from source {source.namespace!r} collides with "
                        "another tool; give one of them an explicit alias"
                    )
                tools.append(_detached(binding.tool))
                origins[name] = binding.origin
        yield PreparedRun(
            ToolRegistry(tuple(tools)),
            MappingProxyType(origins),
            tuple(omitted),
            has_sources=bool(agent.toolsets),
            tools=agent.tools,
            toolsets=agent.toolsets,
            identity=(context.principal_id, context.tenant_id) if context else None,
        )


async def _close_bounded(stack: AsyncExitStack) -> None:
    """Reverse-order cleanup that cannot hang a finished run on a wedged server."""
    try:
        async with asyncio.timeout(CLEANUP_TIMEOUT_SECONDS):
            await stack.aclose()
    except TimeoutError:
        logger.error("tool source cleanup exceeded %.0fs", CLEANUP_TIMEOUT_SECONDS)


async def _open_source(source, stack: AsyncExitStack, omitted: list[str]):
    # Entered on a scratch stack first so a source that opens but cannot list
    # its tools is closed immediately instead of lingering until the run ends.
    try:
        async with AsyncExitStack() as scratch:
            prepared = await scratch.enter_async_context(source.open())
            bindings = await prepared.list_tools()
            stack.push_async_exit(scratch.pop_all())
            return bindings
    except Exception as exc:
        if not source.required:
            logger.warning("optional tool source %r omitted: %s", source.namespace, exc)
            omitted.append(source.namespace)
            return ()
        raise PreparationError(
            f"tool source {source.namespace!r} failed to prepare ({type(exc).__name__})"
        ) from exc


def _reject_duplicate_namespaces(agent: Agent) -> None:
    seen: set[str] = set()
    for source in agent.toolsets:
        if source.namespace in seen:
            raise ValueError(f"duplicate tool source namespace {source.namespace!r}")
        seen.add(source.namespace)


def _detached(tool: Tool) -> Tool:
    return replace(
        tool,
        input_schema=copy.deepcopy(tool.input_schema),
        output_schema=copy.deepcopy(tool.output_schema),
    )


class RoutingToolExecutor:
    """Route by prepared binding, never by parsing an alias.

    Direct tools go to the caller's executor (or the bundled async one). Source
    tools always use the bundled executor: they must not receive an injected
    `idempotency_key` their remote schema never declared. In a mixed run the
    caller's `executor=` therefore customises direct tools only.
    """

    def __init__(
        self, prepared: PreparedRun, direct: ToolExecutor | AsyncToolExecutor | None = None
    ) -> None:
        self._prepared = prepared
        self._direct = direct or AsyncLocalToolExecutor()
        self._bound = AsyncLocalToolExecutor(inject_idempotency_key=False)

    def _for(self, tool: Tool):
        if getattr(self._direct, "serves_every_tool", False):
            return self._direct
        return self._bound if self._prepared.origin_of(tool.name) else self._direct

    def validate(self, call: ToolCall, tool: Tool) -> ToolResult | None:
        return self._for(tool).validate(call, tool)

    def execute(self, call: ToolCall, tool: Tool, context: ToolContext):
        return self._for(tool).execute(call, tool, context)


__all__ = [
    "PreparationError",
    "PreparedRun",
    "RoutingToolExecutor",
    "direct_prepared",
    "prepare",
]
