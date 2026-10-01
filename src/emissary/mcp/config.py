"""Client-side description of an MCP server. Pure data: importing this opens nothing.

MCP access is classified by what the client says it is, never inferred from the
transport — localhost can front an external API and a remote endpoint can be an
internal one. Server-supplied annotations are untrusted hints and are not read.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

NAMESPACE_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,31}$")


@dataclass(frozen=True)
class Stdio:
    """Launch `command` directly (no shell). The SDK passes a minimal default
    environment; only `env` is added to it, so nothing else is inherited."""

    command: str
    args: tuple[str, ...] = ()
    cwd: str | None = None
    env: tuple[tuple[str, str], ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        if not self.command:
            raise ValueError("stdio command must not be empty")
        if isinstance(self.env, Mapping):
            object.__setattr__(self, "env", tuple(sorted(self.env.items())))

    @property
    def endpoint(self) -> str:
        return f"stdio:{self.command}"


@dataclass(frozen=True)
class StreamableHTTP:
    """`url` is hidden from repr (it may carry userinfo or a key); `auth` is a trusted collaborator handed to the HTTP client (an `httpx2.Auth`,
    e.g. the SDK's OAuth provider). It never appears in repr, schemas, events or errors."""

    url: str = field(repr=False)
    auth: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if urlsplit(self.url).scheme not in ("http", "https"):
            raise ValueError("StreamableHTTP url must be http or https")

    @property
    def endpoint(self) -> str:
        """Identity without userinfo or query, either of which may carry credentials."""
        parts = urlsplit(self.url)
        host = parts.hostname or ""
        netloc = f"{host}:{parts.port}" if parts.port else host
        return urlunsplit((parts.scheme, netloc, parts.path, "", ""))


@dataclass(frozen=True)
class MCPToolPolicy:
    """What the application decides about a tool, independent of what the server claims.

    `max_attempts > 1` is the application asserting the server tolerates the
    repeated call; nothing is added to the arguments to make that safe.
    """

    api_scope: Literal["internal", "external"]
    approval: Literal["never", "always", "policy"] = "never"
    max_attempts: int = 1
    idempotent: bool = False
    side_effect: Literal["none", "local", "external"] = "external"

    def __post_init__(self) -> None:
        if self.api_scope not in ("internal", "external"):
            raise ValueError("an MCP tool always crosses a process boundary: internal or external")


@dataclass(frozen=True)
class MCPToolset:
    namespace: str
    transport: Stdio | StreamableHTTP
    default_policy: MCPToolPolicy
    include_tools: tuple[str, ...] | None = None
    exclude_tools: tuple[str, ...] = ()
    tool_policies: tuple[tuple[str, MCPToolPolicy], ...] = ()
    aliases: tuple[tuple[str, str], ...] = ()
    required: bool = True
    connect_timeout_seconds: float = 30.0
    call_timeout_seconds: float = 60.0
    max_discovery_pages: int = 50

    def __post_init__(self) -> None:
        if not NAMESPACE_PATTERN.match(self.namespace):
            raise ValueError(
                "namespace must start with a letter and use only letters, digits, '_' or '-' "
                "(at most 32 characters)"
            )
        for name in ("tool_policies", "aliases"):
            value = getattr(self, name)
            if isinstance(value, Mapping):
                object.__setattr__(self, name, tuple(value.items()))
        if self.connect_timeout_seconds <= 0 or self.call_timeout_seconds <= 0:
            raise ValueError("timeouts must be positive")
        if self.max_discovery_pages < 1:
            raise ValueError("max_discovery_pages must be positive")

    def policy_for(self, original_name: str) -> MCPToolPolicy:
        return dict(self.tool_policies).get(original_name, self.default_policy)

    def open(self):
        from .client import open_toolset

        return open_toolset(self)


__all__ = ["MCPToolPolicy", "MCPToolset", "Stdio", "StreamableHTTP"]
