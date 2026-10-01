"""Approval and authorization, kept outside prompts and tool implementations.

Approval asks a human (or stand-in) whether a call may proceed. Authorization
asks the application whether *this principal* may make *this exact request*.
They are separate: `approval="never"` never skips authorization, and approval
cannot override a denial.
"""

import hashlib
import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Protocol

from ..llm.decision import ToolCall
from .tooling.sources import ToolOrigin
from .tooling.tools import Tool, idempotency_key


class ApprovalDecision(str, Enum):
    ALLOW = "allow"
    REJECT = "reject"
    PAUSE = "pause"


class Approver(Protocol):
    def __call__(self, call: ToolCall, tool: Tool) -> ApprovalDecision: ...


def approval_for(call: ToolCall, tool: Tool, approver: Approver | None) -> ApprovalDecision:
    if tool.approval == "never":
        return ApprovalDecision.ALLOW
    if approver is None:
        return ApprovalDecision.PAUSE
    return approver(call, tool)


@dataclass(frozen=True)
class AuthorizationContext:
    """Who is acting, as established by the application and never by the model.

    `references` are nonsecret application identifiers; credentials do not
    belong here because this value reaches policy callbacks and event logs.
    """

    principal_id: str
    tenant_id: str | None = None
    references: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.principal_id:
            raise ValueError("principal_id must not be empty")
        if isinstance(self.references, Mapping):
            object.__setattr__(self, "references", tuple(sorted(self.references.items())))


def canonical_json(value: Any) -> bytes:
    """Stable bytes for any JSON value; rejects what JSON cannot faithfully carry."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


@dataclass(frozen=True)
class InvocationRequest:
    """One call, frozen as owned bytes so no later mutation can change what runs.

    Arguments are held as canonical JSON rather than a dict: a frozen dataclass
    around a mutable dict is frozen in name only. Every read of `arguments`
    returns a detached copy.
    """

    run_id: str
    call_id: str
    tool: str
    original_name: str
    source: str | None
    endpoint: str | None
    fingerprint: str
    arguments_json: bytes
    trusted_json: bytes
    context: AuthorizationContext
    attempt: int = 1

    @property
    def arguments(self) -> dict[str, Any]:
        return json.loads(self.arguments_json)

    @property
    def trusted(self) -> dict[str, Any]:
        return json.loads(self.trusted_json)

    @property
    def digest(self) -> str:
        """Identity of the logical request, deliberately independent of `attempt`:
        a retry is the same operation, and the decision is what binds the attempt."""
        envelope = {
            "run_id": self.run_id,
            "call_id": self.call_id,
            "tool": self.tool,
            "original_name": self.original_name,
            "source": self.source,
            "endpoint": self.endpoint,
            "fingerprint": self.fingerprint,
            "arguments": json.loads(self.arguments_json),
            "trusted": json.loads(self.trusted_json),
            "principal_id": self.context.principal_id,
            "tenant_id": self.context.tenant_id,
            "references": [list(pair) for pair in self.context.references],
        }
        return hashlib.sha256(canonical_json(envelope)).hexdigest()

    def for_attempt(self, attempt: int) -> "InvocationRequest":
        return replace(self, attempt=attempt)

    def detached_call(self) -> ToolCall:
        return ToolCall(self.call_id, self.tool, self.arguments)

    def allow(
        self,
        reason: str = "",
        *,
        policy_version: str | None = None,
        expires_at: datetime | None = None,
    ) -> "AuthorizationDecision":
        return AuthorizationDecision(
            True, self.digest, self.attempt, reason, policy_version, expires_at
        )

    def deny(
        self, reason: str = "", *, policy_version: str | None = None
    ) -> "AuthorizationDecision":
        return AuthorizationDecision(False, self.digest, self.attempt, reason, policy_version)


@dataclass(frozen=True)
class AuthorizationDecision:
    allowed: bool
    request_digest: str
    attempt: int
    reason: str = ""
    policy_version: str | None = None
    expires_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.expires_at is not None and self.expires_at.tzinfo is None:
            raise ValueError("expires_at must be timezone-aware")


class AuthorizerFailed(Exception):
    """The application's policy raised. The run fails closed; detail stays server-side."""


class AuthorizationMismatch(Exception):
    """A decision does not belong to the request about to be dispatched."""


def verify_decision(
    request: InvocationRequest, decision: AuthorizationDecision, *, now: datetime | None = None
) -> bool:
    """Whether dispatch is allowed; raises if the decision is about something else.

    A mismatch is an error rather than a denial: it means a policy answered a
    different question than the one asked, which is a defect to surface.
    """
    if not isinstance(decision, AuthorizationDecision):
        raise AuthorizationMismatch("authorizer must return an AuthorizationDecision")
    if decision.request_digest != request.digest or decision.attempt != request.attempt:
        raise AuthorizationMismatch("decision does not match the request it was asked about")
    if decision.expires_at is not None:
        try:
            expired = (now or datetime.now(UTC)) >= decision.expires_at
        except TypeError:  # a naive value smuggled past construction
            raise AuthorizationMismatch("authorization expiry is not timezone-aware") from None
        if expired:
            raise AuthorizationMismatch("authorization decision expired before dispatch")
    return decision.allowed


Authorizer = Callable[[InvocationRequest], AuthorizationDecision | Awaitable[AuthorizationDecision]]


class AllowRegisteredTools:
    """The explicit default: the application delegated its whole registered catalog.

    This is not resource authorization. It exists so an absent policy is a named
    choice with a name, rather than a missing check.
    """

    policy_version = "allow_registered_tools"

    def __call__(self, request: InvocationRequest) -> AuthorizationDecision:
        return request.allow("registered tool", policy_version=self.policy_version)


def build_request(
    *,
    run_id: str,
    call: ToolCall,
    tool: Tool,
    origin: ToolOrigin | None,
    context: AuthorizationContext,
) -> InvocationRequest:
    """Snapshot a call. Raises ValueError/TypeError if its arguments are not plain JSON."""
    trusted: dict[str, Any] = {}
    if origin is None and tool.idempotent:
        trusted["idempotency_key"] = idempotency_key(run_id, call.id)
    return InvocationRequest(
        run_id=run_id,
        call_id=call.id,
        tool=call.name,
        original_name=origin.original_name if origin else tool.name,
        source=origin.source if origin else None,
        endpoint=origin.endpoint if origin else None,
        fingerprint=origin.fingerprint if origin else tool.fingerprint,
        arguments_json=canonical_json(call.arguments),
        trusted_json=canonical_json(trusted),
        context=context,
    )


def decision_data(request: InvocationRequest, decision: AuthorizationDecision) -> dict[str, Any]:
    """The loggable facts of a decision: no arguments, nothing a policy could leak."""
    return {
        "call_id": request.call_id,
        "attempt": request.attempt,
        "allowed": decision.allowed,
        "reason": decision.reason,
        "policy_version": decision.policy_version,
        "request_digest": decision.request_digest,
        "principal_id": request.context.principal_id,
        "tenant_id": request.context.tenant_id,
    }


__all__ = [
    "AllowRegisteredTools",
    "ApprovalDecision",
    "Approver",
    "AuthorizationContext",
    "AuthorizationDecision",
    "AuthorizationMismatch",
    "Authorizer",
    "AuthorizerFailed",
    "InvocationRequest",
    "build_request",
    "canonical_json",
    "decision_data",
    "verify_decision",
]
