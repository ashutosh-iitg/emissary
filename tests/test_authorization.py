"""Independent authorization and exact-request binding (mcp-support-plan §6).

The point of every test here is a property a bypass would break: authorization
runs even when approval says `never`, it covers a whole batch before anything
executes, it is repeated for every attempt, and what is dispatched is the
snapshot that was authorized rather than what a callback left behind.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import pytest

from emissary.harness.agent import Agent
from emissary.harness.policy import (
    ApprovalDecision,
    AuthorizationContext,
    AuthorizationDecision,
)
from emissary.harness.runner import arun, run
from emissary.harness.state import RunStatus, StopReason
from emissary.harness.tools import Tool, ToolResult
from emissary.llm.decision import FinalOutput, ModelResult, ToolCall, ToolCalls, Usage

CTX = AuthorizationContext("user-42", "tenant-7")


@dataclass
class Scripted:
    decisions: list

    def __call__(self, *, system, messages, tools=(), settings=None):
        return ModelResult(self.decisions.pop(0), "fake", "scripted", Usage(1, 1))


@dataclass
class Recorder:
    runs: list = field(default_factory=list)

    def tool(self, name="act", **overrides):
        def act(**arguments):
            self.runs.append((name, arguments))
            return {"ok": True}

        return Tool(name, "d", {"type": "object"}, act, api_scope="internal", **overrides)


def _calls(*calls):
    return ToolCalls(tuple(ToolCall(f"c{i}", name, args) for i, (name, args) in enumerate(calls)))


def _agent(*tools):
    return Agent("a", "i", tools=tuple(tools))


def test_denial_blocks_execution_even_when_approval_is_never():
    rec = Recorder()
    caller = Scripted([_calls(("act", {"note": "SECRET-ARG"}))])

    result = run(
        _agent(rec.tool(approval="never")),
        "go",
        caller=caller,
        authorizer=lambda request: request.deny("not your tenant", policy_version="v9"),
        authorization_context=CTX,
    )

    assert (result.status, result.stop_reason) == (
        RunStatus.STOPPED,
        StopReason.AUTHORIZATION_DENIED,
    )
    assert rec.runs == []
    denied = next(e for e in result.events if e.kind == "authorization_resolved")
    assert denied.data["allowed"] is False and denied.data["policy_version"] == "v9"
    assert "SECRET-ARG" not in repr(denied.data)  # arguments never reach the log


def test_approval_cannot_override_a_denial():
    rec = Recorder()
    caller = Scripted([_calls(("act", {}))])

    result = run(
        _agent(rec.tool(approval="always")),
        "go",
        caller=caller,
        approver=lambda call, tool: ApprovalDecision.ALLOW,
        authorizer=lambda request: request.deny(),
        authorization_context=CTX,
    )

    assert result.stop_reason is StopReason.AUTHORIZATION_DENIED
    assert rec.runs == []


def test_a_denied_call_anywhere_in_a_batch_stops_the_whole_batch_before_effects():
    rec = Recorder()
    caller = Scripted([_calls(("act", {"n": 1}), ("other", {"n": 2}))])

    def policy(request):
        return request.deny() if request.tool == "other" else request.allow()

    result = run(
        _agent(rec.tool("act"), rec.tool("other")),
        "go",
        caller=caller,
        authorizer=policy,
        authorization_context=CTX,
    )

    assert result.stop_reason is StopReason.AUTHORIZATION_DENIED
    assert rec.runs == []


def test_every_attempt_is_reauthorized_with_its_own_attempt_number():
    attempts = []

    def flaky(**_):
        return ToolResult("error", "boom", retryable=True)

    tool = Tool(
        "act", "d", {"type": "object"}, flaky,
        api_scope="internal", idempotent=True, max_attempts=3, retry_backoff_seconds=0,
    )  # fmt: skip

    def policy(request):
        attempts.append((request.attempt, request.digest))
        return request.allow()

    run(
        _agent(tool),
        "go",
        caller=Scripted([_calls(("act", {})), FinalOutput(text="x")]),
        authorizer=policy,
        authorization_context=CTX,
    )

    batch_then_attempts = [a for a, _ in attempts]
    assert batch_then_attempts == [1, 1, 2, 3]
    assert len({digest for _, digest in attempts}) == 1  # one logical request


def test_a_policy_that_raises_fails_closed_without_executing():
    rec = Recorder()

    def broken(request):
        raise RuntimeError("db password=hunter2")

    result = run(
        _agent(rec.tool()),
        "go",
        caller=Scripted([_calls(("act", {}))]),
        authorizer=broken,
        authorization_context=CTX,
    )

    assert (result.status, result.stop_reason) == (
        RunStatus.FAILED,
        StopReason.AUTHORIZATION_ERROR,
    )
    assert rec.runs == [] and "hunter2" not in repr(result.events)


def test_a_decision_about_a_different_request_is_an_error_not_an_allow():
    rec = Recorder()
    result = run(
        _agent(rec.tool()),
        "go",
        caller=Scripted([_calls(("act", {}))]),
        authorizer=lambda request: AuthorizationDecision(True, "0" * 64, request.attempt),
        authorization_context=CTX,
    )

    assert result.stop_reason is StopReason.AUTHORIZATION_ERROR
    assert rec.runs == []


def test_an_expired_decision_is_not_honoured():
    rec = Recorder()
    past = datetime.now(UTC) - timedelta(seconds=1)

    result = run(
        _agent(rec.tool()),
        "go",
        caller=Scripted([_calls(("act", {}))]),
        authorizer=lambda request: request.allow(expires_at=past),
        authorization_context=CTX,
    )

    assert result.stop_reason is StopReason.AUTHORIZATION_ERROR
    assert rec.runs == []


def test_callbacks_cannot_change_what_is_dispatched():
    rec = Recorder()
    seen = []

    def approver(call, tool):
        call.arguments["amount"] = 1_000_000
        return ApprovalDecision.ALLOW

    def policy(request):
        view = request.arguments
        view["amount"] = 999
        seen.append(request.arguments["amount"])
        return request.allow()

    run(
        _agent(rec.tool(approval="always")),
        "go",
        caller=Scripted([_calls(("act", {"amount": 5})), FinalOutput(text="x")]),
        approver=approver,
        authorizer=policy,
        authorization_context=CTX,
    )

    assert rec.runs == [("act", {"amount": 5})]
    assert seen == [5, 5]  # once for the batch, once before the attempt


def test_principal_and_tenant_come_from_the_application_never_from_arguments():
    requests = []

    def policy(request):
        requests.append(request)
        return request.allow()

    run(
        _agent(Recorder().tool()),
        "go",
        caller=Scripted(
            [
                _calls(("act", {"tenant_id": "attacker", "principal_id": "root"})),
                FinalOutput(text="x"),
            ]
        ),
        authorizer=policy,
        authorization_context=CTX,
    )

    assert {(r.context.principal_id, r.context.tenant_id) for r in requests} == {
        ("user-42", "tenant-7")
    }
    assert requests[0].arguments["tenant_id"] == "attacker"  # data, not identity


def test_the_digest_binds_arguments_and_identity():
    digests = []

    def policy(request):
        digests.append(request.digest)
        return request.allow()

    for args, ctx in (({"a": 1}, CTX), ({"a": 2}, CTX), ({"a": 1}, AuthorizationContext("other"))):
        run(
            _agent(Recorder().tool()),
            "go",
            caller=Scripted([_calls(("act", args)), FinalOutput(text="x")]),
            authorizer=policy,
            authorization_context=ctx,
        )

    per_run = [digests[0:2], digests[2:4], digests[4:6]]
    assert all(first == second for first, second in per_run)
    assert len({first for first, _ in per_run}) == 3


def test_an_authorizer_without_a_context_is_refused_up_front():
    with pytest.raises(ValueError, match="authorization_context"):
        run(_agent(), "go", caller=Scripted([]), authorizer=lambda r: r.allow())


async def test_an_async_authorizer_is_awaited_by_arun():
    rec = Recorder()

    async def policy(request):
        return request.allow("async")

    result = await arun(
        _agent(rec.tool()),
        "go",
        caller=_AsyncScripted([_calls(("act", {"x": 1})), FinalOutput(text="x")]),
        authorizer=policy,
        authorization_context=CTX,
    )

    assert result.status is RunStatus.COMPLETED
    assert rec.runs == [("act", {"x": 1})]


@dataclass
class _AsyncScripted:
    decisions: list

    async def __call__(self, *, system, messages, tools=(), settings=None):
        return ModelResult(self.decisions.pop(0), "fake", "scripted", Usage(1, 1))


def test_without_an_authorizer_legacy_logs_are_unchanged():
    result = run(
        _agent(Recorder().tool()),
        "go",
        caller=Scripted([_calls(("act", {})), FinalOutput(text="x")]),
    )

    assert not [e for e in result.events if e.kind.startswith("authorization")]


async def test_an_async_policy_that_raises_fails_closed_with_a_terminal_event():
    rec = Recorder()

    async def broken(request):
        raise RuntimeError("password=hunter2")

    result = await arun(
        _agent(rec.tool()),
        "go",
        caller=_AsyncScripted([_calls(("act", {}))]),
        authorizer=broken,
        authorization_context=CTX,
    )

    assert (result.status, result.stop_reason) == (
        RunStatus.FAILED,
        StopReason.AUTHORIZATION_ERROR,
    )
    assert result.events[-1].kind == "run_stopped"
    assert rec.runs == [] and "hunter2" not in repr(result.events)


async def test_an_async_policy_that_times_out_itself_is_an_authorization_error_not_a_crash():
    async def slow(request):
        raise TimeoutError("policy backend")

    result = await arun(
        _agent(Recorder().tool()),
        "go",
        caller=_AsyncScripted([_calls(("act", {}))]),
        authorizer=slow,
        authorization_context=CTX,
    )

    assert result.stop_reason is StopReason.AUTHORIZATION_ERROR


async def test_a_naive_expiry_fails_closed_instead_of_crashing():
    rec = Recorder()
    naive = datetime.now() + timedelta(minutes=5)  # noqa: DTZ005

    result = await arun(
        _agent(rec.tool()),
        "go",
        caller=_AsyncScripted([_calls(("act", {}))]),
        authorizer=lambda request: request.allow(expires_at=naive),
        authorization_context=CTX,
    )

    assert result.stop_reason is StopReason.AUTHORIZATION_ERROR
    assert result.events[-1].kind == "run_stopped" and rec.runs == []
