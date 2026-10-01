"""Tool sources: preparation, routing, lifecycle (mcp-support-plan phase 1).

Everything here uses fake sources, so it also proves the neutral layer works
without any MCP SDK installed.
"""

from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import pytest

from emissary.harness.agent import Agent
from emissary.harness.events import InMemoryEventSink
from emissary.harness.preparation import PreparationError, prepare
from emissary.harness.runner import arun, run
from emissary.harness.sources import ToolBinding, ToolOrigin
from emissary.harness.state import RunStatus, StopReason
from emissary.harness.tools import Tool
from emissary.llm.decision import FinalOutput, ModelResult, ToolCall, ToolCalls, Usage

SCHEMA = {"type": "object", "properties": {"id": {"type": "string"}}}


@dataclass
class AsyncScripted:
    decisions: list
    seen: list = field(default_factory=list)

    async def __call__(self, *, system, messages, tools=(), settings=None):
        self.seen.append((messages, tools))
        return ModelResult(self.decisions.pop(0), "fake", "scripted", Usage(1, 1))


@dataclass
class FakeSource:
    namespace: str
    tool_names: tuple[str, ...] = ("lookup",)
    required: bool = True
    fail_listing: bool = False
    calls: list = field(default_factory=list)
    opened: int = 0
    closed: int = 0

    @asynccontextmanager
    async def open(self):
        self.opened += 1
        try:
            yield self
        finally:
            self.closed += 1

    async def list_tools(self):
        if self.fail_listing:
            raise RuntimeError("secret-token-123 leaked in transport error")
        return tuple(self._binding(name) for name in reversed(self.tool_names))

    def _binding(self, name):
        async def invoke(**arguments):
            self.calls.append((name, arguments))
            return {"from": self.namespace, "echo": arguments}

        alias = f"{self.namespace}__{name}"
        return ToolBinding(
            Tool(alias, "remote", dict(SCHEMA), invoke, api_scope="external"),
            ToolOrigin(self.namespace, name, f"fake://{self.namespace}", "fp-" + name),
        )


def _call(name, **arguments):
    return ToolCalls((ToolCall("c1", name, arguments),))


async def test_a_prepared_source_tool_is_discovered_called_and_closed():
    source = FakeSource("claims")
    agent = Agent("a", "i", toolsets=(source,))
    caller = AsyncScripted([_call("claims__lookup", id="7"), FinalOutput(text="done")])

    result = await arun(agent, "go", caller=caller)

    assert result.status is RunStatus.COMPLETED
    assert source.calls == [("lookup", {"id": "7"})]
    assert (source.opened, source.closed) == (1, 1)
    assert [t.name for t in caller.seen[0][1]] == ["claims__lookup"]
    kinds = [event.kind for event in result.events]
    assert kinds[:3] == ["run_started", "tool_preparation_started", "tools_prepared"]
    assert kinds.count("run_started") == 1


async def test_catalog_is_ordered_by_alias_not_by_server_discovery_order():
    source = FakeSource("s", tool_names=("b", "a", "c"))
    agent = Agent("a", "i", toolsets=(source,))

    async with prepare(agent) as prepared:
        names = [d.name for d in prepared.registry.definitions]

    assert names == ["s__a", "s__b", "s__c"]


async def test_direct_and_source_tools_share_one_registry_and_the_same_machine():
    direct = Tool("sum", "Add.", {"type": "object"}, lambda a, b: a + b, api_scope="none")
    source = FakeSource("claims")
    agent = Agent("a", "i", tools=(direct,), toolsets=(source,))
    caller = AsyncScripted(
        [
            ToolCalls((ToolCall("1", "claims__lookup", {"id": "x"}),)),
            ToolCalls((ToolCall("2", "sum", {"a": 1, "b": 2}),)),
            FinalOutput(text="ok"),
        ]
    )

    result = await arun(agent, "go", caller=caller)

    assert result.status is RunStatus.COMPLETED
    completed = [
        e.data["result"]["content"] for e in result.events if e.kind == "tool_call_completed"
    ]
    assert completed == [{"from": "claims", "echo": {"id": "x"}}, 3]


async def test_source_tools_never_receive_an_undeclared_idempotency_key():
    source = FakeSource("s")
    binding = (await _list(source))[0]
    idem = Tool(
        binding.tool.name, "d", dict(SCHEMA), binding.tool.execute,
        api_scope="external", idempotent=True, max_attempts=2,
    )  # fmt: skip

    @dataclass
    class OneTool(FakeSource):
        async def list_tools(self):
            return (ToolBinding(idem, binding.origin),)

    one = OneTool("s")
    result = await arun(
        Agent("a", "i", toolsets=(one,)),
        "go",
        caller=AsyncScripted([_call("s__lookup", id="1"), FinalOutput(text="x")]),
    )

    assert result.status is RunStatus.COMPLETED
    assert source.calls == [("lookup", {"id": "1"})]


async def _list(source):
    async with source.open():
        return await source.list_tools()


async def test_async_direct_tools_are_awaited():
    async def fetch(id):
        return {"id": id}

    tool = Tool("fetch", "d", dict(SCHEMA), fetch, api_scope="internal")
    result = await arun(
        Agent("a", "i", tools=(tool,)),
        "go",
        caller=AsyncScripted([_call("fetch", id="9"), FinalOutput(text="x")]),
    )

    assert result.status is RunStatus.COMPLETED
    done = next(e for e in result.events if e.kind == "tool_call_completed")
    assert done.data["result"]["content"] == {"id": "9"}


def test_sync_run_rejects_sources_and_coroutine_tools_with_a_pointer_to_arun():
    async def coro(**_):
        return 1

    with pytest.raises(ValueError, match="arun"):
        run(Agent("a", "i", toolsets=(FakeSource("s"),)), "go", caller=lambda **_: None)
    with pytest.raises(ValueError, match="arun"):
        run(
            Agent("a", "i", tools=(Tool("c", "d", {"type": "object"}, coro, api_scope="none"),)),
            "go",
            caller=lambda **_: None,
        )


async def test_a_required_source_failure_stops_the_run_without_leaking_detail():
    good, bad = FakeSource("good"), FakeSource("bad", fail_listing=True)
    sink = InMemoryEventSink()

    result = await arun(
        Agent("a", "i", toolsets=(good, bad)),
        "go",
        caller=AsyncScripted([]),
        event_sink=sink,
    )

    assert (result.status, result.stop_reason) == (RunStatus.FAILED, StopReason.PREPARATION_FAILED)
    assert "secret-token-123" not in repr(result.events)
    assert (good.opened, good.closed, bad.opened, bad.closed) == (1, 1, 1, 1)
    assert [e.kind for e in sink.events] == [
        "run_started",
        "tool_preparation_started",
        "tool_preparation_failed",
        "run_stopped",
    ]
    assert [e.sequence for e in sink.events] == [1, 2, 3, 4]


async def test_an_optional_source_is_omitted_visibly():
    ok, flaky = FakeSource("ok"), FakeSource("flaky", required=False, fail_listing=True)

    async with prepare(Agent("a", "i", toolsets=(ok, flaky))) as prepared:
        assert [d.name for d in prepared.registry.definitions] == ["ok__lookup"]
        assert prepared.omitted == ("flaky",)

    assert flaky.closed == 1


async def test_name_collisions_fail_preparation_instead_of_picking_a_winner():
    direct = Tool("s__lookup", "d", {"type": "object"}, lambda: 1, api_scope="none")

    with pytest.raises(PreparationError, match="collides"):
        async with prepare(Agent("a", "i", tools=(direct,), toolsets=(FakeSource("s"),))):
            pass


async def test_duplicate_namespaces_are_rejected():
    with pytest.raises(ValueError, match="namespace"):
        async with prepare(Agent("a", "i", toolsets=(FakeSource("s"), FakeSource("s")))):
            pass


async def test_a_shared_prepared_run_serves_several_runs_over_one_connection():
    source = FakeSource("claims")
    agent = Agent("a", "i", toolsets=(source,))

    async with prepare(agent) as prepared:
        for _ in range(2):
            caller = AsyncScripted([_call("claims__lookup", id="1"), FinalOutput(text="x")])
            result = await arun(agent, "go", caller=caller, prepared=prepared)
            assert result.status is RunStatus.COMPLETED
        assert source.closed == 0

    assert (source.opened, source.closed) == (1, 1)


async def test_prepared_schemas_are_detached_from_the_source_objects():
    source = FakeSource("s")
    async with prepare(Agent("a", "i", toolsets=(source,))) as prepared:
        schema = prepared.registry.resolve("s__lookup").input_schema
        schema["properties"]["injected"] = {}

        again = (await _list(source))[0].tool.input_schema

    assert "injected" not in again["properties"]


async def test_a_source_run_replays_with_recorded_authorization_and_no_live_source():
    from emissary.eval.replay import (
        RecordedAuthorizer,
        ReplayModelCaller,
        ReplayToolExecutor,
        trajectory,
    )
    from emissary.harness.policy import AuthorizationContext

    ctx = AuthorizationContext("u", "t")
    live = FakeSource("claims")
    original = await arun(
        Agent("a", "i", toolsets=(live,)),
        "go",
        caller=AsyncScripted([_call("claims__lookup", id="7"), FinalOutput(text="done")]),
        authorizer=lambda request: request.allow("ok", policy_version="v1"),
        authorization_context=ctx,
    )

    offline = FakeSource("claims")  # stands in for a server that is not reachable
    replayed = await arun(
        Agent("a", "i", toolsets=(offline,)),
        "go",
        caller=_AsAsync(ReplayModelCaller(original)),
        executor=ReplayToolExecutor(original),
        authorizer=RecordedAuthorizer(original),
        authorization_context=ctx,
    )

    assert trajectory(replayed) == trajectory(original)
    assert offline.calls == []


@dataclass
class _AsAsync:
    inner: object

    async def __call__(self, **kwargs):
        return self.inner(**kwargs)


async def test_a_catalog_prepared_for_one_agent_cannot_run_another():
    privileged = Agent("admin", "i", toolsets=(FakeSource("admin"),))
    bare = Agent("bare", "i")

    async with prepare(privileged) as prepared:
        with pytest.raises(ValueError, match="not built for agent"):
            await arun(bare, "go", caller=AsyncScripted([]), prepared=prepared)


async def test_a_catalog_prepared_for_one_identity_cannot_serve_another():
    from emissary.harness.policy import AuthorizationContext

    agent = Agent("a", "i", toolsets=(FakeSource("s"),))
    alice = AuthorizationContext("alice", "t1")
    allow = lambda request: request.allow()

    async with prepare(agent, context=alice) as prepared:
        for other in (AuthorizationContext("bob", "t1"), AuthorizationContext("alice", "t2")):
            with pytest.raises(ValueError, match="principal or tenant"):
                await arun(
                    agent, "go", caller=AsyncScripted([]), prepared=prepared,
                    authorizer=allow, authorization_context=other,
                )  # fmt: skip
        ok = await arun(
            agent, "go", caller=AsyncScripted([FinalOutput(text="x")]), prepared=prepared,
            authorizer=allow, authorization_context=alice,
        )  # fmt: skip

    assert ok.status is RunStatus.COMPLETED
