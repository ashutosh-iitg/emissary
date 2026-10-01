"""Real servers: identity, cleanup, timeouts and cancellation (mcp-support-plan §9)."""

import asyncio
import os
import sys

import pytest
from mcp.server.mcpserver import MCPServer
from mcp_support import FILES_SERVER, BearerAuth, ScriptedModel, build_claims_server, serve_http

from emissary.harness.agent import Agent
from emissary.harness.preparation import prepare
from emissary.harness.runner import arun
from emissary.harness.state import RunStatus, StopReason
from emissary.llm.decision import FinalOutput, ToolCall, ToolCalls
from emissary.mcp import MCPToolPolicy, MCPToolset, Stdio, StreamableHTTP

EXTERNAL = MCPToolPolicy(api_scope="external")


def _http(namespace, url, **kwargs):
    return MCPToolset(namespace, StreamableHTTP(url, auth=BearerAuth()), EXTERNAL, **kwargs)


def _stdio(namespace, pid_file, **kwargs):
    env = {"FILES_PID_FILE": str(pid_file)}
    return MCPToolset(
        namespace, Stdio(sys.executable, (str(FILES_SERVER),), env=env), EXTERNAL, **kwargs
    )


def _gone(pid_file) -> bool:
    try:
        os.kill(int(pid_file.read_text()), 0)
    except ProcessLookupError:
        return True
    return False


async def test_two_servers_exposing_the_same_tool_name_stay_distinct():
    first, second = [], []
    async with (
        serve_http(build_claims_server(first)) as url_a,
        serve_http(build_claims_server(second)) as url_b,
    ):
        agent = Agent("a", "i", toolsets=(_http("north", url_a), _http("south", url_b)))
        model = ScriptedModel(
            [
                ToolCalls((ToolCall("1", "south__lookup_claim", {"customer_id": "c"}),)),
                FinalOutput(text="x"),
            ]
        )

        result = await arun(agent, "go", caller=model)

    assert result.status is RunStatus.COMPLETED
    assert first == [] and second == [("lookup_claim", {"customer_id": "c"})]


async def test_origin_not_alias_parsing_decides_the_target():
    calls = []
    async with serve_http(build_claims_server(calls)) as url:
        toolset = _http("claims", url, aliases={"lookup_claim": "look__up__claim"})
        async with prepare(Agent("a", "i", toolsets=(toolset,))) as prepared:
            origin = prepared.origin_of("look__up__claim")

    assert (origin.source, origin.original_name) == ("claims", "lookup_claim")


async def test_a_failing_later_source_still_reaps_the_earlier_subprocess(tmp_path):
    pid_file = tmp_path / "pid"
    broken = _http("broken", "http://127.0.0.1:9/mcp", connect_timeout_seconds=2)

    result = await arun(
        Agent("a", "i", toolsets=(_stdio("files", pid_file), broken)),
        "go",
        caller=ScriptedModel([]),
    )

    assert (result.status, result.stop_reason) == (RunStatus.FAILED, StopReason.PREPARATION_FAILED)
    assert _gone(pid_file)
    assert [e.kind for e in result.events][-2:] == ["tool_preparation_failed", "run_stopped"]


async def test_an_optional_unreachable_source_is_omitted_and_the_run_proceeds():
    calls = []
    async with serve_http(build_claims_server(calls)) as url:
        agent = Agent(
            "a",
            "i",
            toolsets=(
                _http("up", url),
                _http("down", "http://127.0.0.1:9/mcp", required=False, connect_timeout_seconds=2),
            ),
        )
        result = await arun(agent, "go", caller=ScriptedModel([FinalOutput(text="x")]))

    prepared = next(e for e in result.events if e.kind == "tools_prepared")
    assert result.status is RunStatus.COMPLETED and prepared.data["omitted"] == ["down"]


async def test_a_wrong_credential_fails_preparation_without_echoing_it():
    rejected = []
    async with serve_http(build_claims_server([]), rejected) as url:
        toolset = MCPToolset("claims", StreamableHTTP(url), EXTERNAL, connect_timeout_seconds=5)

        result = await arun(Agent("a", "i", toolsets=(toolset,)), "go", caller=ScriptedModel([]))

    assert result.stop_reason is StopReason.PREPARATION_FAILED and rejected


def _slow_server() -> MCPServer:
    server = MCPServer("slow")

    @server.tool()
    async def wait() -> str:
        """Sleeps past any reasonable deadline."""
        await asyncio.sleep(30)
        return "late"

    return server


async def test_a_slow_call_times_out_as_a_non_retryable_model_visible_error():
    async with serve_http(_slow_server()) as url:
        toolset = _http("slow", url, call_timeout_seconds=0.3)
        agent = Agent("a", "i", toolsets=(toolset,))
        model = ScriptedModel(
            [ToolCalls((ToolCall("1", "slow__wait", {}),)), FinalOutput(text="gave up")]
        )

        result = await arun(agent, "go", caller=model)

    outcome = next(e for e in result.events if e.kind == "tool_call_completed").data["result"]
    assert outcome["timed_out"] is True and outcome["retryable"] is False
    assert result.status is RunStatus.COMPLETED


async def test_the_run_deadline_stops_a_hanging_call_and_closes_connections(tmp_path):
    pid_file = tmp_path / "pid"
    async with serve_http(_slow_server()) as url:
        agent = Agent(
            "a", "i",
            toolsets=(_http("slow", url), _stdio("files", pid_file)),
            limits=_short_limits(),
        )  # fmt: skip
        model = ScriptedModel([ToolCalls((ToolCall("1", "slow__wait", {}),))])

        result = await arun(agent, "go", caller=model)

    assert (result.status, result.stop_reason) == (RunStatus.STOPPED, StopReason.MAX_DURATION)
    assert _gone(pid_file)


def _short_limits():
    from emissary.harness.agent import RunLimits

    return RunLimits(max_duration_seconds=1.5)


async def test_cancellation_during_a_call_cleans_up_and_still_ends_the_log(tmp_path):
    pid_file = tmp_path / "pid"
    async with serve_http(_slow_server()) as url:
        from emissary.harness.events import InMemoryEventSink

        sink = InMemoryEventSink()
        agent = Agent("a", "i", toolsets=(_http("slow", url), _stdio("files", pid_file)))
        model = ScriptedModel([ToolCalls((ToolCall("1", "slow__wait", {}),))])
        task = asyncio.create_task(arun(agent, "go", caller=model, event_sink=sink))
        while not any(e.kind == "tool_call_started" for e in sink.events):
            await asyncio.sleep(0.05)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert sink.events[-1].kind == "run_stopped" and sink.events[-1].data["status"] == "cancelled"
    assert _gone(pid_file)


async def test_sync_callers_get_a_clear_pointer_to_arun(tmp_path):
    from emissary.harness.runner import run

    with pytest.raises(ValueError, match="arun"):
        run(Agent("a", "i", toolsets=(_stdio("f", tmp_path / "p"),)), "go", caller=lambda **_: None)
