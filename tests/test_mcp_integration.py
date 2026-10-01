"""Direct tools and both MCP transports through one machine (mcp-support-plan §9).

A scripted model drives real fixture servers — a loopback Streamable HTTP server
behind a bearer check, and a stdio subprocess — so a failure isolates harness
behaviour from model variation.
"""

import os
import sys

from mcp_support import (
    FILES_SERVER,
    TOKEN,
    BearerAuth,
    ScriptedModel,
    build_claims_server,
    serve_http,
)

from emissary.harness.agent import Agent
from emissary.harness.policy import AuthorizationContext
from emissary.harness.runner import arun
from emissary.harness.state import RunStatus
from emissary.harness.tools import Tool
from emissary.llm.decision import FinalOutput, ToolCall, ToolCalls
from emissary.mcp import MCPToolPolicy, MCPToolset, Stdio, StreamableHTTP


def _step(call_id, tool, **arguments):
    return ToolCalls((ToolCall(call_id, tool, arguments),))


def _process_is_gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


async def test_direct_http_mcp_stdio_mcp_and_direct_again_in_one_run(tmp_path):
    claim_calls, rejected, policy_log = [], [], []
    pid_file = tmp_path / "files.pid"

    lookup_customer = Tool(
        "lookup_customer", "Customer id for a name.", {"type": "object"},
        lambda name: {"customer_id": "cust-5"}, api_scope="internal",
    )  # fmt: skip
    calculate_total = Tool(
        "calculate_total", "Add numbers.", {"type": "object"},
        lambda amount, extra: {"total": amount + extra}, api_scope="none",
    )  # fmt: skip

    async with serve_http(build_claims_server(claim_calls), rejected) as url:
        claims = MCPToolset(
            "claims",
            StreamableHTTP(url, auth=BearerAuth()),
            MCPToolPolicy(api_scope="external"),
            include_tools=("lookup_claim", "list_documents"),
        )
        files = MCPToolset(
            "files",
            Stdio(sys.executable, (str(FILES_SERVER),), env={"FILES_PID_FILE": str(pid_file)}),
            MCPToolPolicy(api_scope="internal"),
            include_tools=("read_document",),
        )
        agent = Agent(
            "claims_assistant", "Review the claim.",
            tools=(lookup_customer, calculate_total), toolsets=(claims, files),
        )  # fmt: skip
        model = ScriptedModel(
            [
                _step("1", "lookup_customer", name="Ada"),
                _step("2", "claims__lookup_claim", customer_id="cust-5"),
                _step("3", "files__read_document", document_id="doc-9"),
                _step("4", "calculate_total", amount=1200, extra=5),
                FinalOutput(text="total 1205"),
            ]
        )

        def authorizer(request):
            policy_log.append((request.tool, request.source, request.attempt))
            return request.allow()

        result = await arun(
            agent, "Review claim for Ada", caller=model, authorizer=authorizer,
            authorization_context=AuthorizationContext("user-42", "tenant-7"),
        )  # fmt: skip

    assert result.status is RunStatus.COMPLETED
    assert claim_calls == [("lookup_claim", {"customer_id": "cust-5"})]
    assert rejected == []  # every request carried the credential

    completed = [e for e in result.events if e.kind == "tool_call_completed"]
    assert [e.data["call_id"] for e in completed] == ["1", "2", "3", "4"]
    assert completed[1].data["result"]["content"]["structured"]["document_id"] == "doc-9"
    assert "Water damage" in completed[2].data["result"]["content"]["text"][0]
    assert completed[3].data["result"]["content"] == {"total": 1205}

    expected = [
        ("lookup_customer", None),
        ("claims__lookup_claim", "claims"),
        ("files__read_document", "files"),
        ("calculate_total", None),
    ]
    # Asked once for the batch and once before the attempt, per call.
    assert [(tool, source) for tool, source, _ in policy_log] == [
        pair for pair in expected for _ in range(2)
    ]
    order = [e.kind for e in result.events if e.kind.startswith(("authorization", "tool_call_s"))]
    assert order == ["authorization_resolved", "tool_call_started", "authorization_resolved"] * 4
    kinds = [e.kind for e in result.events]
    assert kinds.count("run_started") == 1 and kinds[-1] == "run_completed"
    # Each later model turn saw the earlier observation, in order.
    assert len(model.contexts) == 5 and len(model.contexts[4]) == 1 + 2 * 4

    assert TOKEN not in repr(result.events) and TOKEN not in repr(model.contexts)
    assert _process_is_gone(int(pid_file.read_text()))
