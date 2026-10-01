"""MCP adapter contracts that need no server: naming, schemas, results, discovery."""

from dataclasses import dataclass, field

import mcp_types as mt
import pytest

from emissary.harness.tools import Tool
from emissary.llm.decision import ToolDefinition
from emissary.llm.provider import parse_spec
from emissary.llm.wire.openai_compatible import _request
from emissary.mcp import MCPToolPolicy, MCPToolset, Stdio, StreamableHTTP
from emissary.mcp.client import _Connection
from emissary.mcp.naming import default_alias, validate_alias
from emissary.mcp.results import normalize_result
from emissary.mcp.schema import UnsupportedSchema, check_schema

POLICY = MCPToolPolicy(api_scope="external")


def test_portable_names_keep_the_readable_namespace_prefix():
    assert default_alias("claims", "lookup_claim") == "claims__lookup_claim"


def test_unportable_names_get_a_stable_hash_so_lookalikes_cannot_collide():
    dotted, underscored = default_alias("s", "a.b"), default_alias("s", "a_b")

    assert dotted != underscored
    assert dotted == default_alias("s", "a.b")
    assert all(len(alias) <= 64 for alias in (dotted, default_alias("s", "x" * 200)))


def test_explicit_aliases_must_be_provider_portable():
    assert validate_alias("ok_name-1") == "ok_name-1"
    with pytest.raises(ValueError):
        validate_alias("has space")


def test_namespace_and_policy_are_validated_at_construction():
    with pytest.raises(ValueError, match="namespace"):
        MCPToolset("bad name", Stdio("x"), POLICY)
    with pytest.raises(ValueError, match="boundary"):
        MCPToolPolicy(api_scope="none")  # type: ignore[arg-type]


def test_configuration_reprs_never_show_credentials():
    http = StreamableHTTP("https://user:pw@host.example:8443/mcp?key=abc", auth=object())
    stdio = Stdio("srv", env={"API_KEY": "hunter2"})

    assert "hunter2" not in repr(stdio) and "pw" not in http.endpoint
    assert http.endpoint == "https://host.example:8443/mcp"
    assert "object" not in repr(http)
    assert "pw" not in repr(http) and "abc" not in repr(http)


def test_structured_output_uses_the_declared_dialect_too():
    schema = {"$schema": "http://json-schema.org/draft-07/schema#", "dependencies": {"a": ["b"]}}

    out = normalize_result("t", _result([], structured_content={"a": 1}), schema)

    assert out.status == "error"


def test_only_local_references_and_known_dialects_are_accepted():
    check_schema(
        {
            "type": "object",
            "$defs": {"a": {"type": "string"}},
            "properties": {"x": {"$ref": "#/$defs/a"}},
        },
        where="t",
    )
    check_schema(
        {"$schema": "http://json-schema.org/draft-07/schema#", "type": "object"}, where="t"
    )

    with pytest.raises(UnsupportedSchema, match="non-local"):
        check_schema({"properties": {"x": {"$ref": "https://evil.example/s.json"}}}, where="t")
    with pytest.raises(UnsupportedSchema, match="dialect|\\$schema"):
        check_schema({"$schema": "http://json-schema.org/draft-04/schema#"}, where="t")


def _result(content, **kwargs):
    return mt.CallToolResult(content=content, **kwargs)


def test_text_and_structured_content_become_one_observation():
    out = normalize_result(
        "t", _result([mt.TextContent(text="hi")], structured_content={"n": 1}), None
    )

    assert out.status == "success"
    assert out.content == {"text": ["hi"], "structured": {"n": 1}}


def test_tool_reported_errors_stay_model_visible_errors():
    out = normalize_result("t", _result([mt.TextContent(text="no such id")], is_error=True), None)

    assert out.status == "error" and out.content["text"] == ["no such id"]


def test_structured_output_is_validated_against_the_declared_schema():
    schema = {"type": "object", "required": ["n"], "properties": {"n": {"type": "integer"}}}

    bad = normalize_result("t", _result([], structured_content={"n": "x"}), schema)
    good = normalize_result("t", _result([], structured_content={"n": 2}), schema)

    assert bad.status == "error" and "invalid output" in bad.summary
    assert good.status == "success"


def test_an_error_result_need_not_satisfy_the_success_schema():
    schema = {"type": "object", "required": ["n"]}

    out = normalize_result("t", _result([mt.TextContent(text="x")], is_error=True), schema)

    assert out.status == "error" and "invalid output" not in out.summary


def test_binary_blocks_are_reported_never_dropped_or_inlined():
    blocks = [
        mt.TextContent(text="see image"),
        mt.ImageContent(data="QUJDRA==", mime_type="image/png"),
        mt.EmbeddedResource(
            resource=mt.BlobResourceContents(uri="file:///a", blob="QUJD", mime_type="x/y")
        ),
    ]

    out = normalize_result("t", _result(blocks), None)

    assert out.status == "warning"
    assert [u["type"] for u in out.content["unsupported"]] == ["image", "resource"]
    assert "QUJD" not in repr(out)


def test_resource_links_are_references_and_embedded_text_is_preserved():
    blocks = [
        mt.ResourceLink(name="doc", uri="file:///a", mime_type="text/plain"),
        mt.EmbeddedResource(resource=mt.TextResourceContents(uri="file:///b", text="body")),
    ]

    out = normalize_result("t", _result(blocks), None)

    assert out.content["resources"][0]["uri"] == "file:///a"
    assert out.content["text"] == ["body"]


@dataclass
class FakeClient:
    pages: list
    tools_capability: bool = True
    requested: list = field(default_factory=list)

    @property
    def server_capabilities(self):
        return mt.ServerCapabilities(tools=mt.ToolsCapability() if self.tools_capability else None)

    async def list_tools(self, *, cursor=None):
        self.requested.append(cursor)
        page = self.pages[len(self.requested) - 1]
        if isinstance(page, Exception):
            raise page
        return page


def _tool(name):
    return mt.Tool(name=name, description="d", input_schema={"type": "object"})


def _page(names, cursor=None):
    return mt.ListToolsResult(tools=[_tool(n) for n in names], next_cursor=cursor)


def _toolset(**kwargs):
    return MCPToolset("srv", Stdio("x"), POLICY, **kwargs)


async def test_every_discovery_page_is_fetched_in_order():
    client = FakeClient([_page(["a"], "p2"), _page(["b"])])

    bindings = await _Connection(_toolset(), client).list_tools()

    assert [b.tool.name for b in bindings] == ["srv__a", "srv__b"]
    assert client.requested == [None, "p2"]


async def test_a_repeated_cursor_is_an_error_not_an_infinite_loop():
    client = FakeClient([_page(["a"], "same"), _page(["b"], "same")])

    with pytest.raises(RuntimeError, match="cursor"):
        await _Connection(_toolset(), client).list_tools()


async def test_a_failed_page_discards_the_partial_catalog():
    client = FakeClient([_page(["a"], "p2"), ConnectionError("boom")])

    with pytest.raises(ConnectionError):
        await _Connection(_toolset(), client).list_tools()


async def test_a_server_without_the_tools_capability_is_refused():
    with pytest.raises(RuntimeError, match="does not offer tools"):
        await _Connection(_toolset(), FakeClient([], tools_capability=False)).list_tools()


async def test_include_exclude_and_explicit_aliases_and_per_tool_policy_apply():
    client = FakeClient([_page(["a", "b", "c"])])
    toolset = _toolset(
        exclude_tools=("c",),
        aliases={"a": "alpha"},
        tool_policies={"b": MCPToolPolicy(api_scope="internal", approval="always")},
    )

    by_name = {b.tool.name: b for b in await _Connection(toolset, client).list_tools()}

    assert set(by_name) == {"alpha", "srv__b"}
    assert by_name["alpha"].origin.original_name == "a"
    assert by_name["srv__b"].tool.approval == "always"
    assert by_name["alpha"].tool.approval == "never"


async def test_server_annotations_cannot_grant_approval_exemptions():
    tool = mt.Tool(
        name="a",
        input_schema={"type": "object"},
        annotations=mt.ToolAnnotations(read_only_hint=True, idempotent_hint=True),
    )
    client = FakeClient([mt.ListToolsResult(tools=[tool])])

    toolset = MCPToolset("srv", Stdio("x"), MCPToolPolicy(api_scope="external", approval="always"))

    (binding,) = await _Connection(toolset, client).list_tools()

    assert binding.tool.approval == "always"
    assert binding.tool.idempotent is False and binding.tool.max_attempts == 1


async def test_unsupported_schemas_fail_naming_the_tool():
    bad = mt.Tool(
        name="a", input_schema={"type": "object", "properties": {"x": {"$ref": "http://x/y"}}}
    )

    with pytest.raises(UnsupportedSchema, match="srv/a"):
        await _Connection(_toolset(), FakeClient([mt.ListToolsResult(tools=[bad])])).list_tools()


def test_strict_mode_is_skipped_for_tools_that_ask_not_to_use_it():
    spec = parse_spec("openai:gpt-5")
    schema = {"type": "object", "properties": {}}
    tools = (
        ToolDefinition("plain", "d", schema),
        ToolDefinition("remote", "d", schema, strict=False),
    )

    sent = _request(spec, "s", (), tools, None)["tools"]

    assert sent[0]["function"]["strict"] is True
    assert "strict" not in sent[1]["function"]


def test_prepared_tools_ask_the_wire_not_to_force_strict():
    tool = Tool("t", "d", {"type": "object"}, lambda: 1, api_scope="external", strict_schema=False)

    assert tool.definition.strict is False


def test_unresolved_local_references_are_rejected_at_discovery():
    with pytest.raises(UnsupportedSchema, match="unresolved"):
        check_schema({"type": "object", "$ref": "#/$defs/missing"}, where="t")
    check_schema(
        {"type": "object", "$defs": {"a": {}}, "properties": {"x": {"$ref": "#/$defs/a"}}},
        where="t",
    )
    with pytest.raises(UnsupportedSchema, match="unresolved"):
        check_schema({"type": "object", "properties": {"x": {"$ref": "#foo"}}}, where="t")
