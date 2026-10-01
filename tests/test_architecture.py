import ast
from pathlib import Path

PACKAGE = Path(__file__).parents[1] / "src" / "emissary"

PROVIDER_SDKS = {"anthropic", "openai", "google", "httpx"}
"""Every vendor package the wire layer may touch — and `httpx`, which the
typesafe wire speaks directly for want of a vendor SDK.

Listed as a set so that admitting a wire forces this to be updated; before
Gemini landed this check named only two SDKs, and `google.genai` could have
leaked anywhere without failing a test.
"""

CREDENTIAL_PROBE = "llm/credentials.py"
"""The one reviewed exception: `GoogleADC.available()` imports `google.auth`.

ADR-0009 exists to keep SDK *request and response vocabulary* out of the
neutral layer. A credential probe carries none of it — and it cannot move into
`wire/`, because `key_present` is reached from `provider.py`, which must not
import a wire. Narrow by design: only this file, only this module.
"""


def test_provider_sdks_are_imported_only_by_wire_adapters():
    violations = []
    for path in PACKAGE.rglob("*.py"):
        if path.parent.name == "wire":
            continue
        if path.relative_to(PACKAGE).as_posix() == CREDENTIAL_PROBE:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = {node.module.split(".")[0]}
            else:
                continue
            if names & PROVIDER_SDKS:
                violations.append(str(path.relative_to(PACKAGE)))

    assert violations == []


def test_the_credential_probe_exception_stays_narrow():
    """The exemption above is a hole in ADR-0009; this pins its exact size.

    If `credentials.py` ever grows a second SDK import, or starts building a
    client rather than answering yes/no, that is a design change and should
    fail here rather than pass unnoticed.
    """
    tree = ast.parse((PACKAGE / CREDENTIAL_PROBE).read_text())
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name.split(".")[0] in PROVIDER_SDKS
    }

    assert imported == {"google.auth"}


def test_harness_core_does_not_depend_on_provider_registry_or_wires():
    violations = []
    for path in (PACKAGE / "harness").rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and (node.module.endswith("provider") or ".wire" in node.module)
            ):
                violations.append(path.name)

    assert violations == []


def test_llm_layer_does_not_depend_on_harness_evaluation_or_storage():
    violations = []
    for path in (PACKAGE / "llm").rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.split(".")[0]
                in {
                    "harness",
                    "eval",
                    "storage",
                    "memory",
                }
            ):
                violations.append(str(path.relative_to(PACKAGE)))

    assert violations == []


def test_harness_core_runs_without_memory():
    """Memory builds on the loop, never the reverse: an agent with no memory must
    not pay for it, and the loop must not grow opinions about what to remember."""
    violations = []
    for path in (PACKAGE / "harness").rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and "memory" in node.module:
                violations.append(path.name)

    assert violations == []


def test_nothing_in_the_library_depends_on_the_improvement_loop():
    """`improve` ships filesystem tools and runs subprocesses (ADR-0029).

    It is opt-in: if the harness or any other layer imported it, every consumer
    would carry an improver's workspace tools whether they asked for them or not.
    """
    importers = []
    for path in PACKAGE.rglob("*.py"):
        relative = path.relative_to(PACKAGE)
        if relative.parts[0] == "improve":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module and "improve" in node.module:
                importers.append(str(relative))

    assert importers == []


MCP_SDK_MODULES = {"mcp", "mcp_types", "httpx2"}
MCP_ADAPTER = "mcp"


def _imported_roots(path: Path) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


def test_the_mcp_sdk_and_its_http_stack_are_imported_only_by_the_mcp_adapter():
    violations = [
        str(path.relative_to(PACKAGE))
        for path in PACKAGE.rglob("*.py")
        if path.relative_to(PACKAGE).parts[0] != MCP_ADAPTER
        and _imported_roots(path) & MCP_SDK_MODULES
    ]

    assert violations == []


def test_nothing_in_the_library_depends_on_the_mcp_adapter():
    """The harness sees only the `ToolSource` protocol. Were it to import the
    adapter, every install would need the optional SDK."""
    importers = []
    for path in PACKAGE.rglob("*.py"):
        relative = path.relative_to(PACKAGE)
        if relative.parts[0] == MCP_ADAPTER:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module:
                absolute = node.module.startswith("emissary.mcp") or (
                    node.level > 0 and node.module.split(".")[0] == MCP_ADAPTER
                )
                if absolute:
                    importers.append(str(relative))

    assert importers == []


def test_importing_emissary_does_not_load_the_optional_mcp_extra():
    import subprocess
    import sys

    probe = (
        "import sys, emissary; "
        "loaded = [m for m in sys.modules if m.split('.')[0] in ('mcp', 'mcp_types', 'httpx2')]; "
        "assert not loaded, loaded; assert 'emissary.mcp' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr


def test_library_imports_use_canonical_harness_modules():
    from importlib.util import resolve_name

    legacy = {
        f"emissary.harness.{name}"
        for name in (
            "runner",
            "machine",
            "effects",
            "tools",
            "sources",
            "preparation",
            "context",
            "events",
            "projection",
        )
    }
    violations = []
    for path in PACKAGE.rglob("*.py"):
        relative = path.relative_to(PACKAGE).with_suffix("")
        parts = ("emissary", *relative.parts)
        package = ".".join(parts[:-1])
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module:
                name = resolve_name("." * node.level + node.module, package)
                if name in legacy:
                    violations.append(str(relative))
            elif isinstance(node, ast.Import):
                if any(alias.name in legacy for alias in node.names):
                    violations.append(str(relative))
    assert violations == []
