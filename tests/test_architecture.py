import ast
from pathlib import Path

PACKAGE = Path(__file__).parents[1] / "src" / "emissary"

PROVIDER_SDKS = {"anthropic", "openai", "google"}
"""Every vendor package the wire layer may touch.

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
    for path in (PACKAGE / "harness").glob("*.py"):
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
                }
            ):
                violations.append(str(path.relative_to(PACKAGE)))

    assert violations == []
