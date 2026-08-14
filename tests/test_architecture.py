import ast
from pathlib import Path

PACKAGE = Path(__file__).parents[1] / "src" / "emissary"


def test_provider_sdks_are_imported_only_by_wire_adapters():
    violations = []
    for path in PACKAGE.rglob("*.py"):
        if path.parent.name == "wire":
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = {node.module.split(".")[0]}
            else:
                continue
            if names & {"anthropic", "openai"}:
                violations.append(str(path.relative_to(PACKAGE)))

    assert violations == []


def test_harness_core_does_not_depend_on_provider_registry_or_wires():
    core = ["agent.py", "context.py", "events.py", "policy.py", "runner.py", "state.py", "tools.py"]
    violations = []
    for name in core:
        tree = ast.parse((PACKAGE / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in {
                "provider",
                "wire",
                ".provider",
                ".wire",
            }:
                violations.append(name)

    assert violations == []
