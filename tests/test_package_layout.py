import importlib
import pickle
from pathlib import Path

import pytest

import emissary
from emissary import eval, harness, llm, storage

PACKAGE = Path(__file__).parents[1] / "src" / "emissary"


def test_public_convenience_api_survives_the_module_reorganization():
    assert emissary.call_model
    assert emissary.call_tool
    assert emissary.Agent
    assert emissary.run
    assert emissary.Tool
    assert emissary.evaluate
    assert emissary.SQLiteRunStore


def test_each_subpackage_exposes_a_cohesive_convenience_surface():
    assert llm.call_model and llm.parse_spec and llm.ToolDefinition
    assert harness.Agent and harness.run and harness.Tool
    assert eval.evaluate and eval.EvaluationScenario
    assert storage.SQLiteRunStore and storage.serialize_run


def test_implementation_is_grouped_by_single_responsibility():
    expected = {
        "llm/model.py",
        "llm/provider.py",
        "llm/messages.py",
        "llm/prompt.py",
        "llm/decision.py",
        "llm/credentials.py",
        "llm/streaming.py",
        "llm/calls.py",
        "llm/retry.py",
        "llm/wire/anthropic.py",
        "llm/wire/gemini.py",
        "llm/wire/openai_compatible.py",
        "llm/wire/thinking.py",
        "harness/execution/runner.py",
        "harness/execution/machine.py",
        "harness/execution/effects.py",
        "harness/conversation/projection.py",
        "harness/tooling/tools.py",
        "harness/tooling/sources.py",
        "harness/tooling/preparation.py",
        "harness/conversation/context.py",
        "harness/policy.py",
        "harness/state.py",
        "harness/conversation/events.py",
        "eval/evaluation.py",
        "eval/replay.py",
        "storage/persistence.py",
    }

    files = {str(path.relative_to(PACKAGE)) for path in PACKAGE.rglob("*.py")}
    assert expected <= files


def test_flat_implementation_modules_are_removed():
    moved = {
        "agent.py",
        "context.py",
        "decision.py",
        "evaluation.py",
        "events.py",
        "messages.py",
        "model.py",
        "persistence.py",
        "policy.py",
        "provider.py",
        "runner.py",
        "state.py",
        "tools.py",
    }

    assert not (moved & {path.name for path in PACKAGE.glob("*.py")})


LEGACY_MODULES = {
    "runner": "execution.runner",
    "machine": "execution.machine",
    "effects": "execution.effects",
    "tools": "tooling.tools",
    "sources": "tooling.sources",
    "preparation": "tooling.preparation",
    "context": "conversation.context",
    "events": "conversation.events",
    "projection": "conversation.projection",
}


@pytest.mark.parametrize("legacy,canonical", LEGACY_MODULES.items())
def test_legacy_deep_imports_share_the_canonical_module(legacy, canonical):
    old = importlib.import_module(f"emissary.harness.{legacy}")
    new = importlib.import_module(f"emissary.harness.{canonical}")
    assert old is new
    assert getattr(harness, legacy) is new


def test_legacy_runner_monkeypatches_reach_the_actual_driver(monkeypatch):
    old = importlib.import_module("emissary.harness.runner")
    new = importlib.import_module("emissary.harness.execution.runner")
    sentinel = object()
    monkeypatch.setattr(old, "_perform", sentinel)
    assert new._perform is sentinel


def test_run_records_still_load_old_pickled_tool_results():
    from emissary.harness.tooling.tools import ToolResult

    # Protocol 0 records the class by its old module path, as pre-refactor callers did.
    original = ToolResult("success", "done")
    payload = pickle.dumps(original, protocol=0).replace(
        b"emissary.harness.tooling.tools", b"emissary.harness.tools"
    )
    assert pickle.loads(payload) == original
