from pathlib import Path

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
        "llm/wire/anthropic.py",
        "llm/wire/gemini.py",
        "llm/wire/openai_compatible.py",
        "llm/wire/thinking.py",
        "harness/runner.py",
        "harness/projection.py",
        "harness/tools.py",
        "harness/context.py",
        "harness/policy.py",
        "harness/state.py",
        "harness/events.py",
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
