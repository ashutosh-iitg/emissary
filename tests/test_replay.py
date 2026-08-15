"""A recorded run must replay to the same trajectory (ADR-0015).

These are the tests that fail when control flow or projection changes in a way
scripted unit tests cannot see, because they compare a whole real run rather
than one branch of it.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from emissary.eval.replay import (
    ReplayExhausted,
    ReplayModelCaller,
    ReplayToolExecutor,
    trajectory,
)
from emissary.harness.agent import Agent, RunLimits
from emissary.harness.runner import run
from emissary.harness.state import RunStatus, StopReason
from emissary.harness.tools import Tool
from emissary.llm.decision import FinalOutput, ModelResult, ToolCall, ToolCalls, Usage
from emissary.storage.persistence import SCHEMA_VERSION, deserialize_run, serialize_run

FIXTURES = Path(__file__).parent / "fixtures" / "replay"

AGENT = Agent(
    "calculator",
    "Use the tool, then answer.",
    tools=(Tool("add", "Add two integers.", {"type": "object"}, lambda a, b: {"sum": a + b}),),
)
TASK = "what is 2 + 3?"


@dataclass
class ScriptedCaller:
    decisions: list
    messages_seen: list = field(default_factory=list)

    def __call__(self, *, system, messages, tools=(), settings=None):
        self.messages_seen.append(messages)
        return ModelResult(self.decisions.pop(0), "fake", "scripted", Usage(2, 1), "stop")


def record():
    caller = ScriptedCaller(
        [ToolCalls((ToolCall("one", "add", {"a": 2, "b": 3}),)), FinalOutput(text="5")]
    )
    return run(AGENT, TASK, caller=caller)


def test_a_recorded_run_replays_to_an_identical_trajectory():
    original = record()

    replayed = run(
        AGENT,
        TASK,
        caller=ReplayModelCaller(original),
        executor=ReplayToolExecutor(original),
    )

    assert trajectory(replayed) == trajectory(original)
    assert replayed.messages == original.messages
    assert (replayed.status, replayed.stop_reason) == (original.status, original.stop_reason)
    assert replayed.usage == original.usage


def test_replay_survives_a_persistence_round_trip():
    """The recording is only useful if it can be stored and reloaded."""
    original = record()

    restored = deserialize_run(serialize_run(original))
    replayed = run(
        AGENT, TASK, caller=ReplayModelCaller(restored), executor=ReplayToolExecutor(restored)
    )

    assert trajectory(replayed) == trajectory(original)
    assert replayed.messages == original.messages


def test_the_committed_fixture_still_replays():
    """Executable documentation of what a run looks like. If a schema change
    strands this file, that must be a visible reviewed act — not a skip."""
    payload = (FIXTURES / "tool_turn.json").read_text()
    assert json.loads(payload)["schema_version"] == SCHEMA_VERSION

    recorded = deserialize_run(payload)
    replayed = run(
        AGENT, TASK, caller=ReplayModelCaller(recorded), executor=ReplayToolExecutor(recorded)
    )

    assert replayed.status is RunStatus.COMPLETED
    assert replayed.output == FinalOutput(text="5")
    assert [kind for _, kind in trajectory(replayed)] == [kind for _, kind in trajectory(recorded)]


def test_a_run_that_outlives_its_recording_fails_loudly():
    """A harness change that adds a turn must not replay as a quiet success —
    silence here would lose the whole point of replay."""
    tight = Agent(AGENT.name, AGENT.instructions, tools=AGENT.tools, limits=RunLimits(max_turns=1))
    recorded = run(
        tight,
        TASK,
        caller=ScriptedCaller([ToolCalls((ToolCall("one", "add", {"a": 2, "b": 3}),))]),
    )
    assert recorded.stop_reason is StopReason.MAX_TURNS

    with pytest.raises(ReplayExhausted, match="the recording holds"):
        run(
            AGENT,
            TASK,
            caller=ReplayModelCaller(recorded),
            executor=ReplayToolExecutor(recorded),
        )
