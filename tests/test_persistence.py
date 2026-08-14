from dataclasses import dataclass

from emissary.agent import Agent
from emissary.decision import FinalOutput, ModelResult, Usage
from emissary.persistence import SQLiteRunStore
from emissary.runner import run


@dataclass
class Caller:
    def __call__(self, **kwargs):
        return ModelResult(FinalOutput(text="done"), "fake", "fake", Usage(2, 1))


def test_sqlite_store_round_trips_a_complete_run_without_runtime_dependencies(tmp_path):
    original = run(Agent("a", "i"), "task", caller=Caller())
    store = SQLiteRunStore(tmp_path / "runs.sqlite3")

    store.save(original)
    restored = store.load(original.run_id)

    assert restored == original
    assert store.load("missing") is None
