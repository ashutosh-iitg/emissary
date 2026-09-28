from pathlib import Path

import pytest

from emissary.improve.workspace import Workspace


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    (tmp_path / "agent").mkdir()
    (tmp_path / "agent" / "tools.py").write_text("def convert():\n    return 1\n")
    (tmp_path / "evals.py").write_text("HOLDOUT = ['secret scenario']\n")
    (tmp_path / "README.md").write_text("docs\n")
    return Workspace(tmp_path, editable=["agent/**"], protected=["evals.py"])


def call(workspace: Workspace, name: str, **arguments):
    tool = next(tool for tool in workspace.tools() if tool.name == name)
    return tool.execute(**arguments)


def test_the_improver_can_change_only_editable_paths(workspace):
    assert (
        call(workspace, "write_file", path="agent/tools.py", content="fixed\n").status == "success"
    )
    assert (workspace.root / "agent" / "tools.py").read_text() == "fixed\n"

    refused = call(workspace, "write_file", path="README.md", content="rewritten\n")

    assert refused.status == "error"
    assert (workspace.root / "README.md").read_text() == "docs\n"


def test_the_improver_cannot_escape_the_worktree(workspace, tmp_path):
    outside = tmp_path.parent / "outside.txt"

    result = call(workspace, "write_file", path="../outside.txt", content="x")

    assert result.status == "error"
    assert not outside.exists()


def test_graders_and_holdout_are_invisible_not_merely_read_only(workspace):
    # An improver that can read the holdout scenarios has turned them into a
    # search set; one that can edit the grader can pass by rewriting it.
    assert call(workspace, "read_file", path="evals.py").status == "error"
    assert call(workspace, "write_file", path="evals.py", content="PASS = True").status == "error"
    assert "secret" not in call(workspace, "search", pattern="secret").content
    assert "evals.py" not in call(workspace, "list_files").content


def test_a_protected_path_stays_protected_even_inside_an_editable_glob(tmp_path):
    (tmp_path / "agent").mkdir()
    workspace = Workspace(tmp_path, editable=["agent/**"], protected=["agent/grader.py"])

    result = call(workspace, "write_file", path="agent/grader.py", content="PASS = True")

    assert result.status == "error"
    assert not (tmp_path / "agent" / "grader.py").exists()


def test_the_git_directory_is_out_of_reach(workspace):
    (workspace.root / ".git").mkdir()
    (workspace.root / ".git" / "config").write_text("[core]\n")

    assert call(workspace, "read_file", path=".git/config").status == "error"
