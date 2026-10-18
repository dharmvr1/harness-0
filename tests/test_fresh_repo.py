"""Starting on a new or non-git folder: set it up after asking, and never commit anything for the user."""
import json
import subprocess
from types import SimpleNamespace

import pytest
from ce_agent.main import run_agent
from ce_agent.workspace import ensure_repo


def sh(*cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True).stdout


def never_ask(_):
    raise AssertionError("should not have asked anything")

def says(answer):
    asked = []
    def ask(question):
        asked.append(question)
        return answer
    ask.asked = asked
    return ask

def has_commits(path):
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, capture_output=True).returncode == 0


def test_missing_folder_is_created_after_asking(tmp_path):
    target = tmp_path / "brand" / "new"
    ask = says("y")
    ensure_repo(str(target), ask)
    assert len(ask.asked) == 1 and "does not exist" in ask.asked[0]
    assert (target / ".git").exists() and not has_commits(target)        # initialised, nothing committed

def test_declining_changes_nothing(tmp_path):
    target = tmp_path / "nope"
    with pytest.raises(RuntimeError, match="Cancelled"):
        ensure_repo(str(target), says("n"))
    assert not target.exists()

def test_folder_with_files_gets_git_init_but_no_commit_and_files_untouched(tmp_path):
    (tmp_path / "app.py").write_text("print('mine')\n")
    (tmp_path / ".env").write_text("SECRET=1\n")
    ask = says("y")
    ensure_repo(str(tmp_path), ask)
    assert "not a git repo" in ask.asked[0]
    assert (tmp_path / ".git").exists() and not has_commits(tmp_path)
    assert (tmp_path / "app.py").read_text() == "print('mine')\n" and (tmp_path / ".env").exists()

def test_declining_git_init_leaves_the_folder_alone(tmp_path):
    (tmp_path / "app.py").write_text("x\n")
    with pytest.raises(RuntimeError, match="Cancelled"):
        ensure_repo(str(tmp_path), says("n"))
    assert not (tmp_path / ".git").exists()

def test_existing_repos_are_left_alone_even_without_commits_or_with_changes(tmp_path):
    sh("git", "init", "-q", cwd=tmp_path)
    (tmp_path / "uncommitted.py").write_text("x\n")
    ensure_repo(str(tmp_path), never_ask)
    assert not has_commits(tmp_path)


class Brain:
    model = "fake"
    def __init__(self, *script):
        self.script = list(script)
    def think(self, messages, tools):
        return self.script.pop(0)

def write_call(path, content):
    c = SimpleNamespace(id="c", function=SimpleNamespace(
        name="write_file", arguments=json.dumps({"file_path": path, "content": content})))
    return SimpleNamespace(content=None, tool_calls=[c])


def test_whole_agent_run_in_a_folder_that_does_not_exist_yet(tmp_path, monkeypatch):
    """The original bug report: fresh project, 'create server.py'."""
    target = tmp_path / "test_project"
    answers = iter(["y", "a"])          # 1st: create the project, 2nd: accept the agent's work
    monkeypatch.setattr("builtins.input", lambda _="": next(answers))
    run_agent(str(target), "create server.py", unsafe=True,
              brain=Brain(write_call("server.py", "print('server')\n"), SimpleNamespace(content="done", tool_calls=None)))
    assert (target / "server.py").read_text() == "print('server')\n"
    assert not has_commits(target)                                       # we never commit for you
    assert "server.py" in sh("git", "status", "--porcelain", cwd=target)


def test_whole_agent_run_on_an_existing_non_git_folder(tmp_path, monkeypatch):
    (tmp_path / "app.py").write_text("print('v1')\n")
    answers = iter(["y", "a"])          # git init? yes; accept? yes
    monkeypatch.setattr("builtins.input", lambda _="": next(answers))
    edit = SimpleNamespace(content=None, tool_calls=[SimpleNamespace(id="c", function=SimpleNamespace(
        name="edit_file", arguments=json.dumps({"file_path": "app.py", "old_text": "v1", "new_text": "v2"})))])
    run_agent(str(tmp_path), "bump version", unsafe=True,
              brain=Brain(edit, SimpleNamespace(content="done", tool_calls=None)))
    assert (tmp_path / "app.py").read_text() == "print('v2')\n"
