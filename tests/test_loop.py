"""Tests for the agent loop using a fake brain, so no API key or network is needed."""
import json
import subprocess
from types import SimpleNamespace

import pytest
import ce_agent.session as session_module
from ce_agent.context import shorten
from ce_agent.main import run_agent


def sh(*cmd, cwd):
    subprocess.run(cmd, cwd=cwd, check=True, capture_output=True)

def branches(repo):
    return subprocess.run(["git", "branch"], cwd=repo, capture_output=True, text=True).stdout

@pytest.fixture
def repo(tmp_path):
    sh("git", "init", "-q", "-b", "main", cwd=tmp_path)
    (tmp_path / "calc.py").write_text("print(1 + 1)\n")
    sh("git", "add", "-A", cwd=tmp_path)
    sh("git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init", cwd=tmp_path)
    return tmp_path


def tool_call(name, **args):
    call = SimpleNamespace(id="c1", function=SimpleNamespace(name=name, arguments=json.dumps(args)))
    return SimpleNamespace(content=None, tool_calls=[call])

def final(text="all done"):
    return SimpleNamespace(content=text, tool_calls=None)


class FakeBrain:
    """Plays back scripted replies. An Exception in the script is raised instead of returned."""
    model = "fake"
    def __init__(self, *script):
        self.script = list(script)
    def think(self, messages, tools):
        step = self.script.pop(0)
        if isinstance(step, BaseException):
            raise step
        return step


def answer(monkeypatch, *choices):
    it = iter(choices)
    def fake_input(prompt=""):
        c = next(it)
        if isinstance(c, BaseException):
            raise c
        return c
    monkeypatch.setattr("builtins.input", fake_input)


def test_agent_edits_runs_and_user_accepts(repo, monkeypatch):
    brain = FakeBrain(tool_call("write_file", file_path="calc.py", content="print(2 + 2)\n"),
                      tool_call("run_shell", command="python3 calc.py"),
                      final())
    answer(monkeypatch, "a")
    run_agent(str(repo), "change it", unsafe=True, brain=brain)
    assert (repo / "calc.py").read_text() == "print(2 + 2)\n"


def test_reject_leaves_repo_alone(repo, monkeypatch):
    brain = FakeBrain(tool_call("write_file", file_path="calc.py", content="oops\n"), final())
    answer(monkeypatch, "r")
    run_agent(str(repo), "change it", unsafe=True, brain=brain)
    assert (repo / "calc.py").read_text() == "print(1 + 1)\n"
    assert "agent/" not in branches(repo)


def test_api_error_mid_run_keeps_the_work(repo, monkeypatch):
    brain = FakeBrain(tool_call("write_file", file_path="new.py", content="x = 1\n"),
                      RuntimeError("rate limited"))
    answer(monkeypatch, "k")
    run_agent(str(repo), "t", unsafe=True, brain=brain)
    assert "agent/" in branches(repo)
    assert not (repo / "new.py").exists()


def test_ctrl_c_in_loop_goes_to_review(repo, monkeypatch):
    brain = FakeBrain(tool_call("write_file", file_path="new.py", content="x = 1\n"), KeyboardInterrupt())
    answer(monkeypatch, "a")
    run_agent(str(repo), "t", unsafe=True, brain=brain)
    assert (repo / "new.py").exists()


def test_ctrl_c_at_prompt_keeps_branch(repo, monkeypatch):
    brain = FakeBrain(tool_call("write_file", file_path="new.py", content="x = 1\n"), final())
    answer(monkeypatch, KeyboardInterrupt())
    run_agent(str(repo), "t", unsafe=True, brain=brain)
    assert "agent/" in branches(repo) and not (repo / "new.py").exists()


def test_bad_tool_calls_do_not_crash_the_loop(repo, monkeypatch):
    bad_json = SimpleNamespace(content=None, tool_calls=[
        SimpleNamespace(id="c1", function=SimpleNamespace(name="write_file", arguments="{not json"))])
    brain = FakeBrain(bad_json, tool_call("no_such_tool"), tool_call("read_file", file_path="../../etc/passwd"), final())
    answer(monkeypatch)
    run_agent(str(repo), "t", unsafe=True, brain=brain)    # no changes -> no prompt, no crash
    assert "agent/" not in branches(repo)


def test_turn_limit_stops_a_looping_agent(repo, monkeypatch):
    monkeypatch.setattr(session_module, "MAX_TURNS", 3)
    brain = FakeBrain(*[tool_call("list_files") for _ in range(10)])
    answer(monkeypatch)
    run_agent(str(repo), "t", unsafe=True, brain=brain)
    assert len(brain.script) == 7                          # stopped after exactly 3 turns


def test_shorten_keeps_start_and_end():
    text = "START" + "x" * 10000 + "THE ERROR IS HERE"
    out = shorten(text, 100)
    assert out.startswith("START") and out.endswith("THE ERROR IS HERE") and len(out) < 200
    assert shorten("short") == "short"


def test_missing_api_key_fails_before_creating_anything(repo, monkeypatch):
    monkeypatch.setenv("PROVIDER", "groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        run_agent(str(repo), "t", unsafe=True)
    assert not (repo / ".agent" / "worktrees").exists() or not list((repo / ".agent" / "worktrees").iterdir())
    assert "agent/" not in branches(repo)
