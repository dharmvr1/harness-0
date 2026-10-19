"""Tests for the trajectory log."""
import json
import subprocess
from types import SimpleNamespace

import pytest
from ce_agent.main import run_agent
from ce_agent.trace import Trace, find_run, show


def sh(*cmd, cwd):
    subprocess.run(cmd, cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    sh("git", "init", "-q", "-b", "main", cwd=tmp_path)
    (tmp_path / "calc.py").write_text("print(1 + 1)\n")
    sh("git", "add", "-A", cwd=tmp_path)
    sh("git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init", cwd=tmp_path)
    return tmp_path


def call(name, **args):
    c = SimpleNamespace(id="c1", function=SimpleNamespace(name=name, arguments=json.dumps(args)))
    return SimpleNamespace(content="thinking...", tool_calls=[c])


class FakeBrain:
    model = "fake-model"
    def __init__(self, *script):
        self.script = list(script)
        self.last_usage = None
    def think(self, messages, tools):
        step = self.script.pop(0)
        if isinstance(step, BaseException):
            raise step
        self.last_usage = {"prompt_tokens": 100, "completion_tokens": 10}
        return step


def events(repo):
    (path,) = (repo / ".agent" / "runs").glob("*.jsonl")
    return [json.loads(l) for l in path.read_text().splitlines()]


def test_full_run_is_logged(repo, monkeypatch):
    brain = FakeBrain(call("write_file", file_path="calc2.py", content="print(2)\n"),
                      call("run_shell", command="python3 calc2.py"),
                      SimpleNamespace(content="finished", tool_calls=None))
    monkeypatch.setattr("builtins.input", lambda _="": "a")
    run_agent(str(repo), "make it two", unsafe=True, brain=brain)

    ev = events(repo)
    assert [e["kind"] for e in ev] == ["run_start", "user_message", "model_call", "tool_call", "model_call",
                                       "tool_call", "model_call", "prompt_end", "run_end"]
    start, end = ev[0], ev[-1]
    assert start["task"] == "make it two" and start["model"] == "fake-model" and start["mode"] == "unsafe"
    assert ev[2]["usage"] == {"prompt_tokens": 100, "completion_tokens": 10}
    assert ev[2]["tool_calls"][0]["name"] == "write_file"
    assert ev[3]["status"] == "ok" and ev[3]["args"]["file_path"] == "calc2.py"
    assert "2" in ev[5]["result"]                        # output of the python run
    assert end["reason"] == "done" and end["decision"] == "accept" and end["turns"] == 3
    assert end["total_prompt_tokens"] == 300 and end["total_completion_tokens"] == 30
    assert "calc2.py" in end["files_changed"]


def test_errors_are_logged_with_the_model_error_text(repo, monkeypatch):
    brain = FakeBrain(call("read_file", file_path="missing.py"), RuntimeError("rate limited"))
    monkeypatch.setattr("builtins.input", lambda _="": "k")
    run_agent(str(repo), "t", unsafe=True, brain=brain)
    ev = events(repo)
    tool = [e for e in ev if e["kind"] == "tool_call"][0]
    assert tool["status"] == "error" and "does not exist" in tool["result"]
    assert ev[-1]["reason"] == "error: RuntimeError: rate limited"      # the run still ended properly in the log


def test_ctrl_c_and_turn_limit_reasons(repo, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _="": "r")
    run_agent(str(repo), "t", unsafe=True, brain=FakeBrain(KeyboardInterrupt()))
    assert events(repo)[-1]["reason"] == "interrupted"


def test_log_is_written_line_by_line(tmp_path):
    t = Trace(tmp_path / "runs" / "x.jsonl")
    t.log("run_start", task="a")
    # not closed yet (as in a crash), but the line is already on disk
    assert json.loads((tmp_path / "runs" / "x.jsonl").read_text())["task"] == "a"
    t.close()


def test_show_prints_a_readable_timeline(repo, monkeypatch):
    brain = FakeBrain(call("read_file", file_path="calc.py"), SimpleNamespace(content="ok", tool_calls=None))
    monkeypatch.setattr("builtins.input", lambda _="": "r")
    run_agent(str(repo), "look at calc", unsafe=True, brain=brain)
    text = show(find_run(str(repo)))
    assert "task: look at calc" in text and "[turn 0]" in text
    assert "read_file" in text and "100 in / 10 out" in text and "you chose: no_changes" in text


def test_find_run_by_id_and_errors(repo, monkeypatch):
    with pytest.raises(FileNotFoundError, match="No runs"):
        find_run(str(repo))
    monkeypatch.setattr("builtins.input", lambda _="": "r")
    run_agent(str(repo), "t", unsafe=True, brain=FakeBrain(SimpleNamespace(content="ok", tool_calls=None)))
    run_id = find_run(str(repo)).stem
    assert find_run(str(repo), run_id).stem == run_id
    with pytest.raises(FileNotFoundError, match="No run nope"):
        find_run(str(repo), "nope")
