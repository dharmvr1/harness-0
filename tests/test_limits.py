"""Hard limits for a session (turns, money) and the status line."""
import json
import subprocess
from types import SimpleNamespace

import ce_agent.session as session_module
import pytest
from ce_agent.main import run_session


def sh(*cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True).stdout

@pytest.fixture
def repo(tmp_path):
    sh("git", "init", "-q", "-b", "main", cwd=tmp_path)
    (tmp_path / "a.txt").write_text("hello\n")
    sh("git", "add", "-A", cwd=tmp_path)
    sh("git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init", cwd=tmp_path)
    return tmp_path

def call(name, arguments, call_id):
    return SimpleNamespace(content=None, tool_calls=[SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))])

class LoopingBrain:
    """Always asks for another tool call, and reports a fixed token usage per call."""
    model = "fake"
    def __init__(self, tokens_in=0, tokens_out=0):
        self.n, self.last_usage = 0, None
        self.tokens = {"prompt_tokens": tokens_in, "completion_tokens": tokens_out}
    def think(self, messages, tools):
        self.n += 1
        self.last_usage = dict(self.tokens)
        return call("list_files", {}, f"c{self.n}")

def run(repo, monkeypatch, brain, prompts=("go",)):
    it = iter(prompts)
    def fake_input(question=""):
        if "[l]ater" in question:
            return "r"
        try:
            return next(it)
        except StopIteration:
            raise EOFError()
    monkeypatch.setattr("builtins.input", fake_input)
    run_session(str(repo), unsafe=True, brain=brain)

def end_reasons(repo):
    path = max((repo / ".agent" / "runs").glob("*.jsonl"), key=lambda f: f.stat().st_mtime)
    return [e["reason"] for e in map(json.loads, path.read_text().splitlines()) if e["kind"] == "prompt_end"]


def test_the_session_stops_at_the_turn_limit(repo, monkeypatch, capsys):
    monkeypatch.setattr(session_module, "SESSION_MAX_TURNS", 3)
    brain = LoopingBrain()
    run(repo, monkeypatch, brain)
    assert brain.n == 3 and end_reasons(repo) == ["turn_budget"]
    assert "session turn limit reached (3" in capsys.readouterr().out

def test_the_turn_limit_covers_the_whole_session_not_one_prompt(repo, monkeypatch):
    monkeypatch.setattr(session_module, "SESSION_MAX_TURNS", 5)
    monkeypatch.setattr(session_module, "MAX_TURNS", 3)
    brain = LoopingBrain()
    run(repo, monkeypatch, brain, prompts=("one", "two", "three"))
    assert brain.n == 5 and end_reasons(repo) == ["turn_limit", "turn_budget", "turn_budget"]

def test_the_session_stops_when_the_money_runs_out(repo, monkeypatch, capsys):
    monkeypatch.setattr(session_module, "PRICE_IN_PER_M", 1.0)
    monkeypatch.setattr(session_module, "PRICE_OUT_PER_M", 2.0)
    monkeypatch.setattr(session_module, "MAX_COST_USD", 0.01)
    brain = LoopingBrain(tokens_in=4000, tokens_out=1000)      # (4000*1 + 1000*2) / 1e6 = $0.006 per call
    run(repo, monkeypatch, brain)
    assert brain.n == 2 and end_reasons(repo) == ["cost_limit"]    # $0.006 after one call, $0.012 after two
    assert "cost limit reached ($0.01 of $0.01)" in capsys.readouterr().out

def test_without_prices_there_is_no_money_limit(repo, monkeypatch):
    monkeypatch.setattr(session_module, "PRICE_IN_PER_M", 0)
    monkeypatch.setattr(session_module, "PRICE_OUT_PER_M", 0)
    monkeypatch.setattr(session_module, "MAX_COST_USD", 0.0)     # would stop at once if it were checked
    brain = LoopingBrain(tokens_in=10000, tokens_out=10000)
    monkeypatch.setattr(session_module, "SESSION_MAX_TURNS", 4)
    run(repo, monkeypatch, brain)
    assert end_reasons(repo) == ["turn_budget"]

def test_the_status_line_shows_plan_turns_context_tokens_and_money(repo, monkeypatch, capsys):
    monkeypatch.setattr(session_module, "PRICE_IN_PER_M", 1.0)
    monkeypatch.setattr(session_module, "PRICE_OUT_PER_M", 1.0)
    monkeypatch.setattr(session_module, "SESSION_MAX_TURNS", 2)
    class PlanFirst(LoopingBrain):
        def think(self, messages, tools):
            if self.n == 0:
                self.n, self.last_usage = 1, dict(self.tokens)
                return call("update_plan", {"steps": "[x] read\n[~] edit\n[ ] test"}, "p")
            return super().think(messages, tools)
    brain = PlanFirst(tokens_in=1000, tokens_out=500)
    run(repo, monkeypatch, brain)
    out = capsys.readouterr().out
    assert "[plan 1/3: edit | turn 1/2 | context " in out
    assert "| tokens 1k/500k | $0.00/$5.00]" in out
