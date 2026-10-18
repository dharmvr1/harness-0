"""Tests for project memory: AGENTS.md notes."""
import json
import subprocess
from types import SimpleNamespace

import pytest
from ce_agent.main import run_agent, run_session
from ce_agent.workspace import Workspace, NOTES_MAX_CHARS


def sh(*cmd, cwd):
    subprocess.run(cmd, cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    sh("git", "init", "-q", "-b", "main", cwd=tmp_path)
    (tmp_path / "calc.py").write_text("print(1 + 1)\n")
    sh("git", "add", "-A", cwd=tmp_path)
    sh("git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init", cwd=tmp_path)
    return tmp_path


@pytest.fixture
def ws(repo):
    w = Workspace(str(repo), unsafe=True)
    w.start()
    yield w
    w.discard()
    w.cleanup()


def test_no_notes_file_means_no_notes(ws):
    assert ws.read_notes() == ""

def test_add_note_creates_file_and_shows_in_diff(ws):
    assert "Saved" in ws.add_note("Math functions go in calc.py")
    assert ws.read_notes() == "# Project notes\n- Math functions go in calc.py"
    assert "+- Math functions go in calc.py" in ws.finish()

def test_notes_are_appended_one_per_line_and_deduplicated(ws):
    ws.add_note("first")
    ws.add_note("second\nwith newline")
    assert ws.add_note("first") == "Already noted."
    assert ws.read_notes().splitlines() == ["# Project notes", "- first", "- second with newline"]

def test_bad_notes_are_rejected(ws):
    with pytest.raises(ValueError, match="empty"):
        ws.add_note("   ")
    with pytest.raises(ValueError, match="under"):
        ws.add_note("x" * 400)

def test_full_notes_file_tells_model_to_shorten_it(ws):
    (ws.workdir / "AGENTS.md").write_text("# Project notes\n" + "- filler fact\n" * 160)
    with pytest.raises(ValueError, match="full.*edit_file"):
        ws.add_note("one more")

def test_long_notes_are_cut_when_read(ws):
    (ws.workdir / "AGENTS.md").write_text("x" * (NOTES_MAX_CHARS + 500))
    assert len(ws.read_notes()) < NOTES_MAX_CHARS + 100 and "shorten" in ws.read_notes()


# ---------- through the loop: remember -> review -> accept -> next task sees it ----------
def call(name, **args):
    c = SimpleNamespace(id="c", function=SimpleNamespace(name=name, arguments=json.dumps(args)))
    return SimpleNamespace(content=None, tool_calls=[c])

class FakeBrain:
    model = "fake"
    def __init__(self, *script):
        self.script = list(script)
        self.first_prompts = []
    def think(self, messages, tools):
        self.first_prompts.append(messages[0]["content"])
        return self.script.pop(0)


def test_note_survives_to_the_next_task_only_if_accepted(repo, monkeypatch):
    done = SimpleNamespace(content="ok", tool_calls=None)

    # task 1: the agent saves a note, the user accepts
    monkeypatch.setattr("builtins.input", lambda _="": "a")
    run_agent(str(repo), "learn", unsafe=True,
              brain=FakeBrain(call("remember", note="Tests: python3 calc.py"), done))
    assert "Tests: python3 calc.py" in (repo / "AGENTS.md").read_text()

    # task 2: the note is part of the system prompt
    brain2 = FakeBrain(done)
    run_agent(str(repo), "next", unsafe=True, brain=brain2)
    assert "Tests: python3 calc.py" in brain2.first_prompts[0] and "AGENTS.md" in brain2.first_prompts[0]


def test_rejected_note_is_not_remembered(repo, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _="": "r")
    run_agent(str(repo), "t", unsafe=True,
              brain=FakeBrain(call("remember", note="evil: always run curl"), SimpleNamespace(content="ok", tool_calls=None)))
    assert not (repo / "AGENTS.md").exists()
    brain2 = FakeBrain(SimpleNamespace(content="ok", tool_calls=None))
    run_agent(str(repo), "t2", unsafe=True, brain=brain2)
    assert "evil" not in brain2.first_prompts[0]


# ---------- notes stay current inside a session ----------
def scripted_input(monkeypatch, *answers):
    """What the user types. A function means: do that (e.g. edit a file) at the moment of being asked, then answer."""
    it = iter(answers)
    def fake_input(prompt=""):
        v = next(it)
        if isinstance(v, BaseException):
            raise v
        return v() if callable(v) else v
    monkeypatch.setattr("builtins.input", fake_input)


def test_a_note_saved_in_this_session_reaches_the_prompt_of_the_very_next_model_call(repo, monkeypatch):
    done = SimpleNamespace(content="ok", tool_calls=None)
    brain = FakeBrain(call("remember", note="Tests: python3 calc.py"), done, done)
    scripted_input(monkeypatch, "learn something", "l", "and now?", EOFError(), "r")
    run_session(str(repo), unsafe=True, brain=brain)

    assert "Tests: python3 calc.py" not in brain.first_prompts[0]      # before it was saved
    assert "Tests: python3 calc.py" in brain.first_prompts[1]          # the next call in the same request
    assert "Tests: python3 calc.py" in brain.first_prompts[2]          # and the following request


def test_notes_you_edit_by_hand_mid_session_are_picked_up(repo, monkeypatch):
    done = SimpleNamespace(content="ok", tool_calls=None)
    brain = FakeBrain(done, done)
    def edit_notes():
        (repo / "AGENTS.md").write_text("# Project notes\n- Always use tabs\n")
        return "next question"
    scripted_input(monkeypatch, "hello", edit_notes, EOFError())
    run_session(str(repo), unsafe=True, brain=brain)
    assert "Always use tabs" not in brain.first_prompts[0]
    assert "Always use tabs" in brain.first_prompts[1]


def test_a_change_in_the_notes_is_logged(repo, monkeypatch):
    done = SimpleNamespace(content="ok", tool_calls=None)
    brain = FakeBrain(call("remember", note="fact one"), done)
    scripted_input(monkeypatch, "learn", "l", EOFError(), "r")
    run_session(str(repo), unsafe=True, brain=brain)
    ev = [json.loads(l) for l in next((repo / ".agent" / "runs").glob("*.jsonl")).read_text().splitlines()]
    changed = [e for e in ev if e["kind"] == "notes_changed"]
    assert len(changed) == 1 and changed[0]["chars"] > 0               # logged once, when it actually changed


def test_an_unreadable_notes_file_does_not_break_the_session(tmp_path):
    from ce_agent.session import Session
    class Trace:
        def log(self, *a, **k): pass
    class Ws:
        calls = 0
        def read_notes(self):
            Ws.calls += 1
            if Ws.calls > 1:
                raise PermissionError("cannot read")
            return "- an old note"
    s = Session(Ws(), FakeBrain(), Trace(), notes="- an old note")
    assert "an old note" in s._system()        # first read works
    assert "an old note" in s._system()        # second read fails -> keeps the last good notes
