"""Tests for sessions: several prompts in one conversation, plan, context limits, interrupts."""
import json
import subprocess
from types import SimpleNamespace

import pytest
import ce_agent.session as session_module
from ce_agent.main import run_session


def sh(*cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True).stdout


@pytest.fixture
def repo(tmp_path):
    sh("git", "init", "-q", "-b", "main", cwd=tmp_path)
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    (tmp_path / "big.txt").write_text("line of text\n" * 400)
    sh("git", "add", "-A", cwd=tmp_path)
    sh("git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init", cwd=tmp_path)
    return tmp_path


def call(name, **args):
    c = SimpleNamespace(id=f"call_{name}_{len(json.dumps(args))}", function=SimpleNamespace(name=name, arguments=json.dumps(args)))
    return SimpleNamespace(content=None, tool_calls=[c])

def say(text):
    return SimpleNamespace(content=text, tool_calls=None)


class FakeBrain:
    """Plays back scripted replies, and keeps a copy of everything it was shown."""
    model = "fake"
    def __init__(self, *script, usage=None):
        self.script = list(script)
        self.seen = []
        self.usage = usage
        self.last_usage = None
    def think(self, messages, tools):
        self.seen.append(json.loads(json.dumps(messages, default=str)))
        step = self.script.pop(0)
        if isinstance(step, BaseException):
            raise step
        self.last_usage = self.usage
        return step


def typed(monkeypatch, *inputs):
    """What the user types: prompts, then EOFError (Ctrl-D), then the review answer."""
    it = iter(inputs)
    def fake_input(prompt=""):
        v = next(it)
        if isinstance(v, BaseException):
            raise v
        return v() if callable(v) else v   # a function = "do this at the moment the user is asked, then answer"
    monkeypatch.setattr("builtins.input", fake_input)


def assert_valid_conversation(messages):
    """The API rejects a conversation where a tool call has no result."""
    for i, m in enumerate(messages):
        if m["role"] == "assistant" and m.get("tool_calls"):
            ids = [c["id"] for c in m["tool_calls"]]
            following = [x["tool_call_id"] for x in messages[i + 1:i + 1 + len(ids)] if x["role"] == "tool"]
            assert sorted(ids) == sorted(following), f"unanswered tool call in {ids}"


# ---------- the session keeps going ----------
def test_second_prompt_sees_the_first_and_workspace_persists(repo, monkeypatch):
    brain = FakeBrain(call("write_file", file_path="a.py", content="x = 1\n"), say("made a.py"),
                      call("read_file", file_path="a.py"), say("a.py contains x = 1"))
    typed(monkeypatch, "create a.py", "l", "what is in it?", EOFError(), "a")
    run_session(str(repo), unsafe=True, brain=brain)

    # the 3rd model call (first of prompt 2) already contains prompt 1 and its tool calls
    texts = json.dumps(brain.seen[2])
    assert "create a.py" in texts and "what is in it?" in texts and "write_file" in texts
    # the file written in prompt 1 was readable in prompt 2 (same workspace)
    assert "x = 1" in json.dumps(brain.seen[3])
    # both prompts end up in the repo after one accept
    assert (repo / "a.py").read_text() == "x = 1\n"


def test_one_commit_per_prompt_on_the_branch(repo, monkeypatch):
    brain = FakeBrain(call("write_file", file_path="a.py", content="1\n"), say("ok"),
                      call("write_file", file_path="b.py", content="2\n"), say("ok"))
    typed(monkeypatch, "first thing", "l", "second thing", "l", EOFError(), "k")     # k: keep the branch to inspect it
    run_session(str(repo), unsafe=True, brain=brain)
    branch = sh("git", "branch", "--list", "agent/*", "--format=%(refname:short)", cwd=repo).split()[0]
    log = sh("git", "log", "--format=%s", branch, cwd=repo)
    assert "agent: first thing" in log and "agent: second thing" in log


def test_a_question_changes_nothing(repo, monkeypatch):
    brain = FakeBrain(call("list_files"), call("read_file", file_path="calc.py"), say("It has an add function."))
    typed(monkeypatch, "what does this repo contain?", EOFError())
    run_session(str(repo), unsafe=True, brain=brain)         # no review prompt expected: nothing changed
    assert sh("git", "status", "--porcelain", cwd=repo) == "" and "agent/" not in sh("git", "branch", cwd=repo)


def test_first_prompt_from_command_line_then_follow_ups(repo, monkeypatch):
    brain = FakeBrain(say("one"), say("two"))
    typed(monkeypatch, "follow up", EOFError())
    run_session(str(repo), "first request", unsafe=True, brain=brain)
    assert "first request" in json.dumps(brain.seen[0]) and "follow up" in json.dumps(brain.seen[1])


def test_blank_lines_are_ignored(repo, monkeypatch):
    brain = FakeBrain(say("hello"))
    typed(monkeypatch, "", "   ", "hi", EOFError())
    run_session(str(repo), unsafe=True, brain=brain)
    assert len(brain.seen) == 1


# ---------- things going wrong ----------
def test_api_error_ends_that_prompt_but_not_the_session(repo, monkeypatch):
    brain = FakeBrain(RuntimeError("rate limited"), say("recovered"))
    typed(monkeypatch, "first", "second", EOFError())
    run_session(str(repo), unsafe=True, brain=brain)
    assert len(brain.seen) == 2
    ev = [json.loads(l) for l in next((repo / ".agent" / "runs").glob("*.jsonl")).read_text().splitlines()]
    ends = [e for e in ev if e["kind"] == "prompt_end"]
    assert ends[0]["reason"].startswith("error: RuntimeError") and ends[1]["reason"] == "done"


def test_ctrl_c_in_the_middle_of_tools_leaves_a_valid_conversation(repo, monkeypatch):
    from ce_agent.registry import Registry
    real_call = Registry.call
    state = {"n": 0}
    def flaky_call(self, name, kwargs):
        state["n"] += 1
        if state["n"] == 2:
            raise KeyboardInterrupt()
        return real_call(self, name, kwargs)
    monkeypatch.setattr(Registry, "call", flaky_call)

    two_tools = SimpleNamespace(content=None, tool_calls=[
        SimpleNamespace(id="t1", function=SimpleNamespace(name="list_files", arguments="{}")),
        SimpleNamespace(id="t2", function=SimpleNamespace(name="list_files", arguments="{}"))])
    brain = FakeBrain(two_tools, say("fine"))
    typed(monkeypatch, "go", "again", EOFError())
    run_session(str(repo), unsafe=True, brain=brain)
    assert len(brain.seen) == 2                       # the session carried on after Ctrl-C
    assert_valid_conversation(brain.seen[1])          # ...with t2 answered as interrupted
    assert "interrupted" in json.dumps(brain.seen[1])


def test_turn_limit_is_per_prompt(repo, monkeypatch):
    monkeypatch.setattr(session_module, "MAX_TURNS", 2)
    brain = FakeBrain(call("list_files"), call("list_files"),      # prompt 1 hits the limit after 2 calls
                      say("done"))                                  # prompt 2 gets a fresh allowance
    typed(monkeypatch, "loop", "ok", EOFError())
    run_session(str(repo), unsafe=True, brain=brain)
    assert len(brain.seen) == 3


# ---------- the plan ----------
def test_plan_is_kept_and_shown_to_the_model_every_turn(repo, monkeypatch):
    plan1 = "[~] read calc.py\n[ ] fix add\n[ ] run tests"
    plan2 = "[x] read calc.py\n[~] fix add\n[ ] run tests"
    brain = FakeBrain(call("update_plan", steps=plan1), call("update_plan", steps=plan2), say("done"))
    typed(monkeypatch, "fix things", EOFError())
    run_session(str(repo), unsafe=True, brain=brain)
    assert "plan" not in brain.seen[0][0]["content"].lower().split("update_plan")[-1]     # no plan yet at first
    assert "[~] read calc.py" in brain.seen[1][0]["content"]                              # in the system prompt
    assert "[x] read calc.py" in brain.seen[2][0]["content"] and "[~] read calc.py" not in brain.seen[2][0]["content"]
    ev = [json.loads(l) for l in next((repo / ".agent" / "runs").glob("*.jsonl")).read_text().splitlines()]
    assert [e["kind"] for e in ev].count("plan_update") == 2


def test_bad_plans_get_a_helpful_error(repo, monkeypatch):
    brain = FakeBrain(call("update_plan", steps="do the thing\nthen the other"), say("ok"))
    typed(monkeypatch, "go", EOFError())
    run_session(str(repo), unsafe=True, brain=brain)
    tool_msg = [m for m in brain.seen[1] if m["role"] == "tool"][0]["content"]
    assert tool_msg.startswith("ERROR") and "[x] finished step" in tool_msg


def test_plan_formats_models_commonly_write_are_accepted():
    class T:
        def log(self, *a, **k): pass
    s = session_module.Session(SimpleNamespace(), FakeBrain(), T())
    s.set_plan("1. [x] one\n- [~] two\n* [ ] three\n[X] four\n[] five")
    assert s.plan == [("x", "one"), ("~", "two"), (" ", "three"), ("x", "four"), (" ", "five")]
    with pytest.raises(ValueError, match="more than 15"):
        s.set_plan("\n".join(f"[ ] step {i}" for i in range(16)))


# ---------- context limits ----------
def test_old_tool_outputs_are_cleared_when_the_context_fills_up(repo, monkeypatch):
    monkeypatch.setattr(session_module, "CONTEXT_TOKENS", 6000)
    reads = [call("read_file", file_path="big.txt", start_line=1 + i, end_line=300) for i in range(5)]
    brain = FakeBrain(*reads, say("done"))
    typed(monkeypatch, "read it a lot", EOFError())
    run_session(str(repo), unsafe=True, brain=brain)

    last = brain.seen[-1]
    results = [m["content"] for m in last if m["role"] == "tool"]
    assert any(r.startswith("[output cleared") for r in results)       # old ones shrunk
    assert not results[-1].startswith("[output cleared")                # newest intact
    ev = [json.loads(l) for l in next((repo / ".agent" / "runs").glob("*.jsonl")).read_text().splitlines()]
    assert any(e["kind"] == "context_cleared" for e in ev)


def test_a_full_conversation_stops_instead_of_calling_the_model(repo, monkeypatch):
    monkeypatch.setattr(session_module, "CONTEXT_TOKENS", 200)          # smaller than the system prompt itself
    brain = FakeBrain(say("never reached"))
    typed(monkeypatch, "hello", EOFError())
    run_session(str(repo), unsafe=True, brain=brain)
    assert brain.seen == []
    ev = [json.loads(l) for l in next((repo / ".agent" / "runs").glob("*.jsonl")).read_text().splitlines()]
    assert [e for e in ev if e["kind"] == "prompt_end"][0]["reason"] == "context_full"


def test_session_token_budget_stops_further_prompts(repo, monkeypatch):
    monkeypatch.setattr(session_module, "TOKEN_BUDGET", 150)
    brain = FakeBrain(say("one"), say("two"), usage={"prompt_tokens": 100, "completion_tokens": 100})
    typed(monkeypatch, "first", "second", EOFError())
    run_session(str(repo), unsafe=True, brain=brain)
    assert len(brain.seen) == 1                                          # 200 tokens used > 150: second prompt refused
    ev = [json.loads(l) for l in next((repo / ".agent" / "runs").glob("*.jsonl")).read_text().splitlines()]
    assert [e["reason"] for e in ev if e["kind"] == "prompt_end"] == ["done", "budget"]
    assert ev[-1]["total_prompt_tokens"] == 100


def test_trajectory_viewer_understands_sessions(repo, monkeypatch):
    from ce_agent.trace import find_run, show
    brain = FakeBrain(call("update_plan", steps="[~] look"), say("one"), say("two"))
    typed(monkeypatch, "first", "second", EOFError())
    run_session(str(repo), unsafe=True, brain=brain)
    text = show(find_run(str(repo)))
    assert "> you: first" in text and "> you: second" in text and "plan:" in text and "prompt ended: done" in text


# ---------- keeping the workspace and your real folder in sync ----------
def test_accept_after_a_prompt_changes_your_folder_immediately(repo, monkeypatch):
    seen_in_repo = []
    brain = FakeBrain(call("write_file", file_path="a.py", content="1\n"), say("made a.py"), say("ok"))
    def at_second_prompt():
        seen_in_repo.append((repo / "a.py").exists())    # BEFORE the session ends
        return "just checking"
    typed(monkeypatch, "create a.py", "a", at_second_prompt, EOFError())
    run_session(str(repo), unsafe=True, brain=brain)
    assert seen_in_repo == [True]
    assert (repo / "a.py").read_text() == "1\n"
    assert "agent/" not in sh("git", "branch", cwd=repo)       # nothing left pending, so no leftover branch


def test_after_an_accept_the_next_review_only_shows_new_changes(repo, monkeypatch, capsys):
    brain = FakeBrain(call("write_file", file_path="a.py", content="1\n"), say("ok"),
                      call("write_file", file_path="b.py", content="2\n"), say("ok"))
    typed(monkeypatch, "make a", "a", "make b", "a", EOFError())
    run_session(str(repo), unsafe=True, brain=brain)
    out = capsys.readouterr().out
    second_review = out[out.rindex("Pending changes"):]
    assert "b.py" in second_review and "a.py" not in second_review    # a.py is already merged, not shown again
    assert (repo / "a.py").exists() and (repo / "b.py").exists()


def test_later_keeps_changes_pending_and_they_pile_up(repo, monkeypatch):
    brain = FakeBrain(call("write_file", file_path="a.py", content="1\n"), say("ok"),
                      call("write_file", file_path="b.py", content="2\n"), say("ok"))
    seen = []
    typed(monkeypatch, "make a", "l", "make b", lambda: (seen.append((repo / "a.py").exists()), "l")[1], EOFError(), "a")
    run_session(str(repo), unsafe=True, brain=brain)
    assert seen == [False]                                              # nothing merged while "later"
    assert (repo / "a.py").exists() and (repo / "b.py").exists()        # both merged at the end


def test_reject_resets_the_workspace_and_tells_the_model(repo, monkeypatch):
    brain = FakeBrain(call("read_file", file_path="calc.py"), call("write_file", file_path="calc.py", content="BROKEN\n"),
                      say("changed calc"),
                      call("read_file", file_path="calc.py"), say("it is back"))
    typed(monkeypatch, "break calc", "r", "read calc again", EOFError())
    run_session(str(repo), unsafe=True, brain=brain)
    assert (repo / "calc.py").read_text().startswith("def add")        # your folder never changed
    tool_result = [m for m in brain.seen[4] if m["role"] == "tool"][-1]["content"]
    assert "def add" in tool_result and "BROKEN" not in tool_result    # the workspace was reset
    assert "rejected your pending changes" in json.dumps(brain.seen[3])  # the model was told


def test_commits_you_make_in_your_folder_reach_the_workspace(repo, monkeypatch):
    brain = FakeBrain(say("first answer"), call("read_file", file_path="from_me.py"), say("saw it"))
    def user_commits_something():
        (repo / "from_me.py").write_text("human = True\n")
        sh("git", "add", "-A", cwd=repo)
        sh("git", "-c", "user.name=me", "-c", "user.email=me@x", "commit", "-qm", "human work", cwd=repo)
        return "read from_me.py"
    typed(monkeypatch, "hello", user_commits_something, EOFError())
    run_session(str(repo), unsafe=True, brain=brain)
    result = [m for m in brain.seen[2] if m["role"] == "tool"][-1]["content"]
    assert "human = True" in result                                    # the agent saw the new commit


def test_your_commit_plus_pending_agent_work_both_end_up_in_your_folder(repo, monkeypatch):
    brain = FakeBrain(call("write_file", file_path="agent.py", content="a = 1\n"), say("ok"), say("ok"))
    def user_commits_something():
        (repo / "human.py").write_text("h = 1\n")
        sh("git", "add", "-A", cwd=repo)
        sh("git", "-c", "user.name=me", "-c", "user.email=me@x", "commit", "-qm", "human work", cwd=repo)
        return "go on"
    typed(monkeypatch, "make agent.py", "l", user_commits_something, EOFError(), "a")
    run_session(str(repo), unsafe=True, brain=brain)
    assert (repo / "agent.py").exists() and (repo / "human.py").exists()
    assert "agent.py" in sh("git", "status", "--porcelain", cwd=repo)       # arrives as an uncommitted change


def test_edits_you_make_mid_session_are_announced_when_picked_up(repo, monkeypatch, capsys):
    brain = FakeBrain(say("ok"), say("ok"))
    def user_edits():
        (repo / "calc.py").write_text("# my unsaved idea\n")
        return "next"
    typed(monkeypatch, "hello", user_edits, EOFError())
    run_session(str(repo), unsafe=True, brain=brain)
    out = capsys.readouterr().out
    assert "includes your latest changes" in out or "latest changes" in out


def test_conflicting_changes_are_warned_about_and_the_session_continues(repo, monkeypatch, capsys):
    brain = FakeBrain(call("read_file", file_path="calc.py"), call("write_file", file_path="calc.py", content="agent version\n"),
                      say("ok"), say("still here"))
    def user_commits_conflict():
        (repo / "calc.py").write_text("human version\n")
        sh("git", "-c", "user.name=me", "-c", "user.email=me@x", "commit", "-qam", "human edit", cwd=repo)
        return "next request"
    # prompt 1 -> review "l" (later); the user commits a conflicting edit; prompt 2 -> sync conflicts (warned)
    typed(monkeypatch, "edit calc", "l", user_commits_conflict, EOFError(), "k")
    run_session(str(repo), unsafe=True, brain=brain)
    out = capsys.readouterr().out
    assert "conflict with the agent's pending changes" in out
    assert (repo / "calc.py").read_text() == "human version\n"          # your work is untouched
    assert "agent/" in sh("git", "branch", cwd=repo)                    # agent's work kept on its branch


# ---------- the model is told when your folder changed under it ----------
def user_messages(conversation):
    return [m["content"] for m in conversation if m["role"] == "user"]


def test_model_is_told_which_files_changed_when_you_edit_mid_session(repo, monkeypatch):
    brain = FakeBrain(say("first"), say("second"))
    def user_changes_things():
        (repo / "calc.py").write_text("def add(a, b):\n    return 0\n")          # modified
        (repo / "brand_new.py").write_text("n = 1\n")                           # new
        (repo / "big.txt").unlink()                                              # deleted
        return "go on"
    typed(monkeypatch, "start", user_changes_things, EOFError())
    run_session(str(repo), unsafe=True, brain=brain)

    last_user = user_messages(brain.seen[1])[-1]
    assert "go on" in last_user and "changed the project" in last_user
    assert "calc.py (modified)" in last_user and "brand_new.py (new)" in last_user and "big.txt (deleted)" in last_user
    assert "re-read" in last_user.lower()
    assert len(user_messages(brain.seen[1])) == 2                                # one merged message, not two in a row

    ev = [json.loads(l) for l in next((repo / ".agent" / "runs").glob("*.jsonl")).read_text().splitlines()]
    assert [e["text"] for e in ev if e["kind"] == "user_message"] == ["start", "go on"]    # the log keeps what you typed
    sync_event = [e for e in ev if e["kind"] == "sync"][0]
    assert sorted(sync_event["changes"]) == ["A\tbrand_new.py", "D\tbig.txt", "M\tcalc.py"]


def test_no_note_when_nothing_changed_in_your_folder(repo, monkeypatch):
    brain = FakeBrain(say("one"), say("two"))
    typed(monkeypatch, "first", "second", EOFError())
    run_session(str(repo), unsafe=True, brain=brain)
    assert user_messages(brain.seen[1])[-1] == "second"


def test_no_note_about_the_agents_own_accepted_changes(repo, monkeypatch):
    brain = FakeBrain(call("write_file", file_path="a.py", content="1\n"), say("made it"), say("ok"))
    typed(monkeypatch, "make a.py", "a", "next", EOFError())
    run_session(str(repo), unsafe=True, brain=brain)
    assert user_messages(brain.seen[2])[-1] == "next"                           # accepting must not look like "you changed files"


def test_model_is_not_told_when_the_sync_failed(repo, monkeypatch):
    brain = FakeBrain(call("read_file", file_path="calc.py"), call("write_file", file_path="calc.py", content="agent version\n"),
                      say("ok"), say("still here"))
    def user_makes_conflicting_edit():
        (repo / "calc.py").write_text("human version\n")
        return "next request"
    typed(monkeypatch, "edit calc", "l", user_makes_conflicting_edit, EOFError(), "k")
    run_session(str(repo), unsafe=True, brain=brain)
    assert "agent/" in sh("git", "branch", cwd=repo)                              # the agent really had pending work
    assert "changed the project" not in json.dumps(brain.seen[3])               # its copy did not change, so no note
