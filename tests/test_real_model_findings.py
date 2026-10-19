"""Behaviours found by running a REAL model, pinned down with scripted ones.

The first real run of the feature task ended with no change at all: the model's write_file arguments arrived cut off,
our error said only "Arguments were not valid JSON", and the model's next reply ("I am going to ...") was taken as 'done'."""
import json
import subprocess
from types import SimpleNamespace

import ce_agent.session as session_module
import pytest
from ce_agent.main import run_session
from ce_agent.session import describe_invalid_arguments
from ce_agent.brain import Brain
from ce_agent.trace import find_run, show

# the real, cut-off arguments from the failed run (71 characters)
TRUNCATED = '{"content": "import re\\n\\ndef word_count(text):\\n    words = re.findall'


def sh(*cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True).stdout


@pytest.fixture
def repo(tmp_path):
    sh("git", "init", "-q", "-b", "main", cwd=tmp_path)
    (tmp_path / "textutils.py").write_text("def slugify(t):\n    return t.lower()\n")
    sh("git", "add", "-A", cwd=tmp_path)
    sh("git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init", cwd=tmp_path)
    return tmp_path


def call(name, raw_arguments, call_id=None):
    arguments = raw_arguments if isinstance(raw_arguments, str) else json.dumps(raw_arguments)
    c = SimpleNamespace(id=call_id or f"call_{abs(hash(arguments)) % 100000}", function=SimpleNamespace(name=name, arguments=arguments))
    return SimpleNamespace(content=None, tool_calls=[c])

def say(text):
    return SimpleNamespace(content=text, tool_calls=None)


class FakeBrain:
    model = "fake"
    last_usage = None
    def __init__(self, *script, finish_reasons=None):
        self.script, self.seen = list(script), []
        self.finish_reasons = list(finish_reasons or [])
        self.last_finish_reason = None
    def think(self, messages, tools):
        self.seen.append(json.loads(json.dumps(messages, default=str)))
        self.last_finish_reason = self.finish_reasons.pop(0) if self.finish_reasons else "tool_calls"
        return self.script.pop(0)


def run(repo, monkeypatch, brain, prompts=("do it",), review="a"):
    """Run a session. The fake user answers each question according to what is being asked."""
    requests = iter(prompts)
    def fake_input(question=""):
        if "[l]ater" in question:                 # the review after a request that changed files
            return "l"
        if "[k]eep branch" in question:           # the final review when the session ends
            return review
        try:
            return next(requests)                 # the > prompt
        except StopIteration:
            raise EOFError()                      # Ctrl-D
    monkeypatch.setattr("builtins.input", fake_input)
    run_session(str(repo), unsafe=True, brain=brain)

def events(repo):
    path = max((repo / ".agent" / "runs").glob("*.jsonl"), key=lambda f: f.stat().st_mtime)
    return [json.loads(l) for l in path.read_text().splitlines()]

def tool_results(messages):
    return [m["content"] for m in messages if m["role"] == "tool"]


# ======================================================================
# an error message the model can act on
# ======================================================================
def error_for(raw, name="write_file"):
    try:
        json.loads(raw)
    except json.JSONDecodeError as e:
        return describe_invalid_arguments(name, raw, e)

def test_the_message_for_the_real_truncated_call_says_what_happened_and_how_to_fix_it():
    text = error_for(TRUNCATED)
    assert "arguments of your write_file call are not valid JSON" in text
    assert "Unterminated string" in text and "character 12" in text and "at at" not in text
    assert "received 71 characters" in text and "words = re.findall" in text            # the tail it actually sent
    assert "looks cut off" in text
    assert "escape backslashes" in text and "smaller pieces" in text and "edit_file" in text

def test_a_complete_but_malformed_call_is_not_called_cut_off():
    text = error_for('{"file_path": "a.py", "content": x}')
    assert "looks cut off" not in text and "Expecting value" in text

def test_a_very_long_tail_is_shortened():
    assert len(error_for('{"content": "' + "x" * 5000)) < 700


# ======================================================================
# in the loop
# ======================================================================
def test_the_model_sees_the_explanation_and_the_broken_call_is_made_valid_in_the_history(repo, monkeypatch):
    brain = FakeBrain(call("write_file", TRUNCATED, "c1"), call("write_file", {"file_path": "new.py", "content": "x = 1\n"}), say("done"))
    run(repo, monkeypatch, brain)
    second_request = brain.seen[1]
    assert "not valid JSON" in tool_results(second_request)[0] and "looks cut off" in tool_results(second_request)[0]
    assistant_call = next(m for m in second_request if m.get("tool_calls"))["tool_calls"][0]
    assert assistant_call["function"]["arguments"] == "{}"                                # still valid JSON for every provider
    for message in brain.seen[2]:                                                          # and it stays valid afterwards
        for c in message.get("tool_calls", []):
            json.loads(c["function"]["arguments"])
    assert (repo / "new.py").read_text() == "x = 1\n"

def test_the_raw_broken_text_is_kept_in_the_trace_for_debugging(repo, monkeypatch):
    run(repo, monkeypatch, FakeBrain(call("write_file", TRUNCATED), say("giving up"), say("x")))
    ev = events(repo)
    assert any(e["kind"] == "model_call" and e["tool_calls"][0]["arguments"] == TRUNCATED for e in ev if e["kind"] == "model_call" and e["tool_calls"])
    tool = [e for e in ev if e["kind"] == "tool_call"][0]
    assert tool["status"] == "error" and tool["args"] == TRUNCATED


# ======================================================================
# a model that gives up after a failure
# ======================================================================
def test_the_exact_failure_from_the_real_run_now_recovers(repo, monkeypatch, capsys):
    brain = FakeBrain(call("write_file", TRUNCATED),                                       # 1. cut-off arguments
                      say("I am going to update the plan to add word_count..."),           # 2. it only announces what it will do
                      call("write_file", {"file_path": "new.py", "content": "def word_count(t):\n    return {}\n"}),   # 3. after the nudge
                      say("Added word_count."))
    run(repo, monkeypatch, brain)
    nudge = [m for m in brain.seen[2] if m["role"] == "user"][-1]["content"]              # what the model saw before its 3rd call
    assert "last tool call failed" in nudge and "not finished" in nudge and "exactly what blocked you" in nudge
    assert (repo / "new.py").exists()
    ev = events(repo)
    assert [e["kind"] for e in ev].count("nudge") == 1
    assert [e for e in ev if e["kind"] == "prompt_end"][0]["reason"] == "done" and [e for e in ev if e["kind"] == "prompt_end"][0]["model_calls"] == 4
    assert "[nudge]" in capsys.readouterr().out
    assert "asked to try again" in show(find_run(str(repo)))

def test_a_real_tool_error_is_nudged_in_the_same_way(repo, monkeypatch):
    brain = FakeBrain(call("edit_file", {"file_path": "textutils.py", "old_text": "does not exist", "new_text": "x"}),
                      say("I will fix that now."),
                      call("edit_file", {"file_path": "textutils.py", "old_text": "t.lower()", "new_text": "t.lower().strip()"}),
                      say("done"))
    run(repo, monkeypatch, brain)
    assert "t.lower().strip()" in (repo / "textutils.py").read_text()

def test_only_one_nudge_per_request(repo, monkeypatch):
    brain = FakeBrain(call("write_file", TRUNCATED), say("I will try."), say("I really will."))
    run(repo, monkeypatch, brain)
    assert len(brain.seen) == 3                                                           # one nudge, then it is allowed to stop
    assert [e["kind"] for e in events(repo)].count("nudge") == 1

def test_the_allowance_is_per_request_not_per_session(repo, monkeypatch):
    brain = FakeBrain(call("write_file", TRUNCATED), say("I will try."), say("ok, stopping"),          # request 1
                      call("write_file", TRUNCATED), say("I will try."), say("ok, stopping"))          # request 2
    run(repo, monkeypatch, brain, prompts=("first", "second"), review="r")
    assert [e["kind"] for e in events(repo)].count("nudge") == 2

def test_no_nudge_when_the_last_call_succeeded(repo, monkeypatch):
    brain = FakeBrain(call("read_file", {"file_path": "nope.py"}),                         # fails
                      call("read_file", {"file_path": "textutils.py"}),                    # then succeeds
                      say("It defines slugify."))
    run(repo, monkeypatch, brain)
    assert len(brain.seen) == 3 and "nudge" not in [e["kind"] for e in events(repo)]

def test_no_nudge_for_a_plain_answer(repo, monkeypatch):
    brain = FakeBrain(say("It has one function."))
    run(repo, monkeypatch, brain)
    assert len(brain.seen) == 1 and "nudge" not in [e["kind"] for e in events(repo)]

def test_a_failure_in_an_earlier_request_does_not_cause_a_nudge_in_the_next_one(repo, monkeypatch):
    brain = FakeBrain(call("read_file", {"file_path": "nope.py"}), say("not found, sorry"), say("it is stated nowhere"),   # request 1 (nudged)
                      say("a plain answer"))                                                                         # request 2
    run(repo, monkeypatch, brain, prompts=("first", "what is this?"), review="r")
    assert len(brain.seen) == 4

def test_a_nudge_counts_against_the_turn_limit(repo, monkeypatch):
    monkeypatch.setattr(session_module, "MAX_TURNS", 2)
    brain = FakeBrain(call("write_file", TRUNCATED), say("I will."), say("never reached"))
    run(repo, monkeypatch, brain)
    assert len(brain.seen) == 2                                                           # call, then the nudged turn: the limit stops it


# ======================================================================
# why a reply ended
# ======================================================================
def test_the_finish_reason_is_logged_and_a_cut_off_answer_is_flagged(repo, monkeypatch, capsys):
    brain = FakeBrain(call("read_file", {"file_path": "textutils.py"}), say("done"), finish_reasons=["tool_calls", "length"])
    run(repo, monkeypatch, brain)
    calls = [e for e in events(repo) if e["kind"] == "model_call"]
    assert [c["finish_reason"] for c in calls] == ["tool_calls", "length"]
    assert "cut off (it hit the output limit)" in capsys.readouterr().out
    assert "finish: length" in show(find_run(str(repo)))

def test_the_brain_remembers_why_the_reply_ended():
    class Client:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=3, completion_tokens=4),
                                           choices=[SimpleNamespace(finish_reason="length", message=SimpleNamespace(content="x", tool_calls=None))])
    brain = Brain.__new__(Brain)
    brain.model, brain.client, brain.last_usage, brain.last_finish_reason = "m", Client, None, None
    brain.max_tokens, brain.extra_body = 100, None
    assert brain.think([], []).content == "x" and brain.last_finish_reason == "length" and brain.last_usage == {"prompt_tokens": 3, "completion_tokens": 4}


def make_brain(monkeypatch, provider, **env):
    monkeypatch.setenv("PROVIDER", provider)
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setenv("GROQ_API_KEY", "k")
    for name in ("MAX_TOKENS", "PROVIDER_ORDER"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    sent = {}
    class Client:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    sent.update(kw)
                    return SimpleNamespace(provider="Groq", usage=None,
                                           choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="x", tool_calls=None))])
    brain = Brain()
    brain.client = Client
    return brain, sent


def test_every_request_has_an_output_cap_and_the_backend_is_recorded(monkeypatch):
    brain, sent = make_brain(monkeypatch, "openrouter")
    brain.think([], [])
    assert sent["max_tokens"] == 4096 and brain.last_backend == "Groq"
    assert sent["extra_body"] is None

def test_the_output_cap_and_backend_order_can_be_set(monkeypatch):
    brain, sent = make_brain(monkeypatch, "openrouter", MAX_TOKENS="999", PROVIDER_ORDER="Groq, DeepInfra")
    brain.think([], [])
    assert sent["max_tokens"] == 999
    assert sent["extra_body"] == {"provider": {"order": ["Groq", "DeepInfra"], "allow_fallbacks": False}}

def test_backend_order_is_only_sent_to_openrouter(monkeypatch):
    brain, sent = make_brain(monkeypatch, "groq", PROVIDER_ORDER="Groq")
    brain.think([], [])
    assert sent["extra_body"] is None

def test_only_the_first_of_several_tool_calls_is_kept(monkeypatch):
    brain, sent = make_brain(monkeypatch, "openrouter")
    two = [SimpleNamespace(id="a"), SimpleNamespace(id="b")]
    brain.client.chat.completions.create = staticmethod(lambda **kw: SimpleNamespace(
        provider=None, usage=None, choices=[SimpleNamespace(finish_reason="tool_calls", message=SimpleNamespace(content=None, tool_calls=two))]))
    assert [c.id for c in brain.think([], []).tool_calls] == ["a"]

def test_requests_ask_for_one_tool_call_at_a_time(monkeypatch):
    brain, sent = make_brain(monkeypatch, "openrouter")
    brain.think([], [])
    assert sent["parallel_tool_calls"] is False

def test_the_groq_default_model_is_one_this_api_serves():
    from ce_agent.brain import PROVIDERS
    assert PROVIDERS["groq"]["model"] == "openai/gpt-oss-120b"


# ======================================================================
# a reply cut off in the middle of a tool call is asked for again
# ======================================================================
def test_a_cut_off_tool_call_is_retried_and_the_good_answer_is_used(repo, monkeypatch):
    brain = FakeBrain(call("write_file", TRUNCATED), call("write_file", {"file_path": "new.py", "content": "x = 1\n"}), say("done"),
                      finish_reasons=[None, "tool_calls", "stop"])
    run(repo, monkeypatch, brain)
    assert len(brain.seen) == 3
    assert not any("not valid JSON" in r for m in brain.seen for r in tool_results(m))
    assert [e["kind"] for e in events(repo)].count("retry") == 1

def test_retrying_stops_after_two_attempts(repo, monkeypatch):
    brain = FakeBrain(*[call("write_file", TRUNCATED)] * 3, say("giving up"), finish_reasons=[None] * 3 + ["stop"])
    run(repo, monkeypatch, brain)
    assert [e["kind"] for e in events(repo)].count("retry") == 2
    assert any("not valid JSON" in r for m in brain.seen for r in tool_results(m))

def test_a_reply_that_hit_the_limit_is_not_retried(repo, monkeypatch):
    brain = FakeBrain(call("write_file", TRUNCATED), say("ok"), finish_reasons=["length", "stop"])
    run(repo, monkeypatch, brain)
    assert "retry" not in [e["kind"] for e in events(repo)]

def test_valid_calls_are_never_retried(repo, monkeypatch):
    brain = FakeBrain(call("list_files", {}), say("ok"), finish_reasons=[None, "stop"])
    run(repo, monkeypatch, brain)
    assert "retry" not in [e["kind"] for e in events(repo)]


# ======================================================================
# a file the model never read must not be overwritten (it deleted two functions in the second real run)
# ======================================================================
TEXTUTILS = ("import re\n\n\ndef slugify(text):\n    return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')\n\n\n"
             "def truncate(text, limit):\n    return text if len(text) <= limit else text[:limit - 3] + '...'\n")


@pytest.fixture
def tools_ws(tmp_path):
    from ce_agent.tools import make_workspace_tools
    from ce_agent.workspace import Workspace
    sh("git", "init", "-q", "-b", "main", cwd=tmp_path)
    (tmp_path / "textutils.py").write_text(TEXTUTILS)
    sh("git", "add", "-A", cwd=tmp_path)
    sh("git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init", cwd=tmp_path)
    ws = Workspace(str(tmp_path), unsafe=True)
    ws.start()
    tools = {f.__name__: f for f, _ in make_workspace_tools(ws)}
    yield ws, tools
    ws.discard()
    ws.cleanup()


def test_an_existing_file_that_was_not_read_cannot_be_overwritten(tools_ws):
    ws, tools = tools_ws
    with pytest.raises(ValueError, match="already exists and you have not read it") as e:
        tools["write_file"]("textutils.py", "def word_count(t): ...\n")
    assert "read_file" in str(e.value) and "edit_file" in str(e.value)
    assert (ws.workdir / "textutils.py").read_text() == TEXTUTILS                           # nothing was lost

def test_after_reading_a_file_it_may_be_replaced(tools_ws):
    ws, tools = tools_ws
    tools["read_file"]("textutils.py")
    tools["write_file"]("textutils.py", "replaced\n")
    assert (ws.workdir / "textutils.py").read_text() == "replaced\n"

def test_reading_just_a_few_lines_counts_as_reading(tools_ws):
    ws, tools = tools_ws
    tools["read_file"]("textutils.py", 1, 2)
    tools["write_file"]("textutils.py", "replaced\n")

def test_editing_a_file_counts_as_knowing_it(tools_ws):
    ws, tools = tools_ws
    tools["edit_file"]("textutils.py", "def truncate", "def cut")
    tools["write_file"]("textutils.py", "replaced\n")

def test_new_files_can_be_created_and_then_rewritten(tools_ws):
    ws, tools = tools_ws
    tools["write_file"]("new.py", "x = 1\n")
    tools["write_file"]("new.py", "x = 2\n")                                                 # it wrote this file itself
    assert (ws.workdir / "new.py").read_text() == "x = 2\n"

def test_a_failed_read_does_not_unlock_anything(tools_ws):
    ws, tools = tools_ws
    with pytest.raises(FileNotFoundError):
        tools["read_file"]("missing.py")
    (ws.workdir / "missing.py").write_text("appeared later\n")
    with pytest.raises(ValueError, match="have not read it"):
        tools["write_file"]("missing.py", "x")

def test_reading_one_file_does_not_unlock_another(tools_ws):
    ws, tools = tools_ws
    (ws.workdir / "other.py").write_text("keep me\n")
    tools["read_file"]("textutils.py")
    with pytest.raises(ValueError, match="have not read it"):
        tools["write_file"]("other.py", "x")
    assert (ws.workdir / "other.py").read_text() == "keep me\n"

def test_the_same_file_spelled_differently_is_the_same_file(tools_ws):
    ws, tools = tools_ws
    tools["read_file"]("./textutils.py")
    tools["write_file"]("sub/../textutils.py", "replaced\n")

def test_path_escapes_are_still_reported_as_such(tools_ws):
    ws, tools = tools_ws
    with pytest.raises(ValueError, match="outside the workspace"):
        tools["write_file"]("../evil.py", "x")

def test_each_session_starts_with_nothing_known(tools_ws):
    from ce_agent.tools import make_workspace_tools
    ws, tools = tools_ws
    tools["read_file"]("textutils.py")
    fresh = {f.__name__: f for f, _ in make_workspace_tools(ws)}
    with pytest.raises(ValueError, match="have not read it"):
        fresh["write_file"]("textutils.py", "x")

def test_the_description_tells_the_model_the_rule():
    from ce_agent.tools import make_workspace_tools
    description = {f.__name__: d for f, d in make_workspace_tools(SimpleNamespace())}["write_file"]
    assert "must be read first" in description and "edit_file" in description


def test_the_blind_overwrite_from_the_real_run_is_refused_and_the_model_recovers(repo, monkeypatch):
    (repo / "textutils.py").write_text(TEXTUTILS)
    sh("git", "add", "-A", cwd=repo)
    sh("git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "textutils", cwd=repo)
    brain = FakeBrain(
        call("write_file", {"file_path": "textutils.py", "content": "def word_count(t):\n    return {}\n"}),     # what the model did
        call("read_file", {"file_path": "textutils.py"}),                                                         # after the refusal
        call("edit_file", {"file_path": "textutils.py", "old_text": "def truncate",
                           "new_text": "def word_count(text):\n    return {}\n\n\ndef truncate"}),
        say("Added word_count."))
    run(repo, monkeypatch, brain)
    assert "already exists and you have not read it" in tool_results(brain.seen[1])[0]
    final = (repo / "textutils.py").read_text()
    assert "def slugify" in final and "def truncate" in final and "def word_count" in final                     # nothing was deleted


def test_the_rule_also_holds_when_the_tools_run_in_a_separate_mcp_server(tools_ws):
    from ce_agent.mcp.client import launch_workspace_server
    ws, _ = tools_ws
    for transport in ("stdio", "http"):
        client = launch_workspace_server(ws, transport=transport)
        try:
            refused = client.call_tool("write_file", {"file_path": "textutils.py", "content": "x\n"})
            assert refused["status"] == "error" and "have not read it" in refused["error"]
            assert client.call_tool("read_file", {"file_path": "textutils.py"})["status"] == "ok"
            assert client.call_tool("write_file", {"file_path": "textutils.py", "content": TEXTUTILS})["status"] == "ok"
        finally:
            client.close()


# ======================================================================
# a tool without parameters called with "null" crashed the whole request (seen with a real model)
# ======================================================================
@pytest.mark.parametrize("raw", ["null", "", "{}"])
def test_a_call_without_arguments_works_however_the_model_writes_it(repo, monkeypatch, raw):
    brain = FakeBrain(call("list_files", raw), say("ok"))
    run(repo, monkeypatch, brain)
    assert "textutils.py" in tool_results(brain.seen[1])[0]
    assert [e for e in events(repo) if e["kind"] == "prompt_end"][0]["reason"] == "done"

@pytest.mark.parametrize("raw", ["[1, 2]", '"text"', "42"])
def test_arguments_that_are_not_an_object_give_the_model_an_error_not_a_crash(repo, monkeypatch, raw):
    brain = FakeBrain(call("list_files", raw), say("ok"), say("ok"))   # the second answer follows the nudge
    run(repo, monkeypatch, brain)
    assert tool_results(brain.seen[1])[0].startswith("ERROR:")
    assert [e for e in events(repo) if e["kind"] == "prompt_end"][0]["reason"] == "done"
