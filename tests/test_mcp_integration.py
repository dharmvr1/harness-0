"""The harness with its workspace tools served over MCP, compared against the in-process path."""
import json
import subprocess
import sys
from types import SimpleNamespace

import ce_agent.main as main_module
import ce_agent.mcp.client as client_module
import pytest
from ce_agent.main import run_agent, run_session
from ce_agent.mcp import protocol as p
from ce_agent.registry import Registry
from ce_agent.trace import find_run, show


def sh(*cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True).stdout


def make_repo(path):
    path.mkdir()
    sh("git", "init", "-q", "-b", "main", cwd=path)
    (path / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    sh("git", "add", "-A", cwd=path)
    sh("git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init", cwd=path)
    return path


@pytest.fixture
def repo(tmp_path):
    return make_repo(tmp_path / "repo")


def docker_ok():
    return subprocess.run(["docker", "ps"], capture_output=True).returncode == 0


def call(name, **args):
    c = SimpleNamespace(id=f"call_{name}_{len(json.dumps(args))}_{abs(hash(json.dumps(args, sort_keys=True))) % 10000}",
                        function=SimpleNamespace(name=name, arguments=json.dumps(args)))
    return SimpleNamespace(content=None, tool_calls=[c])

def say(text):
    return SimpleNamespace(content=text, tool_calls=None)


class Brain:
    model = "fake"
    last_usage = None
    def __init__(self, *script):
        self.script, self.seen = list(script), []
    def think(self, messages, tools):
        self.seen.append(json.loads(json.dumps(messages, default=str)))
        return self.script.pop(0)


def typed(monkeypatch, *answers):
    it = iter(answers)
    def fake_input(prompt=""):
        v = next(it)
        if isinstance(v, BaseException):
            raise v
        return v() if callable(v) else v
    monkeypatch.setattr("builtins.input", fake_input)


def events(repo):
    path = max((repo / ".agent" / "runs").glob("*.jsonl"), key=lambda f: f.stat().st_mtime)
    return [json.loads(l) for l in path.read_text().splitlines()]


# ======================================================================
# the Registry treats remote tools like local ones
# ======================================================================
class FakeMcp:
    def __init__(self):
        self.calls = []
    def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if name == "fails":
            return {"status": "error", "error": "it failed"}
        return {"status": "ok", "content": f"{name} done"}


def test_remote_tools_appear_in_the_llm_tool_list_with_the_servers_schema():
    registry = Registry()
    schema = {"type": "object", "properties": {"path": {"type": "string", "description": "A path"}}, "required": ["path"]}
    registry.register_remote({"name": "read_it", "description": "Reads", "inputSchema": schema}, FakeMcp().call_tool)
    assert registry.get_openai_tools() == [{"type": "function", "function": {"name": "read_it", "description": "Reads", "parameters": schema}}]

def test_calling_a_remote_tool_goes_through_the_client_and_errors_stay_errors():
    mcp, registry = FakeMcp(), Registry()
    for name in ("works", "fails"):
        registry.register_remote({"name": name, "description": "", "inputSchema": {"type": "object", "properties": {}}}, mcp.call_tool)
    assert registry.call("works", {}) == {"status": "ok", "content": "works done"}
    assert registry.call("fails", {}) == {"status": "error", "error": "it failed"}
    assert mcp.calls == [("works", {}), ("fails", {})]

def test_the_registry_still_cleans_up_sloppy_arguments_before_they_go_over_the_wire():
    mcp, registry = FakeMcp(), Registry()
    registry.register_remote({"name": "t", "description": "", "inputSchema": {
        "type": "object", "properties": {"n": {"type": "integer"}, "s": {"type": "string"}}}}, mcp.call_tool)
    registry.call("t", {"n": "4", "s": None})                       # a model sent "4" and null
    assert mcp.calls == [("t", {"n": 4})]

def test_a_tool_with_no_properties_in_its_schema_does_not_crash_the_registry():
    registry = Registry()
    registry.register_remote({"name": "t", "inputSchema": {"type": "object", "additionalProperties": False}}, FakeMcp().call_tool)
    assert registry.call("t", {})["status"] == "ok"


# ======================================================================
# the whole agent over MCP
# ======================================================================
def scripted_brain():
    return Brain(
        call("list_files"),
        call("search", pattern="def"),
        call("read_file", file_path="calc.py"),
        call("edit_file", file_path="calc.py", old_text="a + b", new_text="a - b"),
        call("edit_file", file_path="calc.py", old_text="a + b", new_text="a * b"),       # fails: no longer there
        call("write_file", file_path="new.py", content="x = 1\n"),
        call("run_shell", command="python3 -c 'print(6 * 7)'"),
        call("remember", note="Math lives in calc.py"),
        call("read_file", file_path="missing.py"),                                         # fails: no such file
        call("git_diff"),                                                                   # local tool
        call("update_plan", steps="[x] did everything"),                                    # local tool
        say("all done"),
    )


def tool_messages(brain):
    return [m["content"] for m in brain.seen[-1] if m["role"] == "tool"]


def test_the_agent_behaves_identically_in_process_and_over_mcp(tmp_path, monkeypatch):
    results = {}
    for mode in (None, "stdio"):
        repo = make_repo(tmp_path / f"repo_{mode}")
        brain = scripted_brain()
        typed(monkeypatch, "a")
        run_agent(str(repo), "do everything", unsafe=True, brain=brain, mcp=mode)
        results[mode] = {
            "tool_results": tool_messages(brain),                                          # exactly what the model saw
            "calc": (repo / "calc.py").read_text(),
            "new": (repo / "new.py").read_text(),
            "notes": (repo / "AGENTS.md").read_text(),
        }
    assert results["stdio"] == results[None]
    assert len(results[None]["tool_results"]) == 11
    assert "ERROR" in results[None]["tool_results"][4] and "ERROR" in results[None]["tool_results"][8]    # both failures reached the model
    assert results[None]["calc"] == "def add(a, b):\n    return a - b\n"


def test_the_trace_shows_which_tools_went_over_the_wire(repo, monkeypatch):
    typed(monkeypatch, "r")
    run_agent(str(repo), "go", unsafe=True, brain=scripted_brain(), mcp="stdio")
    ev = events(repo)
    sent = [e["message"]["params"]["name"] for e in ev
            if e["kind"] == "mcp_wire" and e["direction"] == "send" and e["message"]["method"] == "tools/call"]
    assert sent == ["list_files", "search", "read_file", "edit_file", "edit_file", "write_file", "run_shell", "remember", "read_file"]
    assert "git_diff" not in sent and "update_plan" not in sent                              # these stay in the harness
    methods = [e["message"]["method"] for e in ev if e["kind"] == "mcp_wire" and e["direction"] == "send"]
    assert methods[:2] == ["server/discover", "tools/list"]                                  # hello first, then the tool list
    connected = [e for e in ev if e["kind"] == "mcp_connected"][0]
    assert connected["server"].startswith("ce-agent-tools") and connected["version"] == "2026-07-28"
    assert [e for e in ev if e["kind"] == "run_start"][0]["mcp"] == "stdio"
    assert "tools are served over MCP by ce-agent-tools" in show(find_run(str(repo)))

def call_text(repo):
    """The full text of the read_file result as recorded in the tool_call event (not clipped like the wire log)."""
    return [e for e in events(repo) if e["kind"] == "tool_call"][0]["result"]


def test_long_messages_are_clipped_in_the_wire_log(repo, monkeypatch):
    (repo / "big.txt").write_text("x" * 20000 + "\n")
    sh("git", "add", "-A", cwd=repo)
    sh("git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "big", cwd=repo)
    typed(monkeypatch)
    run_agent(str(repo), "go", unsafe=True, brain=Brain(call("read_file", file_path="big.txt"), say("ok")), mcp="stdio")
    wire = [e["message"] for e in events(repo) if e["kind"] == "mcp_wire" and e["direction"] == "recv"]
    file_reply = [m for m in wire if "content" in m.get("result", {})][0]                  # the reply carrying the 20000-char file
    text = file_reply["result"]["content"][0]["text"]
    assert len(text) < 700 and "more characters]" in text                                    # clipped in the log...
    assert len(call_text(repo)) > 15000                                                      # ...while the model got the whole file

def test_the_workspace_tools_are_exactly_what_the_model_is_offered(repo, monkeypatch):
    brain = Brain(say("hi"))
    typed(monkeypatch)
    seen_tools = []
    real_think = brain.think
    brain.think = lambda messages, tools: (seen_tools.append([t["function"]["name"] for t in tools]), real_think(messages, tools))[1]
    run_agent(str(repo), "hello", unsafe=True, brain=brain, mcp="stdio")
    assert seen_tools[0] == ["list_files", "search", "read_file", "edit_file", "write_file", "run_shell", "remember", "git_diff", "update_plan"]


def test_the_server_process_is_gone_when_the_session_ends(repo, monkeypatch):
    launched = []
    real_launch = client_module.launch_workspace_server
    def spy(ws, **kwargs):
        client = real_launch(ws, **kwargs)
        launched.append(client.transport.process)
        return client
    monkeypatch.setattr(main_module, "launch_workspace_server", spy)
    typed(monkeypatch)
    run_agent(str(repo), "hello", unsafe=True, brain=Brain(say("hi")), mcp="stdio")
    assert len(launched) == 1 and launched[0].poll() is not None


def test_a_server_that_cannot_start_fails_cleanly_and_leaves_nothing_behind(repo, monkeypatch):
    monkeypatch.setattr(client_module, "server_command", lambda ws: [sys.executable, "-c", "import sys; sys.stderr.write('bad config\\n'); sys.exit(3)"])
    with pytest.raises(p.McpConnectionError, match="bad config"):
        run_agent(str(repo), "hello", unsafe=True, brain=Brain(say("never reached")), mcp="stdio")
    assert "agent/" not in sh("git", "branch", cwd=repo)
    assert not list((repo / ".agent" / "worktrees").glob("*"))                                # worktree removed

def test_the_cli_reports_a_broken_server_as_an_error_message(repo, monkeypatch, capsys):
    monkeypatch.setattr(client_module, "server_command", lambda ws: ["/definitely/not/a/program"])
    monkeypatch.setattr(main_module, "Brain", lambda: Brain(say("x")))
    monkeypatch.setattr(sys, "argv", ["ce_agent.main", str(repo), "hello", "--once", "--unsafe-local", "--mcp"])
    main_module.main()
    assert "Error: could not start the MCP server" in capsys.readouterr().out

def test_the_mcp_flag_reaches_run_session(monkeypatch):
    captured = {}
    monkeypatch.setattr(main_module, "run_session", lambda *a, **kw: captured.update(kw))
    monkeypatch.setattr(sys, "argv", ["ce_agent.main", "somewhere", "--mcp"])
    main_module.main()
    assert captured["mcp"] == "stdio"
    monkeypatch.setattr(sys, "argv", ["ce_agent.main", "somewhere"])
    main_module.main()
    assert captured["mcp"] is None


@pytest.mark.parametrize("first_call_after_the_kill", ["write", "read"])
def test_the_agent_survives_the_tool_server_being_killed_between_requests(repo, monkeypatch, first_call_after_the_kill):
    clients = []
    real_launch = client_module.launch_workspace_server
    def spy(ws, **kwargs):
        client = real_launch(ws, **kwargs)
        clients.append(client)
        return client
    monkeypatch.setattr(main_module, "launch_workspace_server", spy)

    read = call("read_file", file_path="calc.py")
    write_b = call("write_file", file_path="b.py", content="2\n")
    write_c = call("write_file", file_path="c.py", content="3\n")
    if first_call_after_the_kill == "write":
        script = [say("first"), write_b, read, write_c, say("second")]
    else:
        script = [say("first"), read, write_b, say("second")]
    brain = Brain(*script)

    def kill_the_server():
        clients[0].transport.process.kill()
        clients[0].transport.process.wait()
        return "go on"
    typed(monkeypatch, "hello", kill_the_server, "l", EOFError(), "a")
    run_session(str(repo), unsafe=True, brain=brain, mcp="stdio")
    results = [m["content"] for m in brain.seen[-1] if m["role"] == "tool"]

    if first_call_after_the_kill == "write":
        # The write hit a dead server. The client restarted it but must NOT repeat the write: it may have happened.
        assert "ERROR" in results[0] and "may or may not have completed" in results[0] and "write_file" in results[0]
        assert "def add" in results[1]                                  # the restarted server works
        assert results[2] == "Wrote 2 characters to c.py"
        assert not (repo / "b.py").exists() and (repo / "c.py").read_text() == "3\n"
    else:
        # A read is harmless to repeat, so the model never even noticed the crash.
        assert "def add" in results[0] and "ERROR" not in results[0]
        assert results[1] == "Wrote 2 characters to b.py"
        assert (repo / "b.py").read_text() == "2\n"


@pytest.mark.skipif(not docker_ok(), reason="Docker not running")
def test_commands_run_inside_the_container_when_served_over_mcp(repo, monkeypatch):
    brain = Brain(call("run_shell", command="cat /etc/os-release | head -1; id -u; wget -q -T 2 -O- http://1.1.1.1 || echo NO_NETWORK"), say("ok"))
    typed(monkeypatch)
    run_agent(str(repo), "look around", unsafe=False, brain=brain, mcp="stdio")
    output = brain.seen[-1][-1]["content"]
    assert "Debian" in output                                               # the container's OS, not the host's
    assert "NO_NETWORK" in output                                           # and the container has no network
