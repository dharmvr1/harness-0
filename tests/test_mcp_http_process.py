"""The HTTP tool server as a real subprocess started by the harness, and the whole agent running over HTTP."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import ce_agent
import ce_agent.main as main_module
import ce_agent.mcp.client as client_module
import pytest
from ce_agent.main import run_agent
from ce_agent.mcp import protocol as p
from ce_agent.mcp.client import McpClient, launch_workspace_server, server_command
from ce_agent.mcp.stdio import server_env
from ce_agent.mcp.streamable_http import HttpServerProcess
from ce_agent.workspace import Workspace

PROJECT_ROOT = Path(ce_agent.__file__).parent.parent


def sh(*cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True).stdout


@pytest.fixture
def repo(tmp_path):
    sh("git", "init", "-q", "-b", "main", cwd=tmp_path)
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    sh("git", "add", "-A", cwd=tmp_path)
    sh("git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init", cwd=tmp_path)
    return tmp_path


@pytest.fixture
def ws(repo):
    workspace = Workspace(str(repo), unsafe=True)
    workspace.start()
    yield workspace
    workspace.discard()
    workspace.cleanup()


@pytest.fixture
def transport(ws):
    t = HttpServerProcess(server_command(ws), env=server_env(), cwd=PROJECT_ROOT, timeout=30)
    yield t
    t.close()


def raw_post(url, token, body):
    import http.client
    from urllib.parse import urlsplit
    parts = urlsplit(url)
    conn = http.client.HTTPConnection(parts.hostname, parts.port, timeout=10)
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream",
               "MCP-Protocol-Version": "2026-07-28", "Mcp-Method": "server/discover"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    conn.request("POST", parts.path, body=body, headers=headers)
    status = conn.getresponse().status
    conn.close()
    return status


DISCOVER = p.encode(p.make_request(1, "server/discover")).encode()


# ======================================================================
# the server process
# ======================================================================
def test_the_server_starts_announces_itself_and_serves(transport):
    client = McpClient(transport)
    transport.start()
    assert transport.url.startswith("http://127.0.0.1:") and transport.url.endswith("/mcp")
    assert client.discover()["supportedVersions"] == ["2026-07-28"]
    assert client.call_tool("read_file", {"file_path": "calc.py"})["content"].startswith("    1  def add")

@pytest.mark.skipif(not os.path.exists("/proc/self/cmdline"), reason="needs /proc")
def test_the_token_is_not_visible_on_the_command_line(transport):
    transport.start()
    cmdline = Path(f"/proc/{transport.process.pid}/cmdline").read_bytes().decode()
    assert transport.token not in cmdline and "--token" not in cmdline
    assert raw_post(transport.url, transport.token, DISCOVER) == 200                       # but it does authenticate

def test_requests_without_the_token_are_refused(transport):
    transport.start()
    assert raw_post(transport.url, None, DISCOVER) == 401
    assert raw_post(transport.url, "guess", DISCOVER) == 401

def test_closing_stdin_stops_the_server_so_it_cannot_outlive_the_harness(transport):
    transport.start()
    process = transport.process
    process.stdin.close()
    assert process.wait(timeout=10) == 0

def test_a_server_started_with_no_stdin_watch_survives_stdin_closing(ws):
    process = subprocess.Popen(server_command(ws) + ["--transport", "http", "--port", "0", "--no-stdin-watch"], stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=PROJECT_ROOT, env=server_env())
    try:
        assert json.loads(process.stdout.readline())["ready"] is True
        time.sleep(1)
        assert process.poll() is None
    finally:
        process.terminate(); process.wait(timeout=10); process.stdout.close(); process.stderr.close()

def test_a_server_with_no_token_warns_on_stderr_and_stays_on_loopback(ws):
    process = subprocess.Popen(server_command(ws) + ["--transport", "http", "--port", "0", "--no-stdin-watch"], stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=PROJECT_ROOT, env=server_env())
    try:
        url = json.loads(process.stdout.readline())["url"]
        assert url.startswith("http://127.0.0.1:")
        assert raw_post(url, None, DISCOVER) == 200
    finally:
        process.terminate()
        _, stderr = process.communicate(timeout=10)
    assert "WARNING: no CE_MCP_TOKEN" in stderr

def test_the_server_refuses_to_listen_beyond_loopback_without_a_token(ws):
    result = subprocess.run(server_command(ws) + ["--transport", "http", "--host", "0.0.0.0"], capture_output=True, text=True,
                            cwd=PROJECT_ROOT, env=server_env(), timeout=30, stdin=subprocess.DEVNULL)
    assert result.returncode != 0 and "without an authentication token" in result.stderr and result.stdout == ""

def test_restarting_gives_a_new_port_and_a_new_token(transport):
    transport.start()
    old_url, old_token = transport.url, transport.token
    transport.restart()
    assert transport.token != old_token
    assert raw_post(transport.url, transport.token, DISCOVER) == 200
    assert raw_post(transport.url, old_token, DISCOVER) == 401                              # the old token is dead
    assert old_url != transport.url or old_token != transport.token

def test_a_killed_server_is_detected_and_a_read_recovers(transport):
    client = McpClient(transport)
    transport.start()
    client.discover()
    transport.process.kill(); transport.process.wait()
    with pytest.raises(p.McpConnectionError, match="not running"):
        transport.request(p.make_request(5, "tools/list"))
    assert client.call_tool("read_file", {"file_path": "calc.py"})["status"] == "ok"        # restarted and retried
    transport.process.kill(); transport.process.wait()
    result = client.call_tool("write_file", {"file_path": "x.txt", "content": "x"})          # a write is not repeated
    assert result["status"] == "error" and "may or may not have completed" in result["error"]

def test_a_server_that_dies_at_startup_reports_why(ws):
    t = HttpServerProcess([sys.executable, "-c", "import sys; sys.stderr.write('bad config\\n'); sys.exit(3)"], timeout=10)
    # the command gets "--transport http --port 0" appended, which the one-liner ignores
    with pytest.raises(p.McpConnectionError, match="(?s)failed to start.*exit code 3.*bad config"):
        t.start()

def test_a_server_that_never_announces_itself_times_out(ws):
    t = HttpServerProcess([sys.executable, "-c", "import time; time.sleep(30)"], timeout=10, startup_timeout=1)
    started = time.monotonic()
    with pytest.raises(p.McpTimeout, match="did not announce itself"):
        t.start()
    assert time.monotonic() - started < 10 and t.process is None

def test_a_program_that_does_not_exist_is_a_connection_error():
    with pytest.raises(p.McpConnectionError, match="could not start"):
        HttpServerProcess(["/definitely/not/a/program"]).start()

def test_garbage_instead_of_the_announcement_is_an_error(ws):
    t = HttpServerProcess([sys.executable, "-c", "print('hello, world')"], timeout=10)
    with pytest.raises(p.McpConnectionError, match="failed to start"):
        t.start()


# ======================================================================
# progress and cancellation of a real command
# ======================================================================
def test_run_shell_reports_progress_over_http(ws):
    client = launch_workspace_server(ws, transport="http")
    seen = []
    client.on_progress = seen.append
    try:
        result = client.call_tool("run_shell", {"command": "sleep 2.5; echo finished"})
    finally:
        client.close()
    assert result["status"] == "ok" and "finished" in result["content"]
    assert len(seen) >= 2
    assert [m["progress"] for m in seen] == sorted(m["progress"] for m in seen)              # only goes up
    assert all(m["message"].startswith("command running for") for m in seen)

def test_a_command_that_finishes_quickly_sends_no_progress(ws):
    client = launch_workspace_server(ws, transport="http")
    seen = []
    client.on_progress = seen.append
    try:
        assert client.call_tool("run_shell", {"command": "echo quick"})["status"] == "ok"
    finally:
        client.close()
    assert seen == []

def test_the_clients_timeout_cancels_the_command_without_restarting_the_server(ws):
    client = launch_workspace_server(ws, transport="http", timeout=1.5)
    marker = ws.workdir / "marker.txt"
    pid_before = client.transport.process.pid
    try:
        result = client.call_tool("run_shell", {"command": "(sleep 4; touch marker.txt) & wait"})
        assert result["status"] == "error" and "within 1.5s" in result["error"]
        assert "timed out and was cancelled" in result["error"] and "may or may not have completed" in result["error"]
        time.sleep(5)                                                                       # long enough for the marker to appear
        assert not marker.exists()                                                          # the process group was killed on cancel
        assert client.transport.process.pid == pid_before                                   # the server was NOT restarted...
        assert client.call_tool("read_file", {"file_path": "calc.py"})["status"] == "ok"     # ...and is free for the next call
    finally:
        client.close()

def test_restarting_a_stuck_stdio_server_stops_the_commands_it_was_running(ws):
    client = launch_workspace_server(ws, transport="stdio", timeout=1)
    marker = ws.workdir / "marker.txt"
    try:
        result = client.call_tool("run_shell", {"command": "(sleep 5; touch marker.txt) & wait"})
        assert result["status"] == "error" and "stopped and was restarted" in result["error"]
        time.sleep(6)
        assert not marker.exists()                                                          # SIGTERM -> clean exit -> commands killed
    finally:
        client.close()

@pytest.mark.parametrize("transport_name", ["stdio", "http"])
def test_sigterm_makes_the_server_exit_cleanly(ws, transport_name):
    extra = ["--transport", "http", "--port", "0"] if transport_name == "http" else []
    process = subprocess.Popen(server_command(ws) + extra, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, cwd=PROJECT_ROOT, env=server_env())
    try:
        if transport_name == "http":
            process.stdout.readline()                                                       # wait until it is up
        else:
            time.sleep(1)
        process.terminate()
        assert process.wait(timeout=10) == 0
    finally:
        for pipe in (process.stdin, process.stdout, process.stderr):
            pipe.close()

def test_stdio_also_reports_progress(ws):
    client = launch_workspace_server(ws, transport="stdio")
    seen = []
    client.on_progress = seen.append
    try:
        result = client.call_tool("run_shell", {"command": "sleep 2.5; echo finished"})
    finally:
        client.close()
    assert result["status"] == "ok" and len(seen) >= 2 and seen[0]["message"].startswith("command running for")


# ======================================================================
# the whole agent over HTTP
# ======================================================================
def call(name, **args):
    c = SimpleNamespace(id=f"call_{name}_{abs(hash(json.dumps(args, sort_keys=True))) % 100000}",
                        function=SimpleNamespace(name=name, arguments=json.dumps(args)))
    return SimpleNamespace(content=None, tool_calls=[c])


class Brain:
    model = "fake"
    last_usage = None
    def __init__(self, *script):
        self.script, self.seen = list(script), []
    def think(self, messages, tools):
        self.seen.append(json.loads(json.dumps(messages, default=str)))
        return self.script.pop(0)


def events(repo):
    path = max((repo / ".agent" / "runs").glob("*.jsonl"), key=lambda f: f.stat().st_mtime)
    return [json.loads(l) for l in path.read_text().splitlines()]


def test_a_session_over_http_logs_the_wire_and_the_progress(repo, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _="": "r")
    brain = Brain(call("run_shell", command="sleep 2.2; echo hi"), SimpleNamespace(content="done", tool_calls=None))
    run_agent(str(repo), "go", unsafe=True, brain=brain, mcp="http")
    ev = events(repo)
    assert [e for e in ev if e["kind"] == "run_start"][0]["mcp"] == "http"
    assert [e for e in ev if e["kind"] == "mcp_connected"][0]["server"].startswith("ce-agent-tools")
    progress = [e["message"]["params"] for e in ev if e["kind"] == "mcp_wire" and e["direction"] == "recv"
                and e["message"].get("method") == "notifications/progress"]
    assert len(progress) >= 2 and [m["progress"] for m in progress] == sorted(m["progress"] for m in progress)
    assert "hi" in brain.seen[-1][-1]["content"]

def test_the_http_server_process_is_gone_when_the_session_ends(repo, monkeypatch):
    launched = []
    real = client_module.launch_workspace_server
    def spy(ws, **kwargs):
        client = real(ws, **kwargs)
        launched.append(client.transport.process)
        return client
    monkeypatch.setattr(main_module, "launch_workspace_server", spy)
    monkeypatch.setattr("builtins.input", lambda _="": "r")
    run_agent(str(repo), "hello", unsafe=True, brain=Brain(SimpleNamespace(content="hi", tool_calls=None)), mcp="http")
    assert len(launched) == 1 and launched[0].poll() is not None

def test_a_server_that_cannot_start_over_http_fails_cleanly(repo, monkeypatch):
    monkeypatch.setattr(client_module, "server_command", lambda ws: [sys.executable, "-c", "import sys; sys.stderr.write('bad config\\n'); sys.exit(3)"])
    with pytest.raises(p.McpConnectionError, match="bad config"):
        run_agent(str(repo), "hello", unsafe=True, brain=Brain(SimpleNamespace(content="x", tool_calls=None)), mcp="http")
    assert "agent/" not in sh("git", "branch", cwd=repo) and not list((repo / ".agent" / "worktrees").glob("*"))

def test_an_unknown_transport_name_is_rejected():
    with pytest.raises(ValueError, match="unknown MCP transport"):
        launch_workspace_server(SimpleNamespace(worktree="/x", prefix="", container=None, command_timeout=60), transport="carrier-pigeon")


def docker_ok():
    return subprocess.run(["docker", "ps"], capture_output=True).returncode == 0

@pytest.mark.skipif(not docker_ok(), reason="Docker not running")
def test_commands_run_inside_the_container_when_served_over_http(repo, monkeypatch):
    brain = Brain(call("run_shell", command="cat /etc/os-release | head -1; wget -q -T 2 -O- http://1.1.1.1 || echo NO_NETWORK"),
                  SimpleNamespace(content="ok", tool_calls=None))
    monkeypatch.setattr("builtins.input", lambda _="": "r")
    run_agent(str(repo), "look around", unsafe=False, brain=brain, mcp="http")
    output = brain.seen[-1][-1]["content"]
    assert "Debian" in output and "NO_NETWORK" in output


# ======================================================================
# the command line
# ======================================================================
@pytest.mark.parametrize("flag, expected", [("--mcp", "stdio"), ("--mcp=stdio", "stdio"), ("--mcp=http", "http"), (None, None)])
def test_the_mcp_flag_choices(monkeypatch, flag, expected):
    captured = {}
    monkeypatch.setattr(main_module, "run_session", lambda *a, **kw: captured.update(kw))
    monkeypatch.setattr(sys, "argv", ["ce_agent.main", "somewhere"] + ([flag] if flag else []))
    main_module.main()
    assert captured["mcp"] == expected

def test_an_unknown_mcp_transport_on_the_command_line_is_an_error_message(monkeypatch, capsys):
    called = []
    monkeypatch.setattr(main_module, "run_session", lambda *a, **kw: called.append(1))
    monkeypatch.setattr(sys, "argv", ["ce_agent.main", "somewhere", "--mcp=carrier-pigeon"])
    main_module.main()
    assert "--mcp must be stdio or http" in capsys.readouterr().out and not called
