"""The MCP server as a real subprocess, driven with raw JSON lines (the way you could do it by hand)."""
import json
import subprocess
import sys
from pathlib import Path

import ce_agent
import pytest
from ce_agent.mcp import protocol as p
from ce_agent.workspace import Workspace

PROJECT_ROOT = Path(ce_agent.__file__).parent.parent


def sh(*cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True).stdout


@pytest.fixture
def ws(tmp_path):
    sh("git", "init", "-q", "-b", "main", cwd=tmp_path)
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    sh("git", "add", "-A", cwd=tmp_path)
    sh("git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init", cwd=tmp_path)
    workspace = Workspace(str(tmp_path), unsafe=True)
    workspace.start()
    yield workspace
    workspace.discard()
    workspace.cleanup()


def server_command(ws):
    return [sys.executable, "-m", "ce_agent.mcp", "--worktree", str(ws.worktree), "--prefix", ws.prefix]


def talk(ws, *lines, timeout=30):
    """Send these lines to a fresh server, close its stdin (= shut down), return (parsed stdout lines, stderr, exit code)."""
    proc = subprocess.run(server_command(ws), input="".join(l + "\n" for l in lines), capture_output=True, text=True,
                          cwd=PROJECT_ROOT, timeout=timeout)
    return [json.loads(l) for l in proc.stdout.splitlines()], proc.stderr, proc.returncode


def call(request_id, name, **arguments):
    return p.encode(p.make_request(request_id, "tools/call", {"name": name, "arguments": arguments}))


def test_a_whole_conversation_over_raw_json_lines(ws):
    replies, stderr, code = talk(
        ws,
        p.encode(p.make_request(1, "server/discover")),
        p.encode(p.make_request(2, "tools/list")),
        call(3, "read_file", file_path="calc.py"),
        call(4, "edit_file", file_path="calc.py", old_text="a + b", new_text="a - b"),
        call(5, "read_file", file_path="calc.py"),
        call(6, "run_shell", command="python3 -c 'import calc; print(calc.add(5, 2))'"),
    )
    assert code == 0                                                          # EOF on stdin = clean exit
    assert [r["id"] for r in replies] == [1, 2, 3, 4, 5, 6]                    # one reply per request, in order
    assert replies[0]["result"]["supportedVersions"] == ["2026-07-28"]
    assert [t["name"] for t in replies[1]["result"]["tools"]] == [
        "list_files", "search", "read_file", "edit_file", "write_file", "run_shell", "remember"]   # not git_diff / update_plan
    assert "def add" in replies[2]["result"]["content"][0]["text"]
    assert "Edited calc.py" in replies[3]["result"]["content"][0]["text"]
    assert "a - b" in replies[4]["result"]["content"][0]["text"]
    assert "3" in replies[5]["result"]["content"][0]["text"] and replies[5]["result"]["isError"] is False
    assert "return a - b" in (ws.workdir / "calc.py").read_text()               # the file really changed in the worktree
    assert "[mcp-server] tools/call edit_file: ok" in stderr                    # logs go to stderr


def test_garbage_blank_lines_and_notifications_do_not_break_the_server(ws):
    replies, _, code = talk(
        ws,
        "this is not json",
        "",
        "   ",
        p.encode({"jsonrpc": "2.0", "method": "notifications/whatever"}),
        p.encode([1, 2, 3]),
        p.encode(p.make_request(7, "tools/list")),
    )
    assert code == 0
    assert replies[0]["error"]["code"] == p.PARSE_ERROR and "id" not in replies[0]
    assert replies[1]["error"]["code"] == p.INVALID_REQUEST
    assert replies[2]["id"] == 7 and replies[2]["result"]["tools"]
    assert len(replies) == 3                                                   # blank lines and the notification produced nothing


def test_tool_errors_travel_as_results_and_protocol_errors_as_errors(ws):
    replies, _, _ = talk(
        ws,
        call(1, "read_file", file_path="nope.py"),
        call(2, "edit_file", file_path="calc.py", old_text="zzz", new_text="y"),
        call(3, "no_such_tool"),
        call(4, "read_file"),
    )
    assert replies[0]["result"]["isError"] is True and "does not exist" in replies[0]["result"]["content"][0]["text"]
    assert replies[1]["result"]["isError"] is True and "not found" in replies[1]["result"]["content"][0]["text"]
    assert replies[2]["error"]["code"] == p.INVALID_PARAMS and "Unknown tool" in replies[2]["error"]["message"]
    assert replies[3]["result"]["isError"] is True and "Missing required argument: file_path" in replies[3]["result"]["content"][0]["text"]


def test_path_escape_is_still_blocked_through_mcp(ws):
    replies, _, _ = talk(ws, call(1, "read_file", file_path="/etc/passwd"), call(2, "write_file", file_path="../evil.txt", content="x"))
    for r in replies:
        assert r["result"]["isError"] is True and "outside the workspace" in r["result"]["content"][0]["text"]
    assert not (ws.worktree.parent / "evil.txt").exists()


def test_stdout_carries_nothing_but_protocol_messages(ws):
    # run_shell prints things; they must come back INSIDE a result, never loose on the protocol stream
    proc = subprocess.run(server_command(ws), input=call(1, "run_shell", command="echo STRAY_OUTPUT; echo ERR >&2") + "\n",
                          capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=30)
    lines = proc.stdout.splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["id"] == 1
    assert "STRAY_OUTPUT" in json.loads(lines[0])["result"]["content"][0]["text"]
    assert "STRAY_OUTPUT" not in proc.stderr


def test_protect_stdout_sends_stray_prints_and_child_output_to_stderr():
    script = ("import os, subprocess\n"
              "from ce_agent.mcp.stdio import protect_stdout\n"
              "out = protect_stdout()\n"
              "print('STRAY PRINT')\n"
              "os.write(1, b'STRAY WRITE TO FD 1\\n')\n"
              "subprocess.run(['echo', 'STRAY FROM CHILD'])\n"
              "out.write('PROTOCOL LINE\\n'); out.flush()\n")
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=30)
    assert proc.stdout == "PROTOCOL LINE\n"
    assert "STRAY PRINT" in proc.stderr and "STRAY WRITE TO FD 1" in proc.stderr and "STRAY FROM CHILD" in proc.stderr


def test_the_server_exits_promptly_when_stdin_closes(ws):
    proc = subprocess.Popen(server_command(ws), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, cwd=PROJECT_ROOT)
    proc.stdin.close()
    assert proc.wait(timeout=10) == 0
    proc.stdout.close(); proc.stderr.close()


def test_a_missing_worktree_is_refused_at_startup(tmp_path):
    proc = subprocess.run([sys.executable, "-m", "ce_agent.mcp", "--worktree", str(tmp_path / "nope")],
                          capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=30)
    assert proc.returncode != 0 and "not a directory" in proc.stderr and proc.stdout == ""


# ---------- the attached workspace the server uses ----------
def test_an_attached_workspace_works_on_files_but_never_owns_the_lifecycle(ws):
    view = Workspace.attach(ws.worktree, ws.prefix)
    assert "def add" in view.read_file("calc.py")
    view.write_file("new.txt", "hi\n")
    assert (ws.workdir / "new.txt").read_text() == "hi\n"                       # same files as the harness's workspace
    assert view.run("echo ok")["stdout"].strip() == "ok"
    with pytest.raises(RuntimeError, match="owned by the harness"):
        view.start()
    view.cleanup()                                                              # must NOT remove anything
    assert ws.worktree.exists() and (ws.workdir / "calc.py").exists()
