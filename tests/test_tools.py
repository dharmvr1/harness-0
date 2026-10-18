"""Tests for the tools: read_file, edit_file, search, git_diff, and the registry that exposes them."""
import json
import subprocess
from types import SimpleNamespace

import pytest
from ce_agent.main import run_agent
from ce_agent.tools import make_tools
from ce_agent.registry import Registry
from ce_agent.workspace import Workspace

CALC = "def add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return a - b\n"


def sh(*cmd, cwd):
    subprocess.run(cmd, cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    sh("git", "init", "-q", "-b", "main", cwd=tmp_path)
    (tmp_path / "calc.py").write_text(CALC)
    (tmp_path / "dup.py").write_text("x = 1\nx = 1\n")
    (tmp_path / "crlf.py").write_text("a = 1\r\nb = 2\r\n", newline="")
    (tmp_path / "blob.bin").write_bytes(b"\x00\x01\xff\xfe return a + b")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "util.py").write_text("def helper():\n    return 'a + b'\n")
    (tmp_path / ".gitignore").write_text("secret.txt\n")
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


# ---------- read_file ----------
def test_read_file_has_line_numbers(ws):
    out = ws.read_file("calc.py")
    assert "    1  def add(a, b):" in out and "    5      return a - b" in out
    assert out.endswith("[lines 1-5 of 5]")

def test_read_file_range(ws):
    out = ws.read_file("calc.py", start_line=4, end_line=5)
    assert "def add" not in out and "def sub" in out and out.endswith("[lines 4-5 of 5]")

def test_read_file_errors_are_helpful(ws):
    with pytest.raises(FileNotFoundError, match="search"):
        ws.read_file("nope.py")
    with pytest.raises(ValueError, match="binary"):
        ws.read_file("blob.bin")
    with pytest.raises(ValueError, match="out of range"):
        ws.read_file("calc.py", start_line=99)


# ---------- edit_file ----------
def test_edit_file_replaces_exactly_one_place(ws):
    msg = ws.edit_file("calc.py", "return a - b", "return a - b  # subtract")
    assert "line 5" in msg
    assert ws.read_file("calc.py").count("# subtract") == 1
    assert "def add" in ws.read_file("calc.py")             # rest of the file untouched

def test_edit_file_not_found(ws):
    with pytest.raises(ValueError, match="not found.*read_file"):
        ws.edit_file("calc.py", "def multiply", "def mul")

def test_edit_file_ambiguous(ws):
    with pytest.raises(ValueError, match="matches 2 places"):
        ws.edit_file("dup.py", "x = 1", "x = 2")
    assert (ws.workdir / "dup.py").read_text() == "x = 1\nx = 1\n"   # nothing changed

def test_edit_file_other_errors(ws):
    with pytest.raises(FileNotFoundError, match="write_file"):
        ws.edit_file("new.py", "a", "b")
    with pytest.raises(ValueError, match="empty"):
        ws.edit_file("calc.py", "", "b")
    with pytest.raises(ValueError, match="same"):
        ws.edit_file("calc.py", "def add", "def add")
    with pytest.raises(ValueError, match="CRLF"):
        ws.edit_file("crlf.py", "a = 1", "a = 2")
    with pytest.raises(ValueError, match="outside"):
        ws.edit_file("../evil.py", "a", "b")


# ---------- search ----------
def test_search_finds_matches_with_file_and_line(ws):
    out = ws.search("def sub")
    assert out == "calc.py:4:def sub(a, b):"

def test_search_regex_and_path_filter(ws):
    assert "sub/util.py" in ws.search(r"return .a \+ b.")
    only_sub = ws.search("return", path="sub")
    assert "util.py" in only_sub and "calc.py" not in only_sub

def test_search_skips_binary_and_ignored_but_sees_new_files(ws):
    (ws.workdir / "secret.txt").write_text("return a + b")         # in .gitignore
    ws.write_file("brand_new.py", "return a + b\n")                # not committed yet
    out = ws.search(r"return a \+ b")
    assert "blob.bin" not in out and "secret.txt" not in out and "brand_new.py" in out

def test_search_no_match_and_bad_regex(ws):
    assert ws.search("zzz_nothing") == "No matches."
    with pytest.raises(ValueError, match="valid regex"):
        ws.search("(unclosed")

def test_search_caps_results(ws):
    ws.write_file("many.txt", "".join(f"hit {i}\n" for i in range(80)))
    out = ws.search("hit", max_results=10)
    assert len(out.splitlines()) == 11 and "70 more matches" in out


# ---------- git_diff ----------
def test_diff_now(ws):
    assert ws.diff_now() == "No changes yet."
    ws.edit_file("calc.py", "return a + b", "return a + b + 0")
    ws.write_file("new.py", "x = 1\n")
    diff = ws.diff_now()
    assert "+    return a + b + 0" in diff and "new.py" in diff


# ---------- registry ----------
def test_registry_handles_optional_sloppy_args(ws):
    reg = Registry()
    for f, d in make_tools(ws):
        reg.register(f, d)
    # model sends numbers as strings and null for the optional arg
    r = reg.call("read_file", {"file_path": "calc.py", "start_line": "4", "end_line": None})
    assert r["status"] == "ok" and "def sub" in r["content"] and "def add" not in r["content"]
    r = reg.call("read_file", {"file_path": "calc.py", "start_line": "four"})
    assert r["status"] == "error" and "whole number" in r["error"]
    # tool errors come back as text, not exceptions
    r = reg.call("edit_file", {"file_path": "dup.py", "old_text": "x = 1", "new_text": "y"})
    assert r["status"] == "error" and "2 places" in r["error"]


# ---------- through the agent loop ----------
def test_model_sees_error_and_recovers(repo, monkeypatch):
    def call(name, **args):
        c = SimpleNamespace(id="c", function=SimpleNamespace(name=name, arguments=json.dumps(args)))
        return SimpleNamespace(content=None, tool_calls=[c])

    seen = []
    script = [call("edit_file", file_path="dup.py", old_text="x = 1", new_text="x = 2"),        # ambiguous -> error
              call("edit_file", file_path="dup.py", old_text="x = 1\nx = 1", new_text="x = 2\nx = 2"),
              call("git_diff"),
              SimpleNamespace(content="done", tool_calls=None)]

    class Brain:
        model = "fake"
        def think(self, messages, tools):
            seen.append(list(messages))
            assert {t["function"]["name"] for t in tools} == {
                "list_files", "search", "read_file", "edit_file", "write_file", "run_shell", "remember", "git_diff", "update_plan"}
            return script.pop(0)

    monkeypatch.setattr("builtins.input", lambda _="": "a")
    run_agent(str(repo), "t", unsafe=True, brain=Brain())
    # after the first call the model was shown the helpful error text
    assert "matches 2 places" in seen[1][-1]["content"]
    assert (repo / "dup.py").read_text() == "x = 2\nx = 2\n"
