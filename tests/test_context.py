"""Tests for the context helpers (plain functions on the messages list)."""
import json
from types import SimpleNamespace

from ce_agent.context import as_message, clear_old_tool_outputs, estimate_tokens, repair_messages


def assistant_call(call_id, name="read_file", **args):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}

def tool_result(call_id, text):
    return {"role": "tool", "tool_call_id": call_id, "content": text}

def conversation(n, size=1000):
    msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}]
    for i in range(n):
        msgs += [assistant_call(f"c{i}", file_path=f"f{i}.py"), tool_result(f"c{i}", "x" * size)]
    return msgs


def test_as_message_from_sdk_like_objects():
    call = SimpleNamespace(id="c1", function=SimpleNamespace(name="list_files", arguments="{}"))
    m = as_message(SimpleNamespace(content=None, tool_calls=[call]))
    assert m == {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "list_files", "arguments": "{}"}}]}
    assert as_message(SimpleNamespace(content="hi", tool_calls=None)) == {"role": "assistant", "content": "hi"}

def test_estimate_grows_with_content():
    assert estimate_tokens(conversation(10)) > estimate_tokens(conversation(2)) > 0
    assert estimate_tokens(conversation(2), tools=[{"x": "y" * 3000}]) > estimate_tokens(conversation(2))


def test_clear_keeps_newest_results_and_says_what_was_cleared():
    msgs = conversation(10)
    assert clear_old_tool_outputs(msgs, keep_last=3) == 7
    results = [m for m in msgs if m["role"] == "tool"]
    assert all(m["content"].startswith("[output cleared") for m in results[:7])
    assert "read_file" in results[0]["content"] and "f0.py" in results[0]["content"]   # you can tell what it was
    assert all(m["content"] == "x" * 1000 for m in results[7:])                         # newest untouched
    assert results[0]["tool_call_id"] == "c0"                                            # pairing preserved

def test_clear_leaves_small_results_and_other_messages_alone():
    msgs = conversation(5, size=100)
    before = json.dumps(msgs)
    assert clear_old_tool_outputs(msgs, keep_last=1) == 0 and json.dumps(msgs) == before

def test_clear_twice_does_nothing_the_second_time():
    msgs = conversation(10)
    clear_old_tool_outputs(msgs, keep_last=2)
    assert clear_old_tool_outputs(msgs, keep_last=2) == 0

def test_clear_shrinks_the_estimate():
    msgs = conversation(10, size=3000)
    before = estimate_tokens(msgs)
    clear_old_tool_outputs(msgs, keep_last=2)
    assert estimate_tokens(msgs) < before / 2


def test_repair_adds_missing_tool_results_in_the_right_place():
    two_calls = {"role": "assistant", "content": "", "tool_calls": [
        {"id": "a", "type": "function", "function": {"name": "list_files", "arguments": "{}"}},
        {"id": "b", "type": "function", "function": {"name": "list_files", "arguments": "{}"}}]}
    msgs = [{"role": "user", "content": "go"}, two_calls, tool_result("a", "ok")]      # interrupted before "b"
    assert repair_messages(msgs) == 1
    assert [m["role"] for m in msgs] == ["user", "assistant", "tool", "tool"]
    assert msgs[3]["tool_call_id"] == "b" and "interrupted" in msgs[3]["content"]

def test_repair_with_no_results_at_all_and_when_nothing_is_missing():
    msgs = [{"role": "user", "content": "go"}, assistant_call("a"), {"role": "user", "content": "next"}]
    assert repair_messages(msgs) == 1 and msgs[2]["role"] == "tool" and msgs[3]["role"] == "user"
    assert repair_messages(msgs) == 0
    assert repair_messages(conversation(3)) == 0
