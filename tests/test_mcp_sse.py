"""The Server-Sent Events parser."""
import json

from ce_agent.mcp.sse import parse_sse


def events(text):
    return list(parse_sse(text.splitlines(keepends=True)))


def test_one_event():
    assert events("data: hello\n\n") == [{"event": "message", "data": "hello", "id": None}]

def test_several_events_in_order():
    assert [e["data"] for e in events("data: one\n\ndata: two\n\ndata: three\n\n")] == ["one", "two", "three"]

def test_multiple_data_lines_are_joined_with_a_newline():
    assert events("data: line1\ndata: line2\ndata: line3\n\n")[0]["data"] == "line1\nline2\nline3"

def test_only_one_space_after_the_colon_is_removed():
    assert events("data:no space\n\n")[0]["data"] == "no space"
    assert events("data:  two spaces\n\n")[0]["data"] == " two spaces"

def test_comments_are_ignored_even_in_the_middle_of_an_event():   # keep-alives look like ": ping"
    assert [e["data"] for e in events(": keep-alive\n\ndata: a\n: another comment\ndata: b\n\n:\n\n")] == ["a\nb"]

def test_crlf_line_endings():
    assert events("data: one\r\n\r\ndata: two\r\n\r\n") == [
        {"event": "message", "data": "one", "id": None}, {"event": "message", "data": "two", "id": None}]

def test_event_names_and_ids():
    out = events("event: progress\nid: 7\ndata: x\n\ndata: y\n\n")
    assert out[0] == {"event": "progress", "data": "x", "id": "7"}
    assert out[1] == {"event": "message", "data": "y", "id": "7"}          # the event name resets, the last id sticks

def test_an_id_containing_nul_is_ignored():
    assert events("id: a\0b\ndata: x\n\n")[0]["id"] is None

def test_blank_lines_without_data_produce_nothing():
    assert events("\n\n\nevent: x\n\n") == []

def test_an_empty_data_line_is_still_an_event():
    assert events("data:\n\n")[0]["data"] == ""

def test_unknown_fields_and_retry_are_ignored():
    assert [e["data"] for e in events("retry: 3000\nfoo: bar\ndata: ok\n\n")] == ["ok"]

def test_an_event_cut_off_before_its_blank_line_is_dropped():       # the stream ended mid-event
    assert events("data: complete\n\ndata: cut off") == [{"event": "message", "data": "complete", "id": None}]

def test_works_with_lines_without_line_endings():
    assert [e["data"] for e in parse_sse(["data: a", "", "data: b", ""])] == ["a", "b"]

def test_a_realistic_mcp_stream():
    progress = {"jsonrpc": "2.0", "method": "notifications/progress", "params": {"progressToken": 1, "progress": 1.0, "message": "running"}}
    final = {"jsonrpc": "2.0", "id": 1, "result": {"resultType": "complete", "content": [], "isError": False}}
    stream = f"data: {json.dumps(progress)}\n\n: keep-alive\n\ndata: {json.dumps(final)}\n\n"
    messages = [json.loads(e["data"]) for e in events(stream)]
    assert messages == [progress, final]
