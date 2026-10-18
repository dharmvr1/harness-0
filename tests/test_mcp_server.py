"""Tests for McpServer.handle(): dicts in, dicts out, no I/O. Fixtures marked SPEC come from the specification pages."""
import json
from typing import Annotated

import pytest
from ce_agent.mcp import protocol as p
from ce_agent.mcp.server import McpServer, check_arguments
from ce_agent.registry import Registry


def build_registry():
    registry = Registry()

    def echo(text: Annotated[str, "Text to repeat"], times: Annotated[int, "How many times"] = 1):
        return text * times

    def boom():
        raise ValueError("it broke: old_text matches 2 places")

    def nothing():
        return "ok"

    def silent_failure():
        raise KeyError()

    registry.register(echo, "Repeat some text")
    registry.register(boom, "Always fails")
    registry.register(nothing, "Takes no arguments")
    registry.register(silent_failure, "Fails with an empty message")
    return registry


@pytest.fixture
def server():
    return McpServer(build_registry(), instructions="Test tools.", ttl_ms=5000)


def req(method, params=None, id=1, **meta):
    return p.make_request(id, method, params, client_info={"name": "ExampleClient", "version": "1.0.0"}, **meta)


def error_of(response):
    assert "error" in response and "result" not in response, response
    return response["error"]


# ---------- server/discover ----------
def test_discover_matches_the_spec_example(server):                                    # SPEC: discover page
    r = server.handle(p.make_request("discover-1", "server/discover"))
    assert r["jsonrpc"] == "2.0" and r["id"] == "discover-1"
    result = r["result"]
    assert result["resultType"] == "complete"
    assert result["supportedVersions"] == ["2026-07-28"]
    assert result["capabilities"] == {"tools": {}}
    assert result["instructions"] == "Test tools."
    assert result["_meta"][p.META_SERVER_INFO] == {"name": "ce-agent-tools", "version": "0.1.0"}
    assert isinstance(result["ttlMs"], int) and result["ttlMs"] >= 0 and result["cacheScope"] == "public"   # caching MUSTs

def test_discover_without_instructions_leaves_the_field_out():
    r = McpServer(build_registry()).handle(req("server/discover"))
    assert "instructions" not in r["result"]


# ---------- tools/list ----------
def test_list_tools_matches_the_spec_shape(server):                                    # SPEC: tools page
    result = server.handle(req("tools/list"))["result"]
    assert result["resultType"] == "complete" and result["ttlMs"] == 5000 and result["cacheScope"] == "public"
    assert "nextCursor" not in result                                                   # we never paginate
    names = [t["name"] for t in result["tools"]]
    assert names == ["echo", "boom", "nothing", "silent_failure"]                        # deterministic: registration order
    echo = result["tools"][0]
    assert echo["description"] == "Repeat some text"
    assert echo["inputSchema"] == {"type": "object", "required": ["text"], "properties": {
        "text": {"type": "string", "description": "Text to repeat"},
        "times": {"type": "integer", "description": "How many times"}}}

def test_every_input_schema_is_an_object_schema(server):
    for tool in server.handle(req("tools/list"))["result"]["tools"]:
        assert tool["inputSchema"]["type"] == "object"                                  # MUST be a valid object schema, not null

def test_list_tools_is_stable_between_calls(server):
    assert server.handle(req("tools/list", id=1))["result"]["tools"] == server.handle(req("tools/list", id=2))["result"]["tools"]

def test_a_cursor_is_invalid_because_we_never_issue_one(server):
    assert error_of(server.handle(req("tools/list", {"cursor": "abc"})))["code"] == p.INVALID_PARAMS


# ---------- tools/call ----------
def test_call_success_matches_the_spec_shape(server):                                  # SPEC: tools/call response
    r = server.handle(req("tools/call", {"name": "echo", "arguments": {"text": "ab", "times": 2}}, id=2))
    assert r["id"] == 2
    assert r["result"]["resultType"] == "complete" and r["result"]["isError"] is False
    assert r["result"]["content"] == [{"type": "text", "text": "abab"}]

def test_optional_arguments_may_be_left_out(server):
    r = server.handle(req("tools/call", {"name": "echo", "arguments": {"text": "x"}}))
    assert r["result"]["content"][0]["text"] == "x"

def test_arguments_may_be_left_out_entirely_for_tools_without_parameters(server):
    assert server.handle(req("tools/call", {"name": "nothing"}))["result"]["content"][0]["text"] == "ok"

def test_a_failing_tool_is_a_tool_error_not_a_protocol_error(server):                  # SPEC: tool execution errors
    r = server.handle(req("tools/call", {"name": "boom"}))
    assert "error" not in r
    assert r["result"]["isError"] is True and "matches 2 places" in r["result"]["content"][0]["text"]

def test_a_failure_with_no_message_still_says_something(server):
    text = server.handle(req("tools/call", {"name": "silent_failure"}))["result"]["content"][0]["text"]
    assert text == "KeyError"

def test_unknown_tool_is_a_protocol_error(server):                                     # SPEC: -32602 "Unknown tool"
    e = error_of(server.handle(req("tools/call", {"name": "invalid_tool_name"}, id=3)))
    assert e["code"] == -32602 and e["message"] == "Unknown tool: invalid_tool_name"

def test_missing_or_bad_name_and_arguments(server):
    assert error_of(server.handle(req("tools/call", {})))["code"] == p.INVALID_PARAMS
    assert error_of(server.handle(req("tools/call", {"name": 5})))["code"] == p.INVALID_PARAMS
    assert error_of(server.handle(req("tools/call", {"name": "echo", "arguments": ["x"]})))["code"] == p.INVALID_PARAMS


# ---------- argument validation: strict, and the message tells the model what to do ----------
@pytest.mark.parametrize("arguments, fragment", [
    ({}, "Missing required argument: text"),
    ({"text": "a", "colour": "red"}, "Unexpected argument: colour. Valid arguments: text, times"),
    ({"text": 5}, "Argument text must be of type string, got int"),
    ({"text": "a", "times": "2"}, "Argument times must be of type integer, got str"),       # strict: no "4" -> 4 here
    ({"text": "a", "times": True}, "Argument times must be of type integer, got bool"),     # JSON true is not a number
    ({"text": "a", "times": None}, "Argument times must be of type integer, got NoneType"),
])
def test_bad_arguments_are_tool_errors_with_a_helpful_message(server, arguments, fragment):
    r = server.handle(req("tools/call", {"name": "echo", "arguments": arguments}))
    assert r["result"]["isError"] is True and fragment in r["result"]["content"][0]["text"]

def test_a_whole_number_sent_as_a_float_is_accepted():
    cleaned, error = check_arguments({"properties": {"n": {"type": "integer"}}, "required": ["n"]}, {"n": 4.0})
    assert error is None and cleaned == {"n": 4} and isinstance(cleaned["n"], int)
    _, error = check_arguments({"properties": {"n": {"type": "integer"}}}, {"n": 4.5})
    assert "must be of type integer" in error


# ---------- the rest of the protocol surface ----------
def test_unknown_method(server):
    e = error_of(server.handle(req("resources/list")))
    assert e["code"] == p.METHOD_NOT_FOUND and "resources/list" in e["message"]

def test_notifications_get_no_reply(server):
    assert server.handle({"jsonrpc": "2.0", "method": "notifications/anything"}) is None
    assert server.handle({"jsonrpc": "2.0", "method": "tools/list"}) is None            # even a known method, without an id

def test_legacy_initialize_is_refused_and_names_the_versions_we_speak(server):
    e = error_of(server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-11-25"}}))
    assert e["code"] == p.METHOD_NOT_FOUND and "2026-07-28" in e["message"] and e["data"] == {"supported": ["2026-07-28"]}

def test_unsupported_protocol_version_matches_the_spec_example(server):                # SPEC: versioning page
    e = error_of(server.handle(req("tools/list", version="1900-01-01", id=1)))
    assert e["code"] == -32022 and e["message"] == "Unsupported protocol version"
    assert e["data"] == {"supported": ["2026-07-28"], "requested": "1900-01-01"}

def test_requests_without_meta_are_rejected(server):
    for params in ({}, {"_meta": {}}, {"_meta": {p.META_VERSION: "2026-07-28"}}):
        e = error_of(server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": params}))
        assert e["code"] == p.INVALID_PARAMS

def test_the_server_keeps_no_state_between_requests(server):
    # a request with proper _meta works whether or not anything came before it, in any order
    fresh = McpServer(build_registry())
    assert fresh.handle(req("tools/call", {"name": "nothing"}))["result"]["content"][0]["text"] == "ok"
    assert server.handle(req("tools/list"))["result"]["tools"]
    assert fresh.handle(req("tools/list"))["result"]["tools"] == server.handle(req("tools/list"))["result"]["tools"]

def test_malformed_messages_never_crash_the_server(server):
    for message in ([], "x", 5, None, {"jsonrpc": "2.0"}, {"jsonrpc": "2.0", "id": None, "method": "x"},
                    {"jsonrpc": "2.0", "id": 1, "method": "x", "params": [1]}, {"jsonrpc": "2.0", "id": 1, "result": {}}):
        response = server.handle(message)
        assert response["error"]["code"] in (p.INVALID_REQUEST, p.INVALID_PARAMS)

def test_ids_are_echoed_exactly(server):
    for request_id in (1, 0, "abc", "discover-1", 2**40):
        assert server.handle(req("tools/list", id=request_id))["id"] == request_id

def test_every_result_carries_server_info(server):
    for method, params in (("server/discover", None), ("tools/list", None), ("tools/call", {"name": "nothing"})):
        assert server.handle(req(method, params))["result"]["_meta"][p.META_SERVER_INFO]["name"] == "ce-agent-tools"

def test_a_bug_inside_the_server_becomes_an_internal_error_and_is_logged():
    logged = []
    server = McpServer(build_registry(), log=logged.append)
    server.methods["tools/list"] = lambda params: 1 / 0
    e = error_of(server.handle(req("tools/list", id=9)))
    assert e["code"] == p.INTERNAL_ERROR and e["message"] == "Internal error: ZeroDivisionError"
    assert any("ZeroDivisionError" in line for line in logged)
    assert server.handle(req("server/discover"))["result"]["supportedVersions"]          # and it keeps serving


# ---------- text in, text out (what the transport calls) ----------
def test_handle_line_round_trip(server):
    line = p.encode(req("tools/call", {"name": "echo", "arguments": {"text": "line1\nline2"}}))
    reply = server.handle_line(line)
    assert "\n" not in reply                                                              # framing: one line out
    assert json.loads(reply)["result"]["content"][0]["text"] == "line1\nline2"             # the newline survived inside the string

def test_handle_line_bad_json_is_a_parse_error_without_an_id(server):
    response = json.loads(server.handle_line("{this is not json"))
    assert response["error"]["code"] == p.PARSE_ERROR and "id" not in response

def test_handle_line_returns_nothing_for_notifications(server):
    assert server.handle_line(p.encode({"jsonrpc": "2.0", "method": "notifications/x"})) is None
