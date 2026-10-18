"""Tests for the MCP wire format. The fixtures marked SPEC are copied from the specification pages."""
import json

import pytest
from ce_agent.mcp import protocol as p


def meta(**override):
    base = {p.META_VERSION: "2026-07-28", p.META_CLIENT_CAPS: {}}
    base.update(override)
    return base


# ---------- framing ----------
def test_encode_is_always_a_single_line():
    line = p.encode({"text": "line one\nline two\r\nand é 世"})
    assert "\n" not in line and "\r" not in line
    assert json.loads(line)["text"] == "line one\nline two\r\nand é 世"      # nothing lost

def test_decode_bad_json_is_a_parse_error():
    with pytest.raises(p.ProtocolError) as e:
        p.decode("{not json")
    assert e.value.code == p.PARSE_ERROR
    assert "id" not in e.value.to_response()                                           # id unknown, so left out


# ---------- building messages ----------
def test_make_request_adds_the_required_meta():
    msg = p.make_request(7, "tools/list", client_info={"name": "t", "version": "1"})
    assert msg["id"] == 7 and msg["jsonrpc"] == "2.0" and msg["method"] == "tools/list"
    m = msg["params"]["_meta"]
    assert m[p.META_VERSION] == "2026-07-28" and m[p.META_CLIENT_CAPS] == {} and m[p.META_CLIENT_INFO]["name"] == "t"

def test_make_request_keeps_params():
    msg = p.make_request(1, "tools/call", {"name": "x", "arguments": {"a": 1}})
    assert msg["params"]["name"] == "x" and msg["params"]["arguments"] == {"a": 1} and "_meta" in msg["params"]

def test_result_response_defaults_and_server_info():
    r = p.result_response("abc", {"tools": []}, server_info={"name": "s", "version": "1"})
    assert r["id"] == "abc" and r["result"]["resultType"] == "complete"
    assert r["result"]["_meta"][p.META_SERVER_INFO] == {"name": "s", "version": "1"}

def test_error_response_shapes():
    assert p.error_response(1, -32601, "nope") == {"jsonrpc": "2.0", "id": 1, "error": {"code": -32601, "message": "nope"}}
    assert p.error_response(1, -32602, "bad", {"x": 1})["error"]["data"] == {"x": 1}

def test_tool_result_shape():                                                           # SPEC: tools/call response
    assert p.tool_result("hello") == {"resultType": "complete", "content": [{"type": "text", "text": "hello"}], "isError": False}
    assert p.tool_result("boom", is_error=True)["isError"] is True


# ---------- checking requests ----------
def test_valid_request_and_notification():
    kind, rid, method, params = p.check_request({"jsonrpc": "2.0", "id": 3, "method": "tools/list", "params": {"x": 1}})
    assert (kind, rid, method, params) == ("request", 3, "tools/list", {"x": 1})
    assert p.check_request({"jsonrpc": "2.0", "method": "notifications/cancelled"})[0] == "notification"

@pytest.mark.parametrize("message", [
    [],                                                      # not an object
    "tools/list",
    {"id": 1, "method": "x"},                                # no jsonrpc
    {"jsonrpc": "1.0", "id": 1, "method": "x"},              # wrong version
    {"jsonrpc": "2.0", "id": 1},                             # no method
    {"jsonrpc": "2.0", "id": 1, "method": ""},               # empty method
    {"jsonrpc": "2.0", "id": 1, "method": 5},                # method not a string
    {"jsonrpc": "2.0", "id": 1, "result": {}},               # a response, clients may not send those
])
def test_malformed_requests_are_invalid_request(message):
    with pytest.raises(p.ProtocolError) as e:
        p.check_request(message)
    assert e.value.code == p.INVALID_REQUEST

@pytest.mark.parametrize("bad_id", [None, True, False, 1.5, [], {}])
def test_ids_must_be_string_or_integer_never_null(bad_id):
    with pytest.raises(p.ProtocolError) as e:
        p.check_request({"jsonrpc": "2.0", "id": bad_id, "method": "x"})
    assert e.value.code == p.INVALID_REQUEST and e.value.request_id is None

def test_error_keeps_the_id_when_it_is_readable():
    with pytest.raises(p.ProtocolError) as e:
        p.check_request({"jsonrpc": "2.0", "id": "req-9", "method": 5})
    assert e.value.to_response()["id"] == "req-9"

def test_params_must_be_an_object():
    with pytest.raises(p.ProtocolError) as e:
        p.check_request({"jsonrpc": "2.0", "id": 1, "method": "x", "params": [1, 2]})
    assert e.value.code == p.INVALID_PARAMS and e.value.request_id == 1


# ---------- checking _meta ----------
def test_spec_tools_call_request_passes():                                              # SPEC: tools page + _meta
    message = {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
               "params": {"name": "get_weather", "arguments": {"location": "New York"},
                          "_meta": {p.META_VERSION: "2026-07-28", p.META_CLIENT_INFO: {"name": "ExampleClient", "version": "1.0.0"},
                                    p.META_CLIENT_CAPS: {}}}}
    _, rid, _, params = p.check_request(message)
    assert p.check_meta(params, rid)[p.META_CLIENT_INFO]["name"] == "ExampleClient"

def test_missing_meta_is_invalid_params():
    for params in ({}, {"_meta": "x"}, {"_meta": None}):
        with pytest.raises(p.ProtocolError) as e:
            p.check_meta(params, 5)
        assert e.value.code == p.INVALID_PARAMS and e.value.request_id == 5

def test_each_required_meta_field_is_required():
    with pytest.raises(p.ProtocolError) as e:
        p.check_meta({"_meta": {p.META_CLIENT_CAPS: {}}})
    assert e.value.code == p.INVALID_PARAMS and p.META_VERSION in e.value.message
    with pytest.raises(p.ProtocolError) as e:
        p.check_meta({"_meta": {p.META_VERSION: "2026-07-28"}})
    assert e.value.code == p.INVALID_PARAMS and p.META_CLIENT_CAPS in e.value.message
    with pytest.raises(p.ProtocolError):
        p.check_meta({"_meta": meta(**{p.META_CLIENT_CAPS: "nope"})})
    with pytest.raises(p.ProtocolError):
        p.check_meta({"_meta": meta(**{p.META_CLIENT_INFO: "nope"})})

def test_client_info_is_optional():
    assert p.check_meta({"_meta": meta()})

def test_unsupported_version_matches_the_spec_example():                                # SPEC: versioning page
    with pytest.raises(p.ProtocolError) as e:
        p.check_meta({"_meta": meta(**{p.META_VERSION: "1900-01-01"})}, request_id=1)
    response = e.value.to_response()
    assert response["id"] == 1 and response["error"]["code"] == -32022
    assert response["error"]["message"] == "Unsupported protocol version"
    assert response["error"]["data"] == {"supported": p.SUPPORTED_VERSIONS, "requested": "1900-01-01"}

def test_version_is_checked_before_the_other_fields():
    # a future version may define the other fields differently, so an unknown version must win
    with pytest.raises(p.ProtocolError) as e:
        p.check_meta({"_meta": {p.META_VERSION: "2099-01-01"}})
    assert e.value.code == p.UNSUPPORTED_PROTOCOL_VERSION

def test_error_codes_match_the_spec():
    assert (p.PARSE_ERROR, p.INVALID_REQUEST, p.METHOD_NOT_FOUND, p.INVALID_PARAMS, p.INTERNAL_ERROR) == (-32700, -32600, -32601, -32602, -32603)
    assert (p.HEADER_MISMATCH, p.MISSING_CLIENT_CAPABILITY, p.UNSUPPORTED_PROTOCOL_VERSION) == (-32020, -32021, -32022)
