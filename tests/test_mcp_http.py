"""The Streamable HTTP transport: a real server on a real port, driven with raw HTTP and with HttpClient."""
import http.client
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Annotated

import pytest
from ce_agent.mcp import protocol as p
from ce_agent.mcp.client import McpClient
from ce_agent.mcp.headers import encode_header_value
from ce_agent.mcp.progress import RequestCancelled, report_progress
from ce_agent.mcp.server import McpServer
from ce_agent.mcp.streamable_http import HttpClient, accepts_json_and_sse, serve_http
from ce_agent.registry import Registry

TOKEN = "test-token-123"


# ======================================================================
# a server with a few tools, running in a thread
# ======================================================================
def build_registry(state):
    registry = Registry()

    def echo(text: Annotated[str, "Text to return"]):
        state["echoed"] = state.get("echoed", 0) + 1
        return text

    def boom():
        raise ValueError("it broke")

    def slow(steps: Annotated[int, "How many steps"] = 5, pause: Annotated[float, "Seconds per step"] = 0.05):
        state["started"] = state.get("started", 0) + 1
        try:
            for i in range(steps):
                time.sleep(pause)
                report_progress(i + 1, steps, f"step {i + 1}")     # also notices a client that has gone away
        except RequestCancelled:
            state["cancelled"] = True
            raise
        state["finished"] = True
        return "done"

    registry.register(echo, "Return the text")
    registry.register(boom, "Always fails")
    registry.register(slow, "Takes a while and reports progress")
    # tools whose parameters are copied into headers: Registry cannot express x-mcp-header, so add them by hand
    registry.tools["execute_sql"] = {
        "func": lambda region, query: f"{region}|{query}", "description": "Run SQL",
        "schema": {"type": "object", "required": ["region", "query"], "properties": {
            "region": {"type": "string", "x-mcp-header": "Region"}, "query": {"type": "string"}}}}
    registry.tools["limited"] = {
        "func": lambda limit, flag=False: f"{limit}|{flag}", "description": "Typed headers",
        "schema": {"type": "object", "required": ["limit"], "properties": {
            "limit": {"type": "integer", "x-mcp-header": "Limit"}, "flag": {"type": "boolean", "x-mcp-header": "Flag"}}}}
    registry.tools["bad_annotation"] = {
        "func": lambda x: x, "description": "Annotation breaks the rules",
        "schema": {"type": "object", "properties": {"x": {"type": "number", "x-mcp-header": "X"}}}}
    return registry


class Running:
    def __init__(self, httpd, state, logs):
        self.httpd, self.state, self.logs = httpd, state, logs
        self.host, self.port = httpd.server_address[:2]
        self.url, self.token = httpd.url, TOKEN


@pytest.fixture
def make_server():
    started = []
    def make(**options):
        state, logs = {}, []
        mcp = McpServer(build_registry(state), log=logs.append)
        httpd = serve_http(mcp, token=options.pop("token", TOKEN), **options)
        threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
        started.append(httpd)
        return Running(httpd, state, logs)
    yield make
    for httpd in started:
        httpd.shutdown()
        httpd.server_close()


@pytest.fixture
def srv(make_server):
    return make_server()


# ======================================================================
# raw HTTP helpers
# ======================================================================
class Reply:
    def __init__(self, status, headers, body):
        self.status, self.headers, self.body = status, headers, body

    @property
    def json(self):
        return json.loads(self.body)


def send(srv, method="POST", path="/mcp", body=b"", headers=None):
    connection = http.client.HTTPConnection(srv.host, srv.port, timeout=15)
    try:
        connection.request(method, path, body=body if method == "POST" else None, headers=headers or {})
        response = connection.getresponse()
        return Reply(response.status, {k.lower(): v for k, v in response.getheaders()}, response.read())
    finally:
        connection.close()


def good_headers(srv, message, **changes):
    """The headers a correct client sends for `message`. A value of None in `changes` removes that header."""
    params = message.get("params", {})
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream",
               "Authorization": f"Bearer {srv.token}", "MCP-Protocol-Version": params.get("_meta", {}).get(p.META_VERSION, "2026-07-28"),
               "Mcp-Method": message["method"]}
    if message["method"] == "tools/call":
        headers["Mcp-Name"] = params["name"]
    headers.update(changes)
    return {k: v for k, v in headers.items() if v is not None}


def rpc(srv, method, params=None, *, id=1, version="2026-07-28", progress=None, **header_changes):
    message = p.make_request(id, method, params, version=version, progress_token=progress)
    return send(srv, body=p.encode(message).encode(), headers=good_headers(srv, message, **header_changes))


def call(srv, name, arguments=None, **kw):
    return rpc(srv, "tools/call", {"name": name, "arguments": arguments or {}}, **kw)


# ======================================================================
# the happy path
# ======================================================================
def test_discover_list_and_call_over_http(srv):
    discover = rpc(srv, "server/discover")
    assert discover.status == 200 and discover.headers["content-type"] == "application/json"
    assert discover.json["result"]["supportedVersions"] == ["2026-07-28"]
    tools = rpc(srv, "tools/list", id=2).json["result"]["tools"]
    assert [t["name"] for t in tools][:3] == ["echo", "boom", "slow"]
    reply = call(srv, "echo", {"text": "hi"}, id=3)
    assert reply.status == 200 and reply.json["id"] == 3
    assert reply.json["result"]["content"][0]["text"] == "hi" and reply.json["result"]["isError"] is False

def test_content_length_matches_the_body_and_the_server_does_not_leak_its_python_version(srv):
    reply = rpc(srv, "tools/list")
    assert int(reply.headers["content-length"]) == len(reply.body)
    assert reply.headers["server"] == "ce-agent-mcp"

def test_a_tool_error_is_http_200_with_is_error(srv):
    reply = call(srv, "boom")
    assert reply.status == 200 and reply.json["result"]["isError"] is True and "it broke" in reply.json["result"]["content"][0]["text"]

def test_the_http_server_listens_on_loopback_only(srv):
    assert srv.host == "127.0.0.1"
    assert srv.url == f"http://127.0.0.1:{srv.port}/mcp"

def test_a_large_body_is_accepted(srv):
    text = "x" * 1_000_000
    reply = call(srv, "echo", {"text": text})
    assert reply.status == 200 and len(reply.json["result"]["content"][0]["text"]) == 1_000_000


# ======================================================================
# which requests are accepted at all
# ======================================================================
def test_an_accepted_notification_gets_202_and_no_body(srv):
    body = p.encode(p.make_notification("notifications/whatever")).encode()
    reply = send(srv, body=body, headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream",
                                          "Authorization": f"Bearer {srv.token}"})
    assert reply.status == 202 and reply.body == b""

@pytest.mark.parametrize("method", ["GET", "DELETE", "PUT", "PATCH", "OPTIONS", "HEAD"])
def test_only_post_is_allowed_on_the_endpoint(srv, method):
    reply = send(srv, method=method)
    assert reply.status == 405 and reply.headers["allow"] == "POST"
    if method == "HEAD":
        assert reply.body == b""

def test_another_path_is_404(srv):
    assert send(srv, path="/other", body=b"{}", headers={"Content-Type": "application/json"}).status == 404
    assert send(srv, path="/mcp/extra", body=b"{}", headers={"Content-Type": "application/json"}).status == 404

def test_a_query_string_does_not_change_the_endpoint(srv):
    message = p.make_request(1, "server/discover")
    reply = send(srv, path="/mcp?x=1", body=p.encode(message).encode(), headers=good_headers(srv, message))
    assert reply.status == 200


# ---------- Origin and Host: DNS rebinding protection ----------
def test_a_request_with_an_unknown_origin_is_403(srv):                                  # SPEC: MUST validate Origin
    reply = rpc(srv, "server/discover", Origin="http://evil.example")
    assert reply.status == 403
    assert srv.state.get("echoed") is None

def test_no_origin_header_is_fine_because_only_browsers_send_one(srv):
    assert rpc(srv, "server/discover").status == 200

def test_an_explicitly_allowed_origin_is_accepted(make_server):
    srv = make_server(allowed_origins=["http://localhost:3000"])
    assert rpc(srv, "server/discover", Origin="http://localhost:3000").status == 200
    assert rpc(srv, "server/discover", Origin="http://localhost:3001").status == 403

def test_an_unexpected_host_header_is_403(srv):                                         # extra hardening, not in the spec
    assert rpc(srv, "server/discover", Host="evil.example:80").status == 403
    assert rpc(srv, "server/discover", Host=f"localhost:{srv.port}").status == 200

def test_listening_beyond_loopback_requires_a_token():
    mcp = McpServer(build_registry({}))
    with pytest.raises(ValueError, match="without an authentication token"):
        serve_http(mcp, host="0.0.0.0")


# ---------- authentication ----------
@pytest.mark.parametrize("authorization", [None, "", "Bearer", "Bearer wrong", f"Basic {TOKEN}", TOKEN, f"Bearer {TOKEN} "])
def test_a_missing_or_wrong_token_is_401_and_nothing_runs(srv, authorization):
    reply = call(srv, "echo", {"text": "x"}, Authorization=authorization)
    assert reply.status == 401 and reply.headers["www-authenticate"] == "Bearer"
    assert srv.state.get("echoed") is None

def test_the_right_token_works_and_the_scheme_name_is_case_insensitive(srv):
    assert call(srv, "echo", {"text": "x"}, Authorization=f"bearer {TOKEN}").status == 200

def test_the_token_is_never_logged(srv):
    call(srv, "echo", {"text": "x"}); call(srv, "echo", {"text": "x"}, Authorization="Bearer wrong")
    assert srv.logs and not any(TOKEN in line or "wrong" in line for line in srv.logs)

def test_a_server_without_a_token_accepts_anyone_on_loopback(make_server):
    assert rpc(make_server(token=None), "server/discover", Authorization=None).status == 200


# ---------- Accept, Content-Type, length ----------
@pytest.mark.parametrize("accept, ok", [
    ("application/json, text/event-stream", True), ("text/event-stream, application/json", True),
    ("*/*", True), ("application/*, text/*", True), ("application/json;q=0.9, text/event-stream;q=0.8", True),
    ("application/json", False), ("text/event-stream", False), ("text/html", False), ("", False),
])
def test_accept_must_cover_both_json_and_event_streams(srv, accept, ok):
    assert accepts_json_and_sse(accept) is ok
    assert (rpc(srv, "server/discover", Accept=accept).status == 200) is ok
    if not ok:
        assert rpc(srv, "server/discover", Accept=accept).status == 406

def test_a_missing_accept_header_is_406(srv):
    assert rpc(srv, "server/discover", Accept=None).status == 406

def test_the_body_must_be_declared_as_json(srv):
    assert rpc(srv, "server/discover", **{"Content-Type": "text/plain"}).status == 415
    assert rpc(srv, "server/discover", **{"Content-Type": None}).status == 415
    assert rpc(srv, "server/discover", **{"Content-Type": "application/json; charset=utf-8"}).status == 200

def test_an_oversized_body_is_413_without_being_read(make_server):
    srv = make_server(max_body=200)
    reply = call(srv, "echo", {"text": "x" * 500})
    assert reply.status == 413 and srv.state.get("echoed") is None

def test_a_missing_or_bad_content_length(srv):
    import socket
    def raw(request):
        with socket.create_connection((srv.host, srv.port), timeout=10) as s:
            s.sendall(request)
            return s.recv(4096).split(b"\r\n")[0]
    head = f"POST /mcp HTTP/1.0\r\nHost: 127.0.0.1:{srv.port}\r\nContent-Type: application/json\r\nAccept: */*\r\nAuthorization: Bearer {TOKEN}\r\n"
    assert b"411" in raw((head + "\r\n").encode())
    assert b"400" in raw((head + "Content-Length: abc\r\n\r\n").encode())
    assert b"400" in raw((head + "Content-Length: -5\r\n\r\n").encode())


# ======================================================================
# JSON-RPC problems: statuses and error bodies
# ======================================================================
def post_body(srv, body, **changes):
    headers = {"Content-Type": "application/json", "Accept": "*/*", "Authorization": f"Bearer {srv.token}",
               "MCP-Protocol-Version": "2026-07-28", "Mcp-Method": "tools/list"}
    headers.update(changes)
    return send(srv, body=body, headers=headers)

def test_invalid_json_is_400_with_a_parse_error_and_no_id(srv):
    reply = post_body(srv, b"{not json")
    assert reply.status == 400 and reply.json["error"]["code"] == p.PARSE_ERROR and "id" not in reply.json

def test_invalid_utf8_is_a_parse_error_too(srv):
    assert post_body(srv, b'{"jsonrpc": "2.0", "id": 1, "method": "\xff\xfe"}').json["error"]["code"] == p.PARSE_ERROR

@pytest.mark.parametrize("body", [b"[1, 2]", b'"hello"', b"5", b'{"jsonrpc": "2.0", "id": null, "method": "x"}', b'{"id": 1, "method": "x"}'])
def test_a_message_that_is_not_a_valid_request_is_400_invalid_request(srv, body):
    reply = post_body(srv, body)
    assert reply.status == 400 and reply.json["error"]["code"] == p.INVALID_REQUEST

def test_batches_are_not_supported(srv):                                                # a JSON array is not a request
    batch = json.dumps([p.make_request(1, "tools/list"), p.make_request(2, "tools/list")]).encode()
    assert post_body(srv, batch).status == 400

def test_an_unknown_method_is_404_with_a_json_rpc_error(srv):                           # SPEC: 404 + -32601
    reply = rpc(srv, "resources/list")
    assert reply.status == 404 and reply.json["error"]["code"] == p.METHOD_NOT_FOUND

def test_the_legacy_initialize_handshake_is_refused_and_names_our_version(srv):
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-11-25"}}).encode()
    reply = post_body(srv, body, **{"Mcp-Method": "initialize"})
    assert reply.status == 404 and reply.json["error"]["data"] == {"supported": ["2026-07-28"]}

def test_missing_meta_fields_are_400_invalid_params(srv):                               # SPEC: -32602, 400
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {"_meta": {p.META_VERSION: "2026-07-28"}}}).encode()
    reply = post_body(srv, body)
    assert reply.status == 400 and reply.json["error"]["code"] == p.INVALID_PARAMS

def test_an_unsupported_version_is_400_and_lists_what_is_supported(srv):                # SPEC: -32022, 400
    reply = rpc(srv, "tools/list", version="1900-01-01")
    assert reply.status == 400 and reply.json["error"]["code"] == p.UNSUPPORTED_PROTOCOL_VERSION
    assert reply.json["error"]["data"] == {"supported": ["2026-07-28"], "requested": "1900-01-01"}

def test_an_unknown_tool_is_a_protocol_error_with_status_400(srv):
    reply = call(srv, "no_such_tool")
    assert reply.status == 400 and reply.json["error"] == {"code": -32602, "message": "Unknown tool: no_such_tool"}

def test_a_bug_in_the_server_is_a_500_with_an_error_body(srv):
    srv.httpd.mcp.methods["tools/list"] = lambda params: 1 / 0
    reply = rpc(srv, "tools/list")
    assert reply.status == 500 and reply.json["error"]["code"] == p.INTERNAL_ERROR
    assert rpc(srv, "server/discover").status == 200                                    # and it keeps serving


# ======================================================================
# headers must agree with the body
# ======================================================================
def error_code(reply):
    return reply.json["error"]["code"]

def test_a_missing_protocol_version_header_is_rejected(srv):                            # SPEC: MUST
    reply = rpc(srv, "server/discover", **{"MCP-Protocol-Version": None})
    assert reply.status == 400 and error_code(reply) == p.HEADER_MISMATCH

def test_a_protocol_version_header_that_disagrees_with_meta_is_rejected(srv):           # SPEC: HeaderMismatch
    reply = rpc(srv, "server/discover", **{"MCP-Protocol-Version": "2025-11-25"})
    assert reply.status == 400 and error_code(reply) == p.HEADER_MISMATCH

def test_the_method_header_is_required_and_must_match(srv):
    assert error_code(rpc(srv, "tools/list", **{"Mcp-Method": None})) == p.HEADER_MISMATCH
    reply = rpc(srv, "tools/list", **{"Mcp-Method": "tools/call"})
    assert reply.status == 400 and error_code(reply) == p.HEADER_MISMATCH
    assert rpc(srv, "tools/list", **{"Mcp-Method": "TOOLS/LIST"}).status == 400        # header VALUES are case-sensitive

def test_the_name_header_is_required_for_tools_call_and_must_match(srv):
    assert error_code(call(srv, "echo", {"text": "x"}, **{"Mcp-Name": None})) == p.HEADER_MISMATCH
    reply = call(srv, "echo", {"text": "x"}, **{"Mcp-Name": "boom"})
    assert reply.status == 400 and "does not match" in reply.json["error"]["message"]
    assert srv.state.get("echoed") is None                                              # the tool never ran

def test_a_smuggling_attempt_is_stopped(srv):
    # a proxy that routes on Mcp-Name sees "echo", but the body asks for another tool
    reply = call(srv, "boom", **{"Mcp-Name": "echo"})
    assert reply.status == 400 and error_code(reply) == p.HEADER_MISMATCH

def test_discover_and_list_need_no_name_header(srv):
    assert rpc(srv, "server/discover").status == 200 and rpc(srv, "tools/list").status == 200

def test_a_base64_encoded_name_is_decoded_before_comparing(srv):
    assert call(srv, "echo", {"text": "x"}, **{"Mcp-Name": encode_header_value("echo")}).status == 200
    assert call(srv, "echo", {"text": "x"}, **{"Mcp-Name": "=?base64?ZWNobw==?="}).status == 200      # "echo"
    assert call(srv, "echo", {"text": "x"}, **{"Mcp-Name": "=?base64?***?="}).status == 400

def test_header_names_are_case_insensitive(srv):
    message = p.make_request(1, "tools/list")
    headers = {k.lower(): v for k, v in good_headers(srv, message).items()}
    assert send(srv, body=p.encode(message).encode(), headers=headers).status == 200


# ---------- x-mcp-header ----------
def test_mirrored_parameters_must_appear_as_headers(srv):
    ok = call(srv, "execute_sql", {"region": "us-west1", "query": "SELECT 1"}, **{"Mcp-Param-Region": "us-west1"})
    assert ok.status == 200 and ok.json["result"]["content"][0]["text"] == "us-west1|SELECT 1"
    missing = call(srv, "execute_sql", {"region": "us-west1", "query": "SELECT 1"})
    assert missing.status == 400 and error_code(missing) == p.HEADER_MISMATCH
    wrong = call(srv, "execute_sql", {"region": "us-west1", "query": "q"}, **{"Mcp-Param-Region": "eu-west1"})
    assert wrong.status == 400 and error_code(wrong) == p.HEADER_MISMATCH

def test_mirrored_integers_compare_numerically_and_booleans_as_words(srv):
    assert call(srv, "limited", {"limit": 42, "flag": True}, **{"Mcp-Param-Limit": "42.0", "Mcp-Param-Flag": "true"}).status == 200
    assert call(srv, "limited", {"limit": 42}, **{"Mcp-Param-Limit": "43"}).status == 400
    assert call(srv, "limited", {"limit": 42, "flag": True}, **{"Mcp-Param-Limit": "42", "Mcp-Param-Flag": "True"}).status == 400

def test_unrelated_param_headers_are_ignored(srv):
    assert call(srv, "echo", {"text": "x"}, **{"Mcp-Param-Anything": "whatever"}).status == 200


# ======================================================================
# streaming: progress over Server-Sent Events, and cancellation
# ======================================================================
def read_events(response):
    from ce_agent.mcp.sse import parse_sse
    def lines():
        while True:
            line = response.readline()
            if not line:
                return
            yield line.decode()
    return [json.loads(e["data"]) for e in parse_sse(lines())]


def test_a_request_with_a_progress_token_streams_progress_then_the_result(srv):
    message = p.make_request(7, "tools/call", {"name": "slow", "arguments": {"steps": 4, "pause": 0.02}}, progress_token="tok-1")
    connection = http.client.HTTPConnection(srv.host, srv.port, timeout=15)
    connection.request("POST", "/mcp", body=p.encode(message).encode(), headers=good_headers(srv, message))
    response = connection.getresponse()
    assert response.status == 200 and response.getheader("Content-Type") == "text/event-stream"
    assert response.getheader("Cache-Control") == "no-cache" and response.getheader("X-Accel-Buffering") == "no"
    messages = read_events(response)
    connection.close()
    *progress, final = messages
    assert [m["method"] for m in progress] == ["notifications/progress"] * 4
    assert [m["params"]["progressToken"] for m in progress] == ["tok-1"] * 4
    assert [m["params"]["progress"] for m in progress] == [1, 2, 3, 4]                  # MUST increase
    assert [m["params"]["message"] for m in progress] == ["step 1", "step 2", "step 3", "step 4"]
    assert all(m["params"]["total"] == 4 for m in progress)
    assert final["id"] == 7 and final["result"]["content"][0]["text"] == "done"

def test_without_a_progress_token_the_answer_is_plain_json_even_for_a_tool_that_reports_progress(srv):
    reply = call(srv, "slow", {"steps": 3, "pause": 0.01})
    assert reply.headers["content-type"] == "application/json" and reply.json["result"]["content"][0]["text"] == "done"

def test_a_tool_that_never_reports_progress_gets_plain_json_even_with_a_token(srv):
    reply = call(srv, "echo", {"text": "x"}, progress="t")
    assert reply.headers["content-type"] == "application/json"

def test_an_error_after_the_stream_started_arrives_as_the_last_event(srv):
    srv.httpd.mcp.registry.tools["slow"]["func"] = lambda steps=1, pause=0: (report_progress(1), 1 / 0)[1]
    message = p.make_request(1, "tools/call", {"name": "slow", "arguments": {}}, progress_token=1)
    connection = http.client.HTTPConnection(srv.host, srv.port, timeout=15)
    connection.request("POST", "/mcp", body=p.encode(message).encode(), headers=good_headers(srv, message))
    response = connection.getresponse()
    messages = read_events(response); connection.close()
    assert response.status == 200                                                        # headers were already sent
    assert messages[-1]["result"]["isError"] is True                                     # tool errors are results anyway


def wait_for(predicate, seconds=5):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_closing_the_event_stream_cancels_the_request(srv):                              # SPEC: closing the stream = cancel
    message = p.make_request(1, "tools/call", {"name": "slow", "arguments": {"steps": 400, "pause": 0.05}}, progress_token=1)   # 20 s
    connection = http.client.HTTPConnection(srv.host, srv.port, timeout=15)
    connection.request("POST", "/mcp", body=p.encode(message).encode(), headers=good_headers(srv, message))
    response = connection.getresponse()
    assert b"notifications/progress" in response.readline()                              # the tool is running and reporting
    response.close(); connection.close()                                                 # the client walks away
    assert wait_for(lambda: srv.state.get("cancelled")), "the tool was not told to stop"
    assert not srv.state.get("finished")
    started = time.monotonic()
    assert call(srv, "echo", {"text": "next"}).status == 200                              # the server is free again
    assert time.monotonic() - started < 2

def test_dropping_the_connection_also_cancels_when_no_progress_was_requested(srv):
    message = p.make_request(1, "tools/call", {"name": "slow", "arguments": {"steps": 400, "pause": 0.05}})
    connection = http.client.HTTPConnection(srv.host, srv.port, timeout=15)
    connection.request("POST", "/mcp", body=p.encode(message).encode(), headers=good_headers(srv, message))
    time.sleep(0.3)
    connection.close()                                                                    # no reply was ever read
    assert wait_for(lambda: srv.state.get("cancelled")) and not srv.state.get("finished")

def test_nothing_more_is_sent_for_a_cancelled_request(srv):
    message = p.make_request(1, "tools/call", {"name": "slow", "arguments": {"steps": 400, "pause": 0.05}})
    connection = http.client.HTTPConnection(srv.host, srv.port, timeout=15)
    connection.request("POST", "/mcp", body=p.encode(message).encode(), headers=good_headers(srv, message))
    time.sleep(0.2); connection.close()
    assert wait_for(lambda: srv.state.get("cancelled"))
    assert any("cancelled by the client" in line for line in srv.logs)

def test_requests_from_several_clients_at_once_all_succeed(srv):
    results = []
    def worker(n):
        results.append(call(srv, "echo", {"text": f"msg{n}"}, id=n).json["result"]["content"][0]["text"])
    threads = [threading.Thread(target=worker, args=(n,)) for n in range(12)]
    [t.start() for t in threads]; [t.join(20) for t in threads]
    assert sorted(results) == sorted(f"msg{n}" for n in range(12))


# ======================================================================
# HttpClient
# ======================================================================
@pytest.fixture
def client(srv):
    c = McpClient(HttpClient(srv.url, srv.token, timeout=15))
    yield c
    c.close()


def test_a_full_mcp_client_over_http(client):
    assert client.discover()["supportedVersions"] == ["2026-07-28"] and client.server_info["name"] == "ce-agent-tools"
    assert "echo" in [t["name"] for t in client.list_tools()]
    assert client.call_tool("echo", {"text": "héllo ✓"}) == {"status": "ok", "content": "héllo ✓"}
    assert client.call_tool("boom") == {"status": "error", "error": "it broke"}
    assert client.call_tool("no_such_tool")["error"] == "MCP error -32602: Unknown tool: no_such_tool"

def test_progress_reaches_the_clients_callback(client):
    seen = []
    client.on_progress = seen.append
    assert client.call_tool("slow", {"steps": 3, "pause": 0.02}) == {"status": "ok", "content": "done"}
    assert [m["progress"] for m in seen] == [1, 2, 3] and all("progressToken" in m for m in seen)

def test_no_progress_token_is_sent_unless_someone_listens(client, srv):
    client.call_tool("slow", {"steps": 2, "pause": 0.01})
    assert client.transport.on_notification is not None and client.on_progress is None
    # (the server answers with plain JSON in that case; test_without_a_progress_token... checks the other side)

def test_the_clients_timeout_closes_the_connection_which_cancels_the_tool(srv):
    c = McpClient(HttpClient(srv.url, srv.token, timeout=1))
    result = c.call_tool("slow", {"steps": 400, "pause": 0.05})
    assert result["status"] == "error" and "may or may not have completed" in result["error"] and "within 1s" in result["error"]
    assert wait_for(lambda: srv.state.get("cancelled")) and not srv.state.get("finished")

def test_a_deadline_applies_to_a_stream_even_when_progress_keeps_arriving(srv):
    c = McpClient(HttpClient(srv.url, srv.token, timeout=1))
    c.on_progress = lambda params: None
    result = c.call_tool("slow", {"steps": 400, "pause": 0.05})
    assert result["status"] == "error" and "within 1s" in result["error"]                 # progress does not extend it
    assert wait_for(lambda: srv.state.get("cancelled"))

def test_a_wrong_token_is_reported_without_restarting_anything(srv):
    c = McpClient(HttpClient(srv.url, "wrong", timeout=5))
    result = c.call_tool("echo", {"text": "x"})
    assert result["status"] == "error" and "401" in result["error"] and "may or may not" not in result["error"]
    with pytest.raises(p.McpHttpError, match="401"):
        c.list_tools()

def test_an_unreachable_server_is_a_connection_error(make_server):
    srv = make_server()
    url = srv.url
    srv.httpd.shutdown(); srv.httpd.server_close()
    c = McpClient(HttpClient(url, TOKEN, timeout=3))
    with pytest.raises(p.McpConnectionError, match="could not reach"):
        c.discover()

def test_version_negotiation_over_http(client, srv):
    client.supported, client.version = ["2099-01-01", "2026-07-28"], "2099-01-01"
    assert client.list_tools() and client.version == "2026-07-28"

def test_mirrored_parameters_are_sent_automatically(client):
    client.list_tools()                                                                   # the client learns the schemas
    assert client.call_tool("execute_sql", {"region": "us-west1", "query": "SELECT 1"}) == {"status": "ok", "content": "us-west1|SELECT 1"}
    assert client.call_tool("execute_sql", {"region": "Hello, 世界", "query": "q"})["content"] == "Hello, 世界|q"   # base64 path
    assert client.call_tool("limited", {"limit": 42, "flag": False})["content"] == "42|False"
    assert client.call_tool("limited", {"limit": 7})["content"] == "7|False"

def test_without_the_schema_a_mirrored_parameter_is_rejected_by_the_server(client):
    # never called list_tools(), so the client cannot know which parameters to mirror
    result = client.call_tool("execute_sql", {"region": "us-west1", "query": "q"})
    assert result["status"] == "error" and "Missing required header Mcp-Param-Region" in result["error"]

def test_tools_with_broken_annotations_are_left_out_over_http(client):
    names = [t["name"] for t in client.list_tools()]
    assert "bad_annotation" not in names and "execute_sql" in names
    assert client.rejected_tools[0][0] == "bad_annotation" and "only string, integer and boolean" in client.rejected_tools[0][1]

def test_over_stdio_the_annotations_are_not_checked():
    # stdio has no header layer, so a client MAY ignore x-mcp-header there: the tool stays in the list
    class Stdioish:
        def request(self, message):
            return {"jsonrpc": "2.0", "id": message["id"], "result": {"resultType": "complete", "ttlMs": 0, "tools": [
                {"name": "bad", "inputSchema": {"type": "object", "properties": {"x": {"type": "number", "x-mcp-header": "X"}}}}]}}
        def restart(self): pass
        def close(self): pass
    assert [t["name"] for t in McpClient(Stdioish()).list_tools()] == ["bad"]


# ---------- servers that are not ours ----------
@pytest.fixture
def stub():
    """A tiny HTTP server that answers every POST with whatever the test configured."""
    config = {"status": 200, "headers": {}, "body": b""}
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.send_response(config["status"])
            for k, v in config["headers"].items():
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(config["body"])))
            self.end_headers()
            self.wfile.write(config["body"])
        def log_message(self, *args):
            pass
    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    config["url"] = f"http://127.0.0.1:{httpd.server_address[1]}/mcp"
    yield config
    httpd.shutdown(); httpd.server_close()


def test_an_html_error_page_from_a_proxy_is_a_clear_error(stub):
    stub.update(status=502, headers={"Content-Type": "text/html"}, body=b"<html><body>Bad   Gateway</body></html>")
    with pytest.raises(p.McpConnectionError, match="HTTP 502 Bad Gateway from the MCP server: .*Bad Gateway"):
        McpClient(HttpClient(stub["url"])).discover()

def test_a_non_json_client_error_is_not_a_connection_error(stub):
    stub.update(status=403, headers={"Content-Type": "text/plain"}, body=b"nope")
    with pytest.raises(p.McpHttpError, match="HTTP 403"):
        McpClient(HttpClient(stub["url"])).discover()

def test_an_error_status_with_a_json_rpc_body_is_understood(stub):
    body = json.dumps({"jsonrpc": "2.0", "error": {"code": -32022, "message": "Unsupported protocol version",
                                                   "data": {"supported": ["1999-01-01"], "requested": "2026-07-28"}}}).encode()
    stub.update(status=400, headers={"Content-Type": "application/json"}, body=body)
    with pytest.raises(p.McpVersionError):
        McpClient(HttpClient(stub["url"])).list_tools()

def test_a_stream_that_ends_without_a_response_is_an_error(stub):
    progress = json.dumps({"jsonrpc": "2.0", "method": "notifications/progress", "params": {"progressToken": 1, "progress": 1}})
    stub.update(status=200, headers={"Content-Type": "text/event-stream"}, body=f"data: {progress}\n\n".encode())
    with pytest.raises(p.McpConnectionError, match="ended without a response"):
        McpClient(HttpClient(stub["url"])).discover()

def test_garbage_events_in_a_stream_are_skipped(stub):
    good = json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"resultType": "complete", "supportedVersions": ["2026-07-28"]}})
    stub.update(status=200, headers={"Content-Type": "text/event-stream"}, body=f": hi\n\ndata: not json\n\ndata: {good}\n\n".encode())
    assert McpClient(HttpClient(stub["url"])).discover()["supportedVersions"] == ["2026-07-28"]

def test_a_server_that_answers_in_html_with_status_200_is_an_error(stub):
    stub.update(status=200, headers={"Content-Type": "text/html"}, body=b"<html>login</html>")
    with pytest.raises(p.McpHttpError, match="HTTP 200"):
        McpClient(HttpClient(stub["url"])).discover()
