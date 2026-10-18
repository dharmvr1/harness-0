"""Tests for the MCP client: the protocol logic (with a fake transport) and the stdio transport (with scripted fake servers)."""
import sys
import time
from pathlib import Path

import pytest
from ce_agent.mcp import protocol as p
from ce_agent.mcp.client import McpClient
from ce_agent.mcp.stdio import StdioClient, server_env

FAKE_SERVER = str(Path(__file__).parent / "fake_mcp_server.py")


# ======================================================================
# McpClient, with a fake transport: dicts in, dicts out
# ======================================================================
class FakeTransport:
    def __init__(self, handler):
        self.handler, self.requests, self.restarts, self.closed = handler, [], 0, False

    def request(self, message):
        self.requests.append(message)
        return self.handler(message)

    def restart(self):
        self.restarts += 1

    def close(self):
        self.closed = True


def ok(message, **result):
    return {"jsonrpc": "2.0", "id": message["id"], "result": {"resultType": "complete", **result}}

def fail(message, code, text, data=None):
    return p.error_response(message["id"], code, text, data)

def tool(name="ping"):
    return {"name": name, "description": f"{name} tool", "inputSchema": {"type": "object", "properties": {}}}


def test_requests_carry_the_required_meta_and_increasing_ids():
    transport = FakeTransport(lambda m: ok(m, supportedVersions=["2026-07-28"]))
    client = McpClient(transport)
    client.discover(); client.discover(); client.discover()
    assert [r["id"] for r in transport.requests] == [1, 2, 3]
    meta = transport.requests[0]["params"]["_meta"]
    assert meta[p.META_VERSION] == "2026-07-28" and meta[p.META_CLIENT_CAPS] == {} and meta[p.META_CLIENT_INFO]["name"] == "ce-agent"
    assert transport.requests[0]["method"] == "server/discover"

def test_discover_records_what_the_server_says():
    transport = FakeTransport(lambda m: ok(m, supportedVersions=["2026-07-28"],
                                           _meta={p.META_SERVER_INFO: {"name": "S", "version": "9"}}))
    client = McpClient(transport)
    client.discover()
    assert client.server_info == {"name": "S", "version": "9"} and client.server_versions == ["2026-07-28"]

def test_discover_fails_when_there_is_no_shared_version():
    client = McpClient(FakeTransport(lambda m: ok(m, supportedVersions=["2025-11-25"])))
    with pytest.raises(p.McpVersionError, match="no shared protocol version"):
        client.discover()


# ---------- version negotiation (-32022) ----------
def test_unsupported_version_error_triggers_one_retry_with_a_shared_version():
    def handler(m):
        if m["params"]["_meta"][p.META_VERSION] != "2026-07-28":
            return fail(m, p.UNSUPPORTED_PROTOCOL_VERSION, "Unsupported protocol version",
                        {"supported": ["2026-07-28", "2025-11-25"], "requested": "2099-01-01"})
        return ok(m, tools=[tool()], ttlMs=0)
    transport = FakeTransport(handler)
    client = McpClient(transport)
    client.supported, client.version = ["2099-01-01", "2026-07-28"], "2099-01-01"      # pretend we are newer
    assert [t["name"] for t in client.list_tools()] == ["ping"]
    assert [r["params"]["_meta"][p.META_VERSION] for r in transport.requests] == ["2099-01-01", "2026-07-28"]
    assert client.version == "2026-07-28"                                                # and we stay on it

def test_no_overlap_in_versions_is_an_error_not_a_loop():
    transport = FakeTransport(lambda m: fail(m, p.UNSUPPORTED_PROTOCOL_VERSION, "Unsupported protocol version",
                                             {"supported": ["1999-01-01"], "requested": "2026-07-28"}))
    with pytest.raises(p.McpVersionError):
        McpClient(transport).list_tools()
    assert len(transport.requests) == 1


# ---------- tools/list and its cache ----------
def test_tool_list_is_cached_for_ttl_and_refetched_after():
    now = [100.0]
    transport = FakeTransport(lambda m: ok(m, tools=[tool()], ttlMs=5000, cacheScope="public"))
    client = McpClient(transport, clock=lambda: now[0])
    client.list_tools(); client.list_tools()
    assert len(transport.requests) == 1                      # second call: still fresh
    now[0] += 4.9
    client.list_tools()
    assert len(transport.requests) == 1
    now[0] += 0.2                                            # past 5 seconds
    client.list_tools()
    assert len(transport.requests) == 2

def test_invalidate_forces_a_refetch():
    transport = FakeTransport(lambda m: ok(m, tools=[tool()], ttlMs=60000))
    client = McpClient(transport)
    client.list_tools(); client.invalidate_tools(); client.list_tools()
    assert len(transport.requests) == 2

def test_zero_or_missing_ttl_means_always_refetch():
    transport = FakeTransport(lambda m: ok(m, tools=[tool()]))
    client = McpClient(transport)
    client.list_tools(); client.list_tools()
    assert len(transport.requests) == 2

def test_pagination_follows_cursors_and_the_shortest_ttl_wins():
    pages = {None: ok_page(["a", "b"], "c2", 5000), "c2": ok_page(["c"], None, 1000)}
    transport = FakeTransport(lambda m: {**pages[m["params"].get("cursor")], "id": m["id"]})
    now = [0.0]
    client = McpClient(transport, clock=lambda: now[0])
    assert [t["name"] for t in client.list_tools()] == ["a", "b", "c"]
    assert [r["params"].get("cursor") for r in transport.requests] == [None, "c2"]
    now[0] = 1.5
    client.list_tools()
    assert len(transport.requests) == 4                      # the 1 s page expired, so everything is refetched

def ok_page(names, next_cursor, ttl):
    result = {"resultType": "complete", "tools": [tool(n) for n in names], "ttlMs": ttl}
    if next_cursor:
        result["nextCursor"] = next_cursor
    return {"jsonrpc": "2.0", "id": 0, "result": result}

@pytest.mark.parametrize("result", [{}, {"tools": "x"}, {"tools": [{"description": "no name"}]}, {"tools": [5]}])
def test_malformed_tool_lists_are_rejected(result):
    with pytest.raises(p.McpError, match="malformed tools/list"):
        McpClient(FakeTransport(lambda m: ok(m, **result))).list_tools()

def test_a_reply_without_a_result_object_is_malformed():
    with pytest.raises(p.McpError, match="malformed reply"):
        McpClient(FakeTransport(lambda m: {"jsonrpc": "2.0", "id": m["id"], "result": "nope"})).discover()


# ---------- tools/call: results become {"status", "content" | "error"} ----------
def call_with(result_or_error):
    def handler(m):
        if "error" in result_or_error:
            return {"jsonrpc": "2.0", "id": m["id"], **result_or_error}
        return {"jsonrpc": "2.0", "id": m["id"], "result": result_or_error}
    return McpClient(FakeTransport(handler)).call_tool("x", {"a": 1})

def test_a_successful_call():
    r = call_with({"resultType": "complete", "content": [{"type": "text", "text": "hello"}], "isError": False})
    assert r == {"status": "ok", "content": "hello"}

def test_arguments_and_name_are_sent_as_the_spec_shows():
    transport = FakeTransport(lambda m: ok(m, content=[{"type": "text", "text": "x"}]))
    McpClient(transport).call_tool("get_weather", {"location": "New York"})
    assert transport.requests[0]["method"] == "tools/call"
    assert transport.requests[0]["params"]["name"] == "get_weather" and transport.requests[0]["params"]["arguments"] == {"location": "New York"}

def test_a_tool_error_keeps_its_text_for_the_model():
    r = call_with({"resultType": "complete", "content": [{"type": "text", "text": "old_text matches 2 places"}], "isError": True})
    assert r == {"status": "error", "error": "old_text matches 2 places"}

def test_several_text_blocks_are_joined_and_other_kinds_are_noted():
    r = call_with({"resultType": "complete", "isError": False, "content": [
        {"type": "text", "text": "one"}, {"type": "image", "data": "...", "mimeType": "image/png"}, {"type": "text", "text": "two"}]})
    assert r == {"status": "ok", "content": "one\n[image content not shown]\ntwo"}

def test_an_empty_error_still_says_something():
    assert call_with({"resultType": "complete", "content": [], "isError": True})["error"] == "the tool failed"

def test_a_missing_result_type_means_complete():                                       # older servers do not send it
    assert call_with({"content": [{"type": "text", "text": "old server"}]}) == {"status": "ok", "content": "old server"}

def test_input_required_is_reported_as_unsupported():
    r = call_with({"resultType": "input_required", "inputRequests": {}})
    assert r["status"] == "error" and "multi round-trip" in r["error"]

def test_protocol_errors_become_error_results_with_the_code():
    r = call_with({"error": {"code": -32602, "message": "Unknown tool: x"}})
    assert r == {"status": "error", "error": "MCP error -32602: Unknown tool: x"}


# ---------- crashes: retry reads, never repeat side effects ----------
class Crashy(FakeTransport):
    """Fails with a connection error on the first N requests, then answers."""
    def __init__(self, failures):
        super().__init__(lambda m: ok(m, content=[{"type": "text", "text": "recovered"}], isError=False))
        self.failures = failures

    def request(self, message):
        self.requests.append(message)
        if self.failures:
            self.failures -= 1
            raise p.McpConnectionError("server died")
        return self.handler(message)

def test_a_read_only_call_is_retried_once_after_a_crash():
    transport = Crashy(failures=1)
    assert McpClient(transport).call_tool("read_file", {"file_path": "a"}) == {"status": "ok", "content": "recovered"}
    assert transport.restarts == 1 and len(transport.requests) == 2

def test_a_read_only_call_that_keeps_failing_gives_up_after_one_retry():
    transport = Crashy(failures=5)
    r = McpClient(transport).call_tool("search", {"pattern": "x"})
    assert r["status"] == "error" and len(transport.requests) == 2

@pytest.mark.parametrize("name", ["write_file", "edit_file", "run_shell", "remember", "anything_else"])
def test_calls_with_side_effects_are_never_repeated(name):
    transport = Crashy(failures=1)
    r = McpClient(transport).call_tool(name, {})
    assert len(transport.requests) == 1 and transport.restarts == 1                   # restarted, but not repeated
    assert r["status"] == "error" and "may or may not have completed" in r["error"] and name in r["error"]

def test_the_read_only_list_can_be_configured():
    transport = Crashy(failures=1)
    assert McpClient(transport, read_only=["my_tool"]).call_tool("my_tool", {})["status"] == "ok"

def test_close_closes_the_transport():
    transport = FakeTransport(lambda m: ok(m))
    McpClient(transport).close()
    assert transport.closed


# ======================================================================
# StdioClient, with scripted fake servers (real subprocesses)
# ======================================================================
def fake(mode, *extra, timeout=20, on_wire=None):
    return StdioClient([sys.executable, FAKE_SERVER, mode, *map(str, extra)], timeout=timeout, on_wire=on_wire)

@pytest.fixture
def clients():
    made = []
    def make(*args, **kwargs):
        c = fake(*args, **kwargs)
        made.append(c)
        return c
    yield make
    for c in made:
        c.close()


def test_stdio_round_trip_and_wire_callback(clients):
    wire = []
    client = clients("ok", on_wire=lambda d, m: wire.append((d, m.get("method") or "reply")))
    reply = client.request(p.make_request(1, "tools/list"))
    assert reply["id"] == 1 and reply["result"]["tools"][0]["name"] == "ping"
    assert wire == [("send", "tools/list"), ("recv", "reply")]

def test_a_full_mcp_client_over_a_real_stdio_pipe(clients):
    client = McpClient(clients("ok"))
    assert client.discover()["supportedVersions"] == ["2026-07-28"] and client.server_info["name"] == "fake"
    assert [t["name"] for t in client.list_tools()] == ["ping"]
    assert client.call_tool("ping") == {"status": "ok", "content": "pong"}

def test_strays_garbage_and_late_replies_are_ignored(clients):
    client = clients("stray")
    for request_id in (1, 2, 3):
        assert client.request(p.make_request(request_id, "tools/call"))["id"] == request_id

def test_a_silent_server_times_out_and_can_be_shut_down(clients):
    client = clients("hang", timeout=1)
    started = time.monotonic()
    with pytest.raises(p.McpTimeout, match="within 1s"):
        client.request(p.make_request(1, "tools/list"))
    assert time.monotonic() - started < 5
    started = time.monotonic()
    client.close()                                                                       # stdin close -> terminate
    assert time.monotonic() - started < 15 and client.process is None

def test_a_crashed_server_is_reported_and_restart_recovers(clients, tmp_path):
    client = clients("crash_once", tmp_path / "counter")
    with pytest.raises(p.McpConnectionError, match="exited"):
        client.request(p.make_request(1, "tools/list"))
    client.restart()
    assert client.request(p.make_request(2, "tools/list"))["id"] == 2

def test_a_dead_server_is_detected_before_sending(clients):
    client = clients("ok")
    client.start()
    client.process.kill(); client.process.wait()
    with pytest.raises(p.McpConnectionError, match="not running"):
        client.request(p.make_request(1, "tools/list"))

def test_the_servers_stderr_is_kept_for_diagnostics(tmp_path):
    script = tmp_path / "dies.py"
    script.write_text("import sys\nsys.stderr.write('boom: bad config\\n')\nsys.stderr.flush()\nsys.exit(3)\n")
    client = StdioClient([sys.executable, str(script)], timeout=10)
    try:
        with pytest.raises(p.McpConnectionError, match="boom: bad config"):
            client.request(p.make_request(1, "tools/list"))
    finally:
        client.close()

def test_noisy_stderr_does_not_block_the_server(clients):
    client = clients("noisy", timeout=30)
    assert client.request(p.make_request(1, "tools/list"))["id"] == 1                  # 300 KB on stderr, still answered
    assert len(client.stderr_tail().splitlines()) == 50                                  # only the last 50 lines are kept

def test_a_command_that_cannot_start_is_a_connection_error():
    client = StdioClient(["/definitely/not/a/program"])
    with pytest.raises(p.McpConnectionError, match="could not start"):
        client.start()

def test_close_is_safe_to_repeat_and_before_start():
    client = StdioClient([sys.executable, "-c", "pass"])
    client.close(); client.close()

def test_version_negotiation_over_a_real_pipe(clients):
    client = McpClient(clients("version"))
    client.supported, client.version = ["2099-01-01", "2026-07-28"], "2099-01-01"
    assert client.list_tools() and client.version == "2026-07-28"


# ---------- the environment handed to the server ----------
def test_server_env_passes_only_what_is_needed_and_never_api_keys():
    environ = {"PATH": "/usr/bin", "HOME": "/home/u", "DOCKER_HOST": "unix:///x", "OPENROUTER_API_KEY": "secret",
               "GROQ_API_KEY": "secret", "AWS_SECRET_ACCESS_KEY": "secret", "RANDOM_THING": "x"}
    env = server_env(environ)
    assert env["PATH"] == "/usr/bin" and env["HOME"] == "/home/u" and env["DOCKER_HOST"] == "unix:///x"
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert not any("KEY" in name or "SECRET" in name for name in env) and "RANDOM_THING" not in env
