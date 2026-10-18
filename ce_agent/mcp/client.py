"""
McpClient: the client side of MCP. It builds requests, reads results, and decides what to do about failures.
How the bytes travel is the transport's job (StdioClient for now), which only needs request(), restart() and close().

    client = launch_workspace_server(ws)      # starts the tool server for a workspace and says hello
    tools  = client.list_tools()              # MCP tool definitions
    result = client.call_tool("read_file", {"file_path": "a.py"})   # {"status": "ok", "content": "..."}
"""

import sys
import time
from pathlib import Path

import ce_agent
from ce_agent.mcp import protocol as p
from ce_agent.mcp.stdio import StdioClient, server_env

PROJECT_ROOT = Path(ce_agent.__file__).parent.parent
CLIENT_INFO = {"name": "ce-agent", "version": "0.1.0"}

# Tools that only read. After a crash it is safe to run them again; repeating write_file or run_shell
# could repeat a side effect, so those are reported to the model instead of retried.
READ_ONLY_TOOLS = ("list_files", "search", "read_file")


class McpClient:
    def __init__(self, transport, read_only=READ_ONLY_TOOLS, client_info=None, clock=time.monotonic):
        self.transport = transport
        self.read_only = set(read_only)
        self.client_info = client_info or CLIENT_INFO
        self.clock = clock                      # a parameter so tests can move time forward
        self.supported = list(p.SUPPORTED_VERSIONS)
        self.version = max(self.supported)      # the version we put in every request
        self.server_info = None                 # filled in by discover()
        self.server_versions = []
        self._next_id = 0
        self._tools = None                      # (list of tools, time until which it is fresh)

    # ------------------------------------------------------------------
    # one request
    # ------------------------------------------------------------------
    def _send(self, method, params):
        self._next_id += 1
        message = p.make_request(self._next_id, method, params, version=self.version, client_info=self.client_info)
        return self.transport.request(message)

    def _once(self, method, params):
        reply = self._send(method, params)
        if "error" in reply:
            error = reply["error"] if isinstance(reply["error"], dict) else {}
            raise p.McpRpcError(error.get("code", p.INTERNAL_ERROR), error.get("message", "unknown error"), error.get("data"))
        if not isinstance(reply.get("result"), dict):
            raise p.McpError(f"malformed reply to {method}: no result object")
        return reply["result"]

    def _rpc(self, method, params=None):
        """Send one request and return its result. If the server says it does not speak our protocol version,
        pick a version both sides know and try once more (the spec's version negotiation)."""
        try:
            return self._once(method, params)
        except p.McpRpcError as e:
            if e.code != p.UNSUPPORTED_PROTOCOL_VERSION:
                raise
            self.version = self._pick_version((e.data or {}).get("supported", []))
            return self._once(method, params)

    def _pick_version(self, server_versions):
        common = [v for v in server_versions if v in self.supported]
        if not common:
            raise p.McpVersionError(f"no shared protocol version: the server supports {server_versions}, "
                                    f"this client supports {self.supported}")
        return max(common)                      # dates sort the same way as time, so max() is the newest

    # ------------------------------------------------------------------
    # the three methods
    # ------------------------------------------------------------------
    def discover(self):
        """Ask the server who it is and what it speaks. Optional in the protocol, but it makes a failing
        server fail early and lets us choose a version before the first real request."""
        result = self._rpc("server/discover")
        self.server_versions = result.get("supportedVersions", [])
        self.server_info = (result.get("_meta") or {}).get(p.META_SERVER_INFO)
        if self.version not in self.server_versions:
            self.version = self._pick_version(self.server_versions)
        return result

    def list_tools(self):
        """The server's tool definitions. Cached for as long as the server's ttlMs says the list is fresh."""
        if self._tools and self.clock() < self._tools[1]:
            return list(self._tools[0])
        tools, ttl_ms, cursor = [], None, None
        while True:
            result = self._rpc("tools/list", {"cursor": cursor} if cursor else {})
            page = result.get("tools")
            if not isinstance(page, list) or not all(isinstance(t, dict) and "name" in t for t in page):
                raise p.McpError("malformed tools/list result: expected a list of tools with names")
            tools += page
            page_ttl = result.get("ttlMs", 0)
            ttl_ms = page_ttl if ttl_ms is None else min(ttl_ms, page_ttl)     # the shortest-lived page decides
            cursor = result.get("nextCursor")
            if not cursor:
                break
        self._tools = (tools, self.clock() + max(ttl_ms or 0, 0) / 1000)
        return list(tools)

    def invalidate_tools(self):
        """Forget the cached tool list (what a notifications/tools/list_changed would trigger)."""
        self._tools = None

    def call_tool(self, name, arguments=None):
        """Call a tool. Always returns {"status": "ok", "content": text} or {"status": "error", "error": text},
        the same shape the in-process Registry uses, so the rest of the harness cannot tell the difference."""
        params = {"name": name, "arguments": arguments or {}}
        try:
            try:
                result = self._rpc("tools/call", params)
            except p.McpConnectionError:
                self.transport.restart()
                if name not in self.read_only:
                    raise
                result = self._rpc("tools/call", params)        # reading again is harmless
        except p.McpRpcError as e:
            return {"status": "error", "error": str(e)}
        except p.McpConnectionError as e:
            return {"status": "error", "error": (
                f"The tool server stopped while running '{name}' and was restarted: {e}. "
                "The call may or may not have completed: check the current state before repeating it.")}
        return self._tool_result(result)

    @staticmethod
    def _tool_result(result):
        if result.get("resultType", "complete") == "input_required":      # no resultType = "complete" (older servers)
            return {"status": "error", "error": "The server asked for more input (multi round-trip requests), "
                                                "which this client does not support yet."}
        blocks = result.get("content") or []
        text = "\n".join(b["text"] if b.get("type") == "text" and "text" in b else f"[{b.get('type', 'unknown')} content not shown]"
                         for b in blocks if isinstance(b, dict))
        if result.get("isError"):
            return {"status": "error", "error": text or "the tool failed"}
        return {"status": "ok", "content": text}

    def close(self):
        self.transport.close()


# ----------------------------------------------------------------------
# starting the tool server for a workspace
# ----------------------------------------------------------------------
def server_command(ws):
    """The command that starts the tool server for this workspace (the harness owns the worktree and container)."""
    command = [sys.executable, "-m", "ce_agent.mcp", "--worktree", str(ws.worktree), "--prefix", ws.prefix]
    if ws.container:
        command += ["--container", ws.container]
    return command


def launch_workspace_server(ws, on_wire=None, timeout=120):
    """Start the stdio tool server for `ws`, say hello with server/discover, and return a ready McpClient.
    The timeout must exceed the 60 s limit that run_shell itself enforces."""
    transport = StdioClient(server_command(ws), env=server_env(), cwd=PROJECT_ROOT, timeout=timeout, on_wire=on_wire)
    client = McpClient(transport)
    try:
        transport.start()
        client.discover()
    except Exception:
        transport.close()
        raise
    return client
