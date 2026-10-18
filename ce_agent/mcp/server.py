"""
McpServer: the server side of MCP, with no I/O at all.

    reply = server.handle(message_dict)       # a dict in, a dict out (or None for notifications)

Transports (stdio.py, later http.py) only move bytes and call handle(). That keeps the protocol logic testable
with plain dicts, and lets one server speak over any transport.

Three methods are implemented: server/discover, tools/list, tools/call.
"""

import time

from ce_agent.mcp import protocol as p

SERVER_INFO = {"name": "ce-agent-tools", "version": "0.1.0"}

# JSON Schema type name -> a function that says whether a Python value has that type.
# bool is a subclass of int in Python, but JSON true/false are not numbers.
TYPE_CHECKS = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: (isinstance(v, int) and not isinstance(v, bool)) or (isinstance(v, float) and v.is_integer()),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
}


def check_arguments(schema, arguments):
    """Strictly check tool arguments against the tool's inputSchema (the spec says servers MUST validate inputs).
    Returns (arguments, error): `error` is a message the model can act on, or None.
    A whole number sent as 4.0 is accepted and turned into 4, because JSON does not tell 4 and 4.0 apart."""
    properties = schema.get("properties", {})
    for name in schema.get("required", []):
        if name not in arguments:
            return arguments, f"Missing required argument: {name}"
    cleaned = {}
    for name, value in arguments.items():
        if name not in properties:
            return arguments, f"Unexpected argument: {name}. Valid arguments: {', '.join(properties) or '(none)'}"
        expected = properties[name].get("type")
        if expected in TYPE_CHECKS and not TYPE_CHECKS[expected](value):
            return arguments, f"Argument {name} must be of type {expected}, got {type(value).__name__}"
        cleaned[name] = int(value) if expected == "integer" else value
    return cleaned, None


class McpServer:
    def __init__(self, registry, server_info=None, instructions=None, ttl_ms=60_000, log=None):
        self.registry = registry                       # the tools: a Registry (name -> function + schema)
        self.server_info = server_info or SERVER_INFO
        self.instructions = instructions
        self.ttl_ms = ttl_ms                           # how long a client may cache discover / tools/list results
        self.log = log or (lambda message: None)
        self.methods = {
            "server/discover": self._discover,
            "tools/list": self._list_tools,
            "tools/call": self._call_tool,
        }

    # ------------------------------------------------------------------
    # entry points
    # ------------------------------------------------------------------
    def handle_line(self, line):
        """One line of text in, one line of text out (or None). This is what the stdio transport calls."""
        try:
            message = p.decode(line)
        except p.ProtocolError as e:
            return p.encode(e.to_response())
        response = self.handle(message)
        return None if response is None else p.encode(response)

    def handle(self, message):
        """One incoming message (a dict) -> the response to send back, or None when no reply is due."""
        try:
            kind, request_id, method, params = p.check_request(message)
        except p.ProtocolError as e:
            return e.to_response()
        if kind == "notification":
            return None            # a notification never gets a reply, and we act on none of them yet

        try:
            if method == "initialize":
                # A client from the older, handshake-based era. Say which versions we do speak (the spec asks us to).
                raise p.ProtocolError(p.METHOD_NOT_FOUND,
                                      f"This server speaks MCP {', '.join(p.SUPPORTED_VERSIONS)} only; "
                                      "'initialize' belongs to older protocol versions",
                                      data={"supported": p.SUPPORTED_VERSIONS})
            p.check_meta(params)
            handler = self.methods.get(method)
            if handler is None:
                raise p.ProtocolError(p.METHOD_NOT_FOUND, f"Method not found: {method}")
            return p.result_response(request_id, handler(params), self.server_info)
        except p.ProtocolError as e:
            e.request_id = request_id
            return e.to_response()
        except Exception as e:     # a bug in the server must never kill the server
            self.log(f"internal error handling {method}: {type(e).__name__}: {e}")
            return p.error_response(request_id, p.INTERNAL_ERROR, f"Internal error: {type(e).__name__}")

    # ------------------------------------------------------------------
    # the three methods
    # ------------------------------------------------------------------
    def _cache_hints(self):
        # Servers MUST say how long discover / tools/list results stay fresh. Our tool set never changes
        # while the process runs, and it is the same for every caller, hence "public".
        return {"ttlMs": self.ttl_ms, "cacheScope": "public"}

    def _discover(self, params):
        result = {"supportedVersions": p.SUPPORTED_VERSIONS, "capabilities": {"tools": {}}, **self._cache_hints()}
        if self.instructions:
            result["instructions"] = self.instructions
        return result

    def _list_tools(self, params):
        if params.get("cursor") is not None:        # we never paginate, so we never issued a cursor
            raise p.ProtocolError(p.INVALID_PARAMS, "Invalid cursor")
        tools = [{"name": name, "description": tool["description"], "inputSchema": tool["schema"]}
                 for name, tool in self.registry.tools.items()]       # registration order: deterministic
        return {"tools": tools, **self._cache_hints()}

    def _call_tool(self, params):
        name = params.get("name")
        if not isinstance(name, str):
            raise p.ProtocolError(p.INVALID_PARAMS, 'Missing tool "name"')
        tool = self.registry.tools.get(name)
        if tool is None:
            raise p.ProtocolError(p.INVALID_PARAMS, f"Unknown tool: {name}")      # a protocol error, as the spec shows
        arguments = params.get("arguments", {})
        if not isinstance(arguments, dict):
            raise p.ProtocolError(p.INVALID_PARAMS, '"arguments" must be an object')

        # From here on, failures are TOOL errors: a normal result with isError=true, which the model can read and fix.
        arguments, problem = check_arguments(tool["schema"], arguments)
        if problem:
            self.log(f"tools/call {name}: rejected: {problem}")
            return p.tool_result(problem, is_error=True)
        started = time.monotonic()
        try:
            text = str(tool["func"](**arguments))
            failed = False
        except Exception as e:
            text, failed = str(e) or type(e).__name__, True
        self.log(f"tools/call {name}: {'error' if failed else 'ok'} in {time.monotonic() - started:.2f}s")
        return p.tool_result(text, is_error=failed)
