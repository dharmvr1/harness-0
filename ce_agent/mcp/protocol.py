"""
The MCP wire format (protocol revision 2026-07-28) on top of JSON-RPC 2.0.

Plain functions and constants only: no I/O. Everything here can be tested with ordinary dicts.
Spec: https://modelcontextprotocol.io/specification/2026-07-28/basic/index
"""

import json

PROTOCOL_VERSION = "2026-07-28"
SUPPORTED_VERSIONS = [PROTOCOL_VERSION]

# Every request carries these in params["_meta"]: the protocol is stateless, nothing is remembered between requests.
META_VERSION = "io.modelcontextprotocol/protocolVersion"        # required
META_CLIENT_CAPS = "io.modelcontextprotocol/clientCapabilities"  # required
META_CLIENT_INFO = "io.modelcontextprotocol/clientInfo"          # optional
META_SERVER_INFO = "io.modelcontextprotocol/serverInfo"          # servers SHOULD put it in every result's _meta

# Standard JSON-RPC 2.0 error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

# Codes defined by MCP itself (-32020 to -32099 is reserved for the specification)
HEADER_MISMATCH = -32020               # used by the HTTP transport
MISSING_CLIENT_CAPABILITY = -32021
UNSUPPORTED_PROTOCOL_VERSION = -32022


class ProtocolError(Exception):
    """A problem with a message, to be reported to the sender as a JSON-RPC error response."""

    def __init__(self, code, message, data=None, request_id=None):
        super().__init__(message)
        self.code, self.message, self.data, self.request_id = code, message, data, request_id

    def to_response(self):
        return error_response(self.request_id, self.code, self.message, self.data)


# ----------------------------------------------------------------------
# bytes <-> messages
# ----------------------------------------------------------------------
def encode(message) -> str:
    """One message as ONE line. stdio framing forbids newlines inside a message; json.dumps escapes them
    (a newline inside a string becomes the two characters backslash-n), so the result never contains a real one."""
    return json.dumps(message, separators=(",", ":"))


def decode(line: str):
    try:
        return json.loads(line)
    except json.JSONDecodeError as e:
        raise ProtocolError(PARSE_ERROR, f"Parse error: {e}")


# ----------------------------------------------------------------------
# building messages
# ----------------------------------------------------------------------
def make_request(request_id, method, params=None, *, version=PROTOCOL_VERSION, client_info=None, capabilities=None):
    """A request with the required _meta fields filled in."""
    meta = {META_VERSION: version, META_CLIENT_CAPS: capabilities if capabilities is not None else {}}
    if client_info:
        meta[META_CLIENT_INFO] = client_info
    return {"jsonrpc": "2.0", "id": request_id, "method": method, "params": {**(params or {}), "_meta": meta}}


def make_notification(method, params=None):
    message = {"jsonrpc": "2.0", "method": method}
    if params:
        message["params"] = params
    return message


def result_response(request_id, result, server_info=None):
    """A success response. resultType defaults to "complete"; serverInfo goes into _meta."""
    result = dict(result)
    result.setdefault("resultType", "complete")
    if server_info:
        result["_meta"] = {**result.get("_meta", {}), META_SERVER_INFO: server_info}
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def error_response(request_id, code, message, data=None):
    """An error response. The id is left out when it is unknown (for example the message was not valid JSON)."""
    error = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    response = {"jsonrpc": "2.0", "error": error}
    if request_id is not None:
        response["id"] = request_id
    return response


def tool_result(text, is_error=False):
    """The result of tools/call: text content, plus isError so the model can tell failure from success."""
    return {"resultType": "complete", "content": [{"type": "text", "text": text}], "isError": is_error}


# ----------------------------------------------------------------------
# checking incoming messages
# ----------------------------------------------------------------------
def is_valid_id(value) -> bool:
    """Ids are strings or integers. Unlike plain JSON-RPC, null is not allowed (and JSON true/false are not numbers)."""
    return isinstance(value, (str, int)) and not isinstance(value, bool)


def check_request(message):
    """Check the JSON-RPC shape of a message sent to a server.
    Returns (kind, id, method, params) where kind is "request" or "notification"."""
    if not isinstance(message, dict):
        raise ProtocolError(INVALID_REQUEST, "A message must be a JSON object")
    request_id = message.get("id") if is_valid_id(message.get("id")) else None
    if message.get("jsonrpc") != "2.0":
        raise ProtocolError(INVALID_REQUEST, 'Missing or wrong "jsonrpc": must be "2.0"', request_id=request_id)
    method = message.get("method")
    if not isinstance(method, str) or not method:
        raise ProtocolError(INVALID_REQUEST, 'Missing "method" (clients may not send responses)', request_id=request_id)
    if "id" in message and request_id is None:
        raise ProtocolError(INVALID_REQUEST, "The id must be a string or an integer (null is not allowed)")
    params = message.get("params", {})
    if not isinstance(params, dict):
        raise ProtocolError(INVALID_PARAMS, "params must be an object", request_id=request_id)
    return ("request" if "id" in message else "notification"), request_id, method, params


def check_meta(params, request_id=None):
    """Check params["_meta"], which every request must carry. Returns the _meta dict."""
    meta = params.get("_meta")
    if not isinstance(meta, dict):
        raise ProtocolError(INVALID_PARAMS, 'Missing "_meta" in params', request_id=request_id)
    version = meta.get(META_VERSION)
    if not isinstance(version, str):
        raise ProtocolError(INVALID_PARAMS, f'Missing required _meta field "{META_VERSION}"', request_id=request_id)
    # The version is checked before the other fields: a future version may define them differently.
    if version not in SUPPORTED_VERSIONS:
        raise ProtocolError(UNSUPPORTED_PROTOCOL_VERSION, "Unsupported protocol version",
                            data={"supported": SUPPORTED_VERSIONS, "requested": version}, request_id=request_id)
    if not isinstance(meta.get(META_CLIENT_CAPS), dict):
        raise ProtocolError(INVALID_PARAMS, f'Missing required _meta field "{META_CLIENT_CAPS}"', request_id=request_id)
    if META_CLIENT_INFO in meta and not isinstance(meta[META_CLIENT_INFO], dict):
        raise ProtocolError(INVALID_PARAMS, f'"{META_CLIENT_INFO}" must be an object', request_id=request_id)
    return meta


# ----------------------------------------------------------------------
# errors raised on the CLIENT side (all are RuntimeError, so the CLI prints them as "Error: ...")
# ----------------------------------------------------------------------
class McpError(RuntimeError):
    """Base class for everything that goes wrong while talking to an MCP server."""


class McpRpcError(McpError):
    """The server answered with a JSON-RPC error response."""

    def __init__(self, code, message, data=None):
        super().__init__(f"MCP error {code}: {message}")
        self.code, self.message, self.data = code, message, data


class McpConnectionError(McpError):
    """The server process died, closed its pipe, or could not be started."""


class McpTimeout(McpConnectionError):
    """No reply arrived in time. The server may still be busy, so the connection is treated as broken."""


class McpVersionError(McpError):
    """Client and server share no protocol version."""
