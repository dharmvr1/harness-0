"""
The Streamable HTTP transport (protocol revision 2026-07-28), hand-written on the standard library.

Server:  serve_http(mcp_server, ...) returns a McpHttpServer. Every JSON-RPC message is its own POST to ONE endpoint
         (default /mcp). The reply is a JSON body, or, if a tool has progress to report, a Server-Sent Events stream
         that ends with the final response.
Client:  HttpClient (talks to any such server) and HttpServerProcess (also starts and stops the server process).

Status codes. Required by the spec:  202 for an accepted notification; 400 for header/body mismatch
(HeaderMismatch -32020), an unsupported version (-32022) and missing _meta fields (-32602); 404 + -32601 for an
unknown method; 403 for a bad Origin; 405 for GET and DELETE. My own choices where the spec is silent:  401 bad or
missing token, 406 Accept without both types, 411/413 body length, 415 not JSON, 400 for the other protocol errors,
500 for -32603, and 200 for tool errors (isError is part of a normal result).

Closing the connection is the cancellation signal: the server notices (see _peer_closed) and the running tool
stops with RequestCancelled.
"""

import hmac
import http.client
import json
import queue
import secrets
import select
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from ce_agent.mcp import protocol as p
from ce_agent.mcp.headers import (check_param_headers, decode_header_value, encode_header_value,
                                  header_values_for_call, mirrored_parameters)
from ce_agent.mcp.progress import RequestCancelled
from ce_agent.mcp.sse import parse_sse

MAX_BODY_BYTES = 10 * 1024 * 1024
LOOPBACK_HOSTS = ("127.0.0.1", "localhost")
# The request field that Mcp-Name mirrors, per method.
NAME_FIELD = {"tools/call": "name", "prompts/get": "name", "resources/read": "uri"}
STATUS_FOR_CODE = {
    p.PARSE_ERROR: 400, p.INVALID_REQUEST: 400, p.INVALID_PARAMS: 400, p.METHOD_NOT_FOUND: 404,
    p.INTERNAL_ERROR: 500, p.HEADER_MISMATCH: 400, p.MISSING_CLIENT_CAPABILITY: 400, p.UNSUPPORTED_PROTOCOL_VERSION: 400,
}


def status_for(response):
    """The HTTP status for a JSON-RPC response: 200 for a result, by error code for an error."""
    if "error" in response:
        return STATUS_FOR_CODE.get(response["error"].get("code"), 400)
    return 200


def accepts_json_and_sse(accept):
    """Does an Accept header allow BOTH application/json and text/event-stream (the spec makes clients list both)?"""
    allowed = {part.split(";")[0].strip().lower() for part in accept.split(",")}
    def ok(kind):
        return kind in allowed or kind.split("/")[0] + "/*" in allowed or "*/*" in allowed
    return ok("application/json") and ok("text/event-stream")


# ======================================================================
# server
# ======================================================================
class McpHttpServer(ThreadingHTTPServer):
    daemon_threads = True            # a request thread never keeps the process alive

    def __init__(self, address, mcp, *, token=None, path="/mcp", allowed_origins=(), allowed_hosts="loopback",
                 max_body=MAX_BODY_BYTES):
        self.mcp, self.token, self.path = mcp, token, path
        self.allowed_origins = set(allowed_origins)
        self.max_body = max_body
        self.tool_lock = threading.Lock()      # tools share one workspace, so one tool runs at a time
        super().__init__(address, McpRequestHandler)
        host, port = self.server_address[:2]
        if allowed_hosts == "loopback":        # defence against DNS rebinding, on top of the Origin check
            allowed_hosts = {f"{name}:{port}" for name in LOOPBACK_HOSTS}
        self.allowed_hosts = None if allowed_hosts is None else set(allowed_hosts)
        self.url = f"http://{host}:{port}{path}"


def serve_http(mcp, host="127.0.0.1", port=0, *, token=None, **options):
    """Create the server (it does not run yet: call serve_forever(), usually in a thread). port=0 picks a free port.
    A non-loopback address is refused without a token, because anyone on the network could then run commands."""
    if host not in LOOPBACK_HOSTS and not token:
        raise ValueError(f"refusing to listen on {host} without an authentication token")
    return McpHttpServer((host, port), mcp, token=token, **options)


class McpRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"      # one request per connection; the end of a response is the end of the connection
    timeout = 30                       # seconds a client may stall while sending its request
    server_version = "ce-agent-mcp"
    sys_version = ""

    def version_string(self):
        return self.server_version

    def log_message(self, format, *args):
        self.server.mcp.log("http " + (format % args))     # only the request line: the Authorization header is never logged

    # ---------- anything that is not POST ----------
    def _method_not_allowed(self):
        self._send_text(405, "This endpoint accepts POST only", {"Allow": "POST"})

    do_GET = do_DELETE = do_PUT = do_PATCH = do_HEAD = do_OPTIONS = _method_not_allowed

    # ---------- POST ----------
    def do_POST(self):
        server = self.server
        if urlsplit(self.path).path != server.path:
            return self._send_text(404, "Not found")
        origin = self.headers.get("Origin")
        if origin is not None and origin not in server.allowed_origins:
            return self._send_text(403, "Forbidden origin")        # DNS rebinding protection (required by the spec)
        if server.allowed_hosts is not None and self.headers.get("Host") not in server.allowed_hosts:
            return self._send_text(403, "Forbidden host")
        if server.token and not self._token_ok():
            return self._send_text(401, "Authentication required", {"WWW-Authenticate": "Bearer"})
        if self.headers.get("Content-Type", "").split(";")[0].strip().lower() != "application/json":
            return self._send_text(415, "Content-Type must be application/json")
        if not accepts_json_and_sse(self.headers.get("Accept", "")):
            return self._send_text(406, "Accept must list application/json and text/event-stream")

        length = self.headers.get("Content-Length")
        if length is None:
            return self._send_text(411, "Content-Length is required")
        if not length.isdigit():
            return self._send_text(400, "Bad Content-Length")
        if int(length) > server.max_body:
            return self._send_text(413, "Request body too large")
        try:
            body = self.rfile.read(int(length))
        except OSError:                                             # the client stalled or left while sending
            self.close_connection = True
            return

        try:
            message = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            return self._send_json(400, p.error_response(None, p.PARSE_ERROR, f"Parse error: {e}"))
        try:
            kind, request_id, method, params = p.check_request(message)
        except p.ProtocolError as e:
            return self._send_json(400, e.to_response())
        if kind == "notification":
            return self._send_empty(202)                          # accepted, no body, nothing more to do

        try:
            self._check_headers(method, params, request_id)
        except p.ProtocolError as e:
            return self._send_json(400, e.to_response())

        sink = _ResponseSink(self)
        with server.tool_lock:
            response = server.mcp.handle(message, notify=sink.notify, is_cancelled=self._peer_closed)
        if response is None:                                      # cancelled: send nothing more for this request
            self.close_connection = True
            return
        sink.finish(response)

    def _token_ok(self):
        header = self.headers.get("Authorization", "")
        supplied = header[7:] if header.lower().startswith("bearer ") else ""
        return hmac.compare_digest(supplied.encode(), self.server.token.encode())     # constant time

    def _check_headers(self, method, params, request_id):
        """The headers must agree with the body, or a proxy and this server could act on different values."""
        def mismatch(text):
            raise p.ProtocolError(p.HEADER_MISMATCH, text, request_id=request_id)

        version = self.headers.get("MCP-Protocol-Version")
        if version is None:
            mismatch("Missing required header MCP-Protocol-Version")
        meta = params.get("_meta")
        body_version = meta.get(p.META_VERSION) if isinstance(meta, dict) else None
        if isinstance(body_version, str) and version != body_version:
            mismatch(f"Header MCP-Protocol-Version '{version}' does not match the protocol version '{body_version}' in _meta")

        header_method = self.headers.get("Mcp-Method")
        if header_method is None:
            mismatch("Missing required header Mcp-Method")
        if header_method != method:
            mismatch(f"Header Mcp-Method '{header_method}' does not match the method '{method}' in the body")

        field = NAME_FIELD.get(method)
        if field and isinstance(params.get(field), str):
            raw = self.headers.get("Mcp-Name")
            if raw is None:
                mismatch("Missing required header Mcp-Name")
            try:
                name = decode_header_value(raw)
            except ValueError:
                mismatch("Header Mcp-Name is not valid")
            if name != params[field]:
                mismatch(f"Header Mcp-Name '{name}' does not match '{params[field]}' in the body")

        if method == "tools/call" and isinstance(params.get("name"), str):
            schema = self.server.mcp.input_schema(params["name"])
            arguments = params.get("arguments")
            if schema and isinstance(arguments, dict):
                problem = check_param_headers(schema, arguments, self.headers)
                if problem:
                    mismatch(problem)

    def _peer_closed(self):
        """Has the client closed its end? A socket that is readable but yields no bytes has reached end-of-file.
        (MSG_PEEK looks without consuming.) This is how a closed connection becomes a cancellation."""
        try:
            readable, _, _ = select.select([self.connection], [], [], 0)
            return bool(readable) and self.connection.recv(1, socket.MSG_PEEK) == b""
        except (OSError, ValueError):
            return True

    # ---------- writing responses ----------
    def _send_json(self, status, message):
        body = p.encode(message).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, status, text, headers=None):
        body = (text + "\n").encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_empty(self, status):
        self.send_response(status)
        self.send_header("Content-Length", "0")
        self.end_headers()


class _ResponseSink:
    """Where one request's output goes. It starts as a plain JSON response and switches to a Server-Sent Events
    stream the first time the tool sends a notification (progress). The headers go out only at that moment, so a
    request that never reports progress gets an ordinary JSON reply."""

    def __init__(self, handler):
        self.handler = handler
        self.streaming = False

    def notify(self, message):
        try:
            if not self.streaming:
                h = self.handler
                h.send_response(200)
                h.send_header("Content-Type", "text/event-stream")
                h.send_header("Cache-Control", "no-cache")
                h.send_header("X-Accel-Buffering", "no")      # tells nginx and friends not to hold the events back
                h.end_headers()
                self.streaming = True
            self._event(message)
        except OSError:                                      # the client closed the stream: that is a cancellation
            raise RequestCancelled()

    def finish(self, response):
        try:
            if self.streaming:
                self._event(response)                        # headers are already sent, so the status stays 200
            else:
                self.handler._send_json(status_for(response), response)
        except OSError:
            pass                                             # the client left before the answer arrived

    def _event(self, message):
        self.handler.wfile.write(f"data: {p.encode(message)}\n\n".encode("utf-8"))
        self.handler.wfile.flush()


# ======================================================================
# client
# ======================================================================
class HttpClient:
    """Sends each JSON-RPC request as one POST and reads the JSON or event-stream reply.
    One connection per request (the protocol is stateless, and the server closes after each reply)."""
    mirrors_headers = True               # an HTTP client must copy x-mcp-header parameters into headers
    cancels_on_timeout = True            # giving up = closing the connection, which cancels the request on the server

    def __init__(self, url, token=None, timeout=120, on_wire=None):
        parts = urlsplit(url)
        self.url, self.token, self.timeout, self.on_wire = url, token, timeout, on_wire
        self.host, self.port, self.path = parts.hostname, parts.port or 80, parts.path or "/"
        self.on_notification = None      # set by McpClient: called with every notification the server sends
        self.schema_lookup = None        # set by McpClient: tool name -> inputSchema (for Mcp-Param-* headers)

    def start(self):
        pass

    def restart(self):
        pass                             # the server is somebody else's; nothing for us to restart

    def close(self):
        pass                             # no connection is kept open

    def _headers(self, message):
        params = message.get("params") or {}
        meta = params.get("_meta") or {}
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": meta.get(p.META_VERSION, ""),
            "Mcp-Method": message["method"],
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        field = NAME_FIELD.get(message["method"])
        if field and isinstance(params.get(field), str):
            headers["Mcp-Name"] = encode_header_value(params[field])
        if message["method"] == "tools/call" and self.schema_lookup and isinstance(params.get("name"), str):
            schema = self.schema_lookup(params["name"])
            if schema and isinstance(params.get("arguments"), dict):
                try:
                    headers.update(header_values_for_call(schema, params["arguments"]))
                except ValueError as e:
                    raise p.McpError(f"cannot send this call over HTTP: {e}")
        return headers

    def request(self, message):
        headers = self._headers(message)
        if self.on_wire:
            self.on_wire("send", message)
        deadline = time.monotonic() + self.timeout
        # The socket timeout bounds every single read; `deadline` bounds the whole request (see _read_stream).
        connection = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        response = None
        try:
            try:
                connection.request("POST", self.path, body=p.encode(message).encode("utf-8"), headers=headers)
                response = connection.getresponse()
            except (socket.timeout, TimeoutError):
                raise p.McpTimeout(f"no reply to {message['method']} within {self.timeout}s")
            except (OSError, http.client.HTTPException) as e:
                raise p.McpConnectionError(f"could not reach the MCP server at {self.url}: {e}")
            content_type = (response.getheader("Content-Type") or "").split(";")[0].strip().lower()
            if content_type == "text/event-stream":
                return self._read_stream(response, message, deadline)
            return self._read_body(response, message)
        finally:
            # Closing the connection while a stream is still open IS the cancellation signal. For a reply that ends
            # when the connection closes, http.client hands the socket to the response, so close that explicitly.
            if response is not None:
                response.close()
            connection.close()

    def _read_body(self, response, message):
        try:
            raw = response.read()
        except (socket.timeout, TimeoutError):
            raise p.McpTimeout(f"no reply to {message['method']} within {self.timeout}s")
        except (OSError, http.client.HTTPException) as e:
            raise p.McpConnectionError(f"connection to the MCP server broke: {e}")
        try:
            reply = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            reply = None
        if isinstance(reply, dict) and ("result" in reply or "error" in reply):
            if self.on_wire:
                self.on_wire("recv", reply)
            return reply                 # also for 4xx/5xx: a JSON-RPC error body explains itself
        snippet = " ".join(raw.decode("utf-8", "replace").split())[:200]
        text = f"HTTP {response.status} {response.reason} from the MCP server" + (f": {snippet}" if snippet else "")
        raise (p.McpConnectionError if response.status >= 500 else p.McpHttpError)(text)

    def _read_stream(self, response, message, deadline):
        def lines():
            while True:
                if time.monotonic() > deadline:       # progress events do not extend the deadline (the spec says so)
                    raise p.McpTimeout(f"no final reply to {message['method']} within {self.timeout}s")
                try:
                    line = response.readline()
                except (socket.timeout, TimeoutError):
                    raise p.McpTimeout(f"no reply to {message['method']} within {self.timeout}s")
                except (OSError, http.client.HTTPException) as e:
                    raise p.McpConnectionError(f"connection to the MCP server broke: {e}")
                if not line:
                    return
                yield line.decode("utf-8", "replace")

        for event in parse_sse(lines()):
            try:
                reply = json.loads(event["data"])
            except json.JSONDecodeError:
                continue                 # not JSON: ignore it
            if self.on_wire:
                self.on_wire("recv", reply)
            if isinstance(reply, dict) and ("result" in reply or "error" in reply):
                return reply
            if isinstance(reply, dict) and "method" in reply and "id" not in reply and self.on_notification:
                try:
                    self.on_notification(reply)
                except Exception:
                    pass
        raise p.McpConnectionError("the event stream ended without a response")


class HttpServerProcess(HttpClient):
    """An HttpClient that also starts the server as a subprocess and stops it again.

    The server picks a free port and announces itself with one JSON line on stdout: {"ready": true, "url": ...}.
    The bearer token is created here and handed over in an environment variable, never on the command line
    (command lines are visible to every user through `ps`). The server also exits when its stdin closes, so it
    cannot outlive a harness that was killed."""

    def __init__(self, command, env=None, cwd=None, timeout=120, on_wire=None, startup_timeout=30):
        super().__init__("http://127.0.0.1:0/mcp", timeout=timeout, on_wire=on_wire)
        self.command, self.env, self.cwd, self.startup_timeout = command, env, cwd, startup_timeout
        self.process = None
        self._stderr = []

    def start(self):
        self.token = secrets.token_urlsafe(32)
        env = {**(self.env or {}), "CE_MCP_TOKEN": self.token}
        try:
            self.process = subprocess.Popen(self.command + ["--transport", "http", "--port", "0"], stdin=subprocess.PIPE,
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8",
                                            env=env, cwd=self.cwd)
        except OSError as e:
            raise p.McpConnectionError(f"could not start the MCP server {self.command[:3]}: {e}")
        self._stderr = []
        process, ready = self.process, queue.Queue()
        threading.Thread(target=lambda: ready.put(process.stdout.readline()), daemon=True).start()
        drain = threading.Thread(target=self._drain_stderr, args=(process,), daemon=True)
        drain.start()
        try:
            line = ready.get(timeout=self.startup_timeout)
        except queue.Empty:
            self.close()
            raise p.McpTimeout(f"the MCP server did not announce itself within {self.startup_timeout}s")
        try:
            announcement = json.loads(line)
            url = announcement["url"]
        except (json.JSONDecodeError, KeyError, TypeError):
            try:
                process.wait(timeout=2)               # it probably exited: give its last words time to arrive
            except subprocess.TimeoutExpired:
                pass
            drain.join(timeout=1)
            code = process.poll()
            tail = self.stderr_tail()
            self.close()
            raise p.McpConnectionError("the MCP server failed to start" + (f" (exit code {code})" if code is not None else "")
                                       + (f"\nserver stderr:\n{tail}" if tail else ""))
        parts = urlsplit(url)
        self.url, self.host, self.port, self.path = url, parts.hostname, parts.port, parts.path

    def _drain_stderr(self, process):
        for line in process.stderr:
            self._stderr.append(line.rstrip())
            del self._stderr[:-50]

    def stderr_tail(self):
        return "\n".join(self._stderr)

    def request(self, message):
        if self.process is None:
            self.start()
        if self.process.poll() is not None:
            tail = self.stderr_tail()
            raise p.McpConnectionError(f"the MCP server is not running (exit code {self.process.returncode})"
                                       + (f"\nserver stderr:\n{tail}" if tail else ""))
        return super().request(message)

    def restart(self):
        self.close(grace=0.5)
        self.start()

    def close(self, grace=3):
        process, self.process = self.process, None
        if process is None:
            return
        try:
            process.stdin.close()                     # the server exits when its stdin closes
        except OSError:
            pass
        try:
            process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for pipe in (process.stdout, process.stderr):
            try:
                pipe.close()
            except OSError:
                pass
