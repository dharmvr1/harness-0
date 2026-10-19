"""
The stdio transport: newline-delimited JSON-RPC over a subprocess's standard streams.

Server side:  serve_stdio(server, stdin, stdout)   (+ protect_stdout() so nothing else can corrupt stdout)
Client side:  StdioClient (a subprocess, plus reader threads so that waiting for a reply can time out).

Rules from the spec: one message per line, no newlines inside a message; the server writes ONLY protocol
messages to stdout (logs go to stderr); closing the server's stdin is the signal to exit.
"""

import collections
import os
import queue
import subprocess
import sys
import threading
import time

from ce_agent.mcp import protocol as p


def protect_stdout():
    """Give back a file for protocol messages, and send everything else that tries to write to stdout to stderr.

    A single stray print() in a tool would put garbage on the protocol stream and break the client. So: keep a
    private copy of the real stdout (dup), then point file descriptor 1 at stderr (dup2). Even a child process that
    inherits fd 1 now writes to stderr."""
    protocol_out = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8", newline="\n")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return protocol_out


def serve_stdio(server, stdin, stdout):
    """Read one request per line, answer on one line, until the client closes our stdin."""
    while True:
        line = stdin.readline()
        if not line:                 # end of file: the client closed our stdin, which means "shut down"
            break
        line = line.strip()
        if not line:
            continue
        def notify(message):         # a running tool can send progress while the request is still being worked on
            stdout.write(p.encode(message) + "\n")
            stdout.flush()
        reply = server.handle_line(line, notify=notify)
        if reply is not None:        # notifications get no reply
            stdout.write(reply + "\n")
            stdout.flush()


# ----------------------------------------------------------------------
# client side
# ----------------------------------------------------------------------
# Environment variables a server process may inherit. API keys are deliberately NOT here:
# the tool server runs model-chosen shell commands, and anything in its environment can end up in the model's prompt.
ENV_ALLOWLIST = ["PATH", "HOME", "LANG", "LC_ALL", "USER", "TMPDIR", "XDG_RUNTIME_DIR",
                 "DOCKER_HOST", "DOCKER_CONFIG", "DOCKER_CONTEXT"]


def server_env(environ=None):
    """A minimal environment for the server subprocess."""
    environ = os.environ if environ is None else environ
    env = {name: environ[name] for name in ENV_ALLOWLIST if name in environ}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


_EOF = object()   # put on the queue when the server's stdout closes


class StdioClient:
    """Talks to an MCP server running as a subprocess. Plain threads and queues, one request at a time.

    A reader thread turns the server's stdout into a queue of lines, so waiting for a reply can have a TIMEOUT
    (a blocking readline() cannot). Another thread drains stderr, because an unread pipe fills up and then
    blocks the server."""

    def __init__(self, command, env=None, cwd=None, timeout=120, on_wire=None):
        self.command, self.env, self.cwd, self.timeout = command, env, cwd, timeout
        self.on_wire = on_wire              # optional: called as on_wire("send" | "recv", message) for every message
        self.on_notification = None         # set by McpClient: called with every notification the server sends
        self.process = None
        self._lines = queue.Queue()
        self._stderr = collections.deque(maxlen=50)

    # ---------- process lifecycle ----------
    def start(self):
        try:
            self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1,
                                            env=self.env, cwd=self.cwd)
        except OSError as e:
            raise p.McpConnectionError(f"could not start the MCP server {self.command[:3]}: {e}")
        self._lines = queue.Queue()         # a fresh queue per process: late output of an old one can't leak in
        threading.Thread(target=self._read_stdout, args=(self.process, self._lines), daemon=True).start()
        threading.Thread(target=self._drain_stderr, args=(self.process,), daemon=True).start()

    def _read_stdout(self, process, lines):
        for line in process.stdout:
            lines.put(line)
        lines.put(_EOF)

    def _drain_stderr(self, process):
        for line in process.stderr:
            self._stderr.append(line.rstrip())

    def stderr_tail(self):
        return "\n".join(self._stderr)

    def restart(self):
        self.close(grace=0.5)          # a server being restarted is broken or stuck: do not wait the full 3 s for it
        self.start()

    def close(self, grace=3):
        """Shut down the way the spec says: close stdin, wait, then terminate, then kill."""
        process, self.process = self.process, None
        if process is None:
            return
        try:
            process.stdin.close()
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

    # ---------- one request, one reply ----------
    def request(self, message):
        """Send a request, wait for the reply that has the same id. Anything else that arrives meanwhile
        (notifications, a late reply to an older request that timed out) is ignored."""
        if self.process is None:
            self.start()
        if self.process.poll() is not None:
            raise p.McpConnectionError(f"the MCP server is not running (exit code {self.process.returncode})"
                                       + self._stderr_note())
        self._send(message)
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                line = self._lines.get(timeout=max(deadline - time.monotonic(), 0))
            except queue.Empty:
                raise p.McpTimeout(f"no reply to {message['method']} within {self.timeout}s")
            if line is _EOF:
                raise p.McpConnectionError("the MCP server closed its output (it exited)" + self._stderr_note())
            try:
                reply = p.decode(line)
            except p.ProtocolError:
                continue                       # not JSON: ignore it, as a robust client should
            if self.on_wire:
                self.on_wire("recv", reply)
            if isinstance(reply, dict) and reply.get("id") == message["id"] and ("result" in reply or "error" in reply):
                return reply
            if isinstance(reply, dict) and "method" in reply and "id" not in reply and self.on_notification:
                try:
                    self.on_notification(reply)
                except Exception:
                    pass                       # a broken callback must not break the connection

    def _send(self, message):
        if self.on_wire:
            self.on_wire("send", message)
        try:
            self.process.stdin.write(p.encode(message) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            raise p.McpConnectionError("could not write to the MCP server (it exited)" + self._stderr_note())

    def _stderr_note(self):
        tail = self.stderr_tail()
        return f"\nserver stderr:\n{tail}" if tail else ""
