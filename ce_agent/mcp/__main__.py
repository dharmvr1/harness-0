"""
Run the tool server:

    python -m ce_agent.mcp --worktree PATH [--prefix SUB] [--container NAME]                 (stdio, the default)
    python -m ce_agent.mcp --worktree PATH --transport http [--host 127.0.0.1] [--port 0]    (Streamable HTTP)

It serves the workspace tools (read_file, edit_file, run_shell, ...) for ONE worktree that the harness created.
The harness owns the worktree and container; this process only uses them.

HTTP mode: the bearer token is read from the environment variable CE_MCP_TOKEN (never from the command line, where
every user could see it). The server prints one line {"ready": true, "url": ...} on stdout, and exits when its
stdin closes, so it cannot outlive whoever started it (disable with --no-stdin-watch).
"""

import argparse
import json
import os
import signal
import sys
import threading
from pathlib import Path

from ce_agent.mcp.server import McpServer
from ce_agent.mcp.stdio import protect_stdout, serve_stdio
from ce_agent.mcp.streamable_http import serve_http
from ce_agent.registry import Registry
from ce_agent.tools import make_workspace_tools
from ce_agent.workspace import Workspace


def build_server(worktree, prefix="", container=None, log=None, command_timeout=None):
    if not Path(worktree).is_dir():
        raise SystemExit(f"error: worktree {worktree!r} is not a directory")
    ws = Workspace.attach(worktree, prefix, container, command_timeout)
    registry = Registry()
    for func, description in make_workspace_tools(ws):
        registry.register(func, description)
    server = McpServer(registry, log=log)
    server.on_shutdown = ws.kill_running       # commands must not outlive the server that was running them
    return server


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m ce_agent.mcp", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--worktree", required=True, help="the git worktree the agent works in")
    parser.add_argument("--prefix", default="", help="sub-folder inside the worktree that is the project root")
    parser.add_argument("--container", help="Docker container to run commands in (default: run them on the host)")
    parser.add_argument("--command-timeout", type=int, help="seconds one run_shell command may take (default 60)")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1", help="http only: address to listen on (loopback unless a token is set)")
    parser.add_argument("--port", type=int, default=0, help="http only: port (0 = pick a free one)")
    parser.add_argument("--path", default="/mcp", help="http only: the URL path of the MCP endpoint")
    parser.add_argument("--allow-origin", action="append", default=[], help="http only: an Origin header value to accept")
    parser.add_argument("--no-stdin-watch", action="store_true", help="http only: do not exit when stdin closes")
    args = parser.parse_args(argv)

    def log(message):                       # logs go to stderr, never to stdout
        print(f"[mcp-server] {message}", file=sys.stderr, flush=True)

    server = build_server(args.worktree, args.prefix, args.container, log, args.command_timeout)
    # SIGTERM normally kills a Python process on the spot. Turn it into a normal exit so the cleanup below runs.
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(0))
    try:
        run_transport(args, server, log)
    finally:
        server.on_shutdown()


def run_transport(args, server, log):
    if args.transport == "stdio":
        serve_stdio(server, sys.stdin, protect_stdout())
        return

    token = os.environ.pop("CE_MCP_TOKEN", None)       # take it out of the environment so children cannot inherit it
    try:
        httpd = serve_http(server, args.host, args.port, token=token, path=args.path, allowed_origins=args.allow_origin)
    except ValueError as e:
        raise SystemExit(f"error: {e}")
    if not token:
        log("WARNING: no CE_MCP_TOKEN set: anyone who can reach this port can run commands (loopback only)")
    out = protect_stdout()
    out.write(json.dumps({"ready": True, "url": httpd.url}) + "\n")
    out.flush()
    if not args.no_stdin_watch:
        def watch_stdin():
            for _ in iter(sys.stdin.readline, ""):
                pass
            httpd.shutdown()                            # stdin closed: whoever started us is gone
        threading.Thread(target=watch_stdin, daemon=True).start()
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
