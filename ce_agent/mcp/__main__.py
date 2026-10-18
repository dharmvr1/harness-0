"""
Run the tool server:  python -m ce_agent.mcp --worktree PATH [--prefix SUB] [--container NAME]

It serves the workspace tools (read_file, edit_file, run_shell, ...) over stdio for ONE worktree that the harness
created. The harness owns the worktree and container; this process only uses them.
"""

import argparse
import sys
from pathlib import Path

from ce_agent.mcp.server import McpServer
from ce_agent.mcp.stdio import protect_stdout, serve_stdio
from ce_agent.registry import Registry
from ce_agent.tools import make_workspace_tools
from ce_agent.workspace import Workspace


def build_server(worktree, prefix="", container=None, log=None):
    if not Path(worktree).is_dir():
        raise SystemExit(f"error: worktree {worktree!r} is not a directory")
    ws = Workspace.attach(worktree, prefix, container)
    registry = Registry()
    for func, description in make_workspace_tools(ws):
        registry.register(func, description)
    return McpServer(registry, log=log)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m ce_agent.mcp", description=__doc__)
    parser.add_argument("--worktree", required=True, help="the git worktree the agent works in")
    parser.add_argument("--prefix", default="", help="sub-folder inside the worktree that is the project root")
    parser.add_argument("--container", help="Docker container to run commands in (default: run them on the host)")
    args = parser.parse_args(argv)

    def log(message):                       # logs go to stderr, never to stdout
        print(f"[mcp-server] {message}", file=sys.stderr, flush=True)

    server = build_server(args.worktree, args.prefix, args.container, log)
    serve_stdio(server, sys.stdin, protect_stdout())


if __name__ == "__main__":
    main()
