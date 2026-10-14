"""CLI entry point: ``ce-agent run <repo> "<task>"``.

Phases build on top of this. Right now it loads config and reports what it
found — a smoke test that the package boots and can see the environment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .types import LoopStatus


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ce-agent",
        description="Terminal-native coding agent (capstone build).",
    )
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run the agent loop against a repo + task.")
    run.add_argument("repo", help="Path to the target repository.")
    run.add_argument("task", help="Natural-language task description.")
    run.add_argument("--workdir", default=None, help="Working directory (default: cwd).")

    sub.add_parser("version", help="Print version and exit.")

    return p


def cmd_version() -> int:
    from . import __version__

    print(f"ce-agent {__version__}")
    return 0


def cmd_run(repo: str, task: str, workdir: str | None) -> int:
    from .loop.engine import AgentEngine

    cfg = load_config(Path(workdir) if workdir else None)
    engine = AgentEngine(cfg)
    status: LoopStatus = engine.run(repo_path=repo, task=task)
    print(f"\n[cli] loop finished: {status.value}")
    return 0 if status == LoopStatus.SUCCESS else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "version":
        return cmd_version()
    if args.command == "run":
        return cmd_run(args.repo, args.task, args.workdir)
    build_parser().print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())