"""
Entry point: starts a session and runs the prompt loop.

    python -m ce_agent.main <repo> ["<first request>"] [--once]

The conversation (session.py) and the sandbox (workspace.py) stay alive across your prompts.
Ctrl-D at the prompt ends the session and lets you review everything the agent changed.
"""

import dataclasses
import subprocess
import sys
import time

from ce_agent.egress import summarize
from ce_agent.environment import flags_to_overrides, resolve_environment
from ce_agent.workspace import Workspace, ensure_repo, recover
from ce_agent.brain import Brain
from ce_agent.hooks import Hooks
from ce_agent.session import Session
from ce_agent.trace import Trace, clip_strings, show, find_run
from ce_agent.mcp.client import launch_workspace_server

DIFF_PREVIEW_CHARS = 3000


def run_session(
    repo_path,
    first_prompt=None,
    unsafe=False,
    network=False,
    brain=None,
    interactive=True,
    mcp=None,
    environment=None,
    env_overrides=None,
):
    brain = (
        brain or Brain()
    )  # first: a missing API key should fail before anything is created
    if unsafe:
        print(
            "!!! UNSAFE MODE: commands run directly on your machine, with no container. !!!"
        )
    ensure_repo(repo_path)  # fresh folder? offer to set it up (asks first)

    # Which image, how much network, which limits: detected from the project, then .agent/env.json, then flags.
    overrides = dict(env_overrides or {})
    if network and "network" not in overrides:  # the older boolean argument
        overrides["network"] = "full"
    environment = environment or resolve_environment(
        repo_path, overrides, top=repo_top(repo_path)
    )
    print(environment.describe(unsafe=unsafe))
    if environment.network == "full" and not unsafe:
        print(
            "!!! FULL NETWORK: anything running in the sandbox can reach the whole internet, "
            "including sending your project files out. Prefer --network=registries. !!!"
        )

    hooks = Hooks.load(repo_top(repo_path))   # a broken hooks.json should stop us before anything is created
    ws = Workspace(repo_path, unsafe=unsafe, environment=environment)
    ws.start()  # cleans up after itself if it fails
    trace = Trace(ws.top / ".agent" / "runs" / f"{ws.task_id}.jsonl")

    session, reason, decision, files_changed = None, "unknown", None, ""
    mcp_client = None
    try:
        notes = (
            ws.read_notes()
        )  # memory from earlier sessions: AGENTS.md in the project
        if mcp in ("stdio", "http"):
            # The workspace tools run in a separate MCP server process; every message is logged in the trace.
            mcp_client = launch_workspace_server(
                ws,
                transport=mcp,
                on_wire=lambda direction, message: trace.log(
                    "mcp_wire", direction=direction, message=clip_strings(message)
                ),
            )
            last_shown = [0.0]

            def show_progress(params):  # the server reports every second; showing every 5th is plenty
                if time.monotonic() - last_shown[0] >= 5:
                    last_shown[0] = time.monotonic()
                    print(f"  [progress] {params.get('message') or params.get('progress')}")

            mcp_client.on_progress = show_progress
            info = mcp_client.server_info or {}
            trace.log(
                "mcp_connected",
                server=f"{info.get('name', '?')} {info.get('version', '')}".strip(),
                version=mcp_client.version,
            )
        session = Session(
            ws,
            brain,
            trace,
            notes,
            mcp_client=mcp_client,
            environment_text=environment.prompt_text(unsafe=unsafe),
            hooks=hooks,
        )
        trace.log(
            "run_start",
            task_id=ws.task_id,
            task=first_prompt or "(interactive session)",
            model=getattr(brain, "model", "?"),
            mode="unsafe" if unsafe else "docker",
            repo=str(ws.repo_path),
            base=ws.base,
            notes_chars=len(notes),
            mcp=mcp,
            environment=dataclasses.asdict(environment),
            environment_text=environment.describe(unsafe=unsafe),
        )
        print(
            f"--- Session {ws.task_id} (model: {getattr(brain, 'model', '?')}, branch: {ws.branch}) ---"
        )
        if interactive:
            print(
                "Type a request or a question. Ctrl-D (or Ctrl-C at the prompt) ends the session and lets you review the changes."
            )

        prompt = first_prompt
        while True:
            if prompt is None:
                if not interactive:
                    break
                try:
                    prompt = input("\n> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if not prompt:
                    prompt = None
                    continue
            reason = session.send(prompt)
            prompt = None
            if not interactive:
                break
            if session.last_prompt_changed:
                review_pending(ws, session, trace)

        decision, files_changed = review(ws)
    finally:
        if mcp_client:
            mcp_client.close()  # stop the tool server before its worktree and container are removed
        report_network(ws, trace)  # while the proxy still exists
        trace.log(
            "run_end",
            reason=reason,
            decision=decision,
            turns=session.model_calls if session else 0,
            files_changed=files_changed,
            total_prompt_tokens=session.total_in if session else 0,
            total_completion_tokens=session.total_out if session else 0,
        )
        trace.close()
        ws.cleanup()  # removes container + worktree; the branch stays if it holds unreviewed work
        if ws.keep_branch:
            print(f"\nYour work is saved on branch {ws.branch}")
        print(f"Trajectory: ce-agent {repo_path} --show {ws.task_id}")
    print("\n--- Session finished ---")


def run_agent(
    repo_path,
    task,
    unsafe=False,
    network=False,
    brain=None,
    mcp=None,
    environment=None,
    env_overrides=None,
):
    """One request, then review (no follow-up prompts)."""
    run_session(
        repo_path,
        task,
        unsafe=unsafe,
        network=network,
        brain=brain,
        interactive=False,
        mcp=mcp,
        environment=environment,
        env_overrides=env_overrides,
    )


def repo_top(path):
    """The root of the git repository that contains `path`, or None."""
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=path,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def report_network(ws, trace):
    """Record and print what the egress proxy allowed and refused (network mode "registries")."""
    try:
        entries = ws.egress_entries()
    except Exception:
        return
    allowed, blocked = summarize(entries)
    if not allowed and not blocked:
        return
    trace.log("egress", allowed=allowed, blocked=blocked, entries=entries[-300:])
    if allowed:
        print("\n[network] allowed: " + ", ".join(f"{h} x{n}" for h, n in allowed.items()))
    for note in blocked:
        print(f"[network] BLOCKED {note}")


def show_environment(repo_path, overrides):
    """`--env`: print the sandbox settings that would be used, without running anything."""
    environment = resolve_environment(repo_path, overrides, top=repo_top(repo_path))
    print(environment.describe())
    print("Image:", environment.image, "| project type:", environment.preset)
    print("Told to the model:", environment.prompt_text())
    if environment.network == "registries":
        print("Allowed hosts:", ", ".join(environment.hosts()))


def show_pending(ws):
    """Print what the agent changed that your repo doesn't have yet. Returns the full diff ('' if none)."""
    diff = ws.finish()
    if diff:
        print("\n--- Pending changes (the agent has these, your folder doesn't) ---")
        print(ws.stat())
        print(diff[:DIFF_PREVIEW_CHARS])
        if len(diff) > DIFF_PREVIEW_CHARS:
            print(f"... showing {DIFF_PREVIEW_CHARS} of {len(diff)} characters. Type v to see everything.")
    return diff


def review_pending(ws, session, trace):
    """After a request that changed files: merge now, reject, or leave it pending."""
    diff = show_pending(ws)
    if not diff:
        return
    while True:
        try:
            choice = input("\n[a]ccept (apply to your folder now) / [r]eject (undo these changes) / "
                           "[l]ater (keep working) / [v]iew full diff: ")
        except (KeyboardInterrupt, EOFError):
            choice = "l"   # can't ask, so change nothing
        choice = choice.strip().lower() or "l"
        if choice == "v":
            print(diff)
        elif choice in ("a", "r", "l"):
            break

    if choice == "a":
        message = ws.accept()
        print(message)
        decision = "accept" if message.startswith("Applied") else "accept_failed"
    elif choice == "r":
        ws.reject_pending()
        session.note_rejected()
        print("Rejected. The workspace is back to what your repo has.")
        decision = "reject"
    else:
        print("Left in the agent's copy. It stays pending until you accept it or end the session.")
        decision = "later"
    trace.log("review", prompt_no=session.prompt_no, decision=decision)


def review(ws):
    """Commit the agent's work on its own branch, show it, and let the human decide.
    Returns (decision, list of changed files)."""
    diff = ws.finish()
    if not diff:
        print("\nNo changes made.")
        ws.discard()
        return "no_changes", ""

    stat = ws.stat()
    print("\n--- Changes made by the agent ---")
    print(stat)
    print(diff[:DIFF_PREVIEW_CHARS])
    if len(diff) > DIFF_PREVIEW_CHARS:
        print(
            f"... showing {DIFF_PREVIEW_CHARS} of {len(diff)} characters. Type v to see everything."
        )

    while True:
        try:
            choice = input(
                f"\n[a]ccept (apply to your folder) / [r]eject / [k]eep branch {ws.branch} / [v]iew full diff: "
            )
        except (KeyboardInterrupt, EOFError):
            choice = "k"  # can't ask, so don't lose anything
        choice = choice.strip().lower()
        if choice == "v":
            print(diff)
        elif choice in ("a", "r", "k"):
            break

    if choice == "a":
        message = ws.accept()
        print(message)
        return ("accept" if message.startswith("Applied") else "accept_failed"), stat
    if choice == "k":
        print(ws.keep())
        return "keep", stat
    ws.discard()
    print("Rejected. Your repo was not touched.")
    return "reject", stat


def main():
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--cleanup" in flags and len(args) == 1:
        try:
            print(recover(args[0]))
        except RuntimeError as e:
            print(f"Error: {e}")
    elif "--show" in flags and len(args) in (1, 2):
        try:
            print(show(find_run(args[0], args[1] if len(args) == 2 else None)))
        except FileNotFoundError as e:
            print(e)
    elif "--env" in flags and len(args) == 1:
        try:
            show_environment(args[0], flags_to_overrides(flags))
        except RuntimeError as e:
            print(f"Error: {e}")
    elif len(args) in (1, 2):
        mcp_flags = [f for f in flags if f == "--mcp" or f.startswith("--mcp=")]
        mcp_choice = None
        if mcp_flags:
            mcp_choice = mcp_flags[0].partition("=")[2] or "stdio"
            if mcp_choice not in ("stdio", "http"):
                print(f"Error: --mcp must be stdio or http, not {mcp_choice!r}")
                return
        try:
            run_session(
                args[0],
                args[1] if len(args) == 2 else None,
                unsafe="--unsafe-local" in flags,
                interactive="--once" not in flags,
                mcp=mcp_choice,
                env_overrides=flags_to_overrides(flags),
            )
        except (
            RuntimeError
        ) as e:  # start-up problems (no API key, no Docker, dirty repo...) are messages, not crashes
            print(f"Error: {e}")
    else:
        print(
            "Usage:\n"
            '  python -m ce_agent.main <repo_path> ["first request"] [--once] [--unsafe-local] [--mcp[=http]]\n'
            "       starts a session; keep typing requests, Ctrl-D to finish and review. --once: do the request and stop\n"
            "       --mcp[=stdio|http]: the workspace tools run in a separate MCP server process instead of in-process\n"
            "       sandbox: --network[=none|registries|full]  --image=IMG  --memory=4g  --timeout=SECONDS  --pids=N\n"
            "                --allow-host=HOST (repeatable)  --preset=rust|node|go|python  --no-cache\n"
            "                (defaults come from the project's files and <repo>/.agent/env.json)\n"
            "  python -m ce_agent.main <repo_path> --env              (show the sandbox settings that would be used)\n"
            "  python -m ce_agent.main <repo_path> --show [task_id]   (what did the agent do? default: latest run)\n"
            "  python -m ce_agent.main <repo_path> --cleanup          (after a crash; run only when no agent is running)"
        )


if __name__ == "__main__":
    main()
