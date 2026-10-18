"""
Entry point: starts a session and runs the prompt loop.

    python -m ce_agent.main <repo> ["<first request>"] [--once]

The conversation (session.py) and the sandbox (workspace.py) stay alive across your prompts.
Ctrl-D at the prompt ends the session and lets you review everything the agent changed.
"""

import sys

from ce_agent.workspace import Workspace, ensure_repo, recover
from ce_agent.brain import Brain
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
):
    brain = (
        brain or Brain()
    )  # first: a missing API key should fail before anything is created
    if unsafe:
        print(
            "!!! UNSAFE MODE: commands run directly on your machine, with no container. !!!"
        )
    ensure_repo(repo_path)  # fresh folder? offer to set it up (asks first)
    ws = Workspace(repo_path, unsafe=unsafe, network=network)
    ws.start()  # cleans up after itself if it fails
    trace = Trace(ws.top / ".agent" / "runs" / f"{ws.task_id}.jsonl")

    session, reason, decision, files_changed = None, "unknown", None, ""
    mcp_client = None
    try:
        notes = (
            ws.read_notes()
        )  # memory from earlier sessions: AGENTS.md in the project
        if mcp == "stdio":
            # The workspace tools run in a separate MCP server process; every message is logged in the trace.
            mcp_client = launch_workspace_server(
                ws,
                on_wire=lambda direction, message: trace.log(
                    "mcp_wire", direction=direction, message=clip_strings(message)
                ),
            )
            info = mcp_client.server_info or {}
            trace.log(
                "mcp_connected",
                server=f"{info.get('name', '?')} {info.get('version', '')}".strip(),
                version=mcp_client.version,
            )
        session = Session(ws, brain, trace, notes, mcp_client=mcp_client)
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
        print(f"Trajectory: python -m ce_agent.main {repo_path} --show {ws.task_id}")
    print("\n--- Session finished ---")


def run_agent(repo_path, task, unsafe=False, network=False, brain=None, mcp=None):
    """One request, then review (no follow-up prompts)."""
    run_session(
        repo_path,
        task,
        unsafe=unsafe,
        network=network,
        brain=brain,
        interactive=False,
        mcp=mcp,
    )


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
    elif len(args) in (1, 2):
        try:
            run_session(
                args[0],
                args[1] if len(args) == 2 else None,
                unsafe="--unsafe-local" in flags,
                network="--network" in flags,
                interactive="--once" not in flags,
                mcp="stdio" if "--mcp" in flags else None,
            )
        except (
            RuntimeError
        ) as e:  # start-up problems (no API key, no Docker, dirty repo...) are messages, not crashes
            print(f"Error: {e}")
    else:
        print(
            "Usage:\n"
            '  python -m ce_agent.main <repo_path> ["first request"] [--once] [--unsafe-local] [--network] [--mcp]\n'
            "       starts a session; keep typing requests, Ctrl-D to finish and review. --once: do the request and stop\n"
            "       --mcp: the workspace tools run in a separate MCP server process (stdio) instead of in-process\n"
            "  python -m ce_agent.main <repo_path> --show [task_id]   (what did the agent do? default: latest run)\n"
            "  python -m ce_agent.main <repo_path> --cleanup          (after a crash; run only when no agent is running)"
        )


if __name__ == "__main__":
    main()
