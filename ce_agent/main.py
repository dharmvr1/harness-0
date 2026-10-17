"""
The agent loop: think (LLM) -> act (tool) -> observe (result) -> repeat.
"""

import json
import sys

from ce_agent.workspace import Workspace, recover
from ce_agent.registry import Registry
from ce_agent.tools import make_tools
from ce_agent.brain import Brain, SYSTEM_PROMPT

# ----------------------
# MAIN LOOP
# ----------------------
MAX_TURNS = 20           # hard stop so the agent can never loop forever
MAX_OUTPUT_CHARS = 4000  # cut long tool output so it does not flood the context
DIFF_PREVIEW_CHARS = 3000

def shorten(text, limit=MAX_OUTPUT_CHARS):
    """Cut the middle, not the end: the useful part of an error is usually the last lines."""
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + f"\n... [{len(text) - limit} characters cut] ...\n" + text[-half:]

def run_agent(repo_path: str, task: str, unsafe: bool = False, network: bool = False, brain=None):
    brain = brain or Brain()   # first: a missing API key should fail before anything is created
    if unsafe:
        print("!!! UNSAFE MODE: commands run directly on your machine, with no container. !!!")
    ws = Workspace(repo_path, unsafe=unsafe, network=network)
    ws.start()                 # cleans up after itself if it fails

    try:
        # Register tools (schema is auto-generated from type hints)
        registry = Registry()
        for func, description in make_tools(ws):
            registry.register(func, description)
        tools = registry.get_openai_tools()

        print(f"--- Task: {task} (model: {getattr(brain, 'model', '?')}) ---")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ]

        # If the loop is interrupted or the API fails, we still go on to the review step,
        # so whatever the agent managed to do is never lost.
        try:
            for turn in range(MAX_TURNS):
                # THINK: ask the model what to do next
                reply = brain.think(messages, tools)
                messages.append(reply)

                # No tool calls means the model thinks it is done
                if not reply.tool_calls:
                    print(f"\n[Done] {reply.content}")
                    break

                # ACT + OBSERVE: run each tool the model asked for
                for call in reply.tool_calls:
                    name = call.function.name
                    try:
                        args = json.loads(call.function.arguments)
                    except json.JSONDecodeError:
                        result = {"status": "error", "error": "Arguments were not valid JSON"}
                    else:
                        print(f"\n[Turn {turn}] {name}({args})")
                        result = registry.call(name, args)

                    # RECOVER: errors go back to the model as text so it can fix itself
                    if result["status"] == "ok":
                        text = result["content"]
                    else:
                        text = "ERROR: " + result["error"]
                    print(text[:300])

                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": shorten(text),
                    })
            else:
                print(f"\n[Stopped] hit the {MAX_TURNS}-turn limit")
        except KeyboardInterrupt:
            print("\n[Interrupted] stopping the agent, moving on to review.")
        except Exception as e:
            print(f"\n[Error] {type(e).__name__}: {e}\nMoving on to review of what was done so far.")

        review(ws)
    finally:
        ws.cleanup()  # removes container + worktree; the branch stays if it holds unreviewed work
        if ws.keep_branch:
            print(f"\nYour work is saved on branch {ws.branch}")
    print("\n--- Loop finished ---")

def review(ws):
    """Commit the agent's work on its own branch, show it, and let the human decide."""
    diff = ws.finish()
    if not diff:
        print("\nNo changes made.")
        ws.discard()
        return

    print("\n--- Changes made by the agent ---")
    print(ws.stat())
    print(diff[:DIFF_PREVIEW_CHARS])
    if len(diff) > DIFF_PREVIEW_CHARS:
        print(f"... showing {DIFF_PREVIEW_CHARS} of {len(diff)} characters. Type v to see everything.")

    while True:
        try:
            choice = input(f"\n[a]ccept (merge into your repo) / [r]eject / [k]eep branch {ws.branch} / [v]iew full diff: ")
        except (KeyboardInterrupt, EOFError):
            choice = "k"   # can't ask, so don't lose anything
        choice = choice.strip().lower()
        if choice == "v":
            print(diff)
        elif choice in ("a", "r", "k"):
            break

    if choice == "a":
        print(ws.accept())
    elif choice == "k":
        print(ws.keep())
    else:
        ws.discard()
        print("Rejected. Your repo was not touched.")

def main():
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--cleanup" in flags and len(args) == 1:
        print(recover(args[0]))
    elif len(args) == 2:
        run_agent(args[0], args[1], unsafe="--unsafe-local" in flags, network="--network" in flags)
    else:
        print("Usage:\n"
              "  python -m ce_agent.main <repo_path> <task> [--unsafe-local] [--network]\n"
              "  python -m ce_agent.main <repo_path> --cleanup    (after a crash; run only when no agent is running)")

if __name__ == "__main__":
    main()
