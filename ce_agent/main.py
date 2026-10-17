"""
The agent loop: think (LLM) -> act (tool) -> observe (result) -> repeat.
"""

import json
import sys
from typing import Annotated

from ce_agent.workspace import Workspace
from ce_agent.registry import Registry
from ce_agent.brain import Brain, SYSTEM_PROMPT

# ----------------------
# 1️⃣ TOOLS WITH TYPE HINTS & ANNOTATIONS
# ----------------------
def make_tools(ws):
    """Build the tools, bound to one workspace. Errors are raised and the registry reports them."""

    def read_file(file_path: Annotated[str, "Path of the file to read, relative to the workspace"]):
        return ws.read_file(file_path)

    def write_file(file_path: Annotated[str, "Path of the file to create or overwrite"],
                   content: Annotated[str, "Full new content of the file"]):
        return ws.write_file(file_path, content)

    def list_files():
        return ws.list_files()

    def run_shell(command: Annotated[str, "Shell command to run in the workspace"]):
        r = ws.run(command)
        # Give the model everything: exit code and both output streams
        return f"exit code: {r['exit_code']}\n--- stdout ---\n{r['stdout']}\n--- stderr ---\n{r['stderr']}"

    return read_file, write_file, list_files, run_shell

# ----------------------
# 2️⃣ MAIN LOOP
# ----------------------
MAX_TURNS = 20          # hard stop so the agent can never loop forever
MAX_OUTPUT_CHARS = 4000  # cut long tool output so it does not flood the context

def run_agent(repo_path: str, task: str, unsafe: bool = False, network: bool = False):
    if unsafe:
        print("!!! UNSAFE MODE: commands run directly on your machine, with no container. !!!")
    ws = Workspace(repo_path, unsafe=unsafe, network=network)
    ws.start()
    brain = Brain()

    # Register tools (schema is auto-generated from type hints)
    registry = Registry()
    read_file, write_file, list_files, run_shell = make_tools(ws)
    registry.register(read_file, "Read a file from the workspace")
    registry.register(write_file, "Create or overwrite a file in the workspace")
    registry.register(list_files, "List all files in the workspace workspace")
    registry.register(run_shell, "Run a shell command in the workspace")
    tools = registry.get_openai_tools()

    print(f"--- Task: {task} (model: {brain.model}) ---")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]

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
                    "content": text[:MAX_OUTPUT_CHARS],
                })
        else:
            print(f"\n[Stopped] hit the {MAX_TURNS}-turn limit")

        # Commit the agent's work on its own branch and let the human decide
        diff = ws.finish()
        print("\n--- Changes made by the agent ---")
        print(diff[:3000] if diff else "No changes made")
        if diff:
            choice = input(f"\n[a]ccept (merge into your repo) / [r]eject / [k]eep branch {ws.branch}: ").lower()
            if choice == "a":
                print(ws.accept())
            elif choice == "k":
                print(ws.keep())
            else:
                print("Rejected. Your repo was not touched.")
    finally:
        ws.cleanup()  # always remove container + worktree, even on errors
    print("\n--- Loop finished ---")

if __name__ == "__main__":
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 2:
        print("Usage: python -m ce_agent.main <repo_path> <task> [--unsafe-local] [--network]")
    else:
        run_agent(args[0], args[1], unsafe="--unsafe-local" in flags, network="--network" in flags)
