"""
The "raw" version of the agent.
Everything is in this one file to make it easy to trace.
"""

import sys
from pathlib import Path

# ----------------------
# 1️⃣  IMPORTS
# ----------------------
from ce_agent.sandbox_factory import sandbox_factory  # factory pattern


# ----------------------
# 2️⃣  CONFIGURATION
# ----------------------
def get_config():
    return {
        "sandbox_root": Path(".agent/sandbox"),
        "state_file": ".agent/state.json",
        "turn_limit": 50,
    }


# ----------------------
# 3️⃣  TOOLS (Our tool surface)
# ----------------------
def tool_read_file(sandbox, file_path):
    """Read a file using the sandbox's read_file method."""
    return sandbox.read_file(file_path)


def tool_list_files(sandbox):
    """List files using the sandbox's list_files method."""
    return sandbox.list_files()


def tool_run_shell(sandbox, command):
    """Run a shell command using the sandbox's run method."""
    return sandbox.run(command)


# ----------------------
# 4️⃣  DISPATCHER
# ----------------------
def dispatch(sandbox, command_str):
    """Mini-MCP dispatcher: routes commands to the appropriate tool.

    Args:
        sandbox: the sandbox object (has .read_file(), .list_files(), .run() methods)
        command_str: the command string (e.g., "read file", "ls", "shell cmd")
    """
    # Returns a dict: {"status": "ok"|"error", "content": str, "error": str|None}

    if command_str.startswith("read "):
        file_path = command_str.replace("read ", "").strip()
        content = tool_read_file(sandbox, file_path)
        if content.startswith("Error:"):
            return {"status": "error", "content": content}
        return {"status": "ok", "content": content}
    elif command_str == "ls":
        content = tool_list_files(sandbox)
        return {"status": "ok", "content": content}
    elif command_str.startswith("shell "):
        cmd = command_str.replace("shell ", "").strip()
        # Use the sandbox object's run method (polymorphic - works for Docker or local)
        result = tool_run_shell(sandbox, cmd)
        # Normalise the result dict to our expected shape
        return {
            "status": "ok" if result.get("status") == "ok" else "error",
            "content": result.get("content", ""),
            "error": result.get("error", ""),
        }
    else:
        # Fallback: run the raw shell command via sandbox
        result = tool_run_shell(sandbox, command_str)
        return {
            "status": "ok" if result.get("status") == "ok" else "error",
            "content": result.get("content", ""),
            "error": result.get("error", ""),
        }


# ----------------------
# 5️⃣  THE LOOP (Plan, Act, Observe, Recover)
# ----------------------
def run_agent(repo_path: str, task: str):
    # ---- Initialise sandbox via factory ----
    sandbox = sandbox_factory()  # auto-detects Docker vs local
    sandbox_dir = sandbox.setup(
        repo_path, "task_1"
    )  # prepares isolated dir, returns Path string (may be unused for some backends)

    print(f"--- Starting loop for task: {task} ---")

    # --- THE PLAN STATE ---
    # In Phase 3, this is rewritten each turn.
    # Currently we initialize it once, but the loop body decides next steps.
    plan = {
        "task": task,
        "steps": ["ls", "read Readme.md"],  # Initial plan
        "done": [],
        "summary": "",  # Will be built up each turn
    }

    cfg = get_config()
    turn_limit = cfg["turn_limit"] if cfg else 50

    for turn in range(turn_limit):
        if not plan["steps"]:
            print("\n[Plan] No more steps. Task appears complete.")
            break

        # --- PLAN REWRITING (Phase 3) ---
        # In a real system, the model would rewrite the full `plan` dict each turn.
        # Here we simulate it: after acting, we re-evaluate what's left.
        remaining = [s for s in plan["steps"] if s not in plan["done"]]

        if not remaining:
            print("\n[Plan] All steps completed.")
            break

        # For the first turn, pick the first step.
        # Later, this would be determined by reading the model's rewritten plan.
        current_step = remaining[0] if remaining else None

        if current_step is None:
            break

        print(f"\n--- Turn {turn} ---")
        print(f"[Plan] Current step: {current_step}")
        print(f"[Plan] Summary so far: {plan['summary'][:100]}...")

        # --- ACT & OBSERVE ---
        result = dispatch(sandbox, current_step)  # <-- uses sandbox methods internally

        # --- RECOVER (Error Handling) ---
        if result["status"] == "error":
            error_msg = result.get("error", result.get("content", "Unknown error"))
            print(f"[ERROR] Tool failed: {error_msg[:200]}")
            # Record in plan that this step failed/fixed
            plan["done"].append(current_step)
            # Rewrite the plan: remove this step, maybe add a retry or skip
            plan["steps"] = [s for s in plan["steps"] if s != current_step]
            plan[
                "summary"
            ] += f"\nTurn {turn}: Error on '{current_step}' — removed from plan."
            # Continue to next turn; the loop will pick the next remaining step
            continue

        # --- OBSERVE ---
        output = result["content"][:300]  # Truncate for display
        print(f"[Output] {output}")

        # Update plan state
        plan["done"].append(current_step)
        plan[
            "summary"
        ] += f"\nTurn {turn}: completed '{current_step}'. Output: {output[:50]}..."

        # --- PLAN REWRITING DECISION ---
        # In a real model-driven loop, the model would look at the observation
        # and decide the next step. Here we just have simple logic:
        # If we just read the Readme, maybe we're done? Or maybe we should list more files?
        if "Readme.md" in current_step and turn >= 1:
            # After reading the readme, the "plan" might shift to exploring more
            if "ls" in plan["steps"] and "ls" not in plan["done"]:
                print("[Plan] Noticing we haven't done a full directory listing yet...")
            else:
                print("[Plan] Shifting: Readme read, now exploring with 'ls -R'...")
                # Add a new step to the plan dynamically
                plan["steps"].append("ls -R")

    print("\n--- Loop finished ---")

    # ---- Teardown ----
    sandbox.teardown()
    print("Sandbox cleaned up.")


# ----------------------
# 6️⃣  CLI
# ----------------------
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python main.py <repo_path> <task>")
    else:
        run_agent(sys.argv[1], sys.argv[2])
