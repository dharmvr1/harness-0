"""
The "raw" version of the agent.
Everything is in this one file to make it easy to trace.
"""

import subprocess
import shutil
from pathlib import Path

# --- 1. CONFIGURATION ---
def get_config():
    return {
        "sandbox_root": Path(".agent/sandbox"),
        "state_file": ".agent/state.json",
        "turn_limit": 50
    }

# --- 2. SANDBOX (The isolation contract) ---
def create_sandbox(repo_path, task_id):
    print("path from create sandbox : " ,repo_path)
    sandbox_path = Path(f".agent/worktrees/{task_id}")
    sandbox_path.mkdir(parents=True, exist_ok=True)

    # IMPORTANT: Ignore our own control directory to prevent infinite recursion
    def ignore_func(dir, files):
        return [".agent", ".git"]

    shutil.copytree(repo_path, sandbox_path, dirs_exist_ok=True, ignore=ignore_func)
    print("sandbox path :" , sandbox_path)
    return sandbox_path

def run_in_sandbox(sandbox_path, command):
    # This is how we run tools INSIDE the sandbox.
    # cwd=sandbox_path is the key -- the tool cannot "see" outside.
    return subprocess.run(
        command,
        cwd=sandbox_path,
        capture_output=True,
        text=True,
        shell=True
    )

# --- 3. TOOLS (Our tool surface) ---
def tool_read_file(sandbox_path, file_path):
    full_path = sandbox_path / file_path
    # Security check: ensure file is inside sandbox
    try:
        full_path.resolve().relative_to(sandbox_path.resolve())
    except ValueError:
        return "Error: Access denied (file outside sandbox)"

    if not full_path.exists():
        return f"Error: File {file_path} not found"
    return full_path.read_text()

def tool_list_files(sandbox_path):
    return "\n".join([str(p.relative_to(sandbox_path)) for p in sandbox_path.rglob("*")])

def tool_run_shell(sandbox_path, command):
    """A dedicated tool for running shell commands with capture."""
    return run_in_sandbox(sandbox_path, command)

# --- 4. DISPATCHER ---
def dispatch(sandbox_path, command_str):
    # This acts as our mini-MCP dispatcher
    # Returns a dict: {"status": "ok"|"error", "content": "..."}

    if command_str.startswith("read "):
        file_path = command_str.replace("read ", "").strip()
        content = tool_read_file(sandbox_path, file_path)
        if content.startswith("Error:"):
            return {"status": "error", "content": content}
        return {"status": "ok", "content": content}
    elif command_str == "ls":
        content = tool_list_files(sandbox_path)
        return {"status": "ok", "content": content}
    elif command_str.startswith("shell "):
        cmd = command_str.replace("shell ", "").strip()
        result = tool_run_shell(sandbox_path, cmd)
        return {"status": "ok" if result.returncode == 0 else "error",
                "content": result.stdout, "error": result.stderr}
    else:
        # Fallback: run the raw shell command
        result = run_in_sandbox(sandbox_path, command_str)
        return {"status": "ok" if result.returncode == 0 else "error",
                "content": result.stdout, "error": result.stderr}

# --- 5. THE LOOP (Plan, Act, Observe, Recover) ---
def run_agent(repo_path, task):
    config = get_config()
    sandbox = create_sandbox(repo_path, "task_1")

    print(f"--- Starting loop for task: {task} ---")

    # --- THE PLAN STATE ---
    # In Phase 3, this is rewritten each turn.
    # Currently we initialize it once, but the loop body decides next steps.
    plan = {
        "task": task,
        "steps": ["ls", "read Readme.md"],  # Initial plan
        "done": [],
        "summary": ""  # Will be built up each turn
    }

    for turn in range(config["turn_limit"]):
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
        result = dispatch(sandbox, current_step)

        # --- RECOVER (Error Handling) ---
        if result["status"] == "error":
            error_msg = result.get("error", result.get("content", "Unknown error"))
            print(f"[ERROR] Tool failed: {error_msg[:200]}")
            # Record in plan that this step failed/fixed
            plan["done"].append(current_step)
            # Rewrite the plan: remove this step, maybe add a retry or skip
            plan["steps"] = [s for s in plan["steps"] if s != current_step]
            plan["summary"] += f"\nTurn {turn}: Error on '{current_step}' — removed from plan."
            # Continue to next turn; the loop will pick the next remaining step
            continue

        # --- OBSERVE ---
        output = result["content"][:300]  # Truncate for display
        print(f"[Output] {output}")

        # Update plan state
        plan["done"].append(current_step)
        plan["summary"] += f"\nTurn {turn}: completed '{current_step}'. Output: {output[:50]}..."

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

    # Cleanup
    shutil.rmtree(sandbox, ignore_errors=True)
    print("Sandbox cleaned up.")

# --- 6. CLI ---
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python main.py <repo_path> <task>")
    else:
        run_agent(sys.argv[1], sys.argv[2])