"""
Trajectory log: one JSON line per event, written straight to disk so it survives a crash.

Where:  <repo>/.agent/runs/<task_id>.jsonl   (outside the worktree, so it outlives the task)
Events: run_start, model_call, tool_call, run_end
Read it with:  python -m ce_agent.main <repo> --show [task_id]
"""

import json
import subprocess
import time
from pathlib import Path


class Trace:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.path, "a")

    def log(self, kind, **data):
        event = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": kind, **data}
        self.file.write(json.dumps(event, default=str) + "\n")
        self.file.flush()   # a crash right after this line must not lose it

    def close(self):
        self.file.close()


def clip_strings(value, limit=500):
    """A copy of a JSON value with long strings cut short, so logging every MCP message stays small."""
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + f"...[{len(value) - limit} more characters]"
    if isinstance(value, list):
        return [clip_strings(v, limit) for v in value]
    if isinstance(value, dict):
        return {k: clip_strings(v, limit) for k, v in value.items()}
    return value


# ----------------------------------------------------------------------
# reading logs back
# ----------------------------------------------------------------------
def runs_dir(repo_path):
    r = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=repo_path, capture_output=True, text=True)
    if r.returncode != 0:
        raise FileNotFoundError(f"{repo_path} is not a git repo, so there are no runs to show.")
    return Path(r.stdout.strip()) / ".agent" / "runs"


def find_run(repo_path, task_id=None):
    """Path of the run with this id, or the most recent run if no id is given."""
    folder = runs_dir(repo_path)
    if task_id:
        path = folder / f"{task_id}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"No run {task_id}. Runs: {', '.join(p.stem for p in folder.glob('*.jsonl'))}")
        return path
    runs = sorted(folder.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise FileNotFoundError(f"No runs recorded yet in {folder}")
    return runs[-1]


def one_line(text, limit=150):
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + "..."


def show(path):
    """Turn a trace file into a readable timeline."""
    out = []
    for line in Path(path).read_text().splitlines():
        e = json.loads(line)
        kind = e["kind"]
        if kind == "run_start":
            out.append(f"run {e['task_id']}  {e['time']}  model={e['model']}  mode={e['mode']}")
            out.append(f"task: {e['task']}")
            if e.get("notes_chars"):
                out.append(f"(loaded {e['notes_chars']} characters of project notes)")
        elif kind == "user_message":
            out.append(f"\n> you: {one_line(e['text'], 200)}")
        elif kind == "plan_update":
            out.append("    plan:")
            out.extend("        " + line for line in e["plan"].splitlines())
        elif kind == "context_cleared":
            out.append(f"    (cleared {e['cleared']} old tool outputs: ~{e['tokens_before']} -> ~{e['tokens_after']} tokens)")
        elif kind == "review":
            out.append(f"    you reviewed the changes: {e['decision']}")
        elif kind == "sync":
            out.append(f"    (synced with your repo: {e['message']})")
            for change in e.get("changes", []):
                out.append(f"        {change.replace(chr(9), ' ')}")
        elif kind == "mcp_connected":
            out.append(f"    (tools are served over MCP by {e['server']}, protocol {e['version']})")
        elif kind == "notes_changed":
            out.append(f"    (project notes changed: now {e['chars']} characters)")
        elif kind == "prompt_end":
            out.append(f"    prompt ended: {e['reason']} after {e['model_calls']} model calls")
        elif kind == "model_call":
            u = e.get("usage") or {}
            tokens = f"{u.get('prompt_tokens', '?')} in / {u.get('completion_tokens', '?')} out"
            out.append(f"\n[turn {e['turn']}] model {e['seconds']}s, tokens {tokens}")
            if e.get("content"):
                out.append(f"    says: {one_line(e['content'])}")
        elif kind == "tool_call":
            out.append(f"    {e['name']}({one_line(json.dumps(e['args']), 100)}) -> {e['status']} ({e['seconds']}s)")
            out.append(f"        {one_line(e['result'])}")
        elif kind == "run_end":
            out.append(f"\nended: {e['reason']}  |  you chose: {e['decision']}  |  turns: {e['turns']}  "
                       f"|  tokens: {e['total_prompt_tokens']} in / {e['total_completion_tokens']} out")
            if e.get("files_changed"):
                out.append(e["files_changed"])
    return "\n".join(out)
