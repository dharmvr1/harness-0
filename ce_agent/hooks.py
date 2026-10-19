"""
Hooks: rules that run BEFORE a tool call and decide: allow, deny or ask the user.

    model asks for run_shell("rm -rf ~")  ->  hook rule matches  ->  deny: the model gets an error and must choose again

Rules come from two places, checked in this order (the first rule that matches wins, no match means allow):
  1. <repo>/.agent/hooks.json   your own rules. It lives in .agent/, which is not part of the agent's copy of the
                                project, so the model cannot change its own rules.
  2. DEFAULT_RULES below        a small guard against the worst shell commands.

hooks.json looks like this:
    {"pre_tool_use": [
        {"tool": "run_shell", "pattern": "git push", "action": "ask",  "message": "pushing publishes your work"},
        {"tool": "write_file", "pattern": "\\.env$",  "action": "deny", "message": "never touch secrets"}
    ]}
  tool     the tool name, or "*" for every tool
  pattern  a regular expression, searched in the tool's main argument (run_shell: command, file tools: file_path,
           search: pattern; for other tools, all arguments as text)
  action   "allow" (stop looking, let it through), "deny" or "ask"

This is a guard against mistakes and obvious bad ideas, NOT a security boundary: a shell command can be written in
endless ways. The real isolation is the container (no network, non-root, only the worktree mounted).
"""

import json
import re
from pathlib import Path

HOOKS_PATH = Path(".agent") / "hooks.json"
ACTIONS = ("allow", "deny", "ask")
MAIN_ARGUMENT = {"run_shell": "command", "read_file": "file_path", "write_file": "file_path",
                 "edit_file": "file_path", "search": "pattern"}

DEFAULT_RULES = [
    {"tool": "run_shell", "action": "deny",
     "pattern": r"\brm\s+(-\w+\s+)*-\w*[rR]\w*\s+(-\w+\s+)*(/\S*|~\S*|\$HOME\S*|\S*\.\.\S*|\*(\s|$))",
     "message": "a recursive delete outside the project (an absolute path, ~, .. or a bare *) is not allowed; delete specific files inside the project instead"},
    {"tool": "run_shell", "action": "deny",
     "pattern": r"\b(curl|wget)\b[^|;&]*\|\s*(sudo\s+)?(ba|z)?sh\b",
     "message": "piping a download into a shell is not allowed; download the file, read it, then run it"},
    {"tool": "run_shell", "action": "deny",
     "pattern": r"\bgit\s+(push|remote\s+(add|set-url))\b",
     "message": "the harness decides what leaves the project; you only change files in the worktree"},
]


class HooksError(Exception):
    pass


class Decision:
    def __init__(self, action="allow", message="", rule=None):
        self.action, self.message, self.rule = action, message, rule


def check_rule(rule, where="rule"):
    if not isinstance(rule, dict):
        raise HooksError(f"{where} must be an object")
    unknown = sorted(set(rule) - {"tool", "pattern", "action", "message"})
    if unknown:
        raise HooksError(f"{where}: unknown key(s) {', '.join(unknown)}")
    for key in ("tool", "pattern", "action"):
        if not isinstance(rule.get(key), str) or not rule[key]:
            raise HooksError(f"{where}: '{key}' is required and must be a string")
    if rule["action"] not in ACTIONS:
        raise HooksError(f"{where}: action must be one of {', '.join(ACTIONS)}, not {rule['action']!r}")
    try:
        re.compile(rule["pattern"])
    except re.error as e:
        raise HooksError(f"{where}: pattern is not a valid regular expression ({e})")
    return rule


class Hooks:
    def __init__(self, rules=None, use_defaults=True):
        own = [check_rule(r, f"rule {i + 1}") for i, r in enumerate(rules or [])]
        self.rules = own + (DEFAULT_RULES if use_defaults else [])

    @classmethod
    def load(cls, folder):
        """Rules from <folder>/.agent/hooks.json (no file means only the defaults)."""
        path = Path(folder) / HOOKS_PATH
        if not path.is_file():
            return cls()
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            raise HooksError(f"cannot read {path}: {e}")
        if not isinstance(data, dict) or set(data) - {"pre_tool_use"}:
            raise HooksError(f"{path} must be an object with only a 'pre_tool_use' list")
        rules = data.get("pre_tool_use", [])
        if not isinstance(rules, list):
            raise HooksError(f"{path}: 'pre_tool_use' must be a list")
        try:
            return cls(rules)
        except HooksError as e:
            raise HooksError(f"{path}: {e}")

    def pre_tool_use(self, name, args):
        """Decide about one tool call. `args` is the dict the model sent."""
        text = main_argument(name, args)
        for rule in self.rules:
            if rule["tool"] not in ("*", name):
                continue
            if re.search(rule["pattern"], text):
                return Decision(rule["action"], rule.get("message", ""), rule)
        return Decision()


def main_argument(name, args):
    if not isinstance(args, dict):
        return str(args)
    key = MAIN_ARGUMENT.get(name)
    if key and isinstance(args.get(key), str):
        return args[key]
    return " ".join(str(v) for v in args.values())
