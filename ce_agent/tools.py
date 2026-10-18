"""
The tools the model can call. Each is a plain function with type hints;
the Registry turns them into schemas for the LLM.
The real work is done by the Workspace, so these stay thin.

Two groups:
  - workspace tools: need only the files on disk. These can be served over MCP (ce_agent/mcp/).
  - local tools: depend on the harness's own state (the plan, the review baseline), so they always run in-process.
"""

from typing import Annotated


def make_workspace_tools(ws):
    """Tools that only need the workspace's files. Returns (function, description) pairs.
    Errors are raised; the registry (or the MCP server) turns them into text for the model."""

    def list_files():
        return ws.list_files()

    def search(
        pattern: Annotated[str, "Regular expression to search for in file contents"],
        path: Annotated[
            str, "Optional file or folder to search in (default: everywhere)"
        ] = "",
    ):
        return ws.search(pattern, path)

    def read_file(
        file_path: Annotated[
            str, "Path of the file to read, relative to the workspace"
        ],
        start_line: Annotated[int, "First line to read (default 1)"] = 1,
        end_line: Annotated[int, "Last line to read (default: end of file)"] = None,
    ):
        return ws.read_file(file_path, start_line, end_line)

    def edit_file(
        file_path: Annotated[str, "Path of the existing file to edit"],
        old_text: Annotated[
            str, "Exact text to replace. Must appear exactly once in the file"
        ],
        new_text: Annotated[str, "Text to put in its place"],
    ):
        return ws.edit_file(file_path, old_text, new_text)

    def write_file(
        file_path: Annotated[str, "Path of the file to create"],
        content: Annotated[str, "Full content of the file"],
    ):
        return ws.write_file(file_path, content)

    def run_shell(command: Annotated[str, "Shell command to run in the workspace"]):
        r = ws.run(command)
        # Give the model everything: exit code and both output streams
        return f"exit code: {r['exit_code']}\n--- stdout ---\n{r['stdout']}\n--- stderr ---\n{r['stderr']}"

    def remember(note: Annotated[str, "One short fact about the project worth remembering in future tasks"]):
        return ws.add_note(note)

    return [
        (list_files, "List all files in the project (skips ignored files)"),
        (search, "Search file contents with a regex. Returns file:line:text matches"),
        (
            read_file,
            "Read a file with line numbers. Use start_line/end_line to read part of a big file",
        ),
        (
            edit_file,
            "Change an existing file by replacing one exact piece of text. Preferred way to modify files",
        ),
        (
            write_file,
            "Create a NEW file, or completely rewrite a small one. Use edit_file to change existing files",
        ),
        (
            run_shell,
            "Run a shell command (e.g. run the code or tests). No internet access",
        ),
        (
            remember,
            "Save one short, lasting fact about this project to AGENTS.md (shown to future tasks)",
        ),
    ]


def make_local_tools(ws, set_plan=None):
    """Tools that depend on the harness's own state, so they never go over MCP.
    git_diff compares against the review baseline (ws.mark) that the harness moves at every sync and accept;
    update_plan changes the session's plan."""

    def git_diff():
        return ws.diff_now()

    def update_plan(steps: Annotated[str, "The whole plan, one step per line: '[ ] todo', '[~] in progress' or '[x] done', then the step text"]):
        return set_plan(steps)

    tools = [
        (
            git_diff,
            "Show all changes you have made so far. Use it to review your work before finishing",
        ),
    ]
    if set_plan:   # only sessions keep a plan
        tools.append((
            update_plan,
            "Write or update your plan for a task with several steps or files. Send the WHOLE plan every time",
        ))
    return tools


def make_tools(ws, set_plan=None):
    """All tools, in-process. Same order as always: workspace tools, then git_diff, then update_plan."""
    return make_workspace_tools(ws) + make_local_tools(ws, set_plan)
