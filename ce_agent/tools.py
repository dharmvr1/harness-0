"""
The tools the model can call. Each is a plain function with type hints;
the Registry turns them into schemas for the LLM.
The real work is done by the Workspace, so these stay thin.

Two groups:
  - workspace tools: need only the files on disk. These can be served over MCP (ce_agent/mcp/).
  - local tools: depend on the harness's own state (the plan, the review baseline), so they always run in-process.
"""

from typing import Annotated

from ce_agent.mcp.progress import report_progress


def make_workspace_tools(ws):
    """Tools that only need the workspace's files. Returns (function, description) pairs.
    Errors are raised; the registry (or the MCP server) turns them into text for the model."""

    # Files the model has read, edited or written in this session. write_file refuses to replace an existing file that
    # is not in here: a real model overwrote a file it had never read and deleted two functions in it.
    known = set()

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
        text = ws.read_file(file_path, start_line, end_line)
        known.add(ws._safe_path(file_path))
        return text

    def edit_file(
        file_path: Annotated[str, "Path of the existing file to edit"],
        old_text: Annotated[
            str, "Exact text to replace. Must appear exactly once in the file"
        ],
        new_text: Annotated[str, "Text to put in its place"],
    ):
        result = ws.edit_file(file_path, old_text, new_text)
        known.add(ws._safe_path(file_path))
        return result

    def write_file(
        file_path: Annotated[str, "Path of the file to create"],
        content: Annotated[str, "Full content of the file"],
    ):
        full = ws._safe_path(file_path)
        if full.exists() and full not in known:
            raise ValueError(
                f"{file_path} already exists and you have not read it in this session, so writing it would throw "
                "away its current content. Read it with read_file first, then change it with edit_file "
                "(use write_file on it only if you really mean to replace all of it)."
            )
        result = ws.write_file(file_path, content)
        known.add(full)
        return result

    def run_shell(command: Annotated[str, "Shell command to run in the workspace"]):
        # While the command runs, tell the client (if it asked) that we are alive; a cancelled request stops here.
        r = ws.run(command, on_tick=lambda seconds: report_progress(seconds, message=f"command running for {seconds:.0f}s"))
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
            "Create a NEW file. An existing file must be read first; to change one, use edit_file (write_file replaces ALL of its content)",
        ),
        (
            run_shell,
            "Run a shell command (e.g. run the code or tests). The system prompt says what the sandbox allows",
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
