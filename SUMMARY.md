# Where the project is

A small coding agent built to learn how agents work. Spec: `Readme.md`.

## How it works
```
python -m ce_agent.main <repo> ["first request"]    (keep typing requests; Ctrl-D ends the session and shows the review)

  brain.py      LLM call (OpenRouter or Groq, OpenAI-compatible API)
  tools.py      the 9 tools the model can call
  trace.py      trajectory log: one JSON line per event in .agent/runs/<id>.jsonl
  registry.py   Python functions -> tool schemas (from type hints)
  workspace.py  git worktree on branch agent/<id> + one Docker container per task
  session.py    one conversation across many prompts: the loop, plan, context limits, token budget
  context.py    helpers for the message list: clear old tool output, repair after Ctrl-C, size estimate
  main.py       starts a session, the prompt loop, and the final review (accept / reject / keep)
```
The agent works only in the worktree. When it finishes you see the diff and choose
accept (apply to your folder), reject, or keep (leave it on its branch).
In a session you get this choice after every request that changed files: accept = applied to your folder right
now as uncommitted changes, reject = workspace reset (the model is told), later = stays pending.

## Done
- MCP, hand-written, no SDK (protocol revision 2026-07-28, `ce_agent/mcp/`): a server for the 7 workspace tools and a client, over stdio. `--mcp` runs the tools in a separate server process; `update_plan` and `git_diff` stay in-process. The server refuses the older `initialize` handshake and says which version it speaks
- Works on a dirty folder: the agent's copy starts from a snapshot of your folder (committed + uncommitted + new files, minus .gitignore), built with a temporary index so your files/index/history are never touched
- Per-prompt review: accept applies only the agent's changes to your folder as a patch (checked first, all or nothing); nothing is ever committed for you
- Two-way sync: edits, commits or branch switches you make mid-session are merged into the agent's copy before the next prompt (3-way, aborts safely on conflict)
- Sessions: the conversation and the workspace persist across prompts; one checkpoint commit per prompt; a Ctrl-C or API error ends only that prompt
- Context management: size estimate, old tool outputs cleared when ~70% full, stop at 95%, session token budget
- Plan tool (`update_plan`): the model rewrites its checklist; it lives in the system prompt so clearing can't erase it
- Questions are fine: the prompt tells the model to explore and answer without changing files
- 9 tools: `list_files`, `search` (git grep), `read_file` (line numbers + ranges), `edit_file` (exact single-match replace), `write_file`, `run_shell`, `remember`, `git_diff`, `update_plan`
- Trajectory log of every run (`--show` prints it as a timeline): model calls, tokens, tool calls, errors, your decision
- Project memory: `AGENTS.md` is re-read before every model call (a note saved with `remember`, or one you edit by hand, is in the very next prompt); the `remember` tool adds a line, which goes through your accept/reject review like any other change
- The model is told which files changed when you edit your folder between requests (so it re-reads instead of trusting old reads)
- Works on fresh projects: a missing or non-git folder gets `git init` after asking (`ensure_repo`); no commit is needed
- Real LLM tool loop with a 20-turn cap and head+tail output truncation
- Git worktree + Docker sandbox (no network, memory/process limits, non-root, caps dropped)
- `--unsafe-local` mode (no container, clean environment so API keys are not exposed)
- Work is never lost: crash or Ctrl-C saves it on the branch; `--cleanup` recovers after a hard kill
- Tests: `tests/test_workspace.py`, `tests/test_loop.py` (loop with a fake brain, no API needed), `tests/test_tools.py`, `tests/test_trace.py`, `tests/test_notes.py`, `tests/test_fresh_repo.py`, `tests/test_snapshot.py`, `tests/test_context.py`, `tests/test_session.py`, `tests/test_mcp_protocol.py`, `tests/test_mcp_server.py`, `tests/test_mcp_stdio.py`, `tests/test_mcp_client.py`, `tests/test_mcp_integration.py` (241 tests in total)

## Not done yet
- Summarising old turns with the model (today we only clear old tool output), `--resume` of a saved session
- MCP: Streamable HTTP transport, progress/cancellation, a legacy (`initialize`) shim. The installed official SDK (1.28.1) speaks only up to 2025-11-25, so it cannot talk to our modern-only server until the shim exists
- Hooks
- Per-project Docker images (the default image has no pytest/git, and the network is off)

## Run
```
export OPENROUTER_API_KEY=...        # or PROVIDER=groq GROQ_API_KEY=...  (or use a .env file)
python -m ce_agent.main ~/myproject                      # interactive session
python -m ce_agent.main ~/myproject "fix the bug"        # first request, then keep going
python -m ce_agent.main ~/myproject "fix the bug" --once # one request, then review
python -m ce_agent.main ~/myproject --mcp                # workspace tools served by a separate MCP server process
# env: CONTEXT_TOKENS (default 32000), SESSION_TOKEN_BUDGET (default 500000)
python -m ce_agent.main sandbox_testing --show      # what did the last run do?
python -m pytest tests
```
Uncommitted changes are fine (the agent sees them). A new or non-git folder is fine too: it offers to run `git init`.
