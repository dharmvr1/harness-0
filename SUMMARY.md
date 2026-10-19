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
- **Sandbox environment** (`environment.py`, `egress.py`, `egress_proxy.py`): the image, network mode, limits and caches follow the project. Detected from its files (Cargo.toml -> rust, package.json -> node, go.mod -> go, else python), overridden by `<repo>/.agent/env.json` (outside the agent's copy, so the model cannot edit its own policy) and by flags. Network `none` (default) / `registries` (only package hosts, through an allowlist proxy on an internal Docker network: no direct route, no outside DNS) / `full` (loud warning). Per-project cache folder at `/cache` survives sessions; the model is told what the sandbox allows; what the proxy allowed and refused is printed and logged. `--env` shows the settings
- MCP, hand-written, no SDK (protocol revision 2026-07-28, `ce_agent/mcp/`): a server for the 7 workspace tools and a client, over stdio or Streamable HTTP. `--mcp[=stdio|http]` runs the tools in a separate server process; `update_plan` and `git_diff` stay in-process. HTTP: Origin/Host checks, bearer token (passed in the environment, never argv), header/body consistency checks incl. `x-mcp-header`, JSON or Server-Sent Events replies, progress for long commands, closing the connection cancels (and kills) the command. The server refuses the older `initialize` handshake and says which version it speaks. `python -m ce_agent.mcp.bench` compares in-process / stdio / HTTP
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
- Tests: `tests/test_workspace.py`, `tests/test_loop.py` (loop with a fake brain, no API needed), `tests/test_tools.py`, `tests/test_trace.py`, `tests/test_notes.py`, `tests/test_fresh_repo.py`, `tests/test_snapshot.py`, `tests/test_context.py`, `tests/test_session.py`, `tests/test_mcp_protocol.py`, `tests/test_mcp_server.py`, `tests/test_mcp_stdio.py`, `tests/test_mcp_client.py`, `tests/test_mcp_integration.py`, `tests/test_mcp_headers.py`, `tests/test_mcp_sse.py`, `tests/test_mcp_http.py`, `tests/test_mcp_http_process.py`, `tests/test_run_progress.py`, `tests/test_mcp_bench.py`, `tests/test_environment.py`, `tests/test_egress_proxy.py`, `tests/test_sandbox_network.py`, `tests/test_environment_wiring.py` (647 tests in total; `test_sandbox_network.py` uses real containers and is skipped without Docker)

## Not done yet
- Summarising old turns with the model (today we only clear old tool output), `--resume` of a saved session
- MCP: `notifications/cancelled` over stdio (the stdio server handles one request at a time, so it cannot hear it while busy), multi round-trip requests, `subscriptions/listen`, keep-alive HTTP connections, a legacy (`initialize`) shim. The installed official SDK (1.28.1) speaks only up to 2025-11-25, so it cannot talk to our modern-only server until the shim exists. Cancelling a command inside a Docker container kills the `docker exec` client but not the process inside it (it ends at its own 60 s limit)
- Hooks
- Rust / Node / Go presets (image, registries, resources) are written from what I know of those tools and are NOT tried end to end here (only Python/PyPI was). Not built: a setup phase or human approval before the network opens, a process tool for dev servers, per-tool network rules (e.g. git over SSH), bandwidth limits

## Run
```
export OPENROUTER_API_KEY=...        # or PROVIDER=groq GROQ_API_KEY=...  (or use a .env file)
python -m ce_agent.main ~/myproject                      # interactive session
python -m ce_agent.main ~/myproject "fix the bug"        # first request, then keep going
python -m ce_agent.main ~/myproject "fix the bug" --once # one request, then review
python -m ce_agent.main ~/myproject --mcp                # workspace tools served by a separate MCP server process (stdio)
python -m ce_agent.main ~/myproject --mcp=http           # ...over Streamable HTTP on a random localhost port
python -m ce_agent.mcp.bench                             # in-process vs stdio vs HTTP: cold start and per-call latency
python -m ce_agent.main ~/myproject --env                # show the sandbox settings that would be used
python -m ce_agent.main ~/myproject --network=registries # packages may be installed from the project's registries only
python -m ce_agent.main ~/myproject --image=rust:1-slim --memory=4g --timeout=900 --network=registries   # (flags use = , no space)
# env: CONTEXT_TOKENS (default 32000), SESSION_TOKEN_BUDGET (default 500000)
python -m ce_agent.main sandbox_testing --show      # what did the last run do?
python -m pytest tests
```
Uncommitted changes are fine (the agent sees them). A new or non-git folder is fine too: it offers to run `git init`.

## Real-model run
First run with a real model (see docs/PROBLEMS.md #38): output cap (`MAX_TOKENS`), backend pinning (`PROVIDER_ORDER`), retry on cut-off tool calls, nudge after a failed call, read-before-overwrite, one tool call per reply. 691 tests pass. A weak model still sometimes claims work it did not do; verification belongs to hooks/evals.

## Hooks
`ce_agent/hooks.py`: PreToolUse allow/deny/ask from `<repo>/.agent/hooks.json` plus built-in guards (docs/PROBLEMS.md #39). Other hook types not done.
