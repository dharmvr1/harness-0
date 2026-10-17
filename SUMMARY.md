# Where the project is

A small coding agent built to learn how agents work. Spec: `Readme.md`.

## How it works
```
python -m ce_agent.main <repo> "<task>"

  brain.py      LLM call (OpenRouter or Groq, OpenAI-compatible API)
  tools.py      the 7 tools the model can call
  registry.py   Python functions -> tool schemas (from type hints)
  workspace.py  git worktree on branch agent/<id> + one Docker container per task
  main.py       loop: think -> act -> observe -> repeat, then review
```
The agent works only in the worktree. When it finishes you see the diff and choose
accept (merge into your repo), reject, or keep (leave it on its branch).

## Done
- 7 tools: `list_files`, `search` (git grep), `read_file` (line numbers + ranges), `edit_file` (exact single-match replace), `write_file`, `run_shell`, `git_diff`
- Real LLM tool loop with a 20-turn cap and head+tail output truncation
- Git worktree + Docker sandbox (no network, memory/process limits, non-root, caps dropped)
- `--unsafe-local` mode (no container, clean environment so API keys are not exposed)
- Work is never lost: crash or Ctrl-C saves it on the branch; `--cleanup` recovers after a hard kill
- Tests: `tests/test_workspace.py`, `tests/test_loop.py` (loop with a fake brain, no API needed), `tests/test_tools.py`

## Not done yet
- Context/cost limits beyond the turn cap
- Plan/todo tool, hooks, trace log
- Per-project Docker images (the default image has no pytest/git, and the network is off)

## Run
```
export OPENROUTER_API_KEY=...        # or PROVIDER=groq GROQ_API_KEY=...  (or use a .env file)
python -m ce_agent.main sandbox_testing "Add a multiply function to calculator.py"
python -m pytest tests
```
The target folder must have no uncommitted changes (a worktree starts from the last commit).
