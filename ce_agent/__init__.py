"""ce-agent: a small terminal coding agent, built to learn how agents work.

brain.py      talks to the LLM (any OpenAI-compatible API)
tools.py      the tools the model can call
registry.py   turns Python functions into tool schemas
workspace.py  git worktree + Docker sandbox, with accept / reject / keep
session.py    one conversation across many prompts
mcp/          a hand-written MCP server and client (the workspace tools can run in a separate process)
main.py       starts a session, the prompt loop, the review
"""

__version__ = "0.1.0"
