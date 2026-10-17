"""ce-agent: a small terminal coding agent, built to learn how agents work.

brain.py      talks to the LLM (any OpenAI-compatible API)
tools.py      the tools the model can call
registry.py   turns Python functions into tool schemas
workspace.py  git worktree + Docker sandbox, with accept / reject / keep
main.py       the loop: think -> act -> observe -> repeat
"""

__version__ = "0.1.0"
