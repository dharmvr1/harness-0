"""
The brain: talks to the LLM.
Works with any OpenAI-compatible API (Groq, OpenRouter, ...).
Pick the provider with the PROVIDER env var (default: openrouter).
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

PROVIDERS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "model": "llama-3.3-70b-versatile",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "model": "meta-llama/llama-3.3-70b-instruct",
    },
}

SYSTEM_PROMPT = (
    "You are a coding agent in a long working session with a developer. You work in a sandboxed copy of their "
    "project. All paths are relative to the project folder. There is no internet access. "
    "The developer may ask questions about the code (then just explore and answer, and do not change files) "
    "or ask for changes, possibly across several files, one request after another. "
    "How to work: use search and list_files to find the relevant code, "
    "read_file to read it (use start_line/end_line for big files), "
    "edit_file to change existing files (write_file is only for new files), "
    "run_shell to run the code or tests and check your work, "
    "and git_diff to review your changes before you finish. "
    "For anything with several steps or several files, first write a short plan with update_plan, "
    "and update it as you finish each step. Before acting, say in one sentence what you are about to do. "
    "Earlier tool outputs may be replaced by '[output cleared]' to save space; call the tool again if you need them. "
    "If a tool returns an error, read it carefully and fix your call. "
    "Use remember to save short, lasting facts about the project (where things live, how to run tests); "
    "do not use it for progress on the current task. "
    "When you are finished with a request, reply with a short summary (or the answer to the question) "
    "and do not call any more tools."
)


def system_prompt(notes=""):
    """The base prompt, plus the project's AGENTS.md notes if there are any."""
    if not notes:
        return SYSTEM_PROMPT
    return SYSTEM_PROMPT + "\n\nNotes about this project (from AGENTS.md):\n" + notes


class Brain:
    def __init__(self):
        name = os.environ.get("PROVIDER", "openrouter")
        if name not in PROVIDERS:
            raise RuntimeError(f"Unknown PROVIDER '{name}'. Choose one of: {', '.join(PROVIDERS)}")
        cfg = PROVIDERS[name]
        api_key = os.environ.get(cfg["key_env"])
        if not api_key:
            raise RuntimeError(f"{cfg['key_env']} is not set. Put it in your environment or in a .env file.")
        self.model = os.environ.get("MODEL", cfg["model"])
        self.last_usage = None   # token counts of the most recent think() call
        # max_retries: the SDK waits and retries on rate limits (429), server errors and network blips
        self.client = OpenAI(base_url=cfg["base_url"], api_key=api_key, max_retries=5, timeout=90)

    def think(self, messages, tools):
        """Send the conversation to the model, return its reply message."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
        )
        usage = response.usage   # some providers don't send it
        self.last_usage = {"prompt_tokens": usage.prompt_tokens,
                           "completion_tokens": usage.completion_tokens} if usage else None
        return response.choices[0].message
