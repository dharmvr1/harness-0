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
    "You are a coding agent working in a sandboxed copy of a project. "
    "All paths are relative to the project folder. There is no internet access. "
    "Work like this: use search and list_files to find the relevant code, "
    "read_file to read it (use start_line/end_line for big files), "
    "edit_file to change existing files (write_file is only for new files), "
    "run_shell to run the code or tests and check your work, "
    "and git_diff to review your changes before you finish. "
    "If a tool returns an error, read it carefully and fix your call. "
    "When you are finished, reply with a short summary and do not call any more tools."
)


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
        # max_retries: the SDK waits and retries on rate limits (429), server errors and network blips
        self.client = OpenAI(base_url=cfg["base_url"], api_key=api_key, max_retries=5, timeout=90)

    def think(self, messages, tools):
        """Send the conversation to the model, return its reply message."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
        )
        return response.choices[0].message
