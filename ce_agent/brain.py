"""
The brain: talks to the LLM.
Works with any OpenAI-compatible API (Groq, OpenRouter, ...).
Pick the provider with the PROVIDER env var (default: groq).
"""

import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

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
    "You are a coding agent working inside a sandbox. "
    "Use the tools to complete the task. "
    "Look at the files first, make your changes, and run the code to check it works. "
    "When you are finished, reply with a short summary and do not call any more tools."
)


class Brain:
    def __init__(self):
        name = os.environ.get("PROVIDER", "openrouter")
        cfg = PROVIDERS[name]
        self.model = os.environ.get("MODEL", cfg["model"])
        self.client = OpenAI(
            base_url=cfg["base_url"],
            api_key=os.environ[cfg["key_env"]],
        )

    def think(self, messages, tools):
        """Send the conversation to the model, return its reply message."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
        )
        return response.choices[0].message
