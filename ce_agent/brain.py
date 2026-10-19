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
        "model": "openai/gpt-oss-120b",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "model": "meta-llama/llama-3.3-70b-instruct",
    },
}

SYSTEM_PROMPT = (
    "You are a coding agent in a long working session with a developer. You work in a sandboxed copy of their "
    "project. All paths are relative to the project folder. "
    "The developer may ask questions about the code (then just explore and answer, and do not change files) "
    "or ask for changes, possibly across several files, one request after another. "
    "How to work: use search and list_files to find the relevant code, "
    "read_file to read it (use start_line/end_line for big files), "
    "edit_file to change existing files (write_file is only for new files), "
    "run_shell to run the code or tests and check your work, "
    "and git_diff to review your changes before you finish. "
    "For anything with several steps or several files, first write a short plan with update_plan, "
    "and update it as you finish each step. You may say in one sentence what you are about to do, but never stop at "
    "that sentence: put the tool call in the same reply. A reply without a tool call ends the request. "
    "Earlier tool outputs may be replaced by '[output cleared]' to save space; call the tool again if you need them. "
    "If a tool returns an error, read it carefully and fix your call. "
    "Use remember to save short, lasting facts about the project (where things live, how to run tests); "
    "do not use it for progress on the current task. "
    "When you are finished with a request, reply with a short summary (or the answer to the question) "
    "and do not call any more tools."
)


def system_prompt(notes="", environment=""):
    """The base prompt, plus what the sandbox allows (image, network, time limit) and the project's AGENTS.md notes."""
    text = SYSTEM_PROMPT
    if environment:
        text += "\n\nYour environment: " + environment
    if notes:
        text += "\n\nNotes about this project (from AGENTS.md):\n" + notes
    return text


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
        self.max_tokens = int(os.environ.get("MAX_TOKENS", "4096"))   # cap per reply: bounds cost, and OpenRouter needs it for credit checks
        # OpenRouter only: "PROVIDER_ORDER=Groq,DeepInfra" pins which backends serve the model (no fallback to others)
        order = [p.strip() for p in os.environ.get("PROVIDER_ORDER", "").split(",") if p.strip()]
        self.extra_body = {"provider": {"order": order, "allow_fallbacks": False}} if order and name == "openrouter" else None
        self.last_backend = None   # which backend served the most recent reply (OpenRouter reports it)
        self.last_finish_reason = None   # why the most recent reply ended: "stop", "tool_calls", "length" (cut off), ...
        # max_retries: the SDK waits and retries on rate limits (429), server errors and network blips
        self.client = OpenAI(base_url=cfg["base_url"], api_key=api_key, max_retries=5, timeout=90)

    def think(self, messages, tools):
        """Send the conversation to the model, return its reply message."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            parallel_tool_calls=False,   # one tool call per reply: simpler to follow, and some backends reject several
            max_tokens=self.max_tokens,
            extra_body=self.extra_body,
        )
        self.last_backend = getattr(response, "provider", None)
        usage = response.usage   # some providers don't send it
        self.last_usage = {"prompt_tokens": usage.prompt_tokens,
                           "completion_tokens": usage.completion_tokens} if usage else None
        choice = response.choices[0]
        self.last_finish_reason = choice.finish_reason
        message = choice.message
        if message.tool_calls and len(message.tool_calls) > 1:
            # parallel_tool_calls=False is not obeyed by every backend, and some (Groq) then reject the history with
            # "only supports single tool-calls at once". Keep the first call; the model asks for the rest next turn.
            message.tool_calls = message.tool_calls[:1]
        return message
