"""
A schema-aware tool registry.
It uses Python type hints to generate JSON schemas automatically.
"""

import inspect
import json
from typing import get_type_hints, Annotated, get_origin, get_args

# Python type -> JSON Schema type
JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean"}

class Registry:
    def __init__(self):
        self.tools = {}

    def register(self, func, description: str):
        """Register a function with a description."""
        self.tools[func.__name__] = {
            "func": func,
            "description": description,
            "schema": self._generate_schema(func)
        }

    def _generate_schema(self, func):
        """Automatically generate JSON Schema from type hints."""
        hints = get_type_hints(func, include_extras=True)
        params = {}
        for name, hint in hints.items():
            if name == "return": continue

            # Extract description from Annotated
            desc = "No description"
            if get_origin(hint) is Annotated:
                args = get_args(hint)
                desc = args[1]
                hint = args[0]

            params[name] = {"type": JSON_TYPES.get(hint, "string"), "description": desc}
        # Parameters with a default value (e.g. start_line: int = 1) are optional
        sig = inspect.signature(func).parameters
        required = [n for n in params if sig[n].default is inspect.Parameter.empty]
        return {"type": "object", "properties": params, "required": required}

    def get_tool_list(self):
        """Return schema for all tools (for the LLM)."""
        return {name: {"description": t["description"], "parameters": t["schema"]}
                for name, t in self.tools.items()}

    def get_openai_tools(self):
        """Same schemas, in the format OpenAI-compatible APIs expect."""
        return [
            {"type": "function",
             "function": {"name": name, "description": t["description"], "parameters": t["schema"]}}
            for name, t in self.tools.items()
        ]

    def call(self, name, kwargs):
        """Execute a tool call with validation."""
        if name not in self.tools:
            return {"status": "error", "error": f"Tool {name} not found"}

        # Models are sloppy: they send null for optional args and numbers as strings. Fix both.
        props = self.tools[name]["schema"]["properties"]
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        for k, v in kwargs.items():
            if props.get(k, {}).get("type") == "integer" and isinstance(v, str):
                try:
                    kwargs[k] = int(v)
                except ValueError:
                    return {"status": "error", "error": f"{k} must be a whole number, got '{v}'"}

        try:
            result = self.tools[name]["func"](**kwargs)
            return {"status": "ok", "content": str(result)}
        except Exception as e:
            return {"status": "error", "error": str(e)}
