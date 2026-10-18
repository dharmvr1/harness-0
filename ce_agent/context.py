"""
Keeping the conversation healthy in a long session.

  - as_message:            turn the model's reply into a plain dict we fully control
  - estimate_tokens:       rough size of what we are about to send
  - clear_old_tool_outputs: shrink old, bulky tool results (the cheapest way to free space)
  - repair_messages:       after an interrupt, make sure every tool call has an answer

All plain functions on the `messages` list, so they are easy to test.
"""

import json


def shorten(text, limit=4000):
    """Cut the middle, not the end: the useful part of an error is usually the last lines."""
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + f"\n... [{len(text) - limit} characters cut] ...\n" + text[-half:]


def as_message(reply):
    """The model's reply as a plain dict (works for the OpenAI SDK object and for test fakes)."""
    message = {"role": "assistant", "content": reply.content or ""}
    if reply.tool_calls:
        message["tool_calls"] = [
            {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}}
            for c in reply.tool_calls
        ]
    return message


def estimate_tokens(messages, tools=None):
    """Rough guess (about 3 characters per token, on the safe side for code)."""
    return (len(json.dumps(messages, default=str)) + len(json.dumps(tools or []))) // 3


def _tool_call_labels(messages):
    """tool_call_id -> 'name(args)' so a cleared result can still say what it was."""
    labels = {}
    for m in messages:
        for c in m.get("tool_calls", []) if m["role"] == "assistant" else []:
            args = " ".join(c["function"]["arguments"].split())
            labels[c["id"]] = f"{c['function']['name']}({args[:80]})"
    return labels


def clear_old_tool_outputs(messages, keep_last=6, min_chars=300):
    """Replace big, old tool results with a short stub. The newest `keep_last` results stay intact.
    Returns how many were cleared."""
    tool_positions = [i for i, m in enumerate(messages) if m["role"] == "tool"]
    labels = _tool_call_labels(messages)
    cleared = 0
    for i in tool_positions[:-keep_last] if keep_last else tool_positions:
        m = messages[i]
        if len(m["content"]) > min_chars and not m["content"].startswith("[output cleared"):
            label = labels.get(m["tool_call_id"], "a tool call")
            m["content"] = (f"[output cleared to save space: {label} returned {len(m['content'])} characters. "
                            "Call the tool again if you need it.]")
            cleared += 1
    return cleared


def repair_messages(messages):
    """After a Ctrl-C or crash in the middle of running tools, some tool calls may have no result.
    The API rejects such a conversation, so answer them with an 'interrupted' result."""
    answered = {m["tool_call_id"] for m in messages if m["role"] == "tool"}
    fixed = 0
    i = 0
    while i < len(messages):
        m = messages[i]
        if m["role"] == "assistant" and m.get("tool_calls"):
            # results must follow the assistant message directly, so insert right after it (and its results)
            j = i + 1
            while j < len(messages) and messages[j]["role"] == "tool":
                j += 1
            missing = [c["id"] for c in m["tool_calls"] if c["id"] not in answered]
            for call_id in missing:
                messages.insert(j, {"role": "tool", "tool_call_id": call_id,
                                    "content": "ERROR: this tool call was interrupted and did not finish."})
                j += 1
                fixed += 1
            i = j
        else:
            i += 1
    return fixed
