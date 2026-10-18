"""
A Session: one conversation with the agent that lasts across many of your prompts.

It owns the model, the workspace, the tools and the growing `messages` list.
send(text) handles ONE prompt: it loops (model call -> tool calls -> model call ...)
until the model answers without calling a tool, then control goes back to you.
"""

import json
import os
import re
import time

from ce_agent.context import (
    as_message,
    clear_old_tool_outputs,
    estimate_tokens,
    repair_messages,
    shorten,
)
from ce_agent.brain import system_prompt
from ce_agent.registry import Registry
from ce_agent.tools import make_local_tools, make_tools

MAX_TURNS = 20  # model calls allowed for ONE prompt, so a prompt can never loop forever
MAX_OUTPUT_CHARS = 4000  # cut long tool output so it does not flood the context
TRACE_RESULT_CHARS = 20000  # the log keeps more of each result than the model sees

# Context window we plan for (tokens). Conservative on purpose: free API tiers have tight limits.
CONTEXT_TOKENS = int(os.environ.get("CONTEXT_TOKENS", 32000))
CLEAR_AT = 0.70  # above this share of the window, clear old tool outputs
STOP_AT = 0.95  # above this share even after clearing, refuse to continue
TOKEN_BUDGET = int(
    os.environ.get("SESSION_TOKEN_BUDGET", 500000)
)  # total tokens (in + out) for the whole session

PLAN_STATUS = {"x": "done", "~": "in progress", " ": "pending"}
PLAN_LINE = re.compile(r"^\s*(?:[-*]|\d+[.)])?\s*\[(.?)\]\s*(\S.*)$")


class Session:
    def __init__(self, ws, brain, trace, notes="", mcp_client=None):
        self.ws, self.brain, self.trace, self.notes = ws, brain, trace, notes
        self.plan = []  # list of (mark, text), mark is one of x ~ and a space

        registry = Registry()
        if mcp_client:
            # The workspace tools live on the MCP server. The two that depend on the harness's own state
            # (update_plan, git_diff) always run in-process.
            for tool in mcp_client.list_tools():
                registry.register_remote(tool, mcp_client.call_tool)
            for func, description in make_local_tools(ws, set_plan=self.set_plan):
                registry.register(func, description)
        else:
            for func, description in make_tools(ws, set_plan=self.set_plan):
                registry.register(func, description)
        self.registry = registry
        self.tools = registry.get_openai_tools()

        self.messages = [
            {"role": "system", "content": ""}
        ]  # filled in before every model call
        self.prompt_no = 0
        self.last_prompt_changed = False  # did the last prompt change any files?
        self.model_calls = 0
        self.total_in = self.total_out = 0

    # ------------------------------------------------------------------
    # the plan (the model rewrites the whole thing each time)
    # ------------------------------------------------------------------
    def set_plan(self, text):
        steps = []
        for line in text.strip().splitlines():
            if not line.strip():
                continue
            m = PLAN_LINE.match(line)
            mark = (m.group(1).lower() or " ") if m else None
            if not m or mark not in PLAN_STATUS or len(m.group(2)) > 150:
                raise ValueError(
                    f"Can't read this plan line: '{line[:80]}'. Write one step per line, like:\n"
                    "[x] finished step\n[~] step in progress\n[ ] step still to do"
                )
            steps.append((mark, m.group(2).strip()))
        if len(steps) > 15:
            raise ValueError(
                "The plan has more than 15 steps. Group them into fewer, bigger steps."
            )
        self.plan = steps
        shown = self.render_plan()
        self.trace.log("plan_update", prompt_no=self.prompt_no, plan=shown)
        print("\n[Plan]\n" + (shown or "(empty)"))
        return "Plan updated:\n" + (shown or "(empty)")

    def render_plan(self):
        return "\n".join(f"[{mark}] {text}" for mark, text in self.plan)

    def _refresh_notes(self):
        """Re-read AGENTS.md before each model call, so a note the model just saved (or one you edited by hand)
        is part of the very next prompt, not only of the next session."""
        try:
            notes = self.ws.read_notes()
        except Exception:
            return   # keep the last notes we could read
        if notes != self.notes:
            self.notes = notes
            self.trace.log("notes_changed", prompt_no=self.prompt_no, chars=len(notes))

    def _system(self):
        self._refresh_notes()
        text = system_prompt(self.notes)
        if (
            self.plan
        ):  # in the system prompt, so clearing old tool output can never erase it
            text += (
                "\n\nYour current plan (change it with update_plan):\n"
                + self.render_plan()
            )
        return text

    # ------------------------------------------------------------------
    # one prompt
    # ------------------------------------------------------------------
    def send(self, text):
        """Handle one user prompt. Returns why it ended: done, turn_limit, interrupted, context_full, budget, error: ..."""
        self.prompt_no += 1
        sync_note = self._sync_with_real_repo()
        # One user message: the heads-up about your edits (if any) goes in front of your request.
        self.messages.append({"role": "user", "content": f"{sync_note}\n\n{text}" if sync_note else text})
        self.trace.log("user_message", prompt_no=self.prompt_no, text=text)   # the log keeps exactly what you typed
        reason, turns_before = "turn_limit", self.model_calls

        # Whatever goes wrong, the session survives and the work done so far is kept.
        try:
            for turn in range(MAX_TURNS):
                stop = self._check_limits()
                if stop:
                    reason = stop
                    break
                reply = self._think(turn)
                if (
                    not reply.tool_calls
                ):  # no tool calls means the model thinks it is done
                    print(f"\n{reply.content}")
                    reason = "done"
                    break
                self._run_tools(reply, turn)
            else:
                print(
                    f"\n[Stopped] hit the {MAX_TURNS}-turn limit for this prompt. Say 'continue' to go on."
                )
        except KeyboardInterrupt:
            print("\n[Interrupted] back to the prompt.")
            reason = "interrupted"
        except Exception as e:
            print(f"\n[Error] {type(e).__name__}: {e}")
            reason = f"error: {type(e).__name__}: {e}"

        repair_messages(
            self.messages
        )  # an interrupt can leave tool calls without results
        self.last_prompt_changed = self._checkpoint(text)
        self.trace.log(
            "prompt_end",
            prompt_no=self.prompt_no,
            reason=reason,
            model_calls=self.model_calls - turns_before,
        )
        print(
            f"\n[context ~{estimate_tokens(self.messages, self.tools) // 1000}k/{CONTEXT_TOKENS // 1000}k tokens"
            f" | session total {self.total_in}+{self.total_out} tokens]"
        )
        return reason

    def _check_limits(self):
        """Before each model call: keep the context small enough, and the session within budget."""
        used = estimate_tokens(self.messages, self.tools)
        if used > CLEAR_AT * CONTEXT_TOKENS:
            before, cleared = used, 0
            for keep_last in (
                6,
                3,
                1,
            ):  # protect the newest results first, dig deeper only if still too full
                cleared += clear_old_tool_outputs(self.messages, keep_last=keep_last)
                used = estimate_tokens(self.messages, self.tools)
                if used <= CLEAR_AT * CONTEXT_TOKENS:
                    break
            if cleared:
                print(
                    f"[context] cleared {cleared} old tool outputs (~{before // 1000}k -> ~{used // 1000}k tokens)"
                )
                self.trace.log(
                    "context_cleared",
                    cleared=cleared,
                    tokens_before=before,
                    tokens_after=used,
                )
        if used > STOP_AT * CONTEXT_TOKENS:
            print(
                f"\n[Stopped] the conversation is full (~{used // 1000}k of {CONTEXT_TOKENS // 1000}k tokens). "
                "Press Ctrl-D to review and finish; what you accept stays in the repo, and AGENTS.md notes carry "
                "over to the next session."
            )
            return "context_full"
        if self.total_in + self.total_out > TOKEN_BUDGET:
            print(
                f"\n[Stopped] session token budget used up ({self.total_in + self.total_out} of {TOKEN_BUDGET})."
            )
            return "budget"
        return None

    def _think(self, turn):
        self.messages[0]["content"] = self._system()
        started = time.time()
        reply = self.brain.think(self.messages, self.tools)
        self.model_calls += 1
        self.messages.append(as_message(reply))
        usage = getattr(self.brain, "last_usage", None)
        if usage:
            self.total_in += usage["prompt_tokens"]
            self.total_out += usage["completion_tokens"]
        self.trace.log(
            "model_call",
            prompt_no=self.prompt_no,
            turn=turn,
            seconds=round(time.time() - started, 2),
            usage=usage,
            content=reply.content,
            tool_calls=[
                {"id": c.id, "name": c.function.name, "arguments": c.function.arguments}
                for c in reply.tool_calls or []
            ],
        )
        return reply

    def _run_tools(self, reply, turn):
        for call in reply.tool_calls:
            name = call.function.name
            started = time.time()
            try:
                args = json.loads(call.function.arguments)
            except json.JSONDecodeError:
                args = call.function.arguments
                result = {"status": "error", "error": "Arguments were not valid JSON"}
            else:
                print(f"\n[Turn {turn}] {name}({args})")
                result = self.registry.call(name, args)

            # errors go back to the model as text so it can fix itself
            text = (
                result["content"]
                if result["status"] == "ok"
                else "ERROR: " + result["error"]
            )
            print(text[:300])
            self.trace.log(
                "tool_call",
                prompt_no=self.prompt_no,
                turn=turn,
                name=name,
                args=args,
                status=result["status"],
                result=shorten(text, TRACE_RESULT_CHARS),
                seconds=round(time.time() - started, 2),
            )
            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": shorten(text, MAX_OUTPUT_CHARS),
                }
            )

    def _checkpoint(self, prompt):
        """Commit what the agent has done so far on its branch (the history shows one commit per prompt).
        Returns True if this prompt changed any files."""
        try:
            return self.ws.checkpoint("agent: " + " ".join(prompt.split())[:60])
        except Exception as e:
            print(f"[warning] could not save a checkpoint: {e}")
            return False

    def _sync_with_real_repo(self):
        """Before each prompt: pick up changes you made in your own folder meanwhile (edits, commits, branch switch).
        Returns a note for the model naming the files that changed ('' if nothing changed or the sync failed),
        because anything it read earlier may now be out of date."""
        try:
            message = self.ws.sync()
            if message:
                print(f"[sync] {message}")
            changes = list(self.ws.last_sync_changes)
            if message:
                self.trace.log("sync", prompt_no=self.prompt_no, message=message, changes=changes)
            return self._describe_changes(changes)
        except Exception as e:
            print(f"[warning] could not sync with your folder: {e}")
            return ""

    @staticmethod
    def _describe_changes(changes, limit=15):
        """'M\tcalc.py' lines -> a note for the model. Empty when there is nothing to tell."""
        if not changes:
            return ""
        words = {"M": "modified", "A": "new", "D": "deleted"}
        lines = []
        for entry in changes[:limit]:
            status, _, path = entry.partition("\t")
            lines.append(f"- {path} ({words.get(status[:1], 'changed')})")
        if len(changes) > limit:
            lines.append(f"- ... and {len(changes) - limit} more files")
        return ("[The developer changed the project since your last request, and your copy now includes these changes:\n"
                + "\n".join(lines)
                + "\nFiles you read earlier may be out of date. Re-read a file before relying on what you remember of it.]")

    def note_rejected(self):
        """Tell the model its pending changes were thrown away, or it would keep believing its edits exist."""
        self.messages.append(
            {
                "role": "user",
                "content": (
                    "[The developer rejected your pending changes. The workspace was reset to the project's current state: "
                    "files you edited or created earlier are back to their original content or gone. "
                    "Re-read any file before relying on it.]"
                ),
            }
        )
