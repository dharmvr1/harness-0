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

# Hard limits for a whole session (README step 7). All can be changed with environment variables.
SESSION_MAX_TURNS = int(os.environ.get("SESSION_MAX_TURNS", 100))   # model calls in the whole session
MAX_COST_USD = float(os.environ.get("MAX_COST_USD", 5.0))           # needs the two prices below, otherwise unchecked
PRICE_IN_PER_M = float(os.environ.get("PRICE_IN_PER_M", 0))         # USD per million prompt tokens
PRICE_OUT_PER_M = float(os.environ.get("PRICE_OUT_PER_M", 0))       # USD per million answer tokens

MAX_NUDGES = 1   # times per request that a model which gave up after a failed tool call is asked to try again
MAX_RETRIES = 2  # extra model calls when a reply was cut off in the middle of a tool call

PLAN_STATUS = {"x": "done", "~": "in progress", " ": "pending"}
PLAN_LINE = re.compile(r"^\s*(?:[-*]|\d+[.)])?\s*\[(.?)\]\s*(\S.*)$")


def describe_invalid_arguments(name, raw, error):
    """Explain a tool call whose arguments are not valid JSON in a way the model can act on.
    A bare "invalid JSON" teaches it nothing: in a real run the model gave up after exactly that message."""
    cut_off = error.msg.startswith("Unterminated") or error.pos >= len(raw)   # the problem is at the very end
    text = (f"The arguments of your {name} call are not valid JSON ({error.msg}, character {error.pos}). "
            f"I received {len(raw)} characters, ending with {raw[-80:]!r}. ")
    if cut_off:
        text += "Your message looks cut off before the JSON was complete. "
    text += ("Send the complete JSON object: close every string, and inside strings escape backslashes as \\\\, "
             "quotes as \\\" and newlines as \\n. If the content is long, write it in smaller pieces "
             "(for an existing file, use edit_file).")
    return text


class Session:
    def __init__(self, ws, brain, trace, notes="", mcp_client=None, environment_text="", hooks=None):
        self.hooks = hooks   # rules checked before every tool call (ce_agent/hooks.py); None means no hooks
        self.ws, self.brain, self.trace, self.notes = ws, brain, trace, notes
        self.environment_text = environment_text  # what the sandbox allows, told to the model in the system prompt
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
        self._last_tool_failed = False   # did the last tool call of the latest batch fail?
        self._nudges = 0                 # how often this request already asked the model to try again
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
        text = system_prompt(self.notes, self.environment_text)
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
        self._nudges, self._last_tool_failed = 0, False

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
                ):  # no tool calls means the model thinks it is done...
                    if self._last_tool_failed and self._nudges < MAX_NUDGES:
                        self._nudge()  # ...unless its last call just failed and it only announced what it will do
                        continue
                    print(f"\n{reply.content}")
                    reason = "done"
                    break
                self._run_tools(reply, turn)
                print(f"\033[2m{self.status_line()}\033[0m")   # dim: the bar after every step
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
        print("\n" + self.status_line())
        return reason

    def cost(self):
        """Money spent so far in USD, or None when no prices are configured (then only tokens and turns are limited)."""
        if not (PRICE_IN_PER_M or PRICE_OUT_PER_M):
            return None
        return (self.total_in * PRICE_IN_PER_M + self.total_out * PRICE_OUT_PER_M) / 1_000_000

    def status_line(self):
        """One line with everything that is limited: plan progress, turns, context, tokens, money."""
        done = sum(1 for mark, _ in self.plan if mark == "x")
        parts = []
        if self.plan:
            current = next((t for m, t in self.plan if m == "~"), None) or next((t for m, t in self.plan if m == " "), "all done")
            parts.append(f"plan {done}/{len(self.plan)}: {current[:40]}")
        parts.append(f"turn {self.model_calls}/{SESSION_MAX_TURNS}")
        parts.append(f"context {estimate_tokens(self.messages, self.tools) // 1000}k/{CONTEXT_TOKENS // 1000}k")
        parts.append(f"tokens {(self.total_in + self.total_out) // 1000}k/{TOKEN_BUDGET // 1000}k")
        cost = self.cost()
        if cost is not None:
            parts.append(f"${cost:.2f}/${MAX_COST_USD:.2f}")
        return "[" + " | ".join(parts) + "]"

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
        if self.model_calls >= SESSION_MAX_TURNS:
            print(f"\n[Stopped] session turn limit reached ({SESSION_MAX_TURNS} model calls). Press Ctrl-D to review and finish.")
            return "turn_budget"
        cost = self.cost()
        if cost is not None and cost >= MAX_COST_USD:
            print(f"\n[Stopped] session cost limit reached (${cost:.2f} of ${MAX_COST_USD:.2f}). Press Ctrl-D to review and finish.")
            return "cost_limit"
        return None

    def _nudge(self):
        """The model answered without a tool call right after a failed one (typically "I am going to ..."). Ask it
        once to try again; otherwise the request would end as 'done' with nothing done."""
        self._nudges += 1
        self._last_tool_failed = False
        print("[nudge] the model stopped after a failed tool call; asking it to continue")
        self.trace.log("nudge", prompt_no=self.prompt_no, count=self._nudges)
        self.messages.append({"role": "user", "content": (
            "[Your last tool call failed, so the request is not finished. Read the error, correct the call and try "
            "again. Only answer without a tool call if you cannot continue, and then say exactly what blocked you.]")})

    def _make_history_valid(self, call_id):
        """Some providers reject a conversation that contains a tool call with invalid JSON arguments, so after we
        have explained the problem to the model the broken arguments in the history become an empty object."""
        for message in reversed(self.messages):
            if message["role"] == "assistant":
                for call in message.get("tool_calls", []):
                    if call["id"] == call_id:
                        call["function"]["arguments"] = "{}"
                        return

    def _think(self, turn):
        self.messages[0]["content"] = self._system()
        started = time.time()
        reply = self.brain.think(self.messages, self.tools)
        self.model_calls += 1
        for _ in range(MAX_RETRIES):
            if not self._looks_truncated(reply):
                break
            # a backend cut the reply off mid tool call (seen on OpenRouter): asking again usually works
            self.trace.log("retry", prompt_no=self.prompt_no, turn=turn, backend=getattr(self.brain, "last_backend", None))
            reply = self.brain.think(self.messages, self.tools)
            self.model_calls += 1
        self.messages.append(as_message(reply))
        usage = getattr(self.brain, "last_usage", None)
        if usage:
            self.total_in += usage["prompt_tokens"]
            self.total_out += usage["completion_tokens"]
        finish_reason = getattr(self.brain, "last_finish_reason", None)
        if finish_reason == "length":
            print("[warning] the model's answer was cut off (it hit the output limit)")
        self.trace.log(
            "model_call",
            prompt_no=self.prompt_no,
            turn=turn,
            seconds=round(time.time() - started, 2),
            usage=usage,
            finish_reason=finish_reason,
            backend=getattr(self.brain, "last_backend", None),
            content=reply.content,
            tool_calls=[
                {"id": c.id, "name": c.function.name, "arguments": c.function.arguments}
                for c in reply.tool_calls or []
            ],
        )
        return reply

    def _hook_blocks(self, name, args, turn):
        """Run the PreToolUse hooks. Returns None to let the call run, or the reason it was blocked (told to the model)."""
        if not self.hooks:
            return None
        decision = self.hooks.pre_tool_use(name, args)
        if decision.action == "allow":
            return None
        why = decision.message or f"it matches the rule {decision.rule['pattern']!r}"
        answer = "deny"
        if decision.action == "ask":
            print(f"[hook] {name} wants to run: {args}\n       {why}")
            try:
                answer = "allow" if input("Allow this? [y/N] ").strip().lower() in ("y", "yes") else "deny"
            except EOFError:
                answer = "deny"
        self.trace.log("hook", prompt_no=self.prompt_no, turn=turn, tool=name, args=args, rule=decision.rule,
                       asked=decision.action == "ask", result=answer)
        if answer == "allow":
            return None
        print(f"[hook] blocked {name}: {why}")
        by = "the user declined it" if decision.action == "ask" else "a safety rule"
        return f"Blocked: {by}. {why}. Do not retry the same call; choose another way or explain what you need."

    def _looks_truncated(self, reply):
        """True if a tool call has broken JSON and the backend did not say the reply was complete or hit the limit.
        Brains that do not report a finish reason (the scripted test brains) are never retried."""
        if not hasattr(self.brain, "last_finish_reason") or self.brain.last_finish_reason in ("tool_calls", "length"):
            return False
        for call in reply.tool_calls or []:
            try:
                json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                return True
        return False

    def _run_tools(self, reply, turn):
        for call in reply.tool_calls:
            name = call.function.name
            started = time.time()
            try:
                # a tool with no parameters may arrive as "", "null" or "{}": all mean "no arguments"
                args = json.loads(call.function.arguments or "{}")
                if args is None:
                    args = {}
                if not isinstance(args, dict):
                    raise json.JSONDecodeError("the arguments must be a JSON object", call.function.arguments, 0)
            except json.JSONDecodeError as e:
                args = call.function.arguments
                result = {"status": "error", "error": describe_invalid_arguments(name, args, e)}
                self._make_history_valid(call.id)
            else:
                print(f"\n[Turn {turn}] {name}({args})")
                blocked = self._hook_blocks(name, args, turn)
                result = {"status": "error", "error": blocked} if blocked else self.registry.call(name, args)

            # errors go back to the model as text so it can fix itself
            self._last_tool_failed = result["status"] == "error"
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
