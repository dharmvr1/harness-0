"""
Progress reporting and cancellation for tools that run for a while.

A tool does not need to know about MCP. It just calls

    report_progress(3.0, message="command running for 3s")

If the client asked for progress (it put a progressToken in the request's _meta), a notifications/progress message
is sent. If the client has gone away (HTTP: it closed the connection; that IS the cancellation signal), the call
raises RequestCancelled so the tool can stop. Outside of an MCP request it does nothing, so the same tool code runs
unchanged in-process.

The "current request" is kept in a ContextVar, which is per thread: every HTTP request thread has its own.
"""

import contextlib
import contextvars

from ce_agent.mcp import protocol as p


class RequestCancelled(Exception):
    """The client no longer wants the answer. The server must stop and must not send anything more for this request."""


_current = contextvars.ContextVar("mcp_request", default=None)


class RequestContext:
    def __init__(self, progress_token=None, notify=None, is_cancelled=None):
        self.progress_token = progress_token     # a string or int chosen by the client, or None
        self.notify = notify                     # function that sends one notification to the client
        self.is_cancelled = is_cancelled         # function: has the client gone away?
        self._last = None                        # the spec says progress MUST increase

    def report(self, progress, total=None, message=None):
        if self.is_cancelled and self.is_cancelled():
            raise RequestCancelled()
        if self.progress_token is None or self.notify is None:
            return                                # the client did not ask for progress
        if self._last is not None and progress <= self._last:
            return                                # not an increase: skip it rather than break the rule
        params = {"progressToken": self.progress_token, "progress": progress}
        if total is not None:
            params["total"] = total
        if message:
            params["message"] = message
        self.notify(p.make_notification("notifications/progress", params))
        self._last = progress


@contextlib.contextmanager
def request_scope(context):
    token = _current.set(context)
    try:
        yield context
    finally:
        _current.reset(token)


def report_progress(progress, total=None, message=None):
    """Called by long-running tools. See the module docstring."""
    context = _current.get()
    if context is not None:
        context.report(progress, total, message)
