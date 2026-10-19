"""
A Server-Sent Events parser (the text/event-stream format), written to the WHATWG rules.

A stream is lines of text. A line is "field: value"; a line starting with ":" is a comment (keep-alives use them);
an empty line ends an event. Several "data:" lines in one event are joined with a newline. The MCP server puts one
JSON-RPC message in each event's data.
"""


def parse_sse(lines):
    """Yield {"event", "data", "id"} for every complete event in an iterable of text lines.

    An event that is not followed by its blank line when the stream ends was cut off, and is dropped, as the
    SSE rules say."""
    data, event, last_id = [], None, None
    for raw in lines:
        line = raw.rstrip("\r\n")
        if line == "":
            if data:
                yield {"event": event or "message", "data": "\n".join(data), "id": last_id}
            data, event = [], None
            continue
        if line.startswith(":"):
            continue                                   # a comment: keep-alive, ignored
        field, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]                          # one optional space after the colon
        if field == "data":
            data.append(value)
        elif field == "event":
            event = value
        elif field == "id" and "\0" not in value:
            last_id = value                            # the last id sticks across events
        # "retry" and unknown fields are ignored
