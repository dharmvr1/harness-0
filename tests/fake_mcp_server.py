"""A scripted fake MCP server for testing the client's failure handling. Not a test file itself.

Usage: python fake_mcp_server.py MODE [COUNTER_FILE]
  ok          answer every request properly
  stray       before each real reply, send a reply to some other id, a notification and a garbage line
  crash_once  the first process to run (no COUNTER_FILE yet) dies without answering; later ones behave like `ok`
  hang        read one request and never answer
  noisy       write about 300 KB to stderr, then answer (an unread stderr pipe would block the server)
  version     answer -32022 unless the request uses protocol version 2026-07-28
"""
import json
import os
import sys
import time

mode = sys.argv[1]
counter = sys.argv[2] if len(sys.argv) > 2 else None


def send(message):
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def result_for(request):
    method = request.get("method")
    if method == "server/discover":
        return {"resultType": "complete", "supportedVersions": ["2026-07-28"], "capabilities": {"tools": {}},
                "ttlMs": 1000, "cacheScope": "public", "_meta": {"io.modelcontextprotocol/serverInfo": {"name": "fake", "version": "0"}}}
    if method == "tools/list":
        return {"resultType": "complete", "ttlMs": 1000, "cacheScope": "public",
                "tools": [{"name": "ping", "description": "Say pong", "inputSchema": {"type": "object", "properties": {}}}]}
    return {"resultType": "complete", "content": [{"type": "text", "text": "pong"}], "isError": False}


for line in sys.stdin:
    request = json.loads(line)
    if mode == "hang":
        time.sleep(60)
    elif mode == "crash_once" and not os.path.exists(counter):
        open(counter, "w").close()
        sys.exit(1)
    elif mode == "noisy":
        for _ in range(3000):
            sys.stderr.write("x" * 100 + "\n")
        sys.stderr.flush()
    elif mode == "stray":
        send({"jsonrpc": "2.0", "id": request["id"] + 1000, "result": {"resultType": "complete", "content": [], "isError": False}})
        send({"jsonrpc": "2.0", "method": "notifications/progress", "params": {"progress": 1}})
        sys.stdout.write("this is not json\n")
        sys.stdout.flush()
    elif mode == "version":
        requested = request["params"]["_meta"]["io.modelcontextprotocol/protocolVersion"]
        if requested != "2026-07-28":
            send({"jsonrpc": "2.0", "id": request["id"], "error": {"code": -32022, "message": "Unsupported protocol version",
                                                                  "data": {"supported": ["2026-07-28"], "requested": requested}}})
            continue
    send({"jsonrpc": "2.0", "id": request["id"], "result": result_for(request)})
