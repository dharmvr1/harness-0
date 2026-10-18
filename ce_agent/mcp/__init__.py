"""A hand-written MCP (Model Context Protocol) server and client, no SDK.

protocol.py   the wire format: JSON-RPC 2.0 messages, the _meta fields, error codes (pure functions)
server.py     McpServer.handle(message) -> response   (no I/O at all, so it is easy to test)
stdio.py      the stdio transport: serve_stdio() for servers, StdioClient for clients
client.py     McpClient: discover(), list_tools(), call_tool()

Implements protocol revision 2026-07-28 (stateless: no initialize handshake, metadata on every request).
"""
