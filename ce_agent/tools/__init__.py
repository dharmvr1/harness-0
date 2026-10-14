"""Tool surface: read_file, edit_file, ripgrep, tree_sitter_symbols, run_shell, git.

Implemented in Phase 3. Each tool returns truncated output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from ..types import ToolCall, ToolResult


class ToolDispatcher:
    """Stub. Real implementation lands in Phase 3."""

    def __init__(self) -> None:
        raise NotImplementedError("ToolDispatcher is implemented in Phase 3.")

    def dispatch(self, call: "ToolCall") -> "ToolResult":
        raise NotImplementedError