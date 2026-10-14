"""Hooks: the eight 2026 hook types.

Implemented in Phase 5. The agent core stays dumb; policy, telemetry, and
guardrails live here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from ..types import HookContext, HookResult


class HookRegistry:
    """Stub. Real implementation lands in Phase 5."""

    def __init__(self) -> None:
        raise NotImplementedError("HookRegistry is implemented in Phase 5.")

    def run(self, ctx: "HookContext") -> "HookResult":
        raise NotImplementedError