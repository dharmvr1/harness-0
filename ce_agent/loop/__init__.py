"""The plan / act / observe / recover loop.

Implemented in Phase 6. This package exists so the CLI can import it before
the engine is built; the stub raises a clear error until then.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from ..config import Config
    from ..types import LoopStatus


class AgentEngine:
    """Stub. Real implementation lands in Phase 6."""

    def __init__(self, config: "Config") -> None:
        self.config = config

    def run(self, repo_path: str, task: str) -> "LoopStatus":
        raise NotImplementedError(
            "AgentEngine.run is implemented in Phase 6 (the loop). "
            "Build the sandbox (Phase 1) and tool surface (Phase 3) first."
        )