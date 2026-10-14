"""Sandbox: the isolation boundary.

Implemented in Phase 1. Each task runs inside a fresh worktree; the host
filesystem is unreachable from inside it. Teardown on any outcome.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from ..config import Config
    from ..types import SandboxConfig


class Sandbox:
    """Stub. Real implementation lands in Phase 1."""

    def __init__(self, config: "SandboxConfig | None" = None) -> None:
        self.config = config

    def __enter__(self) -> "Sandbox":
        raise NotImplementedError("Sandbox.__enter__ is implemented in Phase 1.")

    def __exit__(self, *exc: object) -> None:
        raise NotImplementedError("Sandbox.__exit__ is implemented in Phase 1.")