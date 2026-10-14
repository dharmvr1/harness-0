"""Shared types for ce-agent.

One vocabulary for the whole system. Every surface — sandbox, MCP, tools,
hooks, loop, plan — speaks these types. Keeping them here means there is a
single definition of "what is a tool call", "what is a plan item", "what is a
hook", and the loop can compose any of them without each module inventing its
own shape.
"""

from __future__ import annotations

import enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Plan state (TodoWrite-style)
# ---------------------------------------------------------------------------
class PlanItemStatus(str, enum.Enum):
    """A single plan item's lifecycle state."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"


class PlanItem(BaseModel):
    """One line of the plan. The model rewrites the WHOLE state each turn."""

    id: str
    description: str
    status: PlanItemStatus = PlanItemStatus.PENDING
    notes: str = ""

    model_config = ConfigDict(use_enum_values=True)


class PlanState(BaseModel):
    """The full plan object, persisted to ``.agent/state.json``.

    Rewritten wholesale each turn (not mutated incrementally) so it is
    idempotent and resumable after a crash.
    """

    task: str
    items: list[PlanItem] = Field(default_factory=list)
    summary: str = ""


# ---------------------------------------------------------------------------
# Tool surface
# ---------------------------------------------------------------------------
class ToolCall(BaseModel):
    """A tool invocation the model requested."""

    id: str
    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """The outcome of one tool call."""

    tool_call_id: str
    content: str = ""
    is_error: bool = False
    # When the real output was truncated, this is the count of bytes dropped.
    truncated_bytes: int = 0


class ToolDefinition(BaseModel):
    """A tool's schema, exposed to the model."""

    name: str
    description: str
    input_schema: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Observation (what the loop feeds back to the model)
# ---------------------------------------------------------------------------
class Observation(BaseModel):
    """One turn's captured reality: stdout / stderr / exit code, summarized."""

    tool_call_id: str
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    duration_ms: float = 0.0
    is_error: bool = False


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------
class HookType(str, enum.Enum):
    """The eight 2026 hook types."""

    PRE_TOOL_USE = "PreToolUse"
    POST_TOOL_USE = "PostToolUse"
    SESSION_START = "SessionStart"
    SESSION_END = "SessionEnd"
    USER_PROMPT_SUBMIT = "UserPromptSubmit"
    NOTIFICATION = "Notification"
    STOP = "Stop"
    PRE_COMPACT = "PreCompact"


class HookDecision(str, enum.Enum):
    """What a hook decides about an event."""

    ALLOW = "allow"
    BLOCK = "block"
    MODIFY = "modify"


class HookContext(BaseModel):
    """Payload passed to every hook."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    hook_type: HookType
    session_id: str
    turn: int = 0
    data: dict[str, Any] = Field(default_factory=dict)


class HookResult(BaseModel):
    """Return value of a hook."""

    decision: HookDecision = HookDecision.ALLOW
    reason: str = ""
    modified_data: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Cost control
# ---------------------------------------------------------------------------
class Budget(BaseModel):
    """The three-layer cost ceiling."""

    per_turn_tokens: int = 8192
    per_session_tokens: int = 200_000
    per_session_usd: float = 5.0
    hard_turn_limit: int = 50
    compact_threshold_tokens: int = 150_000


class TokenUsage(BaseModel):
    """Running token accounting."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


# ---------------------------------------------------------------------------
# Sandbox
# ---------------------------------------------------------------------------
class SandboxBackend(str, enum.Enum):
    """Where the isolation boundary physically lives."""

    WORKTREE = "worktree"  # local git worktree in a temp dir (default)
    E2B = "e2b"           # E2B cloud sandbox (deferred)
    DAYTONA = "daytona"   # Daytona devcontainer (deferred)


class SandboxConfig(BaseModel):
    """Sandbox parameters."""

    backend: SandboxBackend = SandboxBackend.WORKTREE
    repo_path: str = ""
    worktrees_root: str = ""
    base_branch: str = "main"
    # Network egress policy.
    allow_network: bool = False
    # Cleanup on any outcome.
    cleanup_on_success: bool = True
    cleanup_on_failure: bool = True


# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------
class LoopStatus(str, enum.Enum):
    """Terminal states of the plan/act/observe/recover loop."""

    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BUDGET_EXHAUSTED = "budget_exhausted"
    TURN_LIMIT = "turn_limit"


class TurnRecord(BaseModel):
    """One iteration of the loop, persisted to the trace."""

    turn: int
    plan_state: PlanState | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    observations: list[Observation] = Field(default_factory=list)
    hook_decisions: list[dict[str, Any]] = Field(default_factory=list)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    elapsed_ms: float = 0.0
    error: str | None = None