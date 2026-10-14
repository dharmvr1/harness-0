"""Runtime configuration: model endpoint, budget, sandbox defaults.

Model access is deliberately a single configurable endpoint so the backing
model can be swapped by changing a value, not a rewrite (capstone Exercise 1).
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

from .types import Budget, SandboxBackend, SandboxConfig


class ModelConfig(BaseModel):
    """Where and how the loop talks to a model."""

    base_url: str = ""
    api_key: str = ""
    model: str = ""
    # Anthropic-compatible headers; OpenRouter needs HTTP-Referer/X-Title.
    extra_headers: dict[str, str] = Field(default_factory=dict)
    timeout_s: float = 120.0


class Config(BaseModel):
    """Top-level runtime configuration."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    budget: Budget = Field(default_factory=Budget)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    worktrees_root: str = ""  # where sandbox worktrees live
    state_dir: str = ".agent"
    trace_dir: str = ".agent/traces"
    log_level: str = "INFO"


def _env_model() -> ModelConfig:
    """Build model config from environment (Anthropic-compatible by default)."""
    base_url = os.environ.get("ANTHROPIC_BASE_URL", "")
    api_key = os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
    model = (
        os.environ.get("ANTHROPIC_DEFAULT_SONNET_MODEL")
        or os.environ.get("ANTHROPIC_MODEL")
        or ""
    )
    extra: dict[str, str] = {}
    referer = os.environ.get("OPENROUTER_HTTP_REFERER")
    title = os.environ.get("OPENROUTER_X_TITLE")
    if referer:
        extra["HTTP-Referer"] = referer
    if title:
        extra["X-Title"] = title
    return ModelConfig(base_url=base_url, api_key=api_key, model=model, extra_headers=extra)


def load_config(workdir: str | Path | None = None) -> Config:
    """Load config: env vars first, then optional ``ce_agent.yaml`` override."""
    workdir = Path(workdir) if workdir else Path.cwd()
    cfg = Config(model=_env_model())
    cfg.worktrees_root = str(workdir / ".agent" / "worktrees")
    cfg.state_dir = str(workdir / ".agent")
    cfg.trace_dir = str(workdir / ".agent" / "traces")

    yaml_path = workdir / "ce_agent.yaml"
    if yaml_path.exists():
        try:
            import yaml

            data = yaml.safe_load(yaml_path.read_text()) or {}
            cfg = Config.model_validate({**cfg.model_dump(), **data})
        except Exception as exc:  # pragma: no cover - config is best-effort
            print(f"[config] warning: failed to read {yaml_path}: {exc}")
    return cfg


def default_sandbox(backend: SandboxBackend | str | None = None) -> SandboxConfig:
    """Convenience: a sandbox config with sensible local defaults."""
    if backend is None:
        backend = SandboxBackend.WORKTREE
    return SandboxConfig(backend=SandboxBackend(backend))