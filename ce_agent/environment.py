"""
The sandbox environment: which image the commands run in, how much network they get, and the limits.

Where the settings come from (later wins):
  1. defaults for the project type, detected from its files (Cargo.toml -> rust, package.json -> node, ...)
  2. <repo>/.agent/env.json     (kept in .agent/, which is not part of the agent's copy of the project,
                                 so the model can neither read nor change its own sandbox policy)
  3. command-line flags         (--image=..., --network=registries, --memory=4g, ...)

Network modes:
  none        no network at all (the default; the safest)
  registries  only package registries (crates.io, pypi.org, ...) through an allowlist proxy; everything else is refused
  full        unrestricted internet access: whatever runs in the sandbox can send anything anywhere

The hosts listed for each project type are the ones I know those package managers use. Only the python one has been
tried end to end here; check the others against your own setup before you rely on them.
"""

import hashlib
import json
import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

from ce_agent.egress_proxy import HOST_PATTERN, normalize_host

NETWORK_MODES = ("none", "registries", "full")
DEFAULT_IMAGE = "python:3.11-slim"
CONFIG_PATH = Path(".agent") / "env.json"
PROXY_VARS = {"HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "NO_PROXY", "no_proxy"}

# name -> defaults. "markers" are the files that identify the project type, "cache_env" points the package
# manager's caches at /cache (a per-project folder that survives between sessions).
PRESETS = {
    "python": {
        "image": DEFAULT_IMAGE, "memory": "512m", "pids_limit": 100, "command_timeout": 60,
        "hosts": ("pypi.org", "files.pythonhosted.org"),
        "markers": ("pyproject.toml", "requirements.txt", "setup.py", "setup.cfg", "Pipfile"),
        "env": {"PIP_DISABLE_PIP_VERSION_CHECK": "1"},
        "cache_env": {"PIP_CACHE_DIR": "/cache/pip", "PYTHONUSERBASE": "/cache/pyuser"},
    },
    "node": {
        "image": "node:20-slim", "memory": "2g", "pids_limit": 512, "command_timeout": 300,
        "hosts": ("registry.npmjs.org", "registry.yarnpkg.com"),
        "markers": ("package.json",),
        "env": {},
        "cache_env": {"npm_config_cache": "/cache/npm", "YARN_CACHE_FOLDER": "/cache/yarn"},
    },
    "rust": {
        "image": "rust:1-slim", "memory": "4g", "pids_limit": 512, "command_timeout": 900,
        "hosts": ("index.crates.io", "crates.io", "static.crates.io"),
        "markers": ("Cargo.toml",),
        "env": {},
        # build output goes to the cache folder too, so it is neither committed with the project nor rebuilt every session
        "cache_env": {"CARGO_HOME": "/cache/cargo", "CARGO_TARGET_DIR": "/cache/target"},
    },
    "go": {
        "image": "golang:1.22", "memory": "2g", "pids_limit": 256, "command_timeout": 600,
        "hosts": ("proxy.golang.org", "sum.golang.org"),
        "markers": ("go.mod",),
        "env": {},
        "cache_env": {"GOPATH": "/cache/go", "GOCACHE": "/cache/go-build", "GOMODCACHE": "/cache/go/pkg/mod"},
    },
}
DETECTION_ORDER = ("rust", "go", "node", "python")        # a project with several markers gets the first match


class EnvironmentConfigError(RuntimeError):
    """A bad setting. A RuntimeError, so the command line prints it as 'Error: ...'."""


@dataclass(frozen=True)
class Environment:
    preset: str = "python"
    image: str = DEFAULT_IMAGE
    network: str = "none"
    allow_hosts: tuple = ()               # extra hosts for "registries" mode, on top of the project type's own
    memory: str = "512m"
    pids_limit: int = 100
    command_timeout: int = 60             # seconds a single command may run
    env: dict = field(default_factory=dict)
    cache: bool = True                    # mount a per-project cache folder at /cache
    allow_private: bool = False           # registries mode: let allowed names resolve to private addresses
    egress_network: str = "bridge"        # the Docker network the proxy uses to reach the internet
    source: str = "defaults"              # where these settings came from, for the banner

    # ---------- derived ----------
    def hosts(self):
        """The hosts the proxy allows (registries mode)."""
        seen = []
        for host in (*PRESETS[self.preset]["hosts"], *self.allow_hosts):
            host = normalize_host(host)
            if host not in seen:
                seen.append(host)
        return tuple(seen)

    def container_env(self):
        """Environment variables for the container (not counting the proxy settings)."""
        preset = PRESETS[self.preset]
        values = dict(preset["env"])
        if self.cache:
            values.update(preset["cache_env"])
        values.update(self.env)
        return values

    def describe(self, unsafe=False):
        if unsafe:
            return "Sandbox: NONE (--unsafe-local): commands run directly on this machine"
        network = {"none": "no network", "full": "FULL internet access",
                   "registries": "package registries only (" + ", ".join(self.hosts()) + ")"}[self.network]
        return (f"Sandbox: {self.image} | network: {network} | memory {self.memory}, {self.pids_limit} processes, "
                f"{self.command_timeout}s per command | cache: {'on' if self.cache else 'off'} | from {self.source}")

    def prompt_text(self, unsafe=False):
        """What the model is told about its surroundings, so it does not waste turns on things that cannot work."""
        if unsafe:
            where = "Commands run directly on the developer's machine, not in a container."
        elif self.network == "none":
            where = (f"Commands run in a Docker container (image {self.image}) with NO internet access, so packages cannot "
                     "be downloaded or installed. Only what is already in the image and the project is available.")
        elif self.network == "registries":
            where = (f"Commands run in a Docker container (image {self.image}). The only internet access is through a proxy "
                     f"that allows just these hosts: {', '.join(self.hosts())}. Everything else is refused, so use the "
                     "package manager's normal registry and nothing else.")
        else:
            where = f"Commands run in a Docker container (image {self.image}) with full internet access."
        limits = f" Each command is stopped after {self.command_timeout} seconds."
        cache = ""
        if self.cache and not unsafe:
            cache = (" Folders under /cache persist between sessions; package and build caches are already set up there "
                     "(do not put project files in /cache).")
        return where + limits + cache


# ======================================================================
# validation
# ======================================================================
def validate(env):
    """Check every setting; raise EnvironmentConfigError with a message that says what to change."""
    def bad(message):
        raise EnvironmentConfigError(message)

    if env.preset not in PRESETS:
        bad(f"unknown preset {env.preset!r}; choose one of: {', '.join(PRESETS)}")
    if env.network not in NETWORK_MODES:
        bad(f"network must be one of {', '.join(NETWORK_MODES)}, not {env.network!r}")
    if not isinstance(env.image, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/:@-]*", env.image):
        bad(f"not a valid image name: {env.image!r}")
    if not isinstance(env.memory, str) or not re.fullmatch(r"[1-9]\d*[bkmg]?", env.memory):
        bad(f"memory must look like 512m or 4g, not {env.memory!r}")
    if not isinstance(env.pids_limit, int) or isinstance(env.pids_limit, bool) or not 1 <= env.pids_limit <= 32768:
        bad(f"pids_limit must be a whole number from 1 to 32768, not {env.pids_limit!r}")
    if not isinstance(env.command_timeout, int) or isinstance(env.command_timeout, bool) or not 1 <= env.command_timeout <= 86400:
        bad(f"command_timeout must be a number of seconds from 1 to 86400, not {env.command_timeout!r}")
    for host in env.allow_hosts:
        if not isinstance(host, str) or not HOST_PATTERN.match(normalize_host(host)):
            bad(f"not a valid host name or '*.domain' pattern: {host!r}")
    if not isinstance(env.env, dict):
        bad("env must be an object of NAME: value")
    for name, value in env.env.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            bad(f"not a valid environment variable name: {name!r}")
        if name in PROXY_VARS:
            bad(f"{name} is set by the sandbox itself; remove it from env")
        if not isinstance(value, str):
            bad(f"the value of env.{name} must be a string")
    if not isinstance(env.egress_network, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", env.egress_network):
        bad(f"not a valid Docker network name: {env.egress_network!r}")
    for flag in ("cache", "allow_private"):
        if not isinstance(getattr(env, flag), bool):
            bad(f"{flag} must be true or false")
    return env


# ======================================================================
# finding the settings
# ======================================================================
def detect_preset(*folders):
    """The project type, from the files in these folders. Returns (preset, the file that decided it) or (None, None)."""
    for name in DETECTION_ORDER:
        for folder in folders:
            for marker in PRESETS[name]["markers"]:
                if folder and (Path(folder) / marker).exists():
                    return name, marker
    return None, None


def from_preset(name, source="defaults"):
    preset = PRESETS[name]
    return Environment(preset=name, image=preset["image"], memory=preset["memory"], pids_limit=preset["pids_limit"],
                       command_timeout=preset["command_timeout"], source=source)


def load_config(*folders):
    """The first .agent/env.json found in these folders. Returns (settings dict, path) or ({}, None)."""
    for folder in folders:
        path = Path(folder) / CONFIG_PATH if folder else None
        if path and path.is_file():
            try:
                data = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError) as e:
                raise EnvironmentConfigError(f"cannot read {path}: {e}")
            if not isinstance(data, dict):
                raise EnvironmentConfigError(f"{path} must contain a JSON object")
            allowed = {"preset", "image", "network", "allow_hosts", "memory", "pids_limit", "command_timeout", "env",
                       "cache", "allow_private", "egress_network"}
            unknown = sorted(set(data) - allowed)
            if unknown:
                raise EnvironmentConfigError(f"unknown setting(s) in {path}: {', '.join(unknown)}. Allowed: {', '.join(sorted(allowed))}")
            return data, path
    return {}, None


def cache_dir_for(top):
    """A cache folder for this project on the host (owned by you), mounted at /cache in the container."""
    base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "ce-agent" / hashlib.sha1(str(Path(top).resolve()).encode()).hexdigest()[:12]


def resolve_environment(repo_path, overrides=None, top=None):
    """Work out the environment for a project. `overrides` are the settings from command-line flags."""
    folders = [repo_path] + ([top] if top and Path(top).resolve() != Path(repo_path).resolve() else [])
    detected, marker = detect_preset(*folders)
    config, config_path = load_config(*folders)
    overrides = {k: v for k, v in (overrides or {}).items() if v is not None}

    name = overrides.get("preset") or config.get("preset") or detected or "python"
    if name not in PRESETS:
        raise EnvironmentConfigError(f"unknown preset {name!r}; choose one of: {', '.join(PRESETS)}")
    parts = [f"{name} ({marker})" if (name == detected and not overrides.get("preset") and not config.get("preset")) else name]
    if config_path:
        parts.append(CONFIG_PATH.as_posix())
    if overrides:
        parts.append("flags")

    env = from_preset(name, source=" + ".join(parts))
    for source in (config, overrides):
        settings = {k: v for k, v in source.items() if k != "preset"}
        if "allow_hosts" in settings:
            if not isinstance(settings["allow_hosts"], (list, tuple)):
                raise EnvironmentConfigError("allow_hosts must be a list of host names")
            settings["allow_hosts"] = tuple(settings["allow_hosts"])
        if "env" in settings:
            if not isinstance(settings["env"], dict):
                raise EnvironmentConfigError("env must be an object of NAME: value")
            settings["env"] = {**env.env, **settings["env"]}
        env = replace(env, **settings)
    return validate(env)


def flags_to_overrides(flags):
    """Command-line flags -> settings. Unknown flags are ignored here (main() handles its own)."""
    overrides = {}
    for flag in flags:
        name, _, value = flag.partition("=")
        if name == "--network":
            overrides["network"] = value or "full"                    # a bare --network has always meant "full"
        elif name == "--image" and value:
            overrides["image"] = value
        elif name == "--memory" and value:
            overrides["memory"] = value
        elif name == "--preset" and value:
            overrides["preset"] = value
        elif name == "--timeout" and value:
            if not value.isdigit():
                raise EnvironmentConfigError(f"--timeout must be a number of seconds, not {value!r}")
            overrides["command_timeout"] = int(value)
        elif name == "--pids" and value:
            if not value.isdigit():
                raise EnvironmentConfigError(f"--pids must be a number, not {value!r}")
            overrides["pids_limit"] = int(value)
        elif name == "--allow-host" and value:
            overrides["allow_hosts"] = (*overrides.get("allow_hosts", ()), value)
        elif name == "--no-cache":
            overrides["cache"] = False
        elif name == "--allow-private":
            overrides["allow_private"] = True
    return overrides
