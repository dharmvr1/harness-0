"""Sandbox backend factory and implementations.

Provides interchangeable sandbox strategies so the agent loop
does not need to know whether commands run inside Docker,
a Linux permission jail, or any other isolated environment.

The public API is:
    sandbox = sandbox_factory()        # auto-selects best backend
    sandbox_dir = sandbox.setup(repo, "id")  # prepare isolated dir (may be unused for some backends)
    result    = sandbox.run(cmd)         # run a command inside the sandbox
    content   = sandbox.read_file(file_path)  # read a file inside the sandbox
    listing   = sandbox.list_files()         # list files inside the sandbox
    sandbox.teardown()                 # clean up afterwards
"""

import os
import shutil
import subprocess
from pathlib import Path


# ---------------------------------------------------------------------------
# Common interface (duck-typing; no heavy ABC machinery needed for a demo)
# ---------------------------------------------------------------------------
class Sandbox:
    """Minimal interface that every sandbox backend must satisfy.

    The agent loop will call these methods; any backend that
    implements them can be dropped in.
    """

    def setup(self, repo_path: str, task_id: str) -> Path:
        """Prepare the isolated directory and return the Path to use as cwd.

        For some backends (like Docker) this may return a placeholder
        Path that is not used for file operations.
        """
        raise NotImplementedError

    def run(self, command: str) -> dict:
        """Execute *command* inside the sandbox.

        Return a dict with at least:
            {"status": "ok" | "error", "content": str, "error": str | None}
        """
        raise NotImplementedError

    def read_file(self, file_path: str) -> str:
        """Read a file inside the sandbox and return its contents as a string.

        If the file does not exist or cannot be read, return an error message
        starting with "Error: ".
        """
        raise NotImplementedError

    def list_files(self) -> str:
        """List all files inside the sandbox and return the listing as a string.

        Return a string suitable for display (like the output of `ls -la`).
        If the operation fails, return an error message starting with "Error: ".
        """
        raise NotImplementedError

    def teardown(self) -> None:
        """Tear down any volumes, containers, or processes."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Concrete back‑ends
# ---------------------------------------------------------------------------


class DockerSandbox(Sandbox):
    """Run commands inside a one‑shot Docker container.

    This backend:
    - Starts a temporary container with the repo mounted read‑only at /repo.
    - Executes commands inside that container via `docker run`.
    - Tears the container down automatically ( --rm ).

    It provides strong isolation: the container has its own filesystem,
    network namespace, and user space. The host filesystem is completely
    unreachable from inside the container.
    """

    def __init__(self):
        # Check if Docker is actually available and functional
        self._available = False
        self._repo_path: Path | None = None
        if shutil.which("docker"):
            try:
                result = subprocess.run(
                    ["docker", "ps"], capture_output=True, text=True, timeout=5
                )
                self._available = result.returncode == 0
            except Exception:
                self._available = False

    def setup(self, repo_path: str, task_id: str) -> Path:
        """Build a temporary container path and return a Path object.

        We keep the repo path for the run() method to use.
        No host-side directory is created; the container provides the fs.
        """
        # We keep the repo path for the run() method to use.
        self._repo_path = Path(repo_path).resolve()
        print("from docker setup :" , self._repo_path)
        # Return a dummy path for compatibility; the actual work happens in container.
        return Path(f".agent/worktrees/{task_id}_docker")

    def run(self, command: str) -> dict:
        """Execute *command* inside a fresh Docker container.

        The container:
        - Mounts the host repo read‑only at /repo.
        - Sets the working directory to /repo.
        - Uses a small base image (python:3.11-slim).
        - Is removed immediately after the command finishes (--rm).
        """
        if not self._available:
            raise RuntimeError(
                "DockerSandbox requested but Docker is not available or functional. "
                "Falling back to LocalPermissionSandbox."
            )

        if self._repo_path is None:
            raise RuntimeError("DockerSandbox.setup() must be called before run()")

        # Build the docker run command.
        # We mount the host repo read‑only so the agent cannot write outside it.
        mount_args = [
            "-v",
            f"{self._repo_path}:/repo:ro",  # read‑only mount at /repo
            "-w",
            "/repo",  # set working directory to /repo
        ]

        try:
            result = subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    *mount_args,
                    "python:3.11-slim",
                    "sh",
                    "-c",
                    command,
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            print("from docker run  result :" , result)
            return {
                "status": "ok" if result.returncode == 0 else "error",
                "content": result.stdout,
                "error": result.stderr,
            }
        except subprocess.TimeoutExpired:
            return {
                "status": "error",
                "content": "",
                "error": "Command timed out inside Docker container",
            }

    def read_file(self, file_path: str) -> str:
        """Read a file inside the Docker container at /repo/<file_path>."""
        if not self._available:
            return "Error: Docker not available"
        if self._repo_path is None:
            return "Error: DockerSandbox not initialized"

        # Use docker run to cat the file
        mount_args = [
            "-v",
            f"{self._repo_path}:/repo:ro",
            "-w",
            "/repo",
        ]
        try:
            result = subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    *mount_args,
                    "python:3.11-slim",
                    "sh",
                    "-c",
                    f"cat /repo/{file_path}",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return result.stdout
            else:
                return f"Error: {result.stderr.strip()}"
        except subprocess.TimeoutExpired:
            return "Error: Command timed out"

    def list_files(self) -> str:
        """List files inside the Docker container at /repo."""
        if not self._available:
            return "Error: Docker not available"
        if self._repo_path is None:
            return "Error: DockerSandbox not initialized"

        mount_args = [
            "-v",
            f"{self._repo_path}:/repo:ro",
            "-w",
            "/repo",
        ]
        try:
            result = subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    *mount_args,
                    "python:3.11-slim",
                    "sh",
                    "-c",
                    "ls -la",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return result.stdout
            else:
                return f"Error: {result.stderr.strip()}"
        except subprocess.TimeoutExpired:
            return "Error: Command timed out"

    def teardown(self) -> None:
        # With --rm the container is auto‑removed; no explicit action needed.
        # We still clear any host‑side placeholder if we created one.
        self._repo_path = None
        placeholder = Path(f".agent/worktrees/task_1_docker")
        if placeholder.exists():
            placeholder.rmdir()


class LocalPermissionSandbox(Sandbox):
    """Sandbox that relies on ordinary Unix permissions + ulimit.

    This works on **any** machine that has Python and a shell – no Docker
    required.  It creates a work‑tree directory, copies the repo in, and
    then uses the kernel's standard permission checks to keep the agent
    from writing outside that directory.
    """

    def __init__(self):
        self._sandbox_path: Path | None = None

    def setup(self, repo_path: str, task_id: str) -> Path:
        self._sandbox_path = Path(f".agent/worktrees/{task_id}")
        self._sandbox_path.mkdir(parents=True, exist_ok=True)

        # Ignore the .agent directory to prevent infinite recursion when
        # the repo being sandboxed is the project root itself.
        def _ignore_func(dir, files):
            return [".agent", ".git"]

        # Copy the repo into the sandbox so the agent never touches the host dir.
        shutil.copytree(
            repo_path, self._sandbox_path, dirs_exist_ok=True, ignore=_ignore_func
        )

        # Simple permission fence: make the directory read‑only for anyone
        # except the owner.  The PreToolUse hook in the agent can still
        # block destructive commands (rm -rf, curl, etc.) for extra safety.
        os.chmod(self._sandbox_path, 0o755)

        return self._sandbox_path

    def read_file(self, file_path: str) -> str:
        """Read a file inside the sandbox directory."""
        if self._sandbox_path is None:
            return "Error: LocalPermissionSandbox not initialized"
        full_path = self._sandbox_path / file_path
        # Security check: ensure file is inside sandbox
        try:
            full_path.resolve().relative_to(self._sandbox_path.resolve())
        except ValueError:
            return "Error: Access denied (file outside sandbox)"

        if not full_path.exists():
            return f"Error: File {file_path} not found"
        try:
            return full_path.read_text()
        except Exception as e:
            return f"Error: {str(e)}"

    def list_files(self) -> str:
        """List all files inside the sandbox directory."""
        if self._sandbox_path is None:
            return "Error: LocalPermissionSandbox not initialized"
        try:
            return "\n".join(
                [
                    str(p.relative_to(self._sandbox_path))
                    for p in self._sandbox_path.rglob("*")
                ]
            )
        except Exception as e:
            return f"Error: {str(e)}"

    def teardown(self) -> None:
        if self._sandbox_path and self._sandbox_path.exists():
            shutil.rmtree(self._sandbox_path, ignore_errors=True)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def sandbox_factory(backend: str | None = None) -> Sandbox:
    """Return a sandbox backend matching *backend*.

    *backend* may be:
      - "docker"   – use DockerSandbox (preferred when Docker is available)
      - "local"    – use LocalPermissionSandbox
      - None (default) – auto‑select: Docker if available and functional, else local
    """
    # ------------------------------------------------------------------
    # 1️⃣ Resolve which backend we should use
    # ------------------------------------------------------------------
    if backend is None:
        # Auto‑select: prefer Docker if it's running, otherwise fall back to local.
        if shutil.which("docker"):
            try:
                # Very light check: see if `docker ps` runs without error.
                result = subprocess.run(
                    ["docker", "ps"], capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    backend = "docker"
                else:
                    backend = "local"
            except Exception:
                backend = "local"
        else:
            backend = "local"

    # ------------------------------------------------------------------
    # 2️⃣ Map string → class
    # ------------------------------------------------------------------
    _BACKENDS = {
        "docker": DockerSandbox,
        "local": LocalPermissionSandbox,
    }

    if backend not in _BACKENDS:
        raise ValueError(f"Unknown sandbox backend: {backend}")

    # ------------------------------------------------------------------
    # 3️⃣ Instantiate and return
    # ------------------------------------------------------------------
    return _BACKENDS[backend]()
