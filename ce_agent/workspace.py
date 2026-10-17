"""
Workspace: where the agent does its work.

  - Files live in a git worktree on a fresh branch (agent/<id>), so your real
    folder is never touched while the agent works.
  - Shell commands run inside a Docker container that mounts that worktree
    (or directly on your machine in --unsafe-local mode).
  - Nothing reaches your real branch until you call accept().

All git commands run on the host, never inside the container.
"""

import os
import subprocess
import uuid
from pathlib import Path

DEFAULT_IMAGE = "python:3.11-slim"


def git(*args, cwd):
    """Run a git command, return its output, raise with git's message on failure."""
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout


class Workspace:
    def __init__(self, repo_path: str, unsafe: bool = False, network: bool = False, image: str = DEFAULT_IMAGE):
        self.repo_path = Path(repo_path).resolve()
        self.unsafe = unsafe      # True = run commands on the host, no Docker
        self.network = network    # False = container has no internet
        self.image = image

        self.task_id = uuid.uuid4().hex[:8]
        self.branch = f"agent/{self.task_id}"
        self.container = None     # container name, set in start()
        self.worktree = None      # the checkout the agent works in
        self.workdir = None       # worktree + the sub-folder we were pointed at
        self.keep_branch = False  # True when we must not delete the branch

    # ------------------------------------------------------------------
    # start / cleanup
    # ------------------------------------------------------------------
    def start(self):
        # Find the git repo that contains repo_path (it may be a sub-folder of it)
        top = Path(git("rev-parse", "--show-toplevel", cwd=self.repo_path).strip())
        prefix = git("rev-parse", "--show-prefix", cwd=self.repo_path).strip()  # e.g. "sandbox_testing/"
        self.top = top

        # A worktree starts from the last commit, so uncommitted work would be invisible to the agent
        dirty = git("status", "--porcelain", "--", ".", cwd=self.repo_path)
        if dirty:
            raise RuntimeError("Uncommitted changes in this folder; the agent can't see them. "
                               "Commit or stash first:\n" + dirty)

        if not self.unsafe and not self._docker_ok():
            raise RuntimeError("Docker is not available. Start Docker, or pass --unsafe-local "
                               "to run commands directly on your machine.")

        self.base = git("rev-parse", "HEAD", cwd=top).strip()
        self.worktree = top / ".agent" / "worktrees" / self.task_id
        git("worktree", "add", "-b", self.branch, str(self.worktree), cwd=top)
        self.workdir = self.worktree / prefix

        if not self.unsafe:
            try:
                self._start_container(prefix)
            except Exception:
                self.cleanup()
                raise

    def _docker_ok(self):
        try:
            r = subprocess.run(["docker", "ps"], capture_output=True, timeout=10)
            return r.returncode == 0
        except Exception:
            return False

    def _start_container(self, prefix):
        self.container = f"ce_agent_{self.task_id}"
        cmd = [
            "docker", "run", "-d", "--name", self.container,
            "--label", "ce_agent=true",                    # so leftovers can be found later
            "--user", f"{os.getuid()}:{os.getgid()}",      # files we create belong to you, not root
            "--memory", "512m", "--pids-limit", "100",
            "-e", "HOME=/tmp", "-e", "PYTHONDONTWRITEBYTECODE=1",
            "-v", f"{self.worktree}:/repo",
            "-w", f"/repo/{prefix}",
        ]
        if not self.network:
            cmd += ["--network", "none"]
        cmd += [self.image, "tail", "-f", "/dev/null"]    # keeps the container idle and alive
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("Could not start container: " + r.stderr.strip())

    def cleanup(self):
        """Remove container, worktree and (unless we must keep it) the branch. Safe to call twice."""
        if self.container:
            subprocess.run(["docker", "rm", "-f", self.container], capture_output=True)
            self.container = None
        if self.worktree and self.worktree.exists():
            subprocess.run(["git", "worktree", "remove", "--force", str(self.worktree)],
                           cwd=self.top, capture_output=True)
        if not self.keep_branch:
            subprocess.run(["git", "branch", "-D", self.branch], cwd=self.top, capture_output=True)

    # ------------------------------------------------------------------
    # files (done on the host, directly in the worktree)
    # ------------------------------------------------------------------
    def _safe_path(self, file_path: str) -> Path:
        full = (self.workdir / file_path).resolve()
        if not full.is_relative_to(self.workdir.resolve()):
            raise ValueError(f"Access denied: {file_path} is outside the workspace")
        return full

    def read_file(self, file_path: str) -> str:
        return self._safe_path(file_path).read_text()

    def write_file(self, file_path: str, content: str) -> str:
        full = self._safe_path(file_path)
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content)
        return f"Wrote {len(content)} characters to {file_path}"

    def list_files(self) -> str:
        # tracked files + new files, skipping anything in .gitignore
        return git("ls-files", "--cached", "--others", "--exclude-standard", cwd=self.workdir)

    # ------------------------------------------------------------------
    # running commands
    # ------------------------------------------------------------------
    def run(self, command: str, timeout: int = 60) -> dict:
        """Run a shell command. Always returns exit_code, stdout and stderr."""
        if self.unsafe:
            args = ["timeout", str(timeout), "sh", "-c", command]
            kwargs = {"cwd": self.workdir}
        else:
            args = ["docker", "exec", self.container, "timeout", str(timeout), "sh", "-c", command]
            kwargs = {}
        r = subprocess.run(args, capture_output=True, text=True, **kwargs)
        return {"exit_code": r.returncode, "stdout": r.stdout, "stderr": r.stderr}  # 124 = timed out

    # ------------------------------------------------------------------
    # finishing: review, then accept or reject
    # ------------------------------------------------------------------
    def finish(self) -> str:
        """Commit whatever the agent changed onto its branch and return the diff against where it started."""
        git("add", "-A", cwd=self.worktree)
        if git("status", "--porcelain", cwd=self.worktree):
            git("-c", "user.name=ce-agent", "-c", "user.email=ce-agent@localhost",
                "commit", "-m", f"agent task {self.task_id}", cwd=self.worktree)
        return git("diff", self.base, self.branch, cwd=self.top)

    def accept(self) -> str:
        """Merge the agent's branch into whatever branch is checked out in your real repo."""
        try:
            git("merge", "--no-ff", "-m", f"Merge agent task {self.task_id}", self.branch, cwd=self.top)
        except RuntimeError as e:
            subprocess.run(["git", "merge", "--abort"], cwd=self.top, capture_output=True)
            self.keep_branch = True   # don't lose the work if the merge failed
            return f"Merge failed, nothing changed. Branch {self.branch} is kept so you can merge it by hand.\n{e}"
        return "Merged into your repo."

    def keep(self) -> str:
        """Leave the work on its branch for later."""
        self.keep_branch = True
        return f"Kept as branch {self.branch}. Review: git diff {self.base[:8]} {self.branch}"
