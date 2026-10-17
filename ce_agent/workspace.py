"""
Workspace: where the agent does its work.

  - Files live in a git worktree on a fresh branch (agent/<id>), so your real
    folder is never touched while the agent works.
  - Shell commands run inside a Docker container that mounts that worktree
    (or directly on your machine in --unsafe-local mode).
  - Nothing reaches your real branch until you call accept().
  - The agent's work is never thrown away by accident: cleanup() saves it on
    its branch unless you called discard() or accept() succeeded.

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


def commit_all(path, message):
    """Commit everything in a worktree (does nothing if there are no changes)."""
    git("add", "-A", cwd=path)
    if git("status", "--porcelain", cwd=path):
        git("-c", "user.name=ce-agent", "-c", "user.email=ce-agent@localhost",
            "commit", "-m", message, cwd=path)


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
        # Safe by default: the branch survives cleanup() unless discard() or a successful accept() says otherwise
        self.keep_branch = True

    # ------------------------------------------------------------------
    # start / cleanup
    # ------------------------------------------------------------------
    def start(self):
        # Find the git repo that contains repo_path (it may be a sub-folder of it)
        top = Path(git("rev-parse", "--show-toplevel", cwd=self.repo_path).strip())
        prefix = git("rev-parse", "--show-prefix", cwd=self.repo_path).strip()  # e.g. "sandbox_testing/"
        self.top = top

        # Hide .agent/ from git for this repo only (not committed), so worktrees never look like "changes"
        exclude = top / git("rev-parse", "--git-path", "info/exclude", cwd=top).strip()
        exclude.parent.mkdir(parents=True, exist_ok=True)
        if ".agent/" not in (exclude.read_text() if exclude.exists() else ""):
            with open(exclude, "a") as f:
                f.write("\n.agent/\n")

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
                self.keep_branch = False   # nothing was done yet, nothing to save
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
            "--label", "ce_agent=true", "--label", f"ce_agent_repo={self.top}",  # so leftovers can be found later
            "--init",                                      # reaps zombie processes
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
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
        """Remove container and worktree. The branch stays if it holds work. Safe to call twice."""
        # Save anything not yet committed first, so a crash or Ctrl-C never loses the agent's work
        if self.keep_branch and self.worktree and self.worktree.exists():
            try:
                commit_all(self.worktree, f"agent task {self.task_id}")
                if git("rev-parse", self.branch, cwd=self.top).strip() == self.base:
                    self.keep_branch = False   # branch has no work on it, don't leave an empty one
            except Exception:
                pass
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

    def read_file(self, file_path: str, start_line: int = 1, end_line: int | None = None) -> str:
        """Return the file with line numbers. start_line/end_line (1-based, inclusive) read just a part."""
        full = self._safe_path(file_path)
        if not full.exists():
            raise FileNotFoundError(f"{file_path} does not exist. Use list_files or search to find the right path.")
        try:
            lines = full.read_text().splitlines()
        except UnicodeDecodeError:
            raise ValueError(f"{file_path} is a binary file, it can't be read as text.")
        total = len(lines)
        end = total if end_line is None else min(end_line, total)
        if start_line < 1 or start_line > max(total, 1):
            raise ValueError(f"start_line {start_line} is out of range, the file has {total} lines.")
        body = "\n".join(f"{n:>5}  {lines[n - 1]}" for n in range(start_line, end + 1))
        return body + f"\n[lines {start_line}-{end} of {total}]"

    def edit_file(self, file_path: str, old_text: str, new_text: str) -> str:
        """Replace one exact piece of text in an existing file. old_text must match exactly once."""
        full = self._safe_path(file_path)
        if not full.exists():
            raise FileNotFoundError(f"{file_path} does not exist. Use write_file to create new files.")
        raw = full.read_bytes()
        if b"\r\n" in raw:
            raise ValueError(f"{file_path} uses Windows line endings (CRLF); edit_file can't edit it safely.")
        text = raw.decode()

        if old_text == "":
            raise ValueError("old_text is empty. Give the exact text you want to replace.")
        count = text.count(old_text)
        if count == 0:
            raise ValueError(f"old_text was not found in {file_path}. Use read_file to see the current content; "
                             "spaces, indentation and blank lines must match exactly.")
        if count > 1:
            raise ValueError(f"old_text matches {count} places in {file_path}. "
                             "Include more surrounding lines so it matches exactly once.")
        if old_text == new_text:
            raise ValueError("old_text and new_text are the same, nothing to change.")

        full.write_text(text.replace(old_text, new_text))
        line = text[:text.index(old_text)].count("\n") + 1
        return f"Edited {file_path} (replaced the text at line {line})"

    def write_file(self, file_path: str, content: str) -> str:
        full = self._safe_path(file_path)
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content)
        return f"Wrote {len(content)} characters to {file_path}"

    def list_files(self) -> str:
        # tracked files + new files, skipping anything in .gitignore
        return git("ls-files", "--cached", "--others", "--exclude-standard", cwd=self.workdir)

    def search(self, pattern: str, path: str = "", max_results: int = 50) -> str:
        """Search file contents with a regex (git grep: skips binary files and anything in .gitignore)."""
        args = ["git", "grep", "-n", "-I", "-E", "--untracked", "--exclude-standard", "-e", pattern]
        if path:
            args += ["--", os.path.relpath(self._safe_path(path), self.workdir)]
        r = subprocess.run(args, cwd=self.workdir, capture_output=True, text=True)
        if r.returncode == 1:
            return "No matches."
        if r.returncode != 0:
            raise ValueError(f"Search failed (is the pattern a valid regex?): {r.stderr.strip()}")
        lines = [l if len(l) <= 200 else l[:200] + "..." for l in r.stdout.splitlines()]
        out = "\n".join(lines[:max_results])
        if len(lines) > max_results:
            out += f"\n[{len(lines) - max_results} more matches not shown. Use a more specific pattern or path.]"
        return out

    def diff_now(self) -> str:
        """Everything changed so far (including new files), compared to where the task started."""
        git("add", "-A", cwd=self.worktree)
        return git("diff", self.base, cwd=self.worktree) or "No changes yet."

    # ------------------------------------------------------------------
    # running commands
    # ------------------------------------------------------------------
    def run(self, command: str, timeout: int = 60) -> dict:
        """Run a shell command. Always returns exit_code, stdout and stderr."""
        if self.unsafe:
            args = ["timeout", str(timeout), "sh", "-c", command]
            # Clean environment: the child must not see our API keys (they would end up in the model's prompt)
            env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": "/tmp", "PYTHONDONTWRITEBYTECODE": "1"}
            kwargs = {"cwd": self.workdir, "env": env}
        else:
            args = ["docker", "exec", self.container, "timeout", str(timeout), "sh", "-c", command]
            kwargs = {}
        r = subprocess.run(args, capture_output=True, text=True, **kwargs)
        return {"exit_code": r.returncode, "stdout": r.stdout, "stderr": r.stderr}  # 124 = timed out

    # ------------------------------------------------------------------
    # finishing: review, then accept, keep or discard
    # ------------------------------------------------------------------
    def finish(self) -> str:
        """Commit whatever the agent changed onto its branch and return the full diff against where it started."""
        commit_all(self.worktree, f"agent task {self.task_id}")
        return git("diff", self.base, self.branch, cwd=self.top)

    def stat(self) -> str:
        """Short summary of which files changed (call after finish())."""
        return git("diff", "--stat", self.base, self.branch, cwd=self.top)

    def accept(self) -> str:
        """Merge the agent's branch into whatever branch is checked out in your real repo."""
        try:
            git("merge", "--no-ff", "-m", f"Merge agent task {self.task_id}", self.branch, cwd=self.top)
        except RuntimeError as e:
            subprocess.run(["git", "merge", "--abort"], cwd=self.top, capture_output=True)
            return f"Merge failed, nothing changed. Branch {self.branch} is kept so you can merge it by hand.\n{e}"
        self.keep_branch = False   # the work is safely in your repo now
        return "Merged into your repo."

    def keep(self) -> str:
        """Leave the work on its branch for later."""
        self.keep_branch = True
        return f"Kept as branch {self.branch}. Review: git diff {self.base[:8]} {self.branch}"

    def discard(self):
        """Throw the work away: cleanup() will delete the branch."""
        self.keep_branch = False


def recover(repo_path: str) -> str:
    """Clean up after a crashed run (killed terminal, power cut...).
    Removes leftover containers, and saves any leftover worktree onto its branch before removing it.
    Only run this when no agent is running on the repo."""
    top = Path(git("rev-parse", "--show-toplevel", cwd=repo_path).strip())
    lines = []

    ids = subprocess.run(["docker", "ps", "-aq", "--filter", f"label=ce_agent_repo={top}"],
                         capture_output=True, text=True).stdout.split()
    if ids:
        subprocess.run(["docker", "rm", "-f", *ids], capture_output=True)
        lines.append(f"Removed {len(ids)} leftover container(s).")

    for wt in sorted((top / ".agent" / "worktrees").glob("*")):
        commit_all(wt, f"agent task {wt.name} (recovered)")
        subprocess.run(["git", "worktree", "remove", "--force", str(wt)], cwd=top, capture_output=True)
        lines.append(f"Saved and removed worktree {wt.name}.")
    git("worktree", "prune", cwd=top)

    branches = git("branch", "--list", "agent/*", "--format=%(refname:short)", cwd=top).split()
    lines.append("Branches with agent work: " + (", ".join(branches) if branches else "none"))
    return "\n".join(lines)
