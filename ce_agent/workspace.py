"""
Workspace: where the agent does its work.

  - The agent works in a git worktree on its own branch (agent/<id>), a copy that starts from a SNAPSHOT of your
    folder as it is right now: committed files, uncommitted edits and new files (minus what .gitignore hides).
    So the agent sees what you see, and your real folder is never touched while it works.
  - Shell commands run inside a Docker container that mounts that copy
    (or directly on your machine in --unsafe-local mode).
  - Nothing reaches your folder until you call accept(), which applies only the agent's changes to your files
    as a patch. They arrive as ordinary uncommitted edits; you commit them when you like.
  - The agent's work is never thrown away by accident: cleanup() keeps the branch if anything is still pending.

All git commands run on the host, never inside the container.
"""

import os
import shutil
import signal
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

from ce_agent.egress import EgressProxy
from ce_agent.environment import DEFAULT_IMAGE, Environment, cache_dir_for  # noqa: F401  (DEFAULT_IMAGE stays importable from here)

RUN_TICK_SECONDS = 1.0   # how often run() calls its on_tick callback while a command is still running
NOTES_FILE = "AGENTS.md"
NOTES_MAX_CHARS = 2000   # the notes are pasted into every prompt, so they must stay short
NOTE_MAX_CHARS = 300


def git(*args, cwd, env=None):
    """Run a git command, return its output, raise with git's message on failure."""
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout


def commit_all(path, message):
    """Commit everything in a worktree. Returns True if there was something to commit."""
    git("add", "-A", cwd=path)
    if git("status", "--porcelain", cwd=path):
        git("-c", "user.name=ce-agent", "-c", "user.email=ce-agent@localhost",
            "commit", "-m", message, cwd=path)
        return True
    return False


def ensure_repo(repo_path: str, ask=None):
    """Make sure repo_path is inside a git repo (the agent's copy is a git worktree). A missing folder or a folder
    that is not a repo yet is set up after asking. No commit is made and none of your files are touched."""
    ask = ask or input
    path = Path(repo_path).expanduser().resolve()

    def confirm(question):
        if ask(f"{question} [y/N] ").strip().lower() != "y":
            raise RuntimeError("Cancelled. The agent needs a git repo to work in.")

    if not path.exists():
        confirm(f"{path} does not exist. Create it as a new git project?")
        path.mkdir(parents=True)
        git("init", cwd=path)
        print(f"Created {path} and ran git init.")
    elif subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=path, capture_output=True).returncode != 0:
        confirm(f"{path} is not a git repo. Run git init here? (No commit is made, your files stay as they are.)")
        git("init", cwd=path)
        print(f"Ran git init in {path}.")


class Workspace:
    def __init__(self, repo_path: str, unsafe: bool = False, network=False, image: str = DEFAULT_IMAGE, environment=None):
        """`environment` (see environment.py) says which image, network mode and limits the container gets.
        The older `network` / `image` arguments still work when no environment is given:
        network may be True (full), False (none) or the name of a mode."""
        self.repo_path = Path(repo_path).resolve()
        self.unsafe = unsafe      # True = run commands on the host, no Docker
        if environment is None:
            mode = network if isinstance(network, str) else ("full" if network else "none")
            environment = Environment(image=image, network=mode)
        self.environment = environment
        self.image = environment.image
        self.command_timeout = environment.command_timeout    # seconds one run_shell command may take
        self._egress = None       # the proxy (network mode "registries"), set in _start_container()

        self.task_id = uuid.uuid4().hex[:8]
        self.branch = f"agent/{self.task_id}"
        self.container = None     # container name, set in start()
        self.worktree = None      # the checkout the agent works in
        self.workdir = None       # worktree + the sub-folder we were pointed at
        # Safe by default: the branch survives cleanup() unless discard() or a successful accept() says otherwise
        self.keep_branch = True
        self.last_sync_changes = []   # files the latest sync() brought in, as "M\tpath" / "A\tpath" / "D\tpath"
        self.prefix = ""              # the sub-folder we were pointed at, e.g. "sandbox_testing/"
        self.attached = False         # True: a view of someone else's worktree (see attach())
        self._running = set()         # commands started by run() that have not finished yet

    # ------------------------------------------------------------------
    # start / cleanup
    # ------------------------------------------------------------------
    @classmethod
    def attach(cls, worktree, prefix="", container=None, command_timeout=None):
        """A Workspace over an EXISTING worktree (and container), for the MCP server process.
        Only the file and command operations work: the harness owns the lifecycle (start, sync, review,
        accept, cleanup), so start() and cleanup() are refused here. Without a container, commands run on the host."""
        ws = cls(str(worktree), unsafe=container is None)
        if command_timeout:
            ws.command_timeout = command_timeout
        ws.worktree = Path(worktree).resolve()
        ws.prefix = prefix
        ws.workdir = ws.worktree / prefix
        ws.container = container
        ws.attached = True
        return ws

    def start(self):
        if self.attached:
            raise RuntimeError("An attached workspace is owned by the harness; it cannot be started here")
        # Find the git repo that contains repo_path (it may be a sub-folder of it)
        top = Path(git("rev-parse", "--show-toplevel", cwd=self.repo_path).strip())
        prefix = git("rev-parse", "--show-prefix", cwd=self.repo_path).strip()  # e.g. "sandbox_testing/"
        self.top = top
        self.prefix = prefix

        # Hide .agent/ from git for this repo only (not committed), so worktrees never look like "changes"
        exclude = top / git("rev-parse", "--git-path", "info/exclude", cwd=top).strip()
        exclude.parent.mkdir(parents=True, exist_ok=True)
        if ".agent/" not in (exclude.read_text() if exclude.exists() else ""):
            with open(exclude, "a") as f:
                f.write("\n.agent/\n")

        if not self.unsafe and not self._docker_ok():
            raise RuntimeError("Docker is not available. Start Docker, or pass --unsafe-local "
                               "to run commands directly on your machine.")

        # The agent's copy starts from a snapshot of your folder as it is now (uncommitted work included)
        head = subprocess.run(["git", "rev-parse", "--verify", "-q", "HEAD"], cwd=top,
                              capture_output=True, text=True).stdout.strip() or None   # None: repo has no commits yet
        self.base = head or ""
        self.uncommitted = git("status", "--porcelain", "--untracked-files=all", cwd=top).rstrip()
        self.mark = self._snapshot(head)   # the state of your folder that the agent's copy is based on
        self.worktree = top / ".agent" / "worktrees" / self.task_id
        git("worktree", "add", "-b", self.branch, str(self.worktree), self.mark, cwd=top)
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
        env = self.environment
        self.container = f"ce_agent_{self.task_id}"
        cmd = [
            "docker", "run", "-d", "--name", self.container,
            "--label", "ce_agent=true", "--label", f"ce_agent_repo={self.top}",  # so leftovers can be found later
            "--init",                                      # reaps zombie processes
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--user", f"{os.getuid()}:{os.getgid()}",      # files we create belong to you, not root
            "--memory", env.memory, "--pids-limit", str(env.pids_limit),
            "-e", "HOME=/tmp", "-e", "PYTHONDONTWRITEBYTECODE=1",
            "-v", f"{self.worktree}:/repo",
            "-w", f"/repo/{prefix}",
        ]
        for name, value in env.container_env().items():
            cmd += ["-e", f"{name}={value}"]
        if env.cache:                                      # package and build caches that survive between sessions
            cache = cache_dir_for(self.top)
            cache.mkdir(parents=True, exist_ok=True)       # created by us, so it belongs to you, not to root
            cmd += ["-v", f"{cache}:/cache"]

        if env.network == "none":
            cmd += ["--network", "none"]
        elif env.network == "registries":
            self._egress = EgressProxy(self.task_id, self.top, env.hosts(), env.allow_private, env.egress_network)
            self._egress.start()
            cmd += ["--network", self._egress.network]
            for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
                cmd += ["-e", f"{name}={self._egress.url}"]
            for name in ("NO_PROXY", "no_proxy"):
                cmd += ["-e", f"{name}=localhost,127.0.0.1,::1"]
        # "full": no network option at all, the container gets Docker's normal bridge network

        cmd += [self.image, "tail", "-f", "/dev/null"]    # keeps the container idle and alive
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("Could not start container: " + r.stderr.strip())

    def egress_entries(self):
        """What the proxy has allowed and refused so far (network mode "registries"); [] otherwise."""
        return self._egress.logs() if self._egress else []

    def cleanup(self):
        """Remove container and worktree. The branch stays if it holds work. Safe to call twice."""
        if self.attached:
            return   # not ours to remove
        # Save anything not yet committed first, so a crash or Ctrl-C never loses the agent's work
        if self.keep_branch and self.worktree and self.worktree.exists():
            try:
                commit_all(self.worktree, f"agent task {self.task_id}")
                if not self._has_pending():
                    self.keep_branch = False   # nothing on the branch that your folder doesn't have
            except Exception:
                pass
        if self.container:
            subprocess.run(["docker", "rm", "-f", self.container], capture_output=True)
            self.container = None
        if self._egress:                                   # after the sandbox, so nothing still uses the network
            self._egress.stop()
            self._egress = None
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
            raise ValueError("old_text is empty. Give the exact text you want to replace. To add code at the end of a file, use its last lines as old_text and put those same lines plus your new code in new_text.")
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

    # ------------------------------------------------------------------
    # project notes (memory across tasks): a normal file, so changes show up in the diff for review
    # ------------------------------------------------------------------
    def read_notes(self) -> str:
        notes = self.workdir / NOTES_FILE
        if not notes.exists():
            return ""
        text = notes.read_text().strip()
        if len(text) > NOTES_MAX_CHARS:
            text = text[:NOTES_MAX_CHARS] + "\n[notes cut: file is too long, shorten it]"
        return text

    def add_note(self, note: str) -> str:
        note = " ".join(note.split())   # one line
        if not note:
            raise ValueError("The note is empty.")
        if len(note) > NOTE_MAX_CHARS:
            raise ValueError(f"The note is {len(note)} characters; keep it under {NOTE_MAX_CHARS}.")
        path = self.workdir / NOTES_FILE
        current = path.read_text() if path.exists() else "# Project notes\n"
        if f"- {note}\n" in current + "\n":
            return "Already noted."
        if len(current) + len(note) > NOTES_MAX_CHARS:
            raise ValueError(f"{NOTES_FILE} is full ({len(current)} characters). "
                             "Shorten it with edit_file first, keeping only what matters.")
        path.write_text(current.rstrip("\n") + f"\n- {note}\n")
        return f"Saved to {NOTES_FILE}. It will be part of the diff you review."

    def diff_now(self) -> str:
        """Everything the agent changed that your folder doesn't have yet (including new files)."""
        git("add", "-A", cwd=self.worktree)
        return git("diff", self.mark, cwd=self.worktree) or "No changes yet."

    # ------------------------------------------------------------------
    # running commands
    # ------------------------------------------------------------------
    def run(self, command: str, timeout: int = None, on_tick=None) -> dict:
        """Run a shell command. Always returns exit_code, stdout and stderr.

        on_tick(seconds_so_far), if given, is called about once a second while the command is still running. It is
        how a long command reports progress, and if it raises (the client cancelled), the command is killed.
        The time limit defaults to the environment's command_timeout."""
        timeout = timeout or self.command_timeout
        if self.unsafe:
            args = ["timeout", str(timeout), "sh", "-c", command]
            # Clean environment: the child must not see our API keys (they would end up in the model's prompt)
            env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": "/tmp", "PYTHONDONTWRITEBYTECODE": "1"}
            kwargs = {"cwd": self.workdir, "env": env}
        else:
            args = ["docker", "exec", self.container, "timeout", str(timeout), "sh", "-c", command]
            kwargs = {}
        # start_new_session makes the command a process group of its own, so a cancelled command can be killed
        # together with everything it started (killing only the first process would leave the rest running).
        process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                   start_new_session=True, **kwargs)
        started = time.monotonic()
        self._running.add(process)
        try:
            while True:
                try:
                    # Calling communicate() again after a timeout loses no output (documented behaviour).
                    stdout, stderr = process.communicate(timeout=RUN_TICK_SECONDS if on_tick else None)
                    break
                except subprocess.TimeoutExpired:
                    on_tick(time.monotonic() - started)
        except BaseException:
            self._kill(process)          # includes the case where on_tick raised because the client cancelled
            process.communicate()
            raise
        finally:
            self._running.discard(process)
        return {"exit_code": process.returncode, "stdout": stdout, "stderr": stderr}  # 124 = timed out

    @staticmethod
    def _kill(process):
        """Kill a command together with everything it started (it leads its own process group)."""
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, AttributeError):
            process.kill()

    def kill_running(self):
        """Stop every command that run() started and that is still going. Called when the tool server shuts down,
        so that commands do not outlive the process that was running them."""
        for process in list(self._running):
            self._kill(process)

    # ------------------------------------------------------------------
    # finishing: review, then accept, keep or discard
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # keeping your folder and the agent's copy in sync
    # ------------------------------------------------------------------
    def _snapshot(self, parent):
        """A commit holding exactly what your folder looks like right now: committed files, uncommitted edits and
        new files, minus what .gitignore hides. It is built with a temporary copy of the git index, so your own
        index and files are never touched."""
        with tempfile.TemporaryDirectory() as tmp:
            index = Path(tmp) / "index"
            real_index = self.top / git("rev-parse", "--git-path", "index", cwd=self.top).strip()
            if real_index.exists():
                shutil.copy(real_index, index)   # keeps the file stat info, so only changed files get re-read
            env = {**os.environ, "GIT_INDEX_FILE": str(index)}
            git("add", "-A", cwd=self.top, env=env)
            tree = git("write-tree", cwd=self.top, env=env).strip()
        args = ["-c", "user.name=ce-agent", "-c", "user.email=ce-agent@localhost",
                "commit-tree", tree, "-m", "snapshot of your folder"]
        if parent:
            args += ["-p", parent]
        return git(*args, cwd=self.top).strip()

    def _tree(self, commit) -> str:
        return git("rev-parse", f"{commit}^{{tree}}", cwd=self.top).strip()

    def _has_pending(self) -> bool:
        """Does the agent's branch hold changes that your folder doesn't have? (When unsure: yes, to be safe.)"""
        r = subprocess.run(["git", "diff", "--quiet", self.mark, self.branch], cwd=self.top, capture_output=True)
        return r.returncode != 0

    def sync(self) -> str:
        """Before each prompt: if you changed your folder meanwhile (edited, committed, switched branch), bring
        those changes into the agent's copy, keeping the agent's pending work. Returns a message, or ''."""
        self.last_sync_changes = []
        try:
            snap = self._snapshot(self.mark)
        except RuntimeError as e:
            return f"[warning] could not look at your folder to sync: {e}"
        if self._tree(snap) == self._tree(self.mark):
            return ""   # your folder is as the agent's copy expects
        r = subprocess.run(["git", "-c", "user.name=ce-agent", "-c", "user.email=ce-agent@localhost",
                            "merge", "--no-edit", snap], cwd=self.worktree, capture_output=True, text=True)
        if r.returncode != 0:
            subprocess.run(["git", "merge", "--abort"], cwd=self.worktree, capture_output=True)
            return ("[warning] your recent changes conflict with the agent's pending changes, so they were NOT "
                    "brought into the agent's copy. The agent keeps working on the older snapshot.")
        self.last_sync_changes = git("diff", "--name-status", self.mark, snap, cwd=self.top).strip().splitlines()
        self.mark = snap
        return "The agent's copy now includes your latest changes."

    def finish(self) -> str:
        """Commit what the agent changed and return the pending diff: what the agent has that your folder doesn't."""
        commit_all(self.worktree, f"agent task {self.task_id}")
        return git("diff", self.mark, self.branch, cwd=self.top)

    def checkpoint(self, message: str) -> bool:
        """Save the agent's work so far as a commit on its branch (also keeps it safe if we crash).
        Returns True if the agent changed anything since the last checkpoint."""
        return commit_all(self.worktree, message)

    def stat(self) -> str:
        """Short summary of which files are pending (call after finish())."""
        return git("diff", "--stat", self.mark, self.branch, cwd=self.top)

    def accept(self) -> str:
        """Apply the agent's pending changes to your folder as a patch. They arrive as ordinary uncommitted edits.
        All or nothing: if the patch doesn't fit what is in your folder now, nothing is changed."""
        patch = subprocess.run(["git", "diff", "--binary", self.mark, self.branch], cwd=self.top,
                               capture_output=True).stdout
        if not patch:
            return "Nothing to apply."
        check = subprocess.run(["git", "apply", "--check", "--whitespace=nowarn"], cwd=self.top,
                               input=patch, capture_output=True)
        if check.returncode != 0:
            saved = self.top / ".agent" / "patches" / f"{self.task_id}.patch"
            saved.parent.mkdir(parents=True, exist_ok=True)
            saved.write_bytes(patch)
            return ("Could not apply the changes: they don't fit what is in your folder now. Nothing was changed, "
                    f"and the changes stay pending in the workspace.\n{check.stderr.decode().strip()}\n"
                    f"The patch is saved at {saved}")
        subprocess.run(["git", "apply", "--whitespace=nowarn"], cwd=self.top, input=patch, check=True, capture_output=True)
        self.mark = git("rev-parse", self.branch, cwd=self.top).strip()   # your folder now matches the branch
        return "Applied to your folder as uncommitted changes. Review them with git diff and commit when you like."

    def reject_pending(self):
        """Throw away everything the agent changed that is not in your folder yet: the agent's copy goes back to
        exactly what your folder had."""
        git("reset", "--hard", self.mark, cwd=self.worktree)
        git("clean", "-fd", cwd=self.worktree)

    def keep(self) -> str:
        """Leave the work on its branch for later."""
        self.keep_branch = True
        return (f"Kept as branch {self.branch}.\n  See the changes:  git diff {self.mark[:10]} {self.branch}\n"
                f"  Apply them:       git diff {self.mark[:10]} {self.branch} | git apply")

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
    networks = subprocess.run(["docker", "network", "ls", "-q", "--filter", f"label=ce_agent_repo={top}"],
                              capture_output=True, text=True).stdout.split()
    if networks:                                           # containers first: a network in use cannot be removed
        subprocess.run(["docker", "network", "rm", *networks], capture_output=True)
        lines.append(f"Removed {len(networks)} leftover sandbox network(s).")

    for wt in sorted((top / ".agent" / "worktrees").glob("*")):
        commit_all(wt, f"agent task {wt.name} (recovered)")
        subprocess.run(["git", "worktree", "remove", "--force", str(wt)], cwd=top, capture_output=True)
        lines.append(f"Saved and removed worktree {wt.name}.")
    git("worktree", "prune", cwd=top)

    branches = git("branch", "--list", "agent/*", "--format=%(refname:short)", cwd=top).split()
    lines.append("Branches with agent work: " + (", ".join(branches) if branches else "none"))
    return "\n".join(lines)
