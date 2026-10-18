"""Uncommitted work: the agent sees it, accept applies a patch, and your folder and the agent's copy stay in sync."""
import subprocess

import pytest
from ce_agent.workspace import Workspace


def sh(*cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True).stdout


@pytest.fixture
def repo(tmp_path):
    sh("git", "init", "-q", "-b", "main", cwd=tmp_path)
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    (tmp_path / "old.py").write_text("old = True\n")
    (tmp_path / ".gitignore").write_text("secret.txt\n")
    sh("git", "add", "-A", cwd=tmp_path)
    sh("git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init", cwd=tmp_path)
    return tmp_path


def start(repo):
    ws = Workspace(str(repo), unsafe=True)
    ws.start()
    return ws

def finish_with(ws):
    ws.discard()
    ws.cleanup()


# ---------- the agent sees what you see ----------
def test_agent_sees_uncommitted_edits_new_files_and_deletions(repo):
    (repo / "calc.py").write_text("def add(a, b):\n    return a + b  # my WIP\n")     # modified, not committed
    (repo / "brand_new.py").write_text("new = 1\n")                                    # untracked
    (repo / "old.py").unlink()                                                          # deleted
    (repo / "secret.txt").write_text("hidden\n")                                       # ignored
    ws = start(repo)
    try:
        assert "my WIP" in ws.read_file("calc.py")
        assert "new = 1" in ws.read_file("brand_new.py")
        assert not (ws.workdir / "old.py").exists()
        assert not (ws.workdir / "secret.txt").exists()
        assert "my WIP" in ws.search("WIP")
        assert ws.uncommitted and "brand_new.py" in ws.uncommitted
    finally:
        finish_with(ws)


def test_starting_never_touches_your_files_index_or_history(repo):
    (repo / "calc.py").write_text("# staged edit\n")
    sh("git", "add", "calc.py", cwd=repo)
    (repo / "calc.py").write_text("# staged edit\n# plus unstaged\n")
    (repo / "untracked.py").write_text("u = 1\n")
    before = (sh("git", "status", "--porcelain", cwd=repo), sh("git", "rev-parse", "HEAD", cwd=repo),
              sh("git", "diff", "--cached", cwd=repo), (repo / "calc.py").read_text())
    ws = start(repo)
    ws.write_file("agent.py", "a = 1\n")
    ws.sync()
    finish_with(ws)
    after = (sh("git", "status", "--porcelain", cwd=repo), sh("git", "rev-parse", "HEAD", cwd=repo),
             sh("git", "diff", "--cached", cwd=repo), (repo / "calc.py").read_text())
    assert before == after                                  # same status, same HEAD, same staging, same file


def test_a_repo_with_no_commits_works(tmp_path):
    sh("git", "init", "-q", cwd=tmp_path)
    (tmp_path / "app.py").write_text("v1\n")
    ws = start(tmp_path)
    try:
        assert ws.read_file("app.py").startswith("    1  v1")
        ws.edit_file("app.py", "v1", "v2")
        assert "v2" in ws.finish()
        assert "Applied" in ws.accept()
    finally:
        finish_with(ws)
    assert (tmp_path / "app.py").read_text() == "v2\n"
    assert subprocess.run(["git", "rev-parse", "HEAD"], cwd=tmp_path, capture_output=True).returncode != 0   # still no commits


# ---------- accept = apply a patch to your files ----------
def test_pending_diff_contains_only_the_agents_work_not_your_uncommitted_edits(repo):
    (repo / "calc.py").write_text("def add(a, b):\n    return a + b  # my WIP\n")
    ws = start(repo)
    try:
        ws.write_file("agent.py", "a = 1\n")
        diff = ws.finish()
        assert "agent.py" in diff and "my WIP" not in diff and "calc.py" not in diff
    finally:
        finish_with(ws)


def test_accept_adds_the_agents_changes_on_top_of_your_uncommitted_work(repo):
    (repo / "calc.py").write_text("def add(a, b):\n    return a + b  # my WIP\n")
    head = sh("git", "rev-parse", "HEAD", cwd=repo)
    ws = start(repo)
    try:
        ws.edit_file("calc.py", "def add(a, b):", "def add(a, b):  # agent was here")
        ws.write_file("agent.py", "a = 1\n")
        ws.finish()
        assert "Applied" in ws.accept()
    finally:
        finish_with(ws)
    text = (repo / "calc.py").read_text()
    assert "my WIP" in text and "agent was here" in text                  # both survive
    assert (repo / "agent.py").exists()
    assert sh("git", "rev-parse", "HEAD", cwd=repo) == head               # no commit was made for you
    assert "calc.py" in sh("git", "status", "--porcelain", cwd=repo)


def test_accept_that_does_not_fit_changes_nothing_and_keeps_the_work_pending(repo):
    ws = start(repo)
    try:
        ws.edit_file("calc.py", "return a + b", "return a + b + 1")
        ws.finish()
        (repo / "calc.py").write_text("def add(a, b):\n    return 42\n")  # you edit the same line meanwhile
        message = ws.accept()
        assert "Could not apply" in message and "Nothing was changed" in message
        assert (repo / "calc.py").read_text() == "def add(a, b):\n    return 42\n"
        assert ws.finish()                                                # still pending
    finally:
        finish_with(ws)


def test_after_accept_nothing_is_pending(repo):
    ws = start(repo)
    try:
        ws.write_file("agent.py", "a = 1\n")
        ws.finish(); ws.accept()
        assert ws.finish() == "" and ws.sync() == ""                      # in sync, nothing to bring either way
        ws.write_file("second.py", "b = 2\n")
        diff = ws.finish()
        assert "second.py" in diff and "agent.py" not in diff
    finally:
        finish_with(ws)


# ---------- reject ----------
def test_reject_goes_back_to_your_snapshot_not_to_the_last_commit(repo):
    (repo / "calc.py").write_text("def add(a, b):\n    return a + b  # my WIP\n")
    ws = start(repo)
    try:
        ws.write_file("calc.py", "BROKEN\n")
        ws.write_file("agent.py", "a = 1\n")
        ws.finish()
        ws.reject_pending()
        assert "my WIP" in ws.read_file("calc.py")                        # your uncommitted edit is still there
        assert not (ws.workdir / "agent.py").exists()
        assert ws.finish() == ""
    finally:
        finish_with(ws)


# ---------- staying in sync while you keep working ----------
def test_edits_you_make_later_reach_the_agent(repo):
    ws = start(repo)
    try:
        (repo / "calc.py").write_text("def add(a, b):\n    return a + b  # edited later\n")
        (repo / "later_new.py").write_text("n = 1\n")
        (repo / "old.py").unlink()
        assert "latest changes" in ws.sync()
        assert "edited later" in ws.read_file("calc.py")
        assert (ws.workdir / "later_new.py").exists() and not (ws.workdir / "old.py").exists()
        assert ws.sync() == ""                                            # nothing new the second time
    finally:
        finish_with(ws)


def test_your_edits_and_the_agents_pending_work_are_combined(repo):
    ws = start(repo)
    try:
        ws.write_file("agent.py", "a = 1\n")
        ws.checkpoint("agent work")
        (repo / "mine.py").write_text("m = 1\n")
        assert "latest changes" in ws.sync()
        assert (ws.workdir / "agent.py").exists() and (ws.workdir / "mine.py").exists()
        diff = ws.finish()
        assert "agent.py" in diff and "mine.py" not in diff                # mine.py is yours, not pending
    finally:
        finish_with(ws)


def test_if_you_undo_an_uncommitted_edit_the_agent_sees_the_undo(repo):
    (repo / "calc.py").write_text("def add(a, b):\n    return a + b  # temporary\n")
    ws = start(repo)
    try:
        ws.write_file("agent.py", "a = 1\n")
        ws.checkpoint("agent work")
        (repo / "calc.py").write_text("def add(a, b):\n    return a + b\n")   # you revert your edit
        ws.sync()
        assert "temporary" not in ws.read_file("calc.py")
    finally:
        finish_with(ws)


def test_conflicting_changes_are_reported_and_nothing_is_lost(repo):
    ws = start(repo)
    try:
        ws.edit_file("calc.py", "return a + b", "return a + b  # agent")
        ws.checkpoint("agent work")
        (repo / "calc.py").write_text("def add(a, b):\n    return a + b  # human\n")
        message = ws.sync()
        assert "conflict" in message
        assert "# agent" in ws.read_file("calc.py")                       # the agent's copy is intact
        assert sh("git", "status", "--porcelain", cwd=ws.worktree) == ""  # and no half-done merge is left behind
    finally:
        finish_with(ws)


def test_switching_branches_in_your_folder_is_picked_up(repo):
    ws = start(repo)
    try:
        sh("git", "checkout", "-q", "-b", "feature", cwd=repo)
        (repo / "feature.py").write_text("f = 1\n")
        sh("git", "add", "-A", cwd=repo)
        sh("git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "feature", cwd=repo)
        ws.sync()
        assert (ws.workdir / "feature.py").exists()
    finally:
        finish_with(ws)


# ---------- cleanup ----------
def test_branch_is_deleted_when_nothing_is_pending_and_kept_when_something_is(repo):
    ws = start(repo)
    ws.write_file("a.py", "1\n"); ws.finish(); ws.accept()
    ws.cleanup()
    assert ws.branch not in sh("git", "branch", cwd=repo)                 # all applied -> nothing to keep

    ws2 = start(repo)
    ws2.write_file("b.py", "2\n")                                         # pending, then "crash"
    ws2.cleanup()
    assert ws2.branch in sh("git", "branch", cwd=repo)
    assert "b.py" in sh("git", "show", "--stat", ws2.branch, cwd=repo)


def test_keep_prints_commands_that_work(repo):
    ws = start(repo)
    ws.write_file("a.py", "1\n"); ws.finish()
    message = ws.keep()
    ws.cleanup()
    command = [l.split(":", 1)[1].strip() for l in message.splitlines() if "See the changes" in l][0]
    assert "a.py" in sh(*command.split(), cwd=repo)


def test_sync_reports_which_files_changed(repo):
    ws = start(repo)
    try:
        assert ws.last_sync_changes == []
        (repo / "calc.py").write_text("def add(a, b):\n    return 0\n")
        (repo / "fresh.py").write_text("f = 1\n")
        (repo / "old.py").unlink()
        ws.sync()
        assert sorted(ws.last_sync_changes) == ["A\tfresh.py", "D\told.py", "M\tcalc.py"]
        ws.sync()
        assert ws.last_sync_changes == []                       # nothing new the second time
    finally:
        finish_with(ws)
