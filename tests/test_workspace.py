"""Tests for Workspace. Needs git; the Docker tests are skipped if Docker is not running."""
import subprocess
import pytest
from ce_agent.workspace import Workspace

def sh(*cmd, cwd):
    subprocess.run(cmd, cwd=cwd, check=True, capture_output=True)

@pytest.fixture
def repo(tmp_path):
    sh("git", "init", "-q", "-b", "main", cwd=tmp_path)
    (tmp_path / "calc.py").write_text("print(1 + 1)\n")
    (tmp_path / ".gitignore").write_text(".agent/\n")
    sh("git", "add", "-A", cwd=tmp_path)
    sh("git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init", cwd=tmp_path)
    return tmp_path

def docker_ok():
    return subprocess.run(["docker", "ps"], capture_output=True).returncode == 0

needs_docker = pytest.mark.skipif(not docker_ok(), reason="Docker not running")

def test_unsafe_edit_reject_leaves_repo_untouched(repo):
    ws = Workspace(str(repo), unsafe=True)
    ws.start()
    ws.write_file("calc.py", "print(2 + 2)\n")
    assert "4" in ws.run("python3 calc.py")["stdout"]
    assert "2 + 2" in ws.finish()
    ws.cleanup()
    assert (repo / "calc.py").read_text() == "print(1 + 1)\n"      # real repo untouched
    assert not (repo / ".agent" / "worktrees" / ws.task_id).exists()  # worktree gone
    assert ws.branch not in subprocess.run(["git", "branch"], cwd=repo, capture_output=True, text=True).stdout

def test_accept_merges_into_real_repo(repo):
    ws = Workspace(str(repo), unsafe=True)
    ws.start()
    ws.write_file("new.py", "x = 1\n")
    ws.finish()
    assert "Merged" in ws.accept()
    ws.cleanup()
    assert (repo / "new.py").exists()

def test_path_escape_blocked(repo):
    ws = Workspace(str(repo), unsafe=True)
    ws.start()
    try:
        with pytest.raises(ValueError):
            ws.write_file("../evil.txt", "x")
        with pytest.raises(ValueError):
            ws.read_file("/etc/passwd")
    finally:
        ws.cleanup()

def test_dirty_folder_refused(repo):
    (repo / "uncommitted.py").write_text("x\n")
    with pytest.raises(RuntimeError, match="Uncommitted"):
        Workspace(str(repo), unsafe=True).start()

@needs_docker
def test_docker_blocks_network_and_runs_as_you(repo):
    ws = Workspace(str(repo))
    ws.start()
    try:
        ws.write_file("calc.py", "print(3 + 3)\n")
        assert "6" in ws.run("python calc.py")["stdout"]                 # sees host edits
        net = ws.run("python -c \"import urllib.request as u; u.urlopen('http://1.1.1.1', timeout=3)\"")
        assert net["exit_code"] != 0                                       # network is off
        ws.run("echo hi > made_in_container.txt")
        assert (ws.workdir / "made_in_container.txt").owner() != "root"   # not root-owned
        assert ws.run("sleep 5", timeout=1)["exit_code"] == 124           # timeout works
    finally:
        ws.cleanup()
    names = subprocess.run(["docker", "ps", "-a", "--filter", "label=ce_agent=true", "-q"],
                           capture_output=True, text=True).stdout
    assert ws.container is None and names.strip() == ""                    # container removed
