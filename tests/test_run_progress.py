"""Workspace.run(on_tick): progress while a command runs, and cancelling it."""
import subprocess
import time

import ce_agent.workspace as workspace_module
import pytest
from ce_agent.workspace import Workspace


@pytest.fixture
def ws(tmp_path):
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    (tmp_path / "a.txt").write_text("x\n")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "i"], cwd=tmp_path, check=True)
    w = Workspace(str(tmp_path), unsafe=True)
    w.start()
    yield w
    w.discard()
    w.cleanup()


def test_on_tick_is_called_about_once_per_tick_while_the_command_runs(ws, monkeypatch):
    monkeypatch.setattr(workspace_module, "RUN_TICK_SECONDS", 0.2)
    ticks = []
    result = ws.run("sleep 1", on_tick=ticks.append)
    assert result["exit_code"] == 0
    assert 3 <= len(ticks) <= 6
    assert ticks == sorted(ticks) and len(set(ticks)) == len(ticks)             # elapsed time only goes up
    assert 0.15 < ticks[0] < 0.5

def test_no_ticks_for_a_command_that_finishes_before_the_first_tick(ws):
    ticks = []
    assert ws.run("echo quick", on_tick=ticks.append)["stdout"] == "quick\n"
    assert ticks == []

def test_output_is_not_lost_across_ticks(ws, monkeypatch):
    monkeypatch.setattr(workspace_module, "RUN_TICK_SECONDS", 0.2)
    result = ws.run("echo first; sleep 0.7; echo second; echo oops >&2", on_tick=lambda s: None)
    assert result == {"exit_code": 0, "stdout": "first\nsecond\n", "stderr": "oops\n"}

def test_the_behaviour_without_on_tick_is_unchanged(ws):
    assert ws.run("echo hi; exit 3") == {"exit_code": 3, "stdout": "hi\n", "stderr": ""}
    assert ws.run("sleep 5", timeout=1)["exit_code"] == 124                      # still times out through `timeout`

def test_the_time_limit_still_applies_while_ticking(ws, monkeypatch):
    monkeypatch.setattr(workspace_module, "RUN_TICK_SECONDS", 0.2)
    ticks = []
    assert ws.run("sleep 5", timeout=1, on_tick=ticks.append)["exit_code"] == 124
    assert ticks

def test_an_exception_from_on_tick_stops_the_command_and_everything_it_started(ws, monkeypatch):
    monkeypatch.setattr(workspace_module, "RUN_TICK_SECONDS", 0.1)
    marker = ws.workdir / "marker.txt"
    class Stop(Exception):
        pass
    def cancel(seconds):
        raise Stop()
    started = time.monotonic()
    # a background child that would create the file after 2 seconds, if it were left running
    with pytest.raises(Stop):
        ws.run("(sleep 2; touch marker.txt) & wait", on_tick=cancel)
    assert time.monotonic() - started < 1.5                                      # we did not wait for it
    time.sleep(2.5)
    assert not marker.exists()                                                   # the whole process group was killed


def test_kill_running_stops_commands_that_are_still_going(ws):
    import threading
    outcome = {}
    def run():
        outcome["result"] = ws.run("(sleep 2; touch marker.txt) & wait")
    thread = threading.Thread(target=run)
    thread.start()
    time.sleep(0.4)
    assert ws._running                                                             # the command is registered while it runs
    ws.kill_running()
    thread.join(timeout=5)
    assert not thread.is_alive() and outcome["result"]["exit_code"] != 0
    time.sleep(2.5)
    assert not (ws.workdir / "marker.txt").exists()                                # the command and its child are gone
    assert not ws._running                                                         # and it is unregistered again

def test_kill_running_with_nothing_running_is_harmless(ws):
    ws.kill_running()
    assert ws.run("echo ok")["stdout"] == "ok\n" and not ws._running
