"""The environment reaching the rest of the harness: the model's prompt, the command line, the banner, the trace."""
import json
import subprocess
import sys
from types import SimpleNamespace

import ce_agent.main as main_module
import pytest
from ce_agent.brain import system_prompt
from ce_agent.environment import EnvironmentConfigError
from ce_agent.main import run_agent, run_session, show_environment
from ce_agent.trace import find_run, show


def sh(*cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True).stdout


def docker_ok():
    try:
        return subprocess.run(["docker", "ps"], capture_output=True, timeout=10).returncode == 0
    except Exception:
        return False


needs_docker = pytest.mark.skipif(not docker_ok(), reason="Docker not running")


@pytest.fixture(autouse=True)
def private_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))        # never write into the real ~/.cache


@pytest.fixture
def repo(tmp_path):
    path = tmp_path / "repo"
    path.mkdir()
    sh("git", "init", "-q", "-b", "main", cwd=path)
    (path / "requirements.txt").write_text("")
    (path / "a.py").write_text("x = 1\n")
    sh("git", "add", "-A", cwd=path)
    sh("git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init", cwd=path)
    return path


class Brain:
    model = "fake"
    last_usage = None
    def __init__(self, *script):
        self.script, self.prompts, self.seen = list(script), [], []
    def think(self, messages, tools):
        self.prompts.append(messages[0]["content"])
        self.seen.append(json.loads(json.dumps(messages, default=str)))
        return self.script.pop(0)


def say(text):
    return SimpleNamespace(content=text, tool_calls=None)

def shell(command):
    return SimpleNamespace(content=None, tool_calls=[SimpleNamespace(id="c1", function=SimpleNamespace(
        name="run_shell", arguments=json.dumps({"command": command})))])

def events(repo):
    path = max((repo / ".agent" / "runs").glob("*.jsonl"), key=lambda f: f.stat().st_mtime)
    return [json.loads(l) for l in path.read_text().splitlines()]


# ======================================================================
# what the model is told
# ======================================================================
def test_the_system_prompt_no_longer_claims_there_is_no_internet():
    assert "no internet" not in system_prompt().lower()

def test_the_environment_text_goes_between_the_base_prompt_and_the_notes():
    text = system_prompt("- a note", "Commands run in a container.")
    assert text.index("Your environment: Commands run in a container.") < text.index("Notes about this project")
    assert "Your environment" not in system_prompt("- a note")

def test_the_model_sees_the_real_environment_in_every_mode(repo, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _="": "r")
    brain = Brain(say("hi"))
    run_agent(str(repo), "hello", unsafe=True, brain=brain)
    assert "directly on the developer's machine" in brain.prompts[0]

@needs_docker
@pytest.mark.parametrize("mode, expected", [("none", "NO internet access"), ("registries", "only internet access is through a proxy"),
                                            ("full", "full internet access")])
def test_the_prompt_matches_the_network_mode_in_a_real_session(repo, monkeypatch, mode, expected):
    monkeypatch.setattr("builtins.input", lambda _="": "r")
    brain = Brain(say("hi"))
    run_agent(str(repo), "hello", brain=brain, env_overrides={"network": mode})
    assert expected in brain.prompts[0] and "stopped after 60 seconds" in brain.prompts[0]

def test_the_run_shell_tool_description_does_not_promise_anything_about_the_network():
    from ce_agent.tools import make_workspace_tools
    descriptions = {f.__name__: d for f, d in make_workspace_tools(SimpleNamespace())}
    assert "No internet" not in descriptions["run_shell"]


# ======================================================================
# the banner and the trace
# ======================================================================
def test_the_banner_shows_the_sandbox_and_the_trace_records_it(repo, monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda _="": "r")
    run_agent(str(repo), "hello", unsafe=True, brain=Brain(say("hi")))
    assert "Sandbox: NONE (--unsafe-local)" in capsys.readouterr().out
    start = [e for e in events(repo) if e["kind"] == "run_start"][0]
    assert start["environment"]["network"] == "none" and start["environment"]["preset"] == "python"
    assert "Sandbox: NONE" in start["environment_text"]
    assert "Sandbox: NONE" in show(find_run(str(repo)))

@needs_docker
def test_full_network_prints_a_loud_warning_and_none_does_not(repo, monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda _="": "r")
    run_agent(str(repo), "hello", brain=Brain(say("hi")), env_overrides={"network": "full"})
    out = capsys.readouterr().out
    assert "FULL NETWORK" in out and "send" in out and "--network=registries" in out and "FULL internet access" in out
    run_agent(str(repo), "hello", brain=Brain(say("hi")))
    assert "FULL NETWORK" not in capsys.readouterr().out

@needs_docker
def test_the_legacy_network_argument_still_means_full(repo, monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda _="": "r")
    run_agent(str(repo), "hello", brain=Brain(say("hi")), network=True)
    assert "FULL NETWORK" in capsys.readouterr().out

@needs_docker
def test_what_the_proxy_blocked_is_printed_logged_and_shown_in_the_timeline(repo, monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda _="": "r")
    fetch = "python3 -c \"import urllib.request as u; u.urlopen('https://tracker.example', timeout=10)\""
    run_agent(str(repo), "try the network", brain=Brain(shell(fetch), say("done")), env_overrides={"network": "registries"})
    out = capsys.readouterr().out
    assert "[network] BLOCKED tracker.example: tracker.example is not on the allowlist" in out
    egress = [e for e in events(repo) if e["kind"] == "egress"][0]
    assert egress["blocked"] == ["tracker.example: tracker.example is not on the allowlist"] and egress["allowed"] == {}
    assert any(entry.get("action") == "deny" for entry in egress["entries"])
    timeline = show(find_run(str(repo)))
    assert "network: BLOCKED tracker.example" in timeline

@needs_docker
def test_no_network_report_when_the_proxy_saw_nothing(repo, monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda _="": "r")
    run_agent(str(repo), "hello", brain=Brain(say("hi")), env_overrides={"network": "registries"})
    assert "[network]" not in capsys.readouterr().out
    assert not [e for e in events(repo) if e["kind"] == "egress"]


# ======================================================================
# bad settings fail early and leave nothing behind
# ======================================================================
def test_a_bad_config_file_fails_before_anything_is_created(repo, monkeypatch):
    (repo / ".agent").mkdir()
    (repo / ".agent" / "env.json").write_text('{"network": "wide-open"}')
    with pytest.raises(EnvironmentConfigError, match="network must be one of"):
        run_agent(str(repo), "hello", unsafe=True, brain=Brain(say("x")))
    assert "agent/" not in sh("git", "branch", cwd=repo)
    assert not list((repo / ".agent").glob("worktrees/*")) and not list((repo / ".agent").glob("runs/*"))

def test_a_bad_flag_value_fails_before_anything_is_created(repo):
    with pytest.raises(EnvironmentConfigError, match="memory must look like"):
        run_agent(str(repo), "hello", unsafe=True, brain=Brain(say("x")), env_overrides={"memory": "lots"})
    assert "agent/" not in sh("git", "branch", cwd=repo)

def test_the_config_file_is_used_by_a_real_session(repo, monkeypatch):
    (repo / ".agent").mkdir()
    (repo / ".agent" / "env.json").write_text(json.dumps({"command_timeout": 7}))
    monkeypatch.setattr("builtins.input", lambda _="": "r")
    brain = Brain(say("hi"))
    run_agent(str(repo), "hello", unsafe=True, brain=brain)
    assert "stopped after 7 seconds" in brain.prompts[0]


# ======================================================================
# the command line
# ======================================================================
def run_cli(monkeypatch, *argv):
    captured = {}
    monkeypatch.setattr(main_module, "run_session", lambda *a, **kw: captured.update(kw, positional=a))
    monkeypatch.setattr(sys, "argv", ["ce_agent.main", "somewhere", *argv])
    main_module.main()
    return captured

def test_flags_reach_the_sandbox_settings(monkeypatch):
    got = run_cli(monkeypatch, "--network=registries", "--image=rust:1.80", "--memory=4g", "--timeout=600", "--pids=300",
                  "--allow-host=a.example.com", "--allow-host=*.b.example.com", "--no-cache", "--preset=rust")
    assert got["env_overrides"] == {"network": "registries", "image": "rust:1.80", "memory": "4g", "command_timeout": 600, "pids_limit": 300,
                                    "allow_hosts": ("a.example.com", "*.b.example.com"), "cache": False, "preset": "rust"}

def test_a_bare_network_flag_means_full_as_before(monkeypatch):
    assert run_cli(monkeypatch, "--network")["env_overrides"] == {"network": "full"}

def test_no_flags_means_no_overrides(monkeypatch):
    assert run_cli(monkeypatch)["env_overrides"] == {}

def test_a_bad_flag_is_an_error_message_not_a_traceback(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["ce_agent.main", "somewhere", "--timeout=soon"])
    main_module.main()
    assert "Error: --timeout must be a number of seconds" in capsys.readouterr().out

def test_env_prints_the_settings_without_running_anything(repo, monkeypatch, capsys):
    ran = []
    monkeypatch.setattr(main_module, "run_session", lambda *a, **kw: ran.append(1))
    (repo / "Cargo.toml").write_text("")
    monkeypatch.setattr(sys, "argv", ["ce_agent.main", str(repo), "--env", "--network=registries", "--memory=8g"])
    main_module.main()
    out = capsys.readouterr().out
    assert not ran
    assert "rust:1-slim" in out and "project type: rust" in out and "memory 8g" in out and "900s per command" in out
    assert "Allowed hosts: index.crates.io, crates.io, static.crates.io" in out and "Told to the model:" in out

def test_env_reports_a_bad_config_as_an_error(repo, monkeypatch, capsys):
    (repo / ".agent").mkdir()
    (repo / ".agent" / "env.json").write_text("{broken")
    monkeypatch.setattr(sys, "argv", ["ce_agent.main", str(repo), "--env"])
    main_module.main()
    assert "Error: cannot read" in capsys.readouterr().out

def test_the_usage_text_lists_the_sandbox_flags(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["ce_agent.main"])
    main_module.main()
    out = capsys.readouterr().out
    for text in ("--network[=none|registries|full]", "--image=IMG", "--memory=4g", "--timeout=SECONDS", "--allow-host=HOST", "--env"):
        assert text in out
