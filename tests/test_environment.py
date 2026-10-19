"""The sandbox environment profile: detection, config file, flags, validation, and the text shown to the model."""
import json
from dataclasses import FrozenInstanceError, replace

import pytest
from ce_agent.environment import (DEFAULT_IMAGE, PRESETS, Environment, EnvironmentConfigError, cache_dir_for, detect_preset,
                                  flags_to_overrides, from_preset, resolve_environment, validate)


def project(tmp_path, *files, config=None, name="proj"):
    folder = tmp_path / name
    folder.mkdir()
    for f in files:
        (folder / f).write_text("")
    if config is not None:
        (folder / ".agent").mkdir()
        (folder / ".agent" / "env.json").write_text(config if isinstance(config, str) else json.dumps(config))
    return folder


# ---------- detection ----------
@pytest.mark.parametrize("file, preset", [
    ("Cargo.toml", "rust"), ("go.mod", "go"), ("package.json", "node"), ("pyproject.toml", "python"),
    ("requirements.txt", "python"), ("setup.py", "python"), ("setup.cfg", "python"), ("Pipfile", "python")])
def test_the_project_type_is_detected_from_its_files(tmp_path, file, preset):
    env = resolve_environment(project(tmp_path, file))
    assert env.preset == preset and env.image == PRESETS[preset]["image"]
    assert f"({file})" in env.source

def test_a_project_with_no_markers_gets_the_python_defaults_which_are_todays_behaviour(tmp_path):
    env = resolve_environment(project(tmp_path))
    assert (env.preset, env.image, env.network, env.memory, env.pids_limit, env.command_timeout) == ("python", DEFAULT_IMAGE, "none", "512m", 100, 60)

def test_when_several_markers_exist_rust_wins_then_go_then_node_then_python(tmp_path):
    assert detect_preset(project(tmp_path, "package.json", "Cargo.toml", "requirements.txt", name="a")) == ("rust", "Cargo.toml")
    assert detect_preset(project(tmp_path, "package.json", "go.mod", "requirements.txt", name="b")) == ("go", "go.mod")
    assert detect_preset(project(tmp_path, "package.json", "requirements.txt", name="c")) == ("node", "package.json")
    assert detect_preset(project(tmp_path, name="d")) == (None, None)

def test_detection_also_looks_at_the_repository_root_when_pointed_at_a_subfolder(tmp_path):
    root = project(tmp_path, "Cargo.toml")
    sub = root / "crates" / "core"
    sub.mkdir(parents=True)
    assert resolve_environment(sub, top=root).preset == "rust"

def test_the_presets_have_sensible_resources_for_building():
    assert PRESETS["rust"]["memory"] == "4g" and PRESETS["rust"]["command_timeout"] == 900 and PRESETS["rust"]["image"] == "rust:1-slim"
    assert all(p["command_timeout"] >= 60 for p in PRESETS.values())
    assert PRESETS["python"]["memory"] == "512m"                                   # unchanged for existing users


# ---------- the config file ----------
def test_the_config_file_overrides_the_detected_defaults(tmp_path):
    env = resolve_environment(project(tmp_path, "Cargo.toml", config={"network": "registries", "memory": "8g", "command_timeout": 1200}))
    assert (env.preset, env.network, env.memory, env.command_timeout, env.image) == ("rust", "registries", "8g", 1200, "rust:1-slim")
    assert env.source == "rust (Cargo.toml) + .agent/env.json"

def test_the_config_can_name_the_preset_and_the_image_freely(tmp_path):
    env = resolve_environment(project(tmp_path, config={"preset": "node", "image": "node:22-slim"}))
    assert env.preset == "node" and env.image == "node:22-slim" and env.memory == "2g"

def test_a_config_in_the_repository_root_applies_to_a_subfolder(tmp_path):
    root = project(tmp_path, config={"network": "registries"})
    sub = root / "sub"; sub.mkdir()
    assert resolve_environment(sub, top=root).network == "registries"

@pytest.mark.parametrize("content, message", [
    ("{not json", "cannot read"), ("[1, 2]", "must contain a JSON object"), ('{"netwrok": "full"}', "unknown setting"),
    ('{"preset": "cobol"}', "unknown preset")])
def test_a_broken_config_file_is_a_clear_error_that_names_the_file(tmp_path, content, message):
    with pytest.raises(EnvironmentConfigError, match=message) as e:
        resolve_environment(project(tmp_path, config=content))
    if message != "unknown preset":
        assert "env.json" in str(e.value)

def test_the_config_file_is_not_part_of_the_agents_copy_of_the_project(tmp_path):
    # .agent/ is excluded from the snapshot (see Workspace.start), so the model can never read or edit its own policy
    import subprocess
    repo = project(tmp_path, "a.py", config={"network": "registries"})
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    from ce_agent.workspace import Workspace
    ws = Workspace(str(repo), unsafe=True); ws.start()
    try:
        assert (repo / ".agent" / "env.json").exists() and not (ws.workdir / ".agent" / "env.json").exists()
        assert ".agent" not in ws.list_files()
    finally:
        ws.discard(); ws.cleanup()


# ---------- flags ----------
def test_flags_override_the_config_file_which_overrides_the_detection(tmp_path):
    folder = project(tmp_path, "Cargo.toml", config={"network": "registries", "memory": "8g"})
    env = resolve_environment(folder, {"network": "full", "command_timeout": 30})
    assert (env.network, env.memory, env.command_timeout) == ("full", "8g", 30)
    assert env.source == "rust (Cargo.toml) + .agent/env.json + flags"

def test_a_preset_flag_beats_the_config_and_the_detection(tmp_path):
    assert resolve_environment(project(tmp_path, "Cargo.toml", config={"preset": "node"}), {"preset": "go"}).preset == "go"

@pytest.mark.parametrize("flags, expected", [
    (["--network"], {"network": "full"}),                                          # a bare --network has always meant full access
    (["--network=registries"], {"network": "registries"}),
    (["--network=none"], {"network": "none"}),
    (["--image=rust:1.80"], {"image": "rust:1.80"}), (["--memory=4g"], {"memory": "4g"}), (["--timeout=600"], {"command_timeout": 600}),
    (["--pids=300"], {"pids_limit": 300}), (["--preset=go"], {"preset": "go"}), (["--no-cache"], {"cache": False}),
    (["--allow-private"], {"allow_private": True}),
    (["--allow-host=a.example.com", "--allow-host=*.b.example.com"], {"allow_hosts": ("a.example.com", "*.b.example.com")}),
    (["--once", "--unsafe-local", "--mcp=http", "--show"], {}),                    # other flags are not ours
    (["--image"], {}), (["--memory="], {}),
])
def test_command_line_flags(flags, expected):
    assert flags_to_overrides(flags) == expected

@pytest.mark.parametrize("flag", ["--timeout=abc", "--timeout=-5", "--pids=x"])
def test_numeric_flags_must_be_numbers(flag):
    with pytest.raises(EnvironmentConfigError, match="must be a"):
        flags_to_overrides([flag])


# ---------- validation ----------
GOOD = Environment()

@pytest.mark.parametrize("change, message", [
    ({"network": "internet"}, "network must be one of"), ({"network": True}, "network must be one of"),
    ({"image": ""}, "not a valid image"), ({"image": "bad image"}, "not a valid image"), ({"image": "x; rm -rf /"}, "not a valid image"),
    ({"image": "--privileged"}, "not a valid image"),
    ({"memory": "lots"}, "memory must look like"), ({"memory": "0m"}, "memory must look like"), ({"memory": "-1g"}, "memory must look like"),
    ({"memory": 512}, "memory must look like"),
    ({"pids_limit": 0}, "pids_limit"), ({"pids_limit": 99999}, "pids_limit"), ({"pids_limit": "10"}, "pids_limit"), ({"pids_limit": True}, "pids_limit"),
    ({"command_timeout": 0}, "command_timeout"), ({"command_timeout": 100000}, "command_timeout"), ({"command_timeout": 1.5}, "command_timeout"),
    ({"allow_hosts": ("evil.com/path",)}, "not a valid host"), ({"allow_hosts": ("localhost",)}, "not a valid host"), ({"allow_hosts": (5,)}, "not a valid host"),
    ({"env": {"1BAD": "x"}}, "environment variable name"), ({"env": {"A B": "x"}}, "environment variable name"), ({"env": {"OK": 5}}, "must be a string"),
    ({"env": {"HTTPS_PROXY": "http://x"}}, "set by the sandbox itself"), ({"env": {"no_proxy": "x"}}, "set by the sandbox itself"), ({"env": ["A"]}, "env must be an object"),
    ({"egress_network": "bad name"}, "Docker network name"), ({"egress_network": "--rm"}, "Docker network name"),
    ({"cache": "yes"}, "cache must be true or false"), ({"allow_private": 1}, "allow_private must be true or false"),
    ({"preset": "cobol"}, "unknown preset"),
])
def test_bad_settings_are_rejected_with_a_message_that_says_what_to_change(change, message):
    with pytest.raises(EnvironmentConfigError, match=message):
        validate(replace(GOOD, **change))

def test_good_settings_pass():
    validate(replace(GOOD, network="registries", image="ghcr.io/org/img:1.2@sha256", memory="4g", allow_hosts=("a.example.com", "*.example.org"),
                     env={"RUST_LOG": "debug"}, egress_network="my-net_1.0"))

def test_validation_runs_on_everything_resolve_environment_returns(tmp_path):
    with pytest.raises(EnvironmentConfigError, match="network must be"):
        resolve_environment(project(tmp_path, config={"network": "wide-open"}))
    with pytest.raises(EnvironmentConfigError, match="allow_hosts must be a list"):
        resolve_environment(project(tmp_path, name="b", config={"allow_hosts": "crates.io"}))
    with pytest.raises(EnvironmentConfigError, match="env must be an object"):
        resolve_environment(project(tmp_path, name="c", config={"env": "A=1"}))

def test_an_environment_cannot_be_changed_after_it_is_validated():
    with pytest.raises(FrozenInstanceError):
        GOOD.network = "full"


# ---------- derived values ----------
def test_hosts_combine_the_projects_registries_with_extra_ones_without_duplicates():
    env = replace(from_preset("rust"), allow_hosts=("EXAMPLE.org.", "crates.io", "*.corp.example.com"))
    assert env.hosts() == ("index.crates.io", "crates.io", "static.crates.io", "example.org", "*.corp.example.com")

def test_the_container_environment_points_caches_at_the_cache_folder():
    env = from_preset("rust")
    assert env.container_env() == {"CARGO_HOME": "/cache/cargo", "CARGO_TARGET_DIR": "/cache/target"}
    assert replace(env, cache=False).container_env() == {}                           # no cache folder, so no variables pointing at it
    assert from_preset("python").container_env()["PIP_CACHE_DIR"] == "/cache/pip"
    assert replace(from_preset("python"), cache=False).container_env() == {"PIP_DISABLE_PIP_VERSION_CHECK": "1"}

def test_user_env_is_added_and_wins_over_the_defaults():
    env = replace(from_preset("rust"), env={"CARGO_TARGET_DIR": "/cache/other", "RUST_LOG": "info"})
    assert env.container_env() == {"CARGO_HOME": "/cache/cargo", "CARGO_TARGET_DIR": "/cache/other", "RUST_LOG": "info"}

def test_config_env_and_flag_env_merge(tmp_path):
    env = resolve_environment(project(tmp_path, config={"env": {"A": "1", "B": "2"}}))
    assert env.env == {"A": "1", "B": "2"}


# ---------- what people and the model are told ----------
def test_the_banner_is_honest_about_each_mode():
    none, reg, full = (replace(from_preset("rust"), network=n) for n in NETWORK_MODES)
    assert "network: no network" in none.describe()
    assert "package registries only (index.crates.io, crates.io, static.crates.io)" in reg.describe()
    assert "FULL internet access" in full.describe() and "rust:1-slim" in full.describe() and "900s per command" in full.describe()
    assert "NONE" in none.describe(unsafe=True) and "directly on this machine" in none.describe(unsafe=True)

from ce_agent.environment import NETWORK_MODES  # noqa: E402

def test_the_model_is_told_what_it_can_and_cannot_do():
    none, reg, full = (replace(from_preset("python"), network=n) for n in NETWORK_MODES)
    assert "NO internet access" in none.prompt_text() and "cannot be downloaded" in none.prompt_text()
    assert "only internet access is through a proxy" in reg.prompt_text() and "pypi.org, files.pythonhosted.org" in reg.prompt_text()
    assert "Everything else is refused" in reg.prompt_text()
    assert "full internet access" in full.prompt_text()
    for env in (none, reg, full):
        assert "stopped after 60 seconds" in env.prompt_text() and "/cache" in env.prompt_text()
    assert "directly on the developer's machine" in none.prompt_text(unsafe=True) and "/cache" not in none.prompt_text(unsafe=True)
    assert "/cache" not in replace(none, cache=False).prompt_text()


# ---------- the cache folder ----------
def test_the_cache_folder_is_stable_per_project_and_different_between_projects(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    assert cache_dir_for(a) == cache_dir_for(a) and cache_dir_for(a) != cache_dir_for(b)
    assert cache_dir_for(a).parent == tmp_path / "xdg" / "ce-agent"
    assert cache_dir_for(a) == cache_dir_for(str(a) + "/.")                             # the same folder spelled differently
