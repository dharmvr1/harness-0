"""The sandbox network modes with real Docker containers. A container named upstream.test plays the package registry."""
import json
import os
import subprocess
import time
import uuid
from dataclasses import replace

import pytest
from ce_agent.egress import summarize
from ce_agent.environment import Environment, cache_dir_for
from ce_agent.mcp.client import launch_workspace_server, server_command
from ce_agent.workspace import Workspace, recover


def docker_ok():
    try:
        return subprocess.run(["docker", "ps"], capture_output=True, timeout=10).returncode == 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not docker_ok(), reason="Docker not running")


def sh(*cmd, cwd=None, check=True):
    return subprocess.run(cmd, cwd=cwd, check=check, capture_output=True, text=True).stdout


def docker(*args):
    return subprocess.run(["docker", *args], capture_output=True, text=True)


def inspect(name):
    return json.loads(docker("inspect", name).stdout)[0]


def make_repo(path):
    path.mkdir(parents=True, exist_ok=True)
    sh("git", "init", "-q", "-b", "main", cwd=path)
    (path / "a.txt").write_text("x\n")
    sh("git", "add", "-A", cwd=path)
    sh("git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init", cwd=path)
    return path


FETCH = """import sys, urllib.request, urllib.error
try:
    r = urllib.request.urlopen(sys.argv[1], timeout=15)
    print("STATUS", r.status); print(r.read().decode()[:200])
except urllib.error.HTTPError as e:
    print("HTTPERROR", e.code, e.read().decode()[:300])
except Exception as e:
    print("ERROR", type(e).__name__, e)
"""
DIRECT = """import socket, sys
try:
    socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=5); print("CONNECTED")
except OSError as e:
    print("FAILED", e)
"""
RESOLVE = """import socket, sys
try:
    print("RESOLVED", socket.gethostbyname(sys.argv[1]))
except OSError as e:
    print("FAILED", e)
"""


def run_script(ws, name, source, *args):
    ws.write_file(name, source)
    return ws.run(f"python3 {name} " + " ".join(args))["stdout"]


# ======================================================================
# a fake registry on its own Docker network
# ======================================================================
@pytest.fixture(scope="module")
def egress_net():
    name = f"ce_test_egress_{uuid.uuid4().hex[:6]}"
    docker("network", "create", name)
    upstream = f"ce_test_upstream_{uuid.uuid4().hex[:6]}"
    docker("run", "-d", "--name", upstream, "--network", name, "--network-alias", "upstream.test", "python:3.11-slim", "sh", "-c",
           "mkdir -p /srv && echo 'hello from upstream' > /srv/hello.txt && python -m http.server 80 --directory /srv")
    deadline = time.monotonic() + 20
    while True:
        probe = docker("exec", upstream, "python", "-c", "import urllib.request as u; print(u.urlopen('http://localhost/hello.txt').read())")
        if probe.returncode == 0:
            break
        if time.monotonic() > deadline:          # fail loudly: a broken fixture must not look like a product bug
            logs = docker("logs", upstream)
            docker("rm", "-f", upstream); docker("network", "rm", name)
            pytest.fail(f"the fake registry container did not start:\n{logs.stdout}{logs.stderr}{probe.stderr}")
        time.sleep(0.3)
    yield name
    docker("rm", "-f", upstream)
    docker("network", "rm", name)


def registries_env(egress_net, **changes):
    return replace(Environment(), network="registries", allow_hosts=("upstream.test",), allow_private=True,
                   egress_network=egress_net, **changes)


@pytest.fixture(scope="module")
def cache_home(tmp_path_factory):
    path = tmp_path_factory.mktemp("xdg")
    old = os.environ.get("XDG_CACHE_HOME")
    os.environ["XDG_CACHE_HOME"] = str(path)
    yield path
    if old is None:
        os.environ.pop("XDG_CACHE_HOME", None)
    else:
        os.environ["XDG_CACHE_HOME"] = old


@pytest.fixture(scope="module")
def sandbox(egress_net, cache_home, tmp_path_factory):
    """One running sandbox in "registries" mode, shared by the tests that only look at its behaviour."""
    ws = Workspace(str(make_repo(tmp_path_factory.mktemp("reg") / "repo")), environment=registries_env(egress_net))
    ws.start()
    yield ws
    ws.discard()
    ws.cleanup()


# ======================================================================
# registries mode: what gets through
# ======================================================================
def test_an_allowed_host_can_be_fetched_through_the_proxy(sandbox):
    out = run_script(sandbox, "fetch.py", FETCH, "http://upstream.test/hello.txt")
    assert "STATUS 200" in out and "hello from upstream" in out

def test_https_style_tunnelling_works_for_an_allowed_host(sandbox):
    script = ("import http.client\nc = http.client.HTTPConnection('egress-proxy', 3128, timeout=15)\nc.set_tunnel('upstream.test', 80)\n"
              "c.request('GET', '/hello.txt')\nprint(c.getresponse().read().decode())\n")
    assert "hello from upstream" in run_script(sandbox, "tunnel.py", script)

def test_a_host_that_is_not_allowed_is_refused_with_the_reason(sandbox):
    out = run_script(sandbox, "fetch.py", FETCH, "http://other.test/")
    assert "HTTPERROR 403" in out and "other.test is not on the allowlist" in out
    out = run_script(sandbox, "fetch.py", FETCH, "https://evil.example/")
    assert "403" in out and "ERROR" in out                                          # "Tunnel connection failed: 403 Forbidden"

def test_the_proxy_refuses_ip_addresses(sandbox):
    out = run_script(sandbox, "fetch.py", FETCH, "http://93.184.216.34/")
    assert "HTTPERROR 403" in out and "IP addresses are not allowed" in out

def test_a_wrong_port_is_refused(sandbox):
    assert "port 8080 is not allowed" in run_script(sandbox, "fetch.py", FETCH, "http://upstream.test:8080/")


# ======================================================================
# registries mode: what cannot be done around the proxy
# ======================================================================
@pytest.mark.parametrize("target", ["1.1.1.1 443", "8.8.8.8 53", "93.184.216.34 80", "127.0.0.1 9"])
def test_direct_connections_that_skip_the_proxy_fail(sandbox, target):
    out = run_script(sandbox, "direct.py", DIRECT, target)
    assert out.startswith("FAILED"), out

def test_outside_names_cannot_even_be_resolved_from_inside(sandbox):
    assert run_script(sandbox, "resolve.py", RESOLVE, "example.com").startswith("FAILED")
    assert run_script(sandbox, "resolve.py", RESOLVE, "upstream.test").startswith("FAILED")        # only the proxy knows it

def test_the_registry_container_is_not_reachable_directly_even_by_its_ip(sandbox, egress_net):
    names = docker("ps", "--filter", f"network={egress_net}", "--format", "{{.Names}}").stdout.split()
    upstream = next(n for n in names if n.startswith("ce_test_upstream"))
    ip = json.loads(docker("inspect", upstream).stdout)[0]["NetworkSettings"]["Networks"][egress_net]["IPAddress"]
    assert run_script(sandbox, "direct.py", DIRECT, ip, "80").startswith("FAILED")

def test_the_sandbox_has_no_route_except_the_internal_network(sandbox):
    info = inspect(sandbox.container)
    assert list(info["NetworkSettings"]["Networks"]) == [sandbox._egress.network]
    net = json.loads(docker("network", "inspect", sandbox._egress.network).stdout)[0]
    assert net["Internal"] is True and len(net["Containers"]) == 2                            # the sandbox and the proxy, nothing else

def test_the_sandbox_is_told_to_use_the_proxy(sandbox):
    env = dict(e.split("=", 1) for e in inspect(sandbox.container)["Config"]["Env"])
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        assert env[name] == "http://egress-proxy:3128"
    assert env["NO_PROXY"] == "localhost,127.0.0.1,::1" and env["no_proxy"] == env["NO_PROXY"]


# ======================================================================
# the proxy container itself
# ======================================================================
def test_the_proxy_container_is_locked_down(sandbox):
    info = inspect(sandbox._egress.container)
    host = info["HostConfig"]
    assert host["ReadonlyRootfs"] is True and host["CapDrop"] == ["ALL"] and host["Memory"] == 128 * 1024 * 1024
    assert "no-new-privileges" in " ".join(host["SecurityOpt"]) and info["Config"]["User"] == "65534:65534"
    assert not host["PortBindings"] and not info["NetworkSettings"]["Ports"].get("3128/tcp")  # nothing is published on the host
    assert sorted(info["NetworkSettings"]["Networks"]) == sorted([sandbox._egress.network, sandbox.environment.egress_network])

def test_the_proxy_allowlist_is_the_projects_registries_plus_the_extra_hosts(sandbox):
    announcement = next(e for e in sandbox._egress.logs() if e.get("event") == "listening")
    assert announcement["allowed"] == ["pypi.org", "files.pythonhosted.org", "upstream.test"] and announcement["allow_private"] is True

def test_the_harness_can_read_what_the_proxy_allowed_and_refused(sandbox):
    run_script(sandbox, "fetch.py", FETCH, "http://upstream.test/hello.txt")
    run_script(sandbox, "fetch.py", FETCH, "http://tracker.test/")
    allowed, blocked = summarize(sandbox.egress_entries())
    assert allowed.get("upstream.test", 0) >= 1
    assert any(b.startswith("tracker.test:") and "not on the allowlist" in b for b in blocked)

def test_summarize_ignores_events_and_groups_repeats():
    allowed, blocked = summarize([{"event": "listening"}, {"host": "a.com", "action": "allow"}, {"host": "a.com", "action": "allow"},
                                  {"host": "b.com", "action": "deny", "reason": "no"}, {"host": "b.com", "action": "deny", "reason": "no"}])
    assert allowed == {"a.com": 2} and blocked == ["b.com: no"]


# ======================================================================
# the container's settings follow the environment
# ======================================================================
def test_memory_processes_env_cache_and_hardening_come_from_the_environment(sandbox, cache_home):
    info = inspect(sandbox.container)
    host = info["HostConfig"]
    assert host["Memory"] == 512 * 1024 * 1024 and host["PidsLimit"] == 100 and host["CapDrop"] == ["ALL"]
    assert info["Config"]["User"] == f"{os.getuid()}:{os.getgid()}"
    env = dict(e.split("=", 1) for e in info["Config"]["Env"])
    assert env["PIP_CACHE_DIR"] == "/cache/pip" and env["PYTHONUSERBASE"] == "/cache/pyuser"
    mounts = {m["Destination"]: m["Source"] for m in info["Mounts"]}
    assert mounts["/cache"] == str(cache_dir_for(sandbox.top)) and mounts["/repo"] == str(sandbox.worktree)

def test_custom_limits_image_and_env_are_applied(egress_net, cache_home, tmp_path):
    env = replace(Environment(), memory="256m", pids_limit=77, env={"MY_SETTING": "hello"}, command_timeout=30)
    ws = Workspace(str(make_repo(tmp_path / "repo")), environment=env)
    ws.start()
    try:
        host = inspect(ws.container)["HostConfig"]
        assert host["Memory"] == 256 * 1024 * 1024 and host["PidsLimit"] == 77
        assert ws.run("echo $MY_SETTING")["stdout"].strip() == "hello"
        assert ws._egress is None                                                           # none mode: no proxy was created
        assert inspect(ws.container)["HostConfig"]["NetworkMode"] == "none"
    finally:
        ws.discard(); ws.cleanup()

def test_full_mode_does_not_put_the_container_on_the_none_network_and_creates_no_proxy(cache_home, tmp_path):
    ws = Workspace(str(make_repo(tmp_path / "repo")), environment=replace(Environment(), network="full"))
    ws.start()
    try:
        info = inspect(ws.container)
        assert info["HostConfig"]["NetworkMode"] in ("default", "bridge") and ws._egress is None
        assert not any(e.startswith(("HTTP_PROXY", "http_proxy")) for e in info["Config"]["Env"])
    finally:
        ws.discard(); ws.cleanup()

def test_the_legacy_boolean_network_argument_still_works(cache_home, tmp_path):
    ws = Workspace(str(make_repo(tmp_path / "repo")), network=True)
    assert ws.environment.network == "full" and Workspace(str(tmp_path), network=False).environment.network == "none"


# ======================================================================
# the cache folder
# ======================================================================
def test_the_cache_survives_between_sessions_and_belongs_to_you(cache_home, tmp_path):
    repo = make_repo(tmp_path / "repo")
    first = Workspace(str(repo)); first.start()
    try:
        assert first.run("echo kept > /cache/marker.txt && mkdir -p /cache/pip && echo wheel > /cache/pip/x")["exit_code"] == 0
    finally:
        first.discard(); first.cleanup()
    second = Workspace(str(repo)); second.start()
    try:
        assert second.run("cat /cache/marker.txt")["stdout"] == "kept\n"
    finally:
        second.discard(); second.cleanup()
    marker = cache_dir_for(first.top) / "marker.txt"
    assert marker.read_text() == "kept\n" and marker.stat().st_uid == os.getuid()            # not root-owned

def test_another_project_does_not_see_the_cache(cache_home, tmp_path):
    a = Workspace(str(make_repo(tmp_path / "a"))); a.start()
    b = Workspace(str(make_repo(tmp_path / "b"))); b.start()
    try:
        assert a.run("echo a-only > /cache/secret.txt")["exit_code"] == 0
        assert "secret.txt" in a.run("ls /cache")["stdout"]                                   # it is there for project a...
        assert "secret.txt" not in b.run("ls /cache")["stdout"]                               # ...and not for project b
    finally:
        for w in (a, b):
            w.discard(); w.cleanup()

def test_no_cache_means_no_mount_and_no_variables_pointing_at_it(cache_home, tmp_path):
    ws = Workspace(str(make_repo(tmp_path / "repo")), environment=replace(Environment(), cache=False)); ws.start()
    try:
        info = inspect(ws.container)
        assert "/cache" not in {m["Destination"] for m in info["Mounts"]}
        assert not any(e.startswith("PIP_CACHE_DIR") for e in info["Config"]["Env"])
    finally:
        ws.discard(); ws.cleanup()


# ======================================================================
# the command time limit
# ======================================================================
def test_the_commands_time_limit_comes_from_the_environment(cache_home, tmp_path):
    ws = Workspace(str(make_repo(tmp_path / "repo")), environment=replace(Environment(), command_timeout=2)); ws.start()
    try:
        started = time.monotonic()
        assert ws.run("sleep 10")["exit_code"] == 124 and time.monotonic() - started < 6
        assert ws.run("sleep 10", timeout=1)["exit_code"] == 124                              # an explicit limit still wins
    finally:
        ws.discard(); ws.cleanup()

def test_the_mcp_server_uses_the_same_time_limit(cache_home, tmp_path):
    ws = Workspace(str(make_repo(tmp_path / "repo")), unsafe=True, environment=replace(Environment(), command_timeout=2)); ws.start()
    try:
        command = server_command(ws)
        assert command[command.index("--command-timeout") + 1] == "2"                        # the server is told the limit
        client = launch_workspace_server(ws)
        assert client.transport.timeout == 62                                                # the limit plus a minute
        started = time.monotonic()
        result = client.call_tool("run_shell", {"command": "sleep 10"})
        assert "exit code: 124" in result["content"] and time.monotonic() - started < 6
        client.close()
    finally:
        ws.discard(); ws.cleanup()


# ======================================================================
# lifecycle: nothing is left behind
# ======================================================================
def leftovers(task_id):
    containers = docker("ps", "-a", "--filter", f"name={task_id}", "--format", "{{.Names}}").stdout.split()
    networks = docker("network", "ls", "--filter", f"name={task_id}", "--format", "{{.Name}}").stdout.split()
    return containers, networks

def test_cleanup_removes_the_sandbox_the_proxy_and_the_network(egress_net, cache_home, tmp_path):
    ws = Workspace(str(make_repo(tmp_path / "repo")), environment=registries_env(egress_net)); ws.start()
    assert len(leftovers(ws.task_id)[0]) == 2 and leftovers(ws.task_id)[1]
    ws.discard(); ws.cleanup()
    assert leftovers(ws.task_id) == ([], [])
    ws.cleanup()                                                                              # and a second call is harmless

def test_a_start_that_fails_halfway_leaves_nothing_behind(egress_net, cache_home, tmp_path):
    env = registries_env(egress_net, image="ce-agent-no-such-image:never")
    ws = Workspace(str(make_repo(tmp_path / "repo")), environment=env)
    with pytest.raises(RuntimeError, match="Could not start container"):
        ws.start()                                                                            # the proxy was already up by then
    assert leftovers(ws.task_id) == ([], [])
    assert "agent/" not in sh("git", "branch", cwd=tmp_path / "repo")

def test_recover_removes_a_proxy_and_network_left_by_a_crashed_run(egress_net, cache_home, tmp_path):
    repo = make_repo(tmp_path / "repo")
    ws = Workspace(str(repo), environment=registries_env(egress_net)); ws.start()
    assert leftovers(ws.task_id)[1]
    message = recover(str(repo))                                                              # "the harness was killed"
    assert "leftover container" in message and "leftover sandbox network" in message
    assert leftovers(ws.task_id) == ([], [])

def test_two_sessions_at_once_get_separate_networks_and_proxies(egress_net, cache_home, tmp_path):
    a = Workspace(str(make_repo(tmp_path / "a")), environment=registries_env(egress_net)); a.start()
    b = Workspace(str(make_repo(tmp_path / "b")), environment=registries_env(egress_net)); b.start()
    try:
        assert a._egress.network != b._egress.network and a._egress.container != b._egress.container
        assert "hello from upstream" in run_script(a, "f.py", FETCH, "http://upstream.test/hello.txt")
        assert "hello from upstream" in run_script(b, "f.py", FETCH, "http://upstream.test/hello.txt")
    finally:
        for w in (a, b):
            w.discard(); w.cleanup()
    assert leftovers(a.task_id) == ([], []) and leftovers(b.task_id) == ([], [])
