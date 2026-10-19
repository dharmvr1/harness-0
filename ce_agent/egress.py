"""
The Docker side of network mode "registries": an isolated network for the sandbox, plus the proxy container.

    sandbox container ──(internal network: no route out)──► proxy container ──(egress network)──► the internet
                                                             allows only the listed hosts

  1. `docker network create --internal`: containers on it can talk to each other but not to anything else.
  2. The proxy container joins that network (under the name `egress-proxy`) AND the normal egress network.
  3. The sandbox container joins only the internal network and is told to use the proxy.

The sandbox cannot resolve outside names or reach any IP directly (checked by hand: "Network is unreachable"), so the
proxy is its only way out. Every name, network and container is labelled so `--cleanup` can find leftovers.
"""

import json
import subprocess
import time
from pathlib import Path

import ce_agent.egress_proxy as egress_proxy

PROXY_IMAGE = "python:3.11-slim"     # the proxy needs nothing but Python
PROXY_ALIAS = "egress-proxy"         # the name the sandbox uses (no underscores: some URL parsers dislike them)
PROXY_PORT = 3128
STARTUP_SECONDS = 20


def docker(*args, check=True, timeout=60):
    result = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)
    if check and result.returncode != 0:
        raise RuntimeError(f"docker {' '.join(args[:3])} failed: {result.stderr.strip()}")
    return result


class EgressProxy:
    def __init__(self, task_id, top, hosts, allow_private=False, egress_network="bridge", image=PROXY_IMAGE):
        self.network = f"ce_net_{task_id}"
        self.container = f"ce_proxy_{task_id}"
        self.top, self.hosts, self.allow_private = str(top), list(hosts), allow_private
        self.egress_network, self.image = egress_network, image
        self.url = f"http://{PROXY_ALIAS}:{PROXY_PORT}"

    def start(self):
        labels = ["--label", "ce_agent=true", "--label", f"ce_agent_repo={self.top}"]
        script = Path(egress_proxy.__file__).resolve()
        try:
            docker("network", "create", "--internal", *labels, self.network)
            command = ["python", "/egress_proxy.py", "--port", str(PROXY_PORT)]
            for host in self.hosts:
                command += ["--allow", host]
            if self.allow_private:
                command.append("--allow-private")
            docker("run", "-d", "--name", self.container, "--network", self.network, "--network-alias", PROXY_ALIAS, *labels,
                   "--init", "--read-only", "--tmpfs", "/tmp", "--memory", "128m", "--pids-limit", "128",
                   "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--user", "65534:65534",
                   "-e", "PYTHONDONTWRITEBYTECODE=1", "-v", f"{script}:/egress_proxy.py:ro", self.image, *command)
            docker("network", "connect", self.egress_network, self.container)    # now it also has a way out
            self._wait_until_listening()
        except Exception:
            self.stop()
            raise

    def _wait_until_listening(self):
        deadline = time.monotonic() + STARTUP_SECONDS
        while time.monotonic() < deadline:
            if any(entry.get("event") == "listening" for entry in self.logs()):
                return
            state = docker("inspect", "--format", "{{.State.Running}}", self.container, check=False).stdout.strip()
            if state == "false":
                raise RuntimeError("the egress proxy exited at startup:\n" + docker("logs", self.container, check=False).stderr.strip())
            time.sleep(0.1)
        raise RuntimeError(f"the egress proxy did not start within {STARTUP_SECONDS}s")

    def logs(self):
        """The proxy's decisions so far, as dicts (one per connection), oldest first."""
        out = docker("logs", self.container, check=False).stdout
        entries = []
        for line in out.splitlines():
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return entries

    def stop(self):
        docker("rm", "-f", self.container, check=False)
        for _ in range(3):                                   # the network can take a moment to become removable
            if docker("network", "rm", self.network, check=False).returncode == 0:
                return
            time.sleep(0.3)


def summarize(entries):
    """Proxy log entries -> (allowed count per host, list of blocked 'host: reason')."""
    allowed, blocked = {}, []
    for entry in entries:
        if entry.get("event") or "host" not in entry:
            continue
        if entry.get("action") == "allow":
            allowed[entry["host"]] = allowed.get(entry["host"], 0) + 1
        else:
            note = f"{entry.get('host')}: {entry.get('reason')}"
            if note not in blocked:
                blocked.append(note)
    return allowed, blocked
