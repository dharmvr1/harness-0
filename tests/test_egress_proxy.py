"""The allowlist egress proxy: the policy in isolation, then the real proxy against a real local upstream."""
import json
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import ce_agent.egress_proxy as proxy_module
import pytest
from ce_agent.egress_proxy import Policy, ProxyServer, is_ip_literal, is_public, normalize_host, parse_host_port

PROXY_SCRIPT = str(Path(proxy_module.__file__))


def resolver_for(mapping):
    """A fake DNS: host -> list of addresses."""
    def resolve(host, port, type=0):
        if host not in mapping:
            raise socket.gaierror(-2, "Name or service not known")
        result = []
        for ip in mapping[host]:
            if ":" in ip:
                result.append((socket.AF_INET6, socket.SOCK_STREAM, 6, "", (ip, port, 0, 0)))
            else:
                result.append((socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port)))
        return result
    return resolve


PUBLIC = "93.184.216.34"


# ======================================================================
# the policy
# ======================================================================
def policy(hosts=("pypi.org", "*.crates.io"), dns=None, **kw):
    dns = {"pypi.org": [PUBLIC], "static.crates.io": [PUBLIC], "crates.io": [PUBLIC]} if dns is None else dns
    return Policy(hosts, resolver=resolver_for(dns), **kw)


def test_an_exact_name_on_the_list_is_allowed_and_the_checked_address_is_returned():
    d = policy().check("pypi.org", 443)
    assert d.allowed and d.address == PUBLIC and d.reason == "allowed"

@pytest.mark.parametrize("host", ["PyPI.org", "pypi.org.", "  pypi.org  ", "PYPI.ORG."])
def test_case_whitespace_and_a_trailing_dot_do_not_matter(host):
    assert policy().check(host, 443).allowed

def test_a_wildcard_matches_subdomains_but_not_the_bare_domain():
    p = policy()
    assert p.check("static.crates.io", 443).allowed
    assert policy(dns={"a.b.crates.io": [PUBLIC]}).check("a.b.crates.io", 443).allowed       # any depth below the wildcard
    assert not Policy(["*.crates.io"], resolver=resolver_for({"crates.io": [PUBLIC]})).check("crates.io", 443).allowed

@pytest.mark.parametrize("host", ["evil.com", "pypi.org.evil.com", "notpypi.org", "pypi.org@evil.com", "xpypi.org", "crates.io.evil.com", "evilcrates.io"])
def test_names_that_only_look_like_allowed_ones_are_refused(host):
    d = policy(dns={h: [PUBLIC] for h in (host,)}).check(host, 443)
    assert not d.allowed and "not on the allowlist" in d.reason

@pytest.mark.parametrize("port", [22, 25, 8080, 0, 65535, 444])
def test_only_the_allowed_ports_pass(port):
    d = policy().check("pypi.org", port)
    assert not d.allowed and f"port {port} is not allowed" in d.reason

def test_ports_are_configurable():
    assert policy(allowed_ports=[8443]).check("pypi.org", 8443).allowed and not policy(allowed_ports=[8443]).check("pypi.org", 443).allowed

@pytest.mark.parametrize("literal", ["1.2.3.4", "127.0.0.1", "::1", "[::1]", "10.0.0.5", "169.254.169.254", "2001:db8::1", "::ffff:127.0.0.1"])
def test_ip_literals_are_never_allowed_even_if_they_look_listed(literal):
    d = Policy([literal.strip("[]")] if False else ["pypi.org"], resolver=resolver_for({})).check(literal, 443)
    assert not d.allowed and "IP addresses are not allowed" in d.reason

def test_a_wildcard_subdomain_that_does_not_resolve_is_just_refused():
    assert not policy().check("a.b.crates.io", 443).allowed

@pytest.mark.parametrize("address", ["10.0.0.5", "172.16.0.1", "192.168.1.1", "127.0.0.1", "169.254.169.254", "0.0.0.0",
                                      "100.64.0.1", "::1", "fd00::1", "fe80::1", "::ffff:127.0.0.1", "::ffff:10.0.0.1", "224.0.0.1"])
def test_an_allowed_name_that_resolves_to_a_non_public_address_is_refused(address):
    d = policy(dns={"pypi.org": [address]}).check("pypi.org", 443)
    assert not d.allowed and "non-public address" in d.reason

def test_one_private_answer_among_public_ones_is_enough_to_refuse():            # DNS rebinding / round-robin tricks
    d = policy(dns={"pypi.org": [PUBLIC, "10.0.0.5"]}).check("pypi.org", 443)
    assert not d.allowed and "10.0.0.5" in d.reason

def test_public_addresses_pass_including_ipv6():
    assert policy(dns={"pypi.org": ["2606:4700:4700::1111"]}).check("pypi.org", 443).allowed

def test_allow_private_is_for_registries_on_a_private_network():
    assert policy(dns={"pypi.org": ["10.0.0.5"]}, allow_private=True).check("pypi.org", 443).allowed
    assert not policy(dns={"pypi.org": ["10.0.0.5"]}, allow_private=True).check("1.2.3.4", 443).allowed     # literals stay refused

def test_a_name_that_does_not_resolve_is_refused_with_a_reason():
    d = policy(dns={}).check("pypi.org", 443)
    assert not d.allowed and "could not resolve" in d.reason

def test_the_checked_address_is_what_gets_connected_to():
    d = policy(dns={"pypi.org": [PUBLIC, "93.184.216.35"]}).check("pypi.org", 443)
    assert d.address == PUBLIC                                                   # first answer, resolved exactly once here

@pytest.mark.parametrize("bad", ["", "evil", "a..b", "-a.com", "a.com/path", "*evil.com", "a*.com", "*.", "ex ample.com", "http://a.com", "a.com:80", "*.*.com"])
def test_invalid_patterns_are_rejected_when_the_policy_is_built(bad):
    with pytest.raises(ValueError, match="not a valid host name"):
        Policy([bad])

def test_helper_functions():
    assert normalize_host(" [::1]. ") == "::1" and normalize_host("A.B.") == "a.b"
    assert is_ip_literal("1.2.3.4") and is_ip_literal("[::1]") and not is_ip_literal("pypi.org") and not is_ip_literal("1.2.3.4.example.com")
    assert is_public("93.184.216.34") and not is_public("192.168.0.1") and not is_public("fe80::1%eth0")
    assert parse_host_port("a.com:8443", 443) == ("a.com", 8443) and parse_host_port("a.com", 443) == ("a.com", 443)
    assert parse_host_port("[::1]:8443", 443) == ("::1", 8443) and parse_host_port("[::1]", 443) == ("::1", 443)
    for bad in ("a.com:abc", "a.com:0", "a.com:99999", "a.com:"):
        if bad == "a.com:":
            assert parse_host_port(bad, 443) == ("a.com", 443)
        else:
            with pytest.raises(ValueError):
                parse_host_port(bad, 443)


# ======================================================================
# the real proxy, in-process, in front of a real upstream
# ======================================================================
class Upstream:
    """A tiny HTTP server that remembers the requests it saw."""
    def __init__(self):
        outer = self
        self.seen = []
        class Handler(BaseHTTPRequestHandler):
            def _record(self, body=b""):
                outer.seen.append({"line": self.requestline, "headers": {k.lower(): v for k, v in self.headers.items()}, "body": body})
            def do_GET(self):
                self._record()
                payload = b"hello from upstream"
                self.send_response(200); self.send_header("Content-Length", str(len(payload))); self.end_headers(); self.wfile.write(payload)
            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                self._record(body)
                self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body[::-1])
            def log_message(self, *a):
                pass
        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()

    def close(self):
        self.httpd.shutdown(); self.httpd.server_close()


class Running:
    def __init__(self, server, upstream, logs):
        self.server, self.upstream, self.logs = server, upstream, logs
        self.port = server.server_address[1]

    def connect(self):
        s = socket.create_connection(("127.0.0.1", self.port), timeout=10)
        return s

    def wait_for_log(self, count=1, seconds=5):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline and len(self.logs) < count:
            time.sleep(0.02)
        return list(self.logs)


@pytest.fixture
def running():
    created = []
    def make(allow_private=True, extra_hosts=(), dns_extra=None):
        upstream = Upstream()
        logs = []
        dns = {"upstream.test": ["127.0.0.1"], "internal.test": ["10.1.2.3"], **(dns_extra or {})}
        pol = Policy(["upstream.test", "internal.test", *extra_hosts], allowed_ports=[upstream.port], allow_private=allow_private,
                     resolver=resolver_for(dns))
        server = ProxyServer(("127.0.0.1", 0), pol, log=logs.append)
        threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
        r = Running(server, upstream, logs)
        created.append(r)
        return r
    yield make
    for r in created:
        r.server.shutdown(); r.server.server_close(); r.upstream.close()


def exchange(sock, data, read_until_close=True):
    sock.sendall(data)
    chunks = []
    while True:
        try:
            chunk = sock.recv(65536)
        except socket.timeout:
            break
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)


def test_https_style_tunnel_to_an_allowed_host(running):
    r = running()
    s = r.connect()
    s.sendall(f"CONNECT upstream.test:{r.upstream.port} HTTP/1.1\r\nHost: upstream.test\r\n\r\n".encode())
    assert s.recv(4096).startswith(b"HTTP/1.1 200 Connection Established")
    reply = exchange(s, b"GET /through-the-tunnel HTTP/1.1\r\nHost: upstream.test\r\nConnection: close\r\n\r\n")
    s.close()
    assert b"200 OK" in reply and b"hello from upstream" in reply
    assert r.upstream.seen[0]["line"].startswith("GET /through-the-tunnel")
    record = r.wait_for_log()[0]
    assert record["action"] == "allow" and record["host"] == "upstream.test" and record["kind"] == "connect"
    assert record["bytes_up"] > 0 and record["bytes_down"] > 0

def test_plain_http_is_forwarded_in_origin_form_without_proxy_headers(running):
    r = running()
    s = r.connect()
    request = (f"GET http://upstream.test:{r.upstream.port}/hello?x=1 HTTP/1.1\r\nHost: upstream.test:{r.upstream.port}\r\n"
               "Proxy-Authorization: Basic c2VjcmV0\r\nProxy-Connection: keep-alive\r\nX-Custom: kept\r\n\r\n")
    reply = exchange(s, request.encode())
    s.close()
    assert b"200 OK" in reply and reply.endswith(b"hello from upstream")
    seen = r.upstream.seen[0]
    assert seen["line"] == "GET /hello?x=1 HTTP/1.1"                                  # the absolute URL became a path
    assert seen["headers"]["x-custom"] == "kept" and seen["headers"]["connection"] == "close"
    assert "proxy-authorization" not in seen["headers"] and "proxy-connection" not in seen["headers"]
    assert r.wait_for_log()[0]["kind"] == "http"

def test_a_request_body_is_forwarded(running):
    r = running()
    s = r.connect()
    body = b"abcdef"
    request = (f"POST http://upstream.test:{r.upstream.port}/echo HTTP/1.1\r\nHost: upstream.test\r\n"
               f"Content-Length: {len(body)}\r\n\r\n").encode() + body
    reply = exchange(s, request)
    s.close()
    assert reply.endswith(b"fedcba") and r.upstream.seen[0]["body"] == body

def test_a_client_that_half_closes_still_gets_the_whole_answer(running):
    r = running()
    s = r.connect()
    s.sendall(f"GET http://upstream.test:{r.upstream.port}/ HTTP/1.1\r\nHost: x\r\n\r\n".encode())
    s.shutdown(socket.SHUT_WR)
    chunks = []
    while True:
        chunk = s.recv(4096)
        if not chunk:
            break
        chunks.append(chunk)
    assert b"hello from upstream" in b"".join(chunks)

def test_a_host_that_is_not_on_the_list_gets_403_with_the_reason_and_is_logged(running):
    r = running()
    s = r.connect()
    reply = exchange(s, f"CONNECT evil.example:{r.upstream.port} HTTP/1.1\r\nHost: evil.example\r\n\r\n".encode())
    s.close()
    assert reply.startswith(b"HTTP/1.1 403 Forbidden") and b"blocked by the sandbox egress policy" in reply and b"evil.example is not on the allowlist" in reply
    record = r.wait_for_log()[0]
    assert record["action"] == "deny" and record["host"] == "evil.example" and "allowlist" in record["reason"]

def test_ip_literals_and_wrong_ports_are_refused(running):
    r = running()
    for target in ("127.0.0.1:%d" % r.upstream.port, "[::1]:%d" % r.upstream.port, "upstream.test:22", "upstream.test:99999"):
        s = r.connect()
        reply = exchange(s, f"CONNECT {target} HTTP/1.1\r\n\r\n".encode())
        s.close()
        assert reply.startswith((b"HTTP/1.1 403", b"HTTP/1.1 400")), target
    assert r.upstream.seen == []

def test_a_private_resolution_is_refused_unless_allowed(running):
    r = running(allow_private=False)
    s = r.connect()
    reply = exchange(s, f"CONNECT upstream.test:{r.upstream.port} HTTP/1.1\r\n\r\n".encode())
    s.close()
    assert reply.startswith(b"HTTP/1.1 403") and b"non-public address (127.0.0.1)" in reply
    assert r.upstream.seen == []

def test_malformed_and_unsupported_requests_get_400(running):
    r = running()
    for raw in (b"HELLO\r\n\r\n", b"GET /relative HTTP/1.1\r\nHost: x\r\n\r\n", b"GET ftp://upstream.test/x HTTP/1.1\r\n\r\n",
                b"GET https://upstream.test/x HTTP/1.1\r\n\r\n"):
        s = r.connect()
        assert exchange(s, raw).startswith(b"HTTP/1.1 400"), raw
        s.close()

def test_an_oversized_request_head_is_dropped_without_a_crash(running):
    r = running()
    s = r.connect()
    try:
        s.sendall(b"GET http://upstream.test/ HTTP/1.1\r\n" + b"X: " + b"a" * 40000 + b"\r\n\r\n")
    except OSError:
        pass
    s.close()
    s = r.connect()                                                                  # the proxy still works
    s.sendall(f"CONNECT upstream.test:{r.upstream.port} HTTP/1.1\r\n\r\n".encode())
    assert s.recv(100).startswith(b"HTTP/1.1 200")
    s.close()

def test_an_unreachable_upstream_is_502(running):
    r = running(dns_extra={"gone.test": ["127.0.0.1"]}, extra_hosts=["gone.test"])
    # pick a port nothing listens on, but allow it
    dead = socket.socket(); dead.bind(("127.0.0.1", 0)); dead_port = dead.getsockname()[1]; dead.close()
    r.server.policy.allowed_ports.add(dead_port)
    s = r.connect()
    assert exchange(s, f"CONNECT gone.test:{dead_port} HTTP/1.1\r\n\r\n".encode()).startswith(b"HTTP/1.1 502")

def test_many_tunnels_at_once(running):
    r = running()
    results = []
    def worker():
        s = r.connect()
        s.sendall(f"CONNECT upstream.test:{r.upstream.port} HTTP/1.1\r\n\r\n".encode())
        ok = s.recv(100).startswith(b"HTTP/1.1 200")
        body = exchange(s, b"GET / HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
        s.close()
        results.append(ok and b"hello from upstream" in body)
    threads = [threading.Thread(target=worker) for _ in range(25)]
    [t.start() for t in threads]; [t.join(30) for t in threads]
    assert results == [True] * 25

def test_too_many_connections_get_503(running, monkeypatch):
    monkeypatch.setattr(proxy_module, "MAX_CONNECTIONS", 2)
    r = running()
    held = []
    for _ in range(2):                                                               # two tunnels held open
        s = r.connect()
        s.sendall(f"CONNECT upstream.test:{r.upstream.port} HTTP/1.1\r\n\r\n".encode())
        assert s.recv(100).startswith(b"HTTP/1.1 200")
        held.append(s)
    third = r.connect()
    assert exchange(third, b"CONNECT upstream.test:1 HTTP/1.1\r\n\r\n").startswith(b"HTTP/1.1 503")
    [s.close() for s in held]; third.close()

def test_an_idle_tunnel_is_closed(running, monkeypatch):
    monkeypatch.setattr(proxy_module, "IDLE_TIMEOUT", 0.4)
    r = running()
    s = r.connect()
    s.settimeout(5)
    s.sendall(f"CONNECT upstream.test:{r.upstream.port} HTTP/1.1\r\n\r\n".encode())
    assert s.recv(100).startswith(b"HTTP/1.1 200")
    started = time.monotonic()
    assert s.recv(100) == b""                                                        # closed by the proxy, no data sent by us
    assert time.monotonic() - started < 3

def test_the_script_runs_standalone_and_announces_itself():
    process = subprocess.Popen([sys.executable, PROXY_SCRIPT, "--host", "127.0.0.1", "--port", "0", "--allow", "pypi.org", "--allow", "*.pythonhosted.org"],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        announcement = json.loads(process.stdout.readline())
    finally:
        process.terminate(); process.communicate(timeout=10)
    assert announcement["event"] == "listening" and announcement["allowed"] == ["pypi.org", "*.pythonhosted.org"]
    assert announcement["ports"] == [80, 443] and announcement["allow_private"] is False and announcement["port"] > 0

def test_the_script_refuses_a_bad_allowlist_entry():
    result = subprocess.run([sys.executable, PROXY_SCRIPT, "--allow", "evil.com/path"], capture_output=True, text=True, timeout=20)
    assert result.returncode != 0 and "not a valid host name" in result.stderr
