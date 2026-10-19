"""
An allowlist forward proxy for the sandbox. Standard library only; it runs in its own container.

The sandbox container has NO route to the internet: it sits on an internal Docker network whose only other member is
this proxy. So everything that leaves the sandbox passes through here, and the policy below decides what may.

  HTTPS:  the client sends  CONNECT host:443  and then speaks TLS through the tunnel. We see only the host name.
  HTTP:   the client sends  GET http://host/path  and we forward it.

A connection is allowed only if ALL of these hold:
  1. the port is allowed (443 and 80 by default);
  2. the target is a host NAME, not an IP address;
  3. the name is on the allowlist (exact, or "*.example.com" for subdomains);
  4. every address the name resolves to is a public one. This stops an allowed name from being used to reach
     internal services (a DNS entry pointing at 10.0.0.5 or 169.254.169.254). We then connect to the very address we
     checked, so a second DNS answer cannot change the outcome (DNS rebinding).

Every decision is written to stdout as one JSON line, which the harness reads when the session ends.

    python egress_proxy.py --port 3128 --allow pypi.org --allow files.pythonhosted.org
"""

import argparse
import ipaddress
import json
import re
import select
import socket
import socketserver
import sys
import threading
import time
from urllib.parse import urlsplit

CONNECT_TIMEOUT = 10        # seconds to open the outgoing connection
IDLE_TIMEOUT = 120          # a tunnel with no traffic for this long is closed
HEAD_TIMEOUT = 30           # seconds a client may take to send its request line and headers
MAX_HEAD_BYTES = 16 * 1024
MAX_CONNECTIONS = 64
DEFAULT_PORTS = (443, 80)
HOST_PATTERN = re.compile(r"^(\*\.)?[a-z0-9]([a-z0-9-]*[a-z0-9])?(\.[a-z0-9]([a-z0-9-]*[a-z0-9])?)+$")
HOP_BY_HOP = {"proxy-connection", "proxy-authorization", "connection", "keep-alive", "te", "trailer", "upgrade"}


def normalize_host(host):
    return host.strip().rstrip(".").strip("[]").rstrip(".").lower()


def is_public(address):
    """True only for addresses that are reachable on the public internet (not private, loopback, link-local, ...)."""
    ip = ipaddress.ip_address(address.split("%")[0])
    if ip.version == 6 and ip.ipv4_mapped:                  # ::ffff:127.0.0.1 is really 127.0.0.1
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast             # is_global alone lets some multicast ranges through


def is_ip_literal(host):
    try:
        ipaddress.ip_address(normalize_host(host))
        return True
    except ValueError:
        return False


class Decision:
    def __init__(self, allowed, reason, address=None):
        self.allowed, self.reason, self.address = allowed, reason, address


class Policy:
    def __init__(self, allowed_hosts, allowed_ports=DEFAULT_PORTS, allow_private=False, resolver=socket.getaddrinfo):
        self.patterns = []
        for entry in allowed_hosts:
            entry = normalize_host(entry)
            if not HOST_PATTERN.match(entry):
                raise ValueError(f"not a valid host name or '*.domain' pattern: {entry!r}")
            self.patterns.append(entry)
        self.allowed_ports = set(allowed_ports)
        self.allow_private = allow_private      # for registries on a private network; off by default
        self.resolver = resolver

    def host_matches(self, host):
        host = normalize_host(host)
        for pattern in self.patterns:
            if pattern == host or (pattern.startswith("*.") and host.endswith(pattern[1:]) and host != pattern[2:]):
                return True
        return False

    def check(self, host, port):
        host = normalize_host(host)
        if port not in self.allowed_ports:
            return Decision(False, f"port {port} is not allowed")
        if is_ip_literal(host):
            return Decision(False, "IP addresses are not allowed, use the host name")
        if not self.host_matches(host):
            return Decision(False, f"{host} is not on the allowlist")
        try:
            infos = self.resolver(host, port, type=socket.SOCK_STREAM)
        except OSError as e:
            return Decision(False, f"could not resolve {host}: {e}")
        addresses = []
        for info in infos:
            ip = info[4][0]
            if ip not in addresses:
                addresses.append(ip)
        if not addresses:
            return Decision(False, f"{host} did not resolve to any address")
        if not self.allow_private:
            for ip in addresses:
                if not is_public(ip):
                    return Decision(False, f"{host} resolves to a non-public address ({ip})")
        return Decision(True, "allowed", addresses[0])


# ======================================================================
# the server
# ======================================================================
def read_head(sock):
    """Read up to the blank line that ends the request head. Returns (head bytes, bytes already read after it)."""
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = sock.recv(4096)
        if not chunk:
            return None, b""
        data += chunk
        if len(data) > MAX_HEAD_BYTES:
            raise ValueError("request head too large")
    head, _, rest = data.partition(b"\r\n\r\n")
    return head, rest


def parse_host_port(target, default_port):
    """'host:443' or '[::1]:443' or 'host' -> (host, port)."""
    if target.startswith("["):
        host, _, rest = target[1:].partition("]")
        port_text = rest.lstrip(":")
    else:
        host, _, port_text = target.rpartition(":") if ":" in target else (target, "", "")
    if not port_text:
        return host, default_port
    if not port_text.isdigit() or not 0 < int(port_text) < 65536:
        raise ValueError(f"bad port in {target!r}")
    return host, int(port_text)


def relay(a, b, idle_timeout):
    """Copy bytes both ways until both sides are done. Returns (bytes a->b, bytes b->a)."""
    open_sockets = {a: b, b: a}                      # each socket -> the one its data goes to
    sent = {a: 0, b: 0}
    while open_sockets:
        readable, _, _ = select.select(list(open_sockets), [], [], idle_timeout)
        if not readable:
            break                                     # idle too long
        for sock in readable:
            if sock not in open_sockets:
                continue
            try:
                data = sock.recv(65536)
            except OSError:
                data = b""
            other = open_sockets[sock]
            if not data:                              # this side finished sending: pass the half-close on
                del open_sockets[sock]
                try:
                    other.shutdown(socket.SHUT_WR)
                except OSError:
                    open_sockets.pop(other, None)
                continue
            try:
                other.sendall(data)
            except OSError:
                return sent[a], sent[b]
            sent[sock] += len(data)
    return sent[a], sent[b]


class ProxyHandler(socketserver.BaseRequestHandler):
    def handle(self):
        server = self.server
        client = self.request
        if not server.slots.acquire(blocking=False):
            self._respond(503, "too many connections")
            return
        started = time.time()
        record = {"host": None, "port": None, "action": "deny", "reason": ""}
        try:
            client.settimeout(HEAD_TIMEOUT)
            head, rest = read_head(client)
            if head is None:
                return
            lines = head.decode("latin-1").split("\r\n")
            parts = lines[0].split(" ")
            if len(parts) != 3:
                record["reason"] = "malformed request line"
                self._respond(400, "malformed request")
                return
            method, target, version = parts
            headers = [line.split(":", 1) for line in lines[1:] if ":" in line]

            try:
                if method.upper() == "CONNECT":
                    host, port = parse_host_port(target, 443)
                    record.update(host=host, port=port, kind="connect")
                else:
                    url = urlsplit(target)
                    if url.scheme != "http" or not url.hostname:
                        record["reason"] = "only CONNECT and absolute http:// requests are supported"
                        self._respond(400, record["reason"])
                        return
                    host, port = url.hostname, url.port or 80      # .port raises ValueError for a bad port
                    record.update(host=host, port=port, kind="http")
            except ValueError as e:
                record["reason"] = f"malformed target: {e}"
                self._respond(400, record["reason"])
                return

            decision = server.policy.check(host, port)
            record["reason"] = decision.reason
            if not decision.allowed:
                self._respond(403, f"blocked by the sandbox egress policy: {decision.reason}")
                return

            try:
                upstream = socket.create_connection((decision.address, port), timeout=CONNECT_TIMEOUT)
            except OSError as e:
                record["reason"] = f"could not connect to {host}: {e}"
                self._respond(502, record["reason"])
                return
            record["action"] = "allow"
            with upstream:
                client.settimeout(None)
                upstream.settimeout(None)
                if record["kind"] == "connect":
                    client.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                else:
                    origin_target = (url.path or "/") + (f"?{url.query}" if url.query else "")
                    out = [f"{method} {origin_target} {version}"]
                    out += [f"{k.strip()}:{v}" for k, v in headers if k.strip().lower() not in HOP_BY_HOP]
                    out.append("Connection: close")
                    upstream.sendall(("\r\n".join(out) + "\r\n\r\n").encode("latin-1"))
                if rest:
                    upstream.sendall(rest)
                up, down = relay(client, upstream, IDLE_TIMEOUT)
                record["bytes_up"], record["bytes_down"] = up, down
        except (ValueError, OSError) as e:
            record["reason"] = record["reason"] or f"error: {e}"
        finally:
            record["seconds"] = round(time.time() - started, 3)
            server.log(record)
            server.slots.release()

    def _respond(self, status, text):
        body = (text + "\n").encode()
        reason = {400: "Bad Request", 403: "Forbidden", 502: "Bad Gateway", 503: "Service Unavailable"}.get(status, "Error")
        head = f"HTTP/1.1 {status} {reason}\r\nContent-Type: text/plain\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n"
        try:
            self.request.sendall(head.encode() + body)
        except OSError:
            pass


class ProxyServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 64

    def __init__(self, address, policy, log=None):
        super().__init__(address, ProxyHandler)
        self.policy = policy
        self.slots = threading.BoundedSemaphore(MAX_CONNECTIONS)
        self._log, self._log_lock = log, threading.Lock()

    def log(self, record):
        record = {"time": round(time.time(), 3), **record}
        if self._log:
            with self._log_lock:
                self._log(record)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=3128)
    parser.add_argument("--allow", action="append", default=[], help="an allowed host name, or *.domain (repeatable)")
    parser.add_argument("--allow-port", action="append", type=int, default=[], help="an allowed port (default 443 and 80)")
    parser.add_argument("--allow-private", action="store_true", help="allow names that resolve to private addresses")
    args = parser.parse_args(argv)

    def emit(record):
        print(json.dumps(record), flush=True)

    policy = Policy(args.allow, args.allow_port or DEFAULT_PORTS, args.allow_private)
    server = ProxyServer((args.host, args.port), policy, log=emit)
    emit({"event": "listening", "port": server.server_address[1], "allowed": policy.patterns,
          "ports": sorted(policy.allowed_ports), "allow_private": args.allow_private})
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    sys.exit(main())
