"""
Benchmark: the same tool calls made in-process, over MCP/stdio and over MCP/Streamable HTTP.

    python -m ce_agent.mcp.bench [--calls 100] [--rounds 5] [--starts 5]

It measures, on a throwaway repo, without Docker (so that only the protocol and process overhead shows):
  cold start   from "launch the server" until the first server/discover has been answered
  per call     latency of list_files (small reply), read_file of 1 KB, and read_file of about 120 KB

Numbers depend on the machine and are only worth comparing with each other. HTTP here opens a new TCP connection for
every request (the server speaks HTTP/1.0), which is part of what is being measured.
"""

import argparse
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

from ce_agent.mcp.client import launch_workspace_server
from ce_agent.registry import Registry
from ce_agent.tools import make_workspace_tools
from ce_agent.workspace import Workspace


def sh(*cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True).stdout


def make_workspace():
    repo = Path(tempfile.mkdtemp(prefix="mcp-bench-"))
    sh("git", "init", "-q", "-b", "main", cwd=repo)
    (repo / "small.txt").write_text("".join(f"line {i}\n" for i in range(80)))                  # about 1 KB
    (repo / "big.txt").write_text("".join(f"line number {i} of the big file\n" for i in range(3500)))   # about 120 KB
    for i in range(30):
        (repo / f"file_{i}.py").write_text("x = 1\n")
    sh("git", "add", "-A", cwd=repo)
    sh("git", "-c", "user.name=b", "-c", "user.email=b@b", "commit", "-qm", "bench", cwd=repo)
    ws = Workspace(str(repo), unsafe=True)
    ws.start()
    return ws


OPERATIONS = [
    ("list_files", {}),
    ("read_file 1 KB", {"file_path": "small.txt"}),
    ("read_file 120 KB", {"file_path": "big.txt"}),
]


def tool_name(label):
    return label.split()[0]


def percentile(sorted_values, fraction):
    return sorted_values[min(len(sorted_values) - 1, int(fraction * len(sorted_values)))]


def time_calls(call, count):
    times = []
    for _ in range(count):
        started = time.perf_counter()
        call()
        times.append((time.perf_counter() - started) * 1000)
    return times


def cold_start(ws, transport, starts):
    """Median time from launching the server until the first server/discover is answered."""
    times = []
    for _ in range(starts):
        started = time.perf_counter()
        client = launch_workspace_server(ws, transport=transport)      # spawn + python start-up + imports + discover
        times.append((time.perf_counter() - started) * 1000)
        client.close()
    return statistics.median(times)


def run(ws, calls, rounds, starts):
    """Measure all three modes in ROUNDS rounds, interleaved (A B C A B C ...) so that machine noise and warm-up
    drift hit them equally. Returns {mode: {"cold_ms", "ops": {label: {"p50", "p95", "low", "high"}}}}."""
    registry = Registry()
    for func, description in make_workspace_tools(ws):
        registry.register(func, description)
    stdio = launch_workspace_server(ws, transport="stdio")
    http = launch_workspace_server(ws, transport="http")
    try:
        modes = {
            "in-process": lambda name, a: registry.call(name, a),
            "mcp / stdio": stdio.call_tool,
            "mcp / http": http.call_tool,
        }
        for call in modes.values():                       # warm up every mode (imports, caches, first connection)
            for label, a in OPERATIONS:
                for _ in range(10):
                    assert call(tool_name(label), a)["status"] == "ok", label
        samples = {(m, l): [] for m in modes for l, _ in OPERATIONS}
        medians = {(m, l): [] for m in modes for l, _ in OPERATIONS}
        for _ in range(rounds):
            for mode, call in modes.items():
                for label, a in OPERATIONS:
                    times = time_calls(lambda c=call, n=tool_name(label), x=a: c(n, x), calls)
                    samples[(mode, label)] += times
                    medians[(mode, label)].append(statistics.median(times))
    finally:
        stdio.close()
        http.close()
    cold = {"in-process": 0.0, "mcp / stdio": cold_start(ws, "stdio", starts), "mcp / http": cold_start(ws, "http", starts)}
    results = {}
    for mode in modes:
        ops = {}
        for label, _ in OPERATIONS:
            pooled = sorted(samples[(mode, label)])
            round_medians = medians[(mode, label)]
            ops[label] = {"p50": statistics.median(round_medians), "p95": percentile(pooled, 0.95),
                          "low": min(round_medians), "high": max(round_medians)}
        results[mode] = {"cold_ms": cold[mode], "ops": ops}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calls", type=int, default=100, help="calls per operation in each round")
    parser.add_argument("--rounds", type=int, default=5, help="interleaved rounds")
    parser.add_argument("--starts", type=int, default=5, help="server start-ups measured for the cold start")
    args = parser.parse_args()

    ws = make_workspace()
    try:
        results = run(ws, args.calls, args.rounds, args.starts)
    finally:
        ws.discard()
        ws.cleanup()

    print(f"\n{args.rounds} interleaved rounds x {args.calls} calls = {args.rounds * args.calls} samples per cell; "
          f"cold start = median of {args.starts}; milliseconds")
    print("p50 = median of the round medians, [lowest .. highest round median]; p95 over all samples\n")
    print(f"{'':14}{'cold start':>11}" + "".join(f"{label:>34}" for label, _ in OPERATIONS))
    for name, result in results.items():
        row = f"{name:14}{result['cold_ms']:>11.1f}"
        for label, _ in OPERATIONS:
            m = result["ops"][label]
            row += f"{m['p50']:>10.2f} [{m['low']:.2f}..{m['high']:.2f}]".rjust(26) + f"{m['p95']:>8.2f}"
        print(row)
    base = results["in-process"]["ops"]
    print("\nMCP overhead per call, p50 difference to in-process (ms):")
    for name in ("mcp / stdio", "mcp / http"):
        print(f"  {name:12}" + "".join(f"  {label}: {results[name]['ops'][label]['p50'] - base[label]['p50']:+.2f}" for label, _ in OPERATIONS))


if __name__ == "__main__":
    main()
