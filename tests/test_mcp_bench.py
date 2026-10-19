"""The benchmark must keep working. These tests do not check timings, which depend on the machine."""
import ce_agent.mcp.bench as bench


def test_the_benchmark_runs_and_has_a_result_for_every_mode_and_operation():
    ws = bench.make_workspace()
    try:
        results = bench.run(ws, calls=3, rounds=2, starts=1)
    finally:
        ws.discard()
        ws.cleanup()
    assert set(results) == {"in-process", "mcp / stdio", "mcp / http"}
    for mode, result in results.items():
        assert set(result["ops"]) == {label for label, _ in bench.OPERATIONS}
        for m in result["ops"].values():
            assert 0 < m["low"] <= m["p50"] <= m["high"] and m["p95"] > 0
    assert results["in-process"]["cold_ms"] == 0 and results["mcp / stdio"]["cold_ms"] > 0 and results["mcp / http"]["cold_ms"] > 0


def test_percentile_and_the_big_file_really_is_big():
    assert bench.percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.95) == 10 and bench.percentile([5], 0.95) == 5
    ws = bench.make_workspace()
    try:
        assert 100_000 < len(ws.read_file("big.txt")) < 200_000
    finally:
        ws.discard()
        ws.cleanup()
