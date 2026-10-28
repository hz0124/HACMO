from __future__ import annotations

from pathlib import Path

from supernode_sim.backends.ramulator import TimedHBMRequest, _write_timed_transactions


def test_timed_trace_preserves_repeated_cacheline_transactions(tmp_path: Path) -> None:
    requests = (
        TimedHBMRequest("a", source_core=0, release_time_ns=0.0, address=0, size_bytes=128),
        TimedHBMRequest("b", source_core=1, release_time_ns=5.0, address=0, size_bytes=128),
    )
    trace = tmp_path / "timed.trace"

    parents, counts, dram_bytes = _write_timed_transactions(
        requests,
        trace,
        cacheline_bytes=128,
        dram_transaction_size_bytes=32,
        tck_ns=1.0,
    )

    assert trace.read_text(encoding="utf-8").splitlines() == [
        "0 0 0 0x0",
        "1 0 0 0x20",
        "2 0 0 0x40",
        "3 0 0 0x60",
        "4 5 1 0x0",
        "5 5 1 0x20",
        "6 5 1 0x40",
        "7 5 1 0x60",
    ]
    assert parents == {
        0: "a",
        1: "a",
        2: "a",
        3: "a",
        4: "b",
        5: "b",
        6: "b",
        7: "b",
    }
    assert counts == {"a": 4, "b": 4}
    assert dram_bytes == 256


def test_timed_trace_expands_unaligned_logical_request(tmp_path: Path) -> None:
    request = TimedHBMRequest(
        "unaligned", source_core=3, release_time_ns=2.0, address=64, size_bytes=656
    )
    trace = tmp_path / "timed.trace"

    _, counts, dram_bytes = _write_timed_transactions(
        (request,),
        trace,
        cacheline_bytes=128,
        dram_transaction_size_bytes=32,
        tck_ns=1.0,
    )

    assert counts["unaligned"] == 24
    assert len(trace.read_text(encoding="utf-8").splitlines()) == 24
    assert dram_bytes == 24 * 32
