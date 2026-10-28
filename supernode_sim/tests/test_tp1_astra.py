from __future__ import annotations

from pathlib import Path

from supernode_sim.backends.tp1_astra import (
    TP1AstraAccess,
    TP1AstraConfig,
    TP1AstraPrefetch,
    TP1AstraWindow,
    _build_graph,
    _merge_window_graphs,
)


def _config(tmp_path: Path) -> TP1AstraConfig:
    return TP1AstraConfig(
        rows=2,
        cols=2,
        astra_sim_dir=Path("astra-sim"),
        output_dir=tmp_path,
        system_config=Path("system.json"),
        remote_memory_config=Path("memory.json"),
        link_bandwidth_gb_per_s=100.0,
        link_latency_ns=2.0,
        die_compute_tflops=100.0,
        flops_per_access=1_000_000.0,
        preceding_phase_us=10.0,
        max_items_per_flow=32,
    )


def test_projection_conserves_work_and_physical_traffic(tmp_path: Path) -> None:
    accesses = (
        TP1AstraAccess("local", 0, "local", 0, 0, 656),
        TP1AstraAccess("remote", 1, "remote", 0, 0, 656),
        TP1AstraAccess(
            "hbm",
            3,
            "hbm",
            None,
            2,
            656,
            hbm_port_core=0,
            hbm_wire_bytes=768,
            hbm_completion_ns=12_000.0,
        ),
    )
    graph, summary = _build_graph(_config(tmp_path), accesses, ())

    assert sum(summary["access_counts"]) == 3
    assert summary["source_counts"] == {
        "local": 1,
        "remote": 1,
        "hbm": 1,
        "prefetch": 0,
    }
    assert sum(flow.size_bytes for flow in graph.flows) == 656 + 768 + 656
    assert {flow.family for flow in graph.flows} == {
        "remote",
        "hbm_demand",
        "demand_fill",
    }


def test_prefetch_is_not_ready_until_hbm_and_noc_complete(tmp_path: Path) -> None:
    prefetch = TP1AstraPrefetch("p0", 2, 0, 656, 768, 4_000.0)
    access = TP1AstraAccess(
        "a0",
        3,
        "prefetch",
        2,
        2,
        656,
        prefetch_transfer_id="p0",
    )
    graph, summary = _build_graph(_config(tmp_path), (access,), (prefetch,))

    assert summary["source_counts"]["prefetch"] == 1
    assert [flow.family for flow in graph.flows] == [
        "prefetch",
        "prefetch_remote",
    ]
    compute = [
        node
        for node in graph.nodes[3]
        if node.kind == "compute" and node.name.startswith("prefetch_compute")
    ]
    assert len(compute) == 1
    assert len(compute[0].deps) == 2


def test_packetization_preserves_all_hbm_bytes(tmp_path: Path) -> None:
    accesses = tuple(
        TP1AstraAccess(
            f"a{index}",
            3,
            "hbm",
            None,
            3,
            656,
            hbm_port_core=0,
            hbm_wire_bytes=768,
            hbm_completion_ns=11_000.0,
        )
        for index in range(65)
    )
    graph, summary = _build_graph(_config(tmp_path), accesses, ())
    hbm_flows = [flow for flow in graph.flows if flow.family == "hbm_demand"]

    assert len(hbm_flows) == 3
    assert sum(flow.item_count for flow in hbm_flows) == 65
    assert sum(flow.size_bytes for flow in hbm_flows) == 65 * 768
    assert sum(summary["access_counts"]) == 65


def test_hbm_packetization_does_not_merge_distinct_readiness_ticks(
    tmp_path: Path,
) -> None:
    accesses = (
        TP1AstraAccess(
            "early",
            3,
            "hbm",
            None,
            3,
            656,
            hbm_port_core=0,
            hbm_wire_bytes=768,
            hbm_completion_ns=11_001.0,
        ),
        TP1AstraAccess(
            "late",
            3,
            "hbm",
            None,
            3,
            656,
            hbm_port_core=0,
            hbm_wire_bytes=768,
            hbm_completion_ns=11_011.0,
        ),
    )

    graph, summary = _build_graph(_config(tmp_path), accesses, ())
    hbm_flows = [flow for flow in graph.flows if flow.family == "hbm_demand"]

    assert summary["hbm_readiness_quantum_ns"] == 10.0
    assert len(hbm_flows) == 2
    assert [flow.item_count for flow in hbm_flows] == [1, 1]


def test_nonlocal_compute_groups_are_ready_independently(tmp_path: Path) -> None:
    accesses = (
        TP1AstraAccess(
            "early",
            3,
            "hbm",
            None,
            3,
            656,
            hbm_port_core=0,
            hbm_wire_bytes=768,
            hbm_completion_ns=11_000.0,
        ),
        TP1AstraAccess(
            "late",
            3,
            "hbm",
            None,
            3,
            656,
            hbm_port_core=1,
            hbm_wire_bytes=768,
            hbm_completion_ns=20_000.0,
        ),
    )
    graph, _ = _build_graph(_config(tmp_path), accesses, ())
    compute_nodes = {
        node.name: (node_id, node)
        for node_id, node in enumerate(graph.nodes[3])
        if node.kind == "compute" and node.name.startswith("hbm_compute")
    }

    assert set(compute_nodes) == {"hbm_compute_early", "hbm_compute_late"}
    compute_ids = {node_id for node_id, _ in compute_nodes.values()}
    assert all(
        compute_ids.isdisjoint(node.deps) for _, node in compute_nodes.values()
    )


def test_sequence_serializes_windows_without_dropping_work(tmp_path: Path) -> None:
    first = TP1AstraWindow(
        "w0", (TP1AstraAccess("a0", 0, "local", 0, 0, 656),)
    )
    second = TP1AstraWindow(
        "w1", (TP1AstraAccess("a1", 1, "remote", 0, 0, 656),)
    )

    graph, summary = _merge_window_graphs(_config(tmp_path), (first, second))

    assert summary["num_windows"] == 2
    assert summary["logical_accesses"] == 2
    assert summary["source_counts"]["local"] == 1
    assert summary["source_counts"]["remote"] == 1
    second_phase = next(
        node for node in graph.nodes[0] if node.name == "w1:preceding_phase"
    )
    assert second_phase.deps
