#!/usr/bin/env python3
"""Focused tests for the TP3 edge-local crossbar model."""

from __future__ import annotations

from pathlib import Path

from supernode_sim.plugins.crossbar import (
    EDGE_BOTTOM,
    EDGE_LEFT,
    EDGE_RIGHT,
    EDGE_TOP,
    allocate_edge_local_ports,
    build_projected_edge_traffic,
    crossbar_comm_us_per_step,
    load_crossbar_config,
)


def test_crossbar_config_keeps_yaml_bandwidth_as_per_port() -> None:
    repo = Path(__file__).resolve().parents[2]
    cfg = load_crossbar_config(
        str(repo / "hacmo/interconnect/configs/network/hierarchical_4w_8x8_mesh_crossbar_900.yaml")
    )

    assert cfg["side"] == 8
    assert cfg["external_ports_per_edge"] == 8
    assert cfg["num_io_ports"] == 32
    assert cfg["per_port_bw_gbps"] == 900.0
    assert cfg["switch_bw_total_gbps"] == 32 * 900.0
    assert cfg["crossbar_latency_ns"] == 20.0


def test_projected_edge_traffic_splits_tied_nearest_edges_and_counts_ingress() -> None:
    dispatch = {
        5: {20: 1200},  # wafer 0 die (1,1) -> wafer 1 die (1,0) on a 4x4 mesh
    }
    projected = build_projected_edge_traffic(dispatch, {}, dies_per_wafer=16, side=4, num_wafers=2)

    # Source die (1,1) is tied between top and left; bytes split equally.
    assert projected[0][EDGE_TOP][1]["exit"] == 600.0
    assert projected[0][EDGE_LEFT][4]["exit"] == 600.0
    assert projected[0][EDGE_TOP][1]["transit"] == 600.0
    assert projected[0][EDGE_LEFT][4]["transit"] == 600.0

    # Destination die (1,0) is already on the left edge; ingress consumes that
    # edge-local external service but adds no mesh byte-hop pressure.
    assert projected[1][EDGE_LEFT][4]["exit"] == 1200.0
    assert projected[1][EDGE_LEFT][4]["transit"] == 0.0

    # No accidental leakage to untied exits.
    assert projected[0][EDGE_RIGHT][7]["exit"] == 0.0
    assert projected[0][EDGE_BOTTOM][13]["exit"] == 0.0


def test_allocation_is_scoped_to_each_wafer_edge() -> None:
    projected = {
        0: {
            EDGE_TOP: {
                0: {"exit": 900.0, "transit": 0.0},
                1: {"exit": 100.0, "transit": 0.0},
                2: {"exit": 0.0, "transit": 0.0},
                3: {"exit": 0.0, "transit": 0.0},
            },
            EDGE_RIGHT: {
                3: {"exit": 100.0, "transit": 0.0},
                7: {"exit": 0.0, "transit": 0.0},
                11: {"exit": 0.0, "transit": 0.0},
                15: {"exit": 0.0, "transit": 0.0},
            },
        },
        1: {
            EDGE_TOP: {
                0: {"exit": 100.0, "transit": 0.0},
                1: {"exit": 100.0, "transit": 0.0},
                2: {"exit": 100.0, "transit": 0.0},
                3: {"exit": 100.0, "transit": 0.0},
            },
            EDGE_RIGHT: {
                3: {"exit": 0.0, "transit": 0.0},
                7: {"exit": 0.0, "transit": 0.0},
                11: {"exit": 0.0, "transit": 0.0},
                15: {"exit": 0.0, "transit": 0.0},
            },
        },
    }

    alloc = allocate_edge_local_ports("naive_crossbar", projected, side=4, per_port_bw_gbps=500.0)

    assert sum(v["ports"] for v in alloc[0][EDGE_TOP].values()) == 4
    assert sum(v["ports"] for v in alloc[0][EDGE_RIGHT].values()) == 4
    assert sum(v["ports"] for v in alloc[1][EDGE_TOP].values()) == 4
    assert alloc[0][EDGE_TOP][0]["ports"] > alloc[0][EDGE_TOP][1]["ports"]
    assert alloc[0][EDGE_RIGHT][3]["ports"] == 4
    assert alloc[1][EDGE_TOP][0]["ports"] == 1
    assert alloc[1][EDGE_TOP][3]["ports"] == 1


def test_dynamic_allocation_can_differ_from_exit_proportional_transit_pressure() -> None:
    projected = {
        0: {
            EDGE_TOP: {
                0: {"exit": 100.0, "transit": 1000.0},
                1: {"exit": 100.0, "transit": 0.0},
                2: {"exit": 100.0, "transit": 0.0},
                3: {"exit": 100.0, "transit": 0.0},
            }
        }
    }

    naive = allocate_edge_local_ports("naive_crossbar", projected, side=8, per_port_bw_gbps=500.0)
    dynamic = allocate_edge_local_ports(
        "dynamic_bw", projected, side=8, per_port_bw_gbps=500.0, beta=1.0, rho=0.0
    )

    assert naive[0][EDGE_TOP][0]["ports"] == 2
    assert dynamic[0][EDGE_TOP][0]["ports"] > naive[0][EDGE_TOP][0]["ports"]


def test_crossbar_latency_parameter_changes_phase_time() -> None:
    cfg = {
        "num_wafers": 2,
        "dies_per_wafer": 16,
        "side": 4,
        "num_io_ports": 16,
        "external_ports_per_edge": 4,
        "per_port_bw_gbps": 500.0,
        "mesh_bw_gbps": 2000.0,
        "switch_latency_ns": 2000.0,
        "mesh_latency_ns": 500.0,
        "crossbar_latency_ns": 0.0,
    }
    dispatch = [{5: {20: 1000}}]
    combine = [{20: {5: 1000}}]

    zero = crossbar_comm_us_per_step(
        dispatch, combine, cfg, "naive_crossbar", batch_scale=1.0, window_size=1, crossbar_latency_ns=0.0
    )
    fifty = crossbar_comm_us_per_step(
        dispatch, combine, cfg, "naive_crossbar", batch_scale=1.0, window_size=1, crossbar_latency_ns=50.0
    )

    assert fifty["comm_us"] > zero["comm_us"]
    assert round(fifty["comm_us"] - zero["comm_us"], 6) == 0.1
