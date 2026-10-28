from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "tp2_core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from build_window_traffic_matrix import build_window_matrices
from compute_model import H200RooflineConfig, compute_rank_roofline_from_rank_expert_tokens, compute_rank_roofline_us
from runtime_scheduler import (
    _adjacent_swap_candidates,
    _are_adjacent_dies,
    _bounded_replica_candidate_dies,
    _cost_aware_window_evaluation,
    _edge_base_weight,
    _migration_activation_window,
    _path_base_cost_weighted,
    build_migration_events,
    _runtime_load_penalty,
    run_runtime,
)
from route_steps import route_activation_counts
from solve_static_mapping import (
    _mapping_objective,
    _ours_static_mapping,
    normalize_objective_parts,
    validate_mapping_capacities,
)
from utils_topology import TopologyCost, load_topology


def test_topology_cost_4w_rail_ports_and_cross_path() -> None:
    tc = TopologyCost.from_yaml(ROOT / "configs/network/hierarchical_4w_8x8_mesh_rail_900.yaml")
    assert tc.rail_port_count == 32
    omega = tc.topo.get_outer_to_inner_weights()
    assert abs(sum(omega.values()) / len(omega) - 1.0) < 1e-12
    edge = [v for d, v in omega.items() if tc.topo.get_edge_distance(d) == 0]
    inner = [v for d, v in omega.items() if tc.topo.get_edge_distance(d) > 0]
    assert min(edge) > min(inner)
    same = tc.estimate(0, 7, 1024)
    assert same.path_edges
    assert all(edge[0] == "mesh" for edge in same.path_edges)
    cross = tc.estimate(0, 64, int(3.5 * 1024 * 1024))
    assert any(edge[0] == "switch" for edge in cross.path_edges)
    assert 3.5 <= cross.serialization_ns / 1000.0 <= 4.5


def test_h200_roofline_scales_with_batch_counts() -> None:
    cfg = H200RooflineConfig()
    mapping = {0: [0]}
    one = compute_rank_roofline_us({"0": 1}, mapping, 1.0, 1, cfg)
    two = compute_rank_roofline_us({"0": 2}, mapping, 1.0, 1, cfg)
    assert two["max_rank_memory_us"] < one["max_rank_memory_us"] * 1.01

    many = compute_rank_roofline_from_rank_expert_tokens([{0: 1024.0}], cfg)
    assert many["max_rank_total_us"] > one["max_rank_total_us"] * 4.0
    assert many["max_rank_compute_us"] > many["max_rank_memory_us"]


def test_sparse_p2p_dispatch_combine_no_padding() -> None:
    topo = load_topology(ROOT / "configs/network/hierarchical_16w_4x4_mesh_rail_500.yaml")
    result = build_window_matrices(
        window_src_expert_counts=[
            {"window_id": 0, "src_expert_counts": {"0": {"1": 2}, "1": {"2": 1}}},
        ],
        mapping={1: [16], 2: [1]},
        topo=topo,
        batch_scale=1.0,
        dispatch_bytes_per_token=4,
        combine_bytes_per_token=8,
        num_ranks=256,
        expert_compute_us_per_token=1.0,
        compute_backend="roofline_h200",
        roofline_cfg=H200RooflineConfig(hidden_size=4, expert_intermediate_size=4),
        window_size=1,
        window_stats=[{"window_id": 0, "expert_counts": {"1": 2, "2": 1}}],
    )
    s = result["summary"]
    assert s["validation"]["sparse_p2p_no_padding"] is True
    assert s["validation"]["dispatch_total_bytes"] == 8
    assert s["validation"]["combine_total_bytes"] == 16


def test_window_matrix_uses_scaled_corner_rail_bandwidth() -> None:
    topo = load_topology(ROOT / "configs/network/hierarchical_16w_4x4_mesh_rail_500.yaml")
    nbytes = 500_000
    common = {
        "window_src_expert_counts": [
            {"window_id": 0, "src_expert_counts": {"16": {"0": 1}}},
        ],
        "topo": topo,
        "batch_scale": 1.0,
        "dispatch_bytes_per_token": nbytes,
        "combine_bytes_per_token": nbytes,
        "num_ranks": 256,
        "expert_compute_us_per_token": 0.0,
        "compute_backend": "constant_per_token",
        "roofline_cfg": H200RooflineConfig(hidden_size=4, expert_intermediate_size=4),
        "window_size": 1,
        "window_stats": [{"window_id": 0, "expert_counts": {"0": 1}}],
        "edge_congestion_factor": 0.0,
    }

    corner = build_window_matrices(mapping={0: [0]}, **common)
    non_corner = build_window_matrices(mapping={0: [1]}, **common)

    assert topo.get_link_bandwidth(("switch", topo.total_dies, 0)) == 1000.0
    assert topo.get_link_bandwidth(("switch", topo.total_dies, 1)) == 500.0
    assert corner["summary"]["avg_step_dispatch_us"] < non_corner["summary"]["avg_step_dispatch_us"]


def test_static_mapping_gamma_changes_congestion_aware_die_choice() -> None:
    topo = load_topology(ROOT / "configs/network/hierarchical_16w_4x4_mesh_rail_500.yaml")
    dies = topo.get_all_dies()
    num_experts = 16
    p_ue = {
        str(src): {str(e): 1.0 for e in range(num_experts)}
        for src in range(16)
    }
    s_u = {str(src): 1000 for src in range(16)}
    h = {str(e): 1.0 for e in range(num_experts)}

    no_cong = _ours_static_mapping(
        num_experts=num_experts,
        dies=dies,
        p_ue=p_ue,
        h=h,
        s_u=s_u,
        topo=topo,
        alpha=1.0,
        gamma=0.0,
        delta=0.0,
        lambda_=0.0,
        topk_hot=0,
        max_replicas=1,
    )
    strong_cong = _ours_static_mapping(
        num_experts=num_experts,
        dies=dies,
        p_ue=p_ue,
        h=h,
        s_u=s_u,
        topo=topo,
        alpha=1.0,
        gamma=1000.0,
        delta=0.0,
        lambda_=0.0,
        topk_hot=0,
        max_replicas=1,
    )

    assert [no_cong[e][0] for e in range(num_experts)] != [
        strong_cong[e][0] for e in range(num_experts)
    ]


def test_static_mapping_load_gradient_places_hot_expert_on_edge() -> None:
    topo = load_topology(ROOT / "configs/network/hierarchical_16w_4x4_mesh_rail_500.yaml")
    dies = topo.get_all_dies()
    num_experts = 16
    p_ue = {"0": {str(e): 1.0 / num_experts for e in range(num_experts)}}
    s_u = {"0": 1600}
    h = {str(e): (100.0 if e == 15 else 1.0) for e in range(num_experts)}

    mapping = _ours_static_mapping(
        num_experts=num_experts,
        dies=dies,
        p_ue=p_ue,
        h=h,
        s_u=s_u,
        topo=topo,
        alpha=0.0,
        gamma=0.0,
        delta=1.0,
        lambda_=0.0,
        topk_hot=0,
        max_replicas=1,
    )

    assert topo.get_edge_distance(mapping[15][0]) == 0


def test_static_mapping_can_add_multiple_source_agnostic_replicas() -> None:
    topo = load_topology(ROOT / "configs/network/hierarchical_16w_4x4_mesh_rail_500.yaml")
    dies = topo.get_all_dies()
    num_experts = 16
    h = {str(e): (100.0 if e == 0 else 0.1) for e in range(num_experts)}
    p_ue = {str(src): {str(e): 1.0 / num_experts for e in range(num_experts)} for src in dies}
    s_u = {str(src): 100 for src in dies}

    mapping = _ours_static_mapping(
        num_experts=num_experts,
        dies=dies,
        p_ue=p_ue,
        h=h,
        s_u=s_u,
        topo=topo,
        alpha=1.0,
        gamma=0.2,
        delta=0.3,
        lambda_=1.0,
        topk_hot=1,
        max_replicas=4,
        source_model="source_agnostic",
    )

    assert len(mapping[0]) > 1
    assert len(mapping[0]) <= 4


def test_static_mapping_reports_full_phi_objective_stats() -> None:
    topo = load_topology(ROOT / "configs/network/hierarchical_16w_4x4_mesh_rail_500.yaml")
    dies = topo.get_all_dies()
    num_experts = 8
    h = {str(e): (10.0 if e == 0 else 1.0) for e in range(num_experts)}
    p_ue = {str(src): {str(e): 1.0 / num_experts for e in range(num_experts)} for src in range(8)}
    s_u = {str(src): 100 for src in range(8)}

    mapping, stats = _ours_static_mapping(
        num_experts=num_experts,
        dies=dies,
        p_ue=p_ue,
        h=h,
        s_u=s_u,
        topo=topo,
        alpha=1.0,
        gamma=0.2,
        delta=0.3,
        lambda_=1.0,
        topk_hot=1,
        max_replicas=2,
        source_model="source_agnostic",
        return_stats=True,
    )

    assert len(mapping) == num_experts
    assert {"phi_comm", "phi_cong", "phi_load", "objective_final"} <= set(stats)
    assert stats["objective_final"] <= stats["objective_before_replicas"] + 1e-12
    assert stats["phase2_refinement"]["objective_final"] <= (
        stats["phase2_refinement"]["objective_initial"] + 1e-12
    )


def test_static_objective_terms_are_dimensionless_before_weighting() -> None:
    normalized = normalize_objective_parts(
        {"comm": 4000.0, "cong": 0.001, "load": 0.002},
        {"comm": 4000.0, "cong": 0.001, "load": 0.002},
    )

    assert normalized == {"comm": 1.0, "cong": 1.0, "load": 1.0}
    assert 1.0 * normalized["comm"] + 0.2 * normalized["cong"] + 0.3 * normalized["load"] == 1.5


def test_static_capacity_validation_counts_every_replica() -> None:
    report = validate_mapping_capacities(
        mapping={0: [0, 1], 1: [0]},
        h={"0": 0.6, "1": 0.4},
        dies=[0, 1],
        expert_weight_bytes=10,
        die_memory_capacity_bytes=15,
        die_compute_capacity=1.0,
    )

    assert report["valid"] is False
    assert report["violations"][0]["die"] == 0
    assert report["per_die"]["0"]["memory_bytes"] == 20


def test_static_replica_load_is_routed_once_and_shared_across_copies() -> None:
    topo = load_topology(ROOT / "configs/network/hierarchical_4w_8x8_mesh_rail_900.yaml")
    dies = topo.get_all_dies()
    mapping = {0: [0, 64]}
    path_cache = {
        (src, dst): topo.get_path(src, dst)
        for src in dies
        for dst in dies
    }

    _, parts = _mapping_objective(
        mapping=mapping,
        p_ue={},
        s_u={},
        h={"0": 1.0},
        dies=dies,
        topo=topo,
        alpha=1.0,
        gamma=1.0,
        delta=1.0,
        lambda_=1.0,
        source_model="source_agnostic",
        path_cache=path_cache,
    )

    assert abs(sum(parts["die_load"].values()) - 1.0) < 1e-12
    assert parts["die_load"][0] > 0.0
    assert parts["die_load"][64] > 0.0


def test_static_capacity_uses_routed_compute_load_not_copy_count() -> None:
    report = validate_mapping_capacities(
        mapping={0: [0, 1], 1: [0]},
        h={"0": 0.6, "1": 0.4},
        dies=[0, 1],
        expert_weight_bytes=10,
        die_memory_capacity_bytes=100,
        die_compute_capacity=1.0,
        routed_die_loads={0: 0.7, 1: 0.3},
    )

    assert report["valid"] is True
    assert report["per_die"]["0"]["compute_requirement"] == 0.7
    assert report["per_die"]["1"]["compute_requirement"] == 0.3
    assert sum(
        die["compute_requirement"] for die in report["per_die"].values()
    ) == 1.0


def test_runtime_scheduler_preserves_static_replica_budget() -> None:
    topo = load_topology(ROOT / "configs/network/hierarchical_16w_4x4_mesh_rail_500.yaml")
    static_mapping = {e: [e] for e in range(16)}
    static_mapping[0] = [0, 16, 32, 48]
    windows = run_runtime(
        static_mapping=static_mapping,
        window_stats=[{"window_id": 0, "expert_counts": {"0": 100}}],
        window_src_expert_counts=[{"window_id": 0, "src_expert_counts": {"0": {"0": 100}}}],
        topo=topo,
        decay=0.8,
        hot_percentile=90.0,
        replica_persistence_windows=2,
        migration_budget_bytes_per_window=0,
        bytes_per_token=14336,
        max_replicas_per_expert=3,
    )

    assert windows[0]["mapping"]["0"] == [0, 16, 32, 48]


def test_runtime_load_penalty_uses_signed_deviation_over_all_dies() -> None:
    assert _runtime_load_penalty({0: 2.0}, {0: 1.0, 1: 1.0}, [0, 1]) == 2.0


def test_path_base_cost_matches_topology_path_edges() -> None:
    for config_name in [
        "hierarchical_4w_8x8_mesh_rail_900.yaml",
        "hierarchical_16w_4x4_mesh_rail_500.yaml",
    ]:
        topo = load_topology(ROOT / "configs/network" / config_name)
        sample_dies = sorted(
            {
                0,
                1,
                topo.cfg.dies_per_wafer - 1,
                topo.cfg.dies_per_wafer,
                topo.total_dies // 2,
                topo.total_dies - 1,
            }
        )
        for src in sample_dies:
            for dst in sample_dies:
                reference = sum(
                    _edge_base_weight(topo, edge)
                    for edge in topo.get_path(src, dst)
                )
                assert _path_base_cost_weighted(topo, src, dst) == reference


def test_runtime_scheduler_estimate_uses_shared_cost_aware_router() -> None:
    topo = load_topology(ROOT / "configs/network/hierarchical_16w_4x4_mesh_rail_500.yaml")
    mapping = {0: [0, 1], 1: [2]}
    counts = {"0": {"0": 4, "1": 2}, "3": {"0": 4}}
    direct = route_activation_counts(
        src_expert_counts=counts,
        mapping=mapping,
        topo=topo,
        dispatch_bytes_per_token=4,
        routing_policy="hacmo_cost_aware",
    )
    evaluated = _cost_aware_window_evaluation(
        mapping=mapping,
        src_expert_counts=counts,
        topo=topo,
        omega=topo.get_outer_to_inner_weights(),
        roofline_cfg=H200RooflineConfig(hidden_size=4, expert_intermediate_size=4),
        batch_scale=1.0,
        dispatch_bytes_per_token=4,
        combine_bytes_per_token=8,
        communication_weight=1.0,
        compute_load_weight=1.0,
        route_estimation_decisions_per_source_expert=1,
    )

    estimated = route_activation_counts(
        src_expert_counts=counts,
        mapping=mapping,
        topo=topo,
        dispatch_bytes_per_token=4,
        routing_policy="hacmo_cost_aware",
        max_decisions_per_count=1,
    )

    assert evaluated["route_load"] == dict(estimated["die_load"])
    assert sum(direct["die_load"].values()) == sum(estimated["die_load"].values())
    assert evaluated["comm"]["total_bytes"] == 10 * 4


def test_runtime_swap_adjacency_and_migration_event_accounting() -> None:
    topo = load_topology(ROOT / "configs/network/hierarchical_16w_4x4_mesh_rail_500.yaml")
    assert _are_adjacent_dies(topo, 0, 1)
    assert not _are_adjacent_dies(topo, 0, 2)
    assert not _are_adjacent_dies(topo, 0, 16)

    events = build_migration_events(
        replicas_added=[{"expert": 0, "source_die": 0, "die": 16}],
        swaps=[
            {
                "expert_hot": 1,
                "from_die": 1,
                "to_die": 2,
                "expert_cold": 2,
                "from_die_cold": 2,
                "to_die_cold": 1,
            }
        ],
        expert_weight_bytes=10,
        topo=topo,
        observed_window=3,
    )

    assert [event["operation"] for event in events] == ["replicate", "swap", "swap"]
    assert sum(event["bytes"] for event in events) == 30
    assert _migration_activation_window(3, migration_us=150.0, slack_us_per_window=100.0) == 5


def test_runtime_scheduler_updates_apply_to_next_window() -> None:
    topo = load_topology(ROOT / "configs/network/hierarchical_16w_4x4_mesh_rail_500.yaml")
    static_mapping = {e: [e] for e in range(16)}
    window_stats = [
        {"window_id": 0, "expert_counts": {"0": 1000}},
        {"window_id": 1, "expert_counts": {"0": 1000}},
        {"window_id": 2, "expert_counts": {"0": 1000}},
    ]
    window_src = [
        {"window_id": 0, "src_expert_counts": {"16": {"0": 1000}}},
        {"window_id": 1, "src_expert_counts": {"16": {"0": 1000}}},
        {"window_id": 2, "src_expert_counts": {"16": {"0": 1000}}},
    ]

    windows = run_runtime(
        static_mapping=static_mapping,
        window_stats=window_stats,
        window_src_expert_counts=window_src,
        topo=topo,
        decay=0.8,
        hot_percentile=90.0,
        replica_persistence_windows=2,
        migration_budget_bytes_per_window=100_000_000,
        bytes_per_token=14336,
        max_replicas_per_expert=2,
    )

    assert windows[0]["mapping"]["0"] == [0]
    assert windows[0]["next_mapping"]["0"] == [0]
    assert len(windows[1]["next_mapping"]["0"]) == 2
    assert windows[1]["next_mapping"]["0"] == windows[2]["mapping"]["0"]
    assert len(windows[2]["mapping"]["0"]) == 2


def test_runtime_mapping_waits_until_migration_completes() -> None:
    topo = load_topology(ROOT / "configs/network/hierarchical_16w_4x4_mesh_rail_500.yaml")
    static_mapping = {e: [e] for e in range(16)}
    window_stats = [
        {"window_id": 0, "expert_counts": {"0": 1000}},
        {"window_id": 1, "expert_counts": {"0": 1000}},
    ]
    window_src = [
        {"window_id": 0, "src_expert_counts": {"16": {"0": 1000}}},
        {"window_id": 1, "src_expert_counts": {"16": {"0": 1000}}},
    ]

    windows = run_runtime(
        static_mapping=static_mapping,
        window_stats=window_stats,
        window_src_expert_counts=window_src,
        topo=topo,
        decay=0.8,
        hot_percentile=90.0,
        replica_persistence_windows=1,
        migration_budget_bytes_per_window=100_000_000,
        bytes_per_token=14336,
        max_replicas_per_expert=2,
        attention_migration_slack_us_per_window=1.0,
    )

    assert len(windows[0]["planned_next_mapping"]["0"]) == 2
    assert windows[0]["next_mapping"]["0"] == [0]
    assert windows[1]["mapping"]["0"] == [0]
    assert windows[1]["update_blocked_by_pending_migration"] is True


def test_runtime_scheduler_rejects_joint_improvement_when_latency_worsens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topo = load_topology(ROOT / "configs/network/hierarchical_16w_4x4_mesh_rail_500.yaml")
    static_mapping = {e: [e] for e in range(16)}

    def fake_evaluation(*, mapping: dict[int, list[int]], **_: object) -> dict:
        replicated = len(mapping[0]) > 1
        return {
            "objective_parts": {
                "comm": 1.0 if replicated else 2.0,
                "cong": 1.0,
                "load": 1.0,
                "critical_load": 1.0,
            },
            "compute_us": 20.0 if replicated else 10.0,
            "route_load": {0: 1.0},
            "comm": {
                "max_edge_port_bytes": 10.0,
                "cross_wafer_bytes": 10.0,
                "total_bytes": 10.0,
                "max_phase_service_ns": 10.0,
                "dispatch_max_link_service_ns": 10.0,
                "combine_max_link_service_ns": 10.0,
                "max_rank_send_bytes": 10.0,
                "max_rank_recv_bytes": 10.0,
            },
        }

    monkeypatch.setattr("runtime_scheduler._cost_aware_window_evaluation", fake_evaluation)
    windows = run_runtime(
        static_mapping=static_mapping,
        window_stats=[{"window_id": 0, "expert_counts": {"0": 100}}],
        window_src_expert_counts=[
            {"window_id": 0, "src_expert_counts": {"16": {"0": 100}}}
        ],
        topo=topo,
        decay=0.8,
        hot_percentile=100.0,
        replica_persistence_windows=1,
        migration_budget_bytes_per_window=100_000_000,
        bytes_per_token=14336,
        max_replicas_per_expert=2,
        gamma=0.0,
        delta=0.0,
    )

    assert windows[0]["mapping"]["0"] == [0]
    assert windows[0]["next_mapping"]["0"] == [0]
    assert windows[0]["objective_after"] == windows[0]["objective_before"]
    assert windows[0]["candidate_search"]["accepted"]["replicate"] == 0
    assert windows[0]["candidate_search"]["rejected"]["aggregate_latency_not_improved"] > 0


def test_runtime_scheduler_rejects_aggregate_gain_that_worsens_step_latency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topo = load_topology(ROOT / "configs/network/hierarchical_16w_4x4_mesh_rail_500.yaml")
    static_mapping = {e: [e] for e in range(16)}

    def fake_evaluation(*, mapping: dict[int, list[int]], **_: object) -> dict:
        replicated = len(mapping[0]) > 1
        service_ns = 1_000.0 if replicated else 20_000.0
        return {
            "objective_parts": {
                "comm": 1.0 if replicated else 2.0,
                "cong": 1.0,
                "load": 1.0,
                "critical_load": 1.0,
            },
            "compute_us": 20.0 if replicated else 10.0,
            "route_load": {0: 1.0},
            "comm": {
                "max_edge_port_bytes": 10.0,
                "cross_wafer_bytes": 10.0,
                "total_bytes": 10.0,
                "max_phase_service_ns": service_ns,
                "dispatch_max_link_service_ns": service_ns,
                "combine_max_link_service_ns": service_ns,
                "max_rank_send_bytes": 10.0,
                "max_rank_recv_bytes": 10.0,
            },
        }

    def fake_step_evaluation(*, mapping: dict[int, list[int]], **_: object) -> dict:
        replicated = len(mapping[0]) > 1
        return {
            "estimated_total_us": 30.0 if replicated else 20.0,
            "compute_us": 20.0 if replicated else 10.0,
            "dispatch_service_us": 5.0,
            "combine_service_us": 5.0,
            "num_steps": 1,
        }

    monkeypatch.setattr("runtime_scheduler._cost_aware_window_evaluation", fake_evaluation)
    monkeypatch.setattr(
        "runtime_scheduler._step_level_window_latency_evaluation",
        fake_step_evaluation,
    )
    windows = run_runtime(
        static_mapping=static_mapping,
        window_stats=[{"window_id": 0, "expert_counts": {"0": 100}}],
        window_src_expert_counts=[
            {"window_id": 0, "src_expert_counts": {"16": {"0": 100}}}
        ],
        step_inputs=[
            {
                "window_id": 0,
                "step_id": 0,
                "src_expert_counts": {"16": {"0": 100}},
            }
        ],
        topo=topo,
        decay=0.8,
        hot_percentile=100.0,
        replica_persistence_windows=1,
        migration_budget_bytes_per_window=100_000_000,
        bytes_per_token=14336,
        max_replicas_per_expert=2,
        gamma=0.0,
        delta=0.0,
    )

    assert windows[0]["next_mapping"]["0"] == [0]
    assert windows[0]["candidate_search"]["accepted"]["replicate"] == 0
    assert windows[0]["candidate_search"]["rejected"]["step_latency_not_improved"] > 0


def test_runtime_scheduler_causally_rolls_back_stale_adjustments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topo = load_topology(ROOT / "configs/network/hierarchical_16w_4x4_mesh_rail_500.yaml")
    static_mapping = {e: [e] for e in range(16)}

    def score(mapping: dict[int, list[int]], phase: int) -> float:
        replicated = len(mapping[0]) > 1
        replica_should_win = phase >= 50
        return 1.0 if replicated == replica_should_win else 2.0

    def fake_evaluation(
        *,
        mapping: dict[int, list[int]],
        src_expert_counts: dict[str, dict[str, int]],
        **_: object,
    ) -> dict:
        aggregate_value = 1.0 if len(mapping[0]) > 1 else 2.0
        service_ns = aggregate_value * 1_000.0
        return {
            "objective_parts": {
                "comm": aggregate_value,
                "cong": 1.0,
                "load": 1.0,
                "critical_load": 1.0,
            },
            "compute_us": aggregate_value,
            "route_load": {0: aggregate_value},
            "comm": {
                "max_edge_port_bytes": aggregate_value,
                "cross_wafer_bytes": aggregate_value,
                "total_bytes": aggregate_value,
                "max_phase_service_ns": service_ns,
                "dispatch_max_link_service_ns": service_ns,
                "combine_max_link_service_ns": service_ns,
                "max_rank_send_bytes": aggregate_value,
                "max_rank_recv_bytes": aggregate_value,
            },
        }

    def fake_step_evaluation(
        *, mapping: dict[int, list[int]], window_steps: list[dict], **_: object
    ) -> dict:
        value = score(mapping, int(window_steps[0]["phase"]))
        return {
            "estimated_total_us": value * 3.0,
            "compute_us": value,
            "dispatch_service_us": value,
            "combine_service_us": value,
            "num_steps": 1,
        }

    monkeypatch.setattr("runtime_scheduler._cost_aware_window_evaluation", fake_evaluation)
    monkeypatch.setattr(
        "runtime_scheduler._step_level_window_latency_evaluation",
        fake_step_evaluation,
    )
    windows = run_runtime(
        static_mapping=static_mapping,
        window_stats=[
            {"window_id": 0, "expert_counts": {"0": 100}},
            {"window_id": 1, "expert_counts": {"0": 1}},
            {"window_id": 2, "expert_counts": {"0": 1}},
        ],
        window_src_expert_counts=[
            {"window_id": 0, "src_expert_counts": {"16": {"0": 100}}},
            {"window_id": 1, "src_expert_counts": {"16": {"0": 1}}},
            {"window_id": 2, "src_expert_counts": {"16": {"0": 1}}},
        ],
        step_inputs=[
            {"window_id": 0, "step_id": 0, "phase": 100},
            {"window_id": 1, "step_id": 1, "phase": 1},
            {"window_id": 2, "step_id": 2, "phase": 1},
        ],
        topo=topo,
        decay=0.8,
        hot_percentile=100.0,
        replica_persistence_windows=1,
        migration_budget_bytes_per_window=100_000_000,
        bytes_per_token=14336,
        max_replicas_per_expert=2,
        gamma=0.0,
        delta=0.0,
    )

    assert len(windows[0]["next_mapping"]["0"]) == 2
    assert len(windows[1]["mapping"]["0"]) == 2
    assert windows[1]["base_rollback"] == {
        "accepted": True,
        "changed_experts": 1,
        "copies_restored": 0,
        "copies_removed": 1,
        "objective_improved": False,
    }
    assert windows[1]["candidate_search"]["accepted"]["rollback"] == 1
    assert windows[1]["next_mapping"]["0"] == [0]
    assert windows[2]["mapping"]["0"] == [0]


def test_runtime_scheduler_can_evict_stale_static_replica(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topo = load_topology(ROOT / "configs/network/hierarchical_16w_4x4_mesh_rail_500.yaml")
    static_mapping = {e: [e] for e in range(16)}
    static_mapping[0] = [0, 1]

    def fake_evaluation(*, mapping: dict[int, list[int]], **_: object) -> dict:
        replicas = len(mapping[0])
        return {
            "objective_parts": {
                "comm": float(replicas),
                "cong": 1.0,
                "load": 1.0,
                "critical_load": 1.0,
            },
            "compute_us": 10.0,
            "route_load": {0: 1.0},
            "comm": {
                "max_edge_port_bytes": 10.0,
                "cross_wafer_bytes": 10.0,
                "total_bytes": 10.0,
                "max_phase_service_ns": 10.0,
                "dispatch_max_link_service_ns": 10.0,
                "combine_max_link_service_ns": 10.0,
                "max_rank_send_bytes": 10.0,
                "max_rank_recv_bytes": 10.0,
            },
        }

    monkeypatch.setattr("runtime_scheduler._cost_aware_window_evaluation", fake_evaluation)
    windows = run_runtime(
        static_mapping=static_mapping,
        window_stats=[{"window_id": 0, "expert_counts": {"1": 100}}],
        window_src_expert_counts=[
            {"window_id": 0, "src_expert_counts": {"2": {"1": 100}}}
        ],
        topo=topo,
        decay=0.8,
        hot_percentile=100.0,
        replica_persistence_windows=1,
        migration_budget_bytes_per_window=100_000_000,
        bytes_per_token=14336,
        max_replicas_per_expert=1,
        gamma=0.0,
        delta=0.0,
    )

    assert len(windows[0]["next_mapping"]["0"]) == 1
    assert windows[0]["replicas_removed"][0]["was_static_replica"] is True
    assert windows[0]["candidate_search"]["evaluated"]["evict"] == 2
    assert windows[0]["candidate_search"]["accepted"]["evict"] == 1


def test_adjacent_swap_candidates_are_not_limited_to_global_top_three() -> None:
    topo = load_topology(ROOT / "configs/network/hierarchical_16w_4x4_mesh_rail_500.yaml")
    mapping = {expert: [expert] for expert in range(16)}
    mapping[1].append(17)
    mapping[5].append(21)
    hotness = {expert: 0.0 for expert in mapping}
    hotness[5] = 100.0
    die_load = {die: 0.0 for die in range(16)}
    die_load.update({5: 1.0, 6: 10.0, 9: 9.0, 10: 8.0, 1: 100.0})

    candidates = _adjacent_swap_candidates(
        mapping=mapping,
        hotness=hotness,
        die_load=die_load,
        topo=topo,
    )

    assert any(
        candidate["from_die"] == 5 and candidate["to_die"] == 1
        for candidate in candidates
    )
    candidate = next(
        candidate
        for candidate in candidates
        if candidate["from_die"] == 5 and candidate["to_die"] == 1
    )
    assert candidate["expert_hot"] == 5
    assert candidate["expert_cold"] == 1


def test_runtime_swap_preserves_multi_replica_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topo = load_topology(ROOT / "configs/network/hierarchical_16w_4x4_mesh_rail_500.yaml")
    static_mapping = {expert: [expert] for expert in range(16)}
    static_mapping[1].append(17)
    static_mapping[5].append(21)

    def fake_evaluation(*, mapping: dict[int, list[int]], **_: object) -> dict:
        swapped = 1 in mapping[5] and 5 in mapping[1]
        score = 0.5 if swapped else 1.0
        service_ns = score * 1_000.0
        return {
            "objective_parts": {
                "comm": score,
                "cong": score,
                "load": score,
                "critical_load": score,
            },
            "compute_us": score,
            "route_load": {0: score},
            "comm": {
                "max_edge_port_bytes": score,
                "cross_wafer_bytes": score,
                "total_bytes": score,
                "max_phase_service_ns": service_ns,
                "dispatch_max_link_service_ns": service_ns,
                "combine_max_link_service_ns": service_ns,
                "max_rank_send_bytes": score,
                "max_rank_recv_bytes": score,
            },
        }

    monkeypatch.setattr("runtime_scheduler._cost_aware_window_evaluation", fake_evaluation)
    monkeypatch.setattr(
        "runtime_scheduler._adjacent_swap_candidates",
        lambda **_: [
            {
                "expert_hot": 5,
                "from_die": 5,
                "to_die": 1,
                "expert_cold": 1,
                "from_die_cold": 1,
                "to_die_cold": 5,
                "hotness_gap": 100.0,
                "load_gap": 0.0,
            }
        ],
    )
    windows = run_runtime(
        static_mapping=static_mapping,
        window_stats=[{"window_id": 0, "expert_counts": {"5": 100}}],
        window_src_expert_counts=[
            {"window_id": 0, "src_expert_counts": {"0": {"5": 100}}}
        ],
        topo=topo,
        decay=0.8,
        hot_percentile=100.0,
        replica_persistence_windows=2,
        migration_budget_bytes_per_window=100_000_000,
        bytes_per_token=14336,
        max_replicas_per_expert=2,
    )

    assert windows[0]["candidate_search"]["accepted"]["swap"] == 1
    assert set(windows[0]["planned_next_mapping"]["5"]) == {1, 21}
    assert set(windows[0]["planned_next_mapping"]["1"]) == {5, 17}
    assert len(windows[0]["planned_next_mapping"]["5"]) == 2
    assert len(windows[0]["planned_next_mapping"]["1"]) == 2


def test_replica_candidates_cover_highest_demand_uncovered_wafer() -> None:
    topo = load_topology(ROOT / "configs/network/hierarchical_4w_8x8_mesh_rail_900.yaml")
    mapping = {0: [0, 64, 192]}
    src_expert_counts = {
        "0": {"0": 100},
        "64": {"0": 100},
        "128": {"0": 1},
        "192": {"0": 100},
    }
    die_load = {
        die: (1_000_000.0 if 128 <= die < 192 else 0.0)
        for die in topo.get_all_dies()
    }

    candidates = _bounded_replica_candidate_dies(
        expert=0,
        mapping=mapping,
        src_expert_counts=src_expert_counts,
        die_load=die_load,
        topo=topo,
        limit=3,
    )

    assert len(candidates) == 3
    assert topo.die_to_wafer_local(candidates[0])[0] == 2


def test_at_cap_hot_experts_do_not_consume_hot_candidate_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    topo = load_topology(ROOT / "configs/network/hierarchical_16w_4x4_mesh_rail_500.yaml")
    static_mapping = {expert: [expert] for expert in range(16)}
    for expert in range(8):
        static_mapping[expert] = [expert, expert + 16]
    counts = {str(expert): 100 - expert for expert in range(9)}
    searched_experts: list[int] = []

    def record_candidates(*, expert: int, **_: object) -> list[int]:
        searched_experts.append(expert)
        return []

    monkeypatch.setattr(
        "runtime_scheduler._bounded_replica_candidate_dies",
        record_candidates,
    )
    windows = run_runtime(
        static_mapping=static_mapping,
        window_stats=[{"window_id": 0, "expert_counts": counts}],
        window_src_expert_counts=[
            {
                "window_id": 0,
                "src_expert_counts": {"32": counts},
            }
        ],
        topo=topo,
        decay=0.8,
        hot_percentile=0.0,
        replica_persistence_windows=1,
        migration_budget_bytes_per_window=100_000_000,
        bytes_per_token=14336,
        max_replicas_per_expert=2,
        max_hot_experts_per_window=1,
    )

    assert searched_experts == [8]
    assert windows[0]["candidate_search"]["rejected"]["replica_cap_reached"] == 8
