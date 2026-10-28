from __future__ import annotations

from collections import Counter
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "tp2_core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from route_steps import build_routed_steps, route_activation_counts
from build_window_traffic_matrix import analyze_routed_steps
from compute_model import H200RooflineConfig
from moentwine_baseline import (
    build_causal_ni_windows,
    build_conventional_tp_collective_segments,
    build_er_source_domains,
    build_moentwine_collective_segments,
    incremental_collective_accounting,
    valid_attention_tp_degrees,
)
from utils_topology import HierarchicalTopology, TopologyConfig, load_topology


def test_hacmo_cost_aware_route_plan_preserves_every_activation() -> None:
    topo = load_topology(
        ROOT / "configs/network/hierarchical_16w_4x4_mesh_rail_500.yaml"
    )
    plan = build_routed_steps(
        steps=[
            {
                "step_id": 0,
                "window_id": 0,
                "src_expert_counts": {"0": {"0": 8}},
                "expert_counts": {"0": 8},
                "tokens_per_die": {"0": 1},
                "active": True,
            }
        ],
        static_mapping={0: [0, 2]},
        window_mappings=None,
        topo=topo,
        dispatch_bytes_per_token=4,
        combine_bytes_per_token=8,
        routing_policy="hacmo_cost_aware",
    )

    step = plan["steps"][0]
    assert sum(item["activations"] for item in step["assignments"]) == 8
    assert set(step["rank_expert_tokens"]) == {"0", "2"}
    assert step["validation"]["all_activations_routed"] is True
    assert step["validation"]["compute_matches_assignments"] is True


def _reference_hacmo_cost_aware_counts(
    *,
    src_expert_counts: dict[str, dict[str, int]],
    mapping: dict[int, list[int]],
    topo: HierarchicalTopology,
    dispatch_bytes_per_token: int,
    communication_weight: float,
    compute_load_weight: float,
    max_decisions_per_count: int | None,
) -> dict:
    input_activations = sum(
        int(count)
        for counts in src_expert_counts.values()
        for count in counts.values()
    )
    ideal_die_load = max(input_activations / max(topo.total_dies, 1), 1.0)
    die_load: Counter[int] = Counter()
    link_load_bytes: Counter[tuple] = Counter()
    assignments: Counter[tuple[int, int, int, int]] = Counter()
    path_cache: dict[tuple[int, int], list[tuple]] = {}

    def path(src: int, dst: int) -> list[tuple]:
        key = (src, dst)
        if key not in path_cache:
            path_cache[key] = topo.get_path(src, dst)
        return path_cache[key]

    def path_latency_ns(route: list[tuple]) -> float:
        return sum(
            topo.cfg.mesh_latency_ns if edge[0] == "mesh" else topo.cfg.switch_latency_ns
            for edge in route
        )

    for src_raw, expert_counts in sorted(
        src_expert_counts.items(), key=lambda item: int(item[0])
    ):
        src = int(src_raw)
        for expert_raw, count in sorted(
            expert_counts.items(), key=lambda item: int(item[0])
        ):
            expert = int(expert_raw)
            count = int(count)
            candidates = mapping[expert]
            if len(candidates) == 1:
                dst = candidates[0]
                assignments[(src, src, expert, dst)] += count
                die_load[dst] += count
                for edge in path(src, dst):
                    link_load_bytes[edge] += count * dispatch_bytes_per_token
                continue

            candidate_profiles = {}
            for die in candidates:
                route = path(src, die)
                candidate_profiles[die] = (
                    route,
                    path_latency_ns(route),
                    [
                        (edge, 1.0 / max(topo.get_link_bandwidth(edge), 1e-9))
                        for edge in route
                    ],
                )
            decision_count = (
                count
                if max_decisions_per_count is None
                else min(count, max_decisions_per_count)
            )
            base_chunk, extra = divmod(count, decision_count)
            for decision in range(decision_count):
                chunk_size = base_chunk + (1 if decision < extra else 0)
                comm_costs = {
                    die: profile[1]
                    + sum(
                        (link_load_bytes[edge] + chunk_size * dispatch_bytes_per_token)
                        * inv_bw
                        for edge, inv_bw in profile[2]
                    )
                    for die, profile in candidate_profiles.items()
                }
                comm_reference = max(max(comm_costs.values(), default=0.0), 1.0)
                dst = min(
                    candidates,
                    key=lambda die: (
                        communication_weight * comm_costs[die] / comm_reference
                        + compute_load_weight
                        * (die_load[die] + chunk_size)
                        / ideal_die_load,
                        die,
                    ),
                )
                assignments[(src, src, expert, dst)] += chunk_size
                die_load[dst] += chunk_size
                for edge in candidate_profiles[dst][0]:
                    link_load_bytes[edge] += chunk_size * dispatch_bytes_per_token

    return {
        "assignments": assignments,
        "die_load": die_load,
        "dispatch_link_load_bytes": link_load_bytes,
        "input_activations": input_activations,
    }


def test_hacmo_cost_aware_incremental_router_matches_naive_reference() -> None:
    topo = load_topology(
        ROOT / "configs/network/hierarchical_16w_4x4_mesh_rail_500.yaml"
    )
    counts = {
        "0": {"0": 9, "1": 3},
        "5": {"0": 7},
        "17": {"0": 5, "1": 4},
    }
    mapping = {0: [0, 1, 16], 1: [2, 6]}
    common = {
        "src_expert_counts": counts,
        "mapping": mapping,
        "topo": topo,
        "dispatch_bytes_per_token": 4,
        "communication_weight": 1.3,
        "compute_load_weight": 0.7,
    }

    for max_decisions in (None, 3):
        optimized = route_activation_counts(
            routing_policy="hacmo_cost_aware",
            max_decisions_per_count=max_decisions,
            **common,
        )
        reference = _reference_hacmo_cost_aware_counts(
            max_decisions_per_count=max_decisions,
            **common,
        )

        assert optimized["assignments"] == reference["assignments"]
        assert optimized["die_load"] == reference["die_load"]
        assert optimized["dispatch_link_load_bytes"] == reference["dispatch_link_load_bytes"]
        assert optimized["input_activations"] == reference["input_activations"]


def test_nearest_baseline_route_does_not_use_hacmo_load_balancing() -> None:
    topo = load_topology(
        ROOT / "configs/network/hierarchical_16w_4x4_mesh_rail_500.yaml"
    )
    plan = build_routed_steps(
        steps=[
            {
                "step_id": 0,
                "window_id": 0,
                "src_expert_counts": {"0": {"0": 8}},
                "expert_counts": {"0": 8},
                "tokens_per_die": {"0": 1},
                "active": True,
            }
        ],
        static_mapping={0: [0, 2]},
        window_mappings=None,
        topo=topo,
        dispatch_bytes_per_token=4,
        combine_bytes_per_token=8,
        routing_policy="nearest",
    )

    assert plan["steps"][0]["assignments"] == [
        {"src_die": 0, "expert": 0, "dst_die": 0, "activations": 8}
    ]


def test_step_analysis_charges_weights_each_step_and_reports_overlap_bound() -> None:
    topo = load_topology(
        ROOT / "configs/network/hierarchical_16w_4x4_mesh_rail_500.yaml"
    )
    plan = build_routed_steps(
        steps=[
            {
                "step_id": step,
                "window_id": 0,
                "src_expert_counts": {"0": {"0": 2}},
                "expert_counts": {"0": 2},
                "tokens_per_die": {"0": 1},
                "active": True,
            }
            for step in range(2)
        ],
        static_mapping={0: [1]},
        window_mappings=None,
        topo=topo,
        dispatch_bytes_per_token=4,
        combine_bytes_per_token=8,
        routing_policy="nearest",
    )
    roofline = H200RooflineConfig(hidden_size=4, expert_intermediate_size=4)

    result = analyze_routed_steps(plan, topo=topo, roofline_cfg=roofline, num_ranks=256)

    assert result["summary"]["num_steps"] == 2
    assert result["summary"]["total_routed_expert_weight_reads"] == 2
    assert (
        result["summary"]["total_routed_expert_weight_bytes"]
        == 2 * roofline.expert_weight_bytes
    )
    assert (
        result["summary"]["avg_step_overlap_total_us"]
        <= result["summary"]["avg_step_serial_upper_us"]
    )


def test_moentwine_er_groups_form_compact_ftds() -> None:
    topo = load_topology(
        ROOT / "configs/network/hierarchical_16w_4x4_mesh_rail_500.yaml"
    )
    domains = build_er_source_domains(topo, tp_rows=2, tp_cols=2, hierarchical=False)

    assert domains[0] == [0, 2, 8, 10]
    assert domains[1] == [1, 3, 9, 11]
    her_domains = build_er_source_domains(topo, tp_rows=2, tp_cols=2, hierarchical=True)
    assert her_domains[0] == [wafer * 16 for wafer in range(16)]
    assert valid_attention_tp_degrees(8, 8) == [1, 2, 4, 8]


def test_moentwine_er_route_uses_tp_group_source_alternative() -> None:
    topo = load_topology(
        ROOT / "configs/network/hierarchical_16w_4x4_mesh_rail_500.yaml"
    )
    plan = build_routed_steps(
        steps=[
            {
                "step_id": 0,
                "window_id": 0,
                "src_expert_counts": {"0": {"0": 1}},
                "expert_counts": {"0": 1},
                "tokens_per_die": {"0": 1},
                "active": True,
            }
        ],
        static_mapping={0: [3]},
        window_mappings=None,
        topo=topo,
        dispatch_bytes_per_token=4,
        combine_bytes_per_token=8,
        routing_policy="moentwine_er_nearest",
        source_domains=build_er_source_domains(
            topo, tp_rows=2, tp_cols=2, hierarchical=False
        ),
    )

    assignment = plan["steps"][0]["assignments"][0]
    assert assignment["nominal_src_die"] == 0
    assert assignment["src_die"] == 2
    assert assignment["dst_die"] == 3


def test_moentwine_her_route_stays_within_destination_wafer() -> None:
    topo = HierarchicalTopology(
        TopologyConfig(
            num_wafers=2,
            dies_per_wafer=4,
            mesh_rows=2,
            mesh_cols=2,
            switch_bw_gbps=500.0,
            mesh_bw_gbps=2000.0,
            switch_latency_ns=1.0,
            mesh_latency_ns=100.0,
            rail_ports=[0],
            rail_bw_scales=[1.0],
            rail_optimized=False,
        )
    )
    plan = build_routed_steps(
        steps=[
            {
                "step_id": 0,
                "window_id": 0,
                "src_expert_counts": {"0": {"0": 1}},
                "expert_counts": {"0": 1},
                "tokens_per_die": {"0": 1},
                "active": True,
            }
        ],
        static_mapping={0: [3]},
        window_mappings=None,
        topo=topo,
        dispatch_bytes_per_token=4,
        combine_bytes_per_token=8,
        routing_policy="moentwine_er_nearest",
        source_domains=build_er_source_domains(
            topo, tp_rows=2, tp_cols=2, hierarchical=True
        ),
        moentwine_her_wafer_local=True,
    )

    assignment = plan["steps"][0]["assignments"][0]
    assert assignment["src_die"] == 0
    assert assignment["dst_die"] == 3
    assert assignment["src_die"] // 4 == assignment["dst_die"] // 4


def test_moentwine_er_tp4_charges_complete_ring_all_reduce() -> None:
    topo = HierarchicalTopology(
        TopologyConfig(
            num_wafers=1,
            dies_per_wafer=16,
            mesh_rows=4,
            mesh_cols=4,
            switch_bw_gbps=500.0,
            mesh_bw_gbps=2000.0,
            switch_latency_ns=2000.0,
            mesh_latency_ns=500.0,
            rail_ports=[0],
            rail_bw_scales=[1.0],
            rail_optimized=True,
        )
    )
    segments = build_moentwine_collective_segments(
        tokens_per_die={str(die): 1 for die in range(topo.total_dies)},
        topo=topo,
        hidden_size=4,
        bytes_per_hidden_elem=2,
        tp_rows=2,
        tp_cols=2,
    )

    assert len(segments) == 6
    assert segments[0]["name"] == "er_intra_wafer_reduce_scatter_0"
    assert segments[3]["name"] == "er_intra_wafer_all_gather_0"
    assert sum(
        nbytes
        for segment in segments
        for dsts in segment["messages"].values()
        for nbytes in dsts.values()
    ) == 768


def test_moentwine_tp4_ring_has_no_diagonal_mesh_edges() -> None:
    topo = HierarchicalTopology(
        TopologyConfig(
            num_wafers=1,
            dies_per_wafer=16,
            mesh_rows=4,
            mesh_cols=4,
            switch_bw_gbps=500.0,
            mesh_bw_gbps=2000.0,
            switch_latency_ns=2000.0,
            mesh_latency_ns=500.0,
            rail_ports=[0],
            rail_bw_scales=[1.0],
            rail_optimized=True,
        )
    )
    segments = build_conventional_tp_collective_segments(
        tokens_per_die={str(die): 1 for die in range(topo.total_dies)},
        topo=topo,
        hidden_size=4,
        bytes_per_hidden_elem=2,
        tp_rows=2,
        tp_cols=2,
    )

    for segment in segments:
        for src, dsts in segment["messages"].items():
            src_rc = topo.local_to_rc(int(src))
            for dst in dsts:
                dst_rc = topo.local_to_rc(int(dst))
                assert src_rc[0] == dst_rc[0] or src_rc[1] == dst_rc[1]


def test_moentwine_moe_only_accounting_charges_only_her_increment() -> None:
    accounting = incremental_collective_accounting(
        no_collective_total_us=50.0,
        conventional_tp_collective_us=8.0,
        full_her_collective_us=12.0,
    )

    assert accounting["incremental_collective_us"] == 4.0
    assert accounting["charged_total_us"] == 54.0
    assert accounting["no_collective_total_us"] == 50.0
    assert accounting["conventional_tp_collective_us"] == 8.0
    assert accounting["full_her_collective_us"] == 12.0


def test_moentwine_her_charges_collective_before_moe_dispatch() -> None:
    topo = load_topology(
        ROOT / "configs/network/hierarchical_16w_4x4_mesh_rail_500.yaml"
    )
    segments = build_moentwine_collective_segments(
        tokens_per_die={str(die): 1 for die in range(topo.total_dies)},
        topo=topo,
        hidden_size=4,
        bytes_per_hidden_elem=2,
        tp_rows=2,
        tp_cols=2,
    )

    assert len(segments) == 3 + 15
    assert segments[0]["name"] == "her_intra_wafer_reduce_scatter_0"
    assert segments[3]["name"] == "her_inter_wafer_all_gather_0"
    assert (
        sum(
            nbytes
            for segment in segments
            for dsts in segment["messages"].values()
            for nbytes in dsts.values()
        )
        > 0
    )

    plan = build_routed_steps(
        steps=[
            {
                "step_id": 0,
                "window_id": 0,
                "src_expert_counts": {"0": {"0": 1}},
                "expert_counts": {"0": 1},
                "tokens_per_die": {"0": 1},
                "active": True,
            }
        ],
        static_mapping={0: [0]},
        window_mappings=None,
        topo=topo,
        dispatch_bytes_per_token=4,
        combine_bytes_per_token=8,
        routing_policy="nearest",
    )
    plan["steps"][0]["pre_dispatch_segments"] = segments
    result = analyze_routed_steps(
        plan,
        topo=topo,
        roofline_cfg=H200RooflineConfig(hidden_size=4, expert_intermediate_size=4),
        num_ranks=256,
    )

    assert result["summary"]["avg_step_collective_us"] > 0
    assert result["summary"]["total_collective_bytes"] > 0
    assert (
        result["summary"]["avg_step_total_us"]
        >= result["summary"]["avg_step_collective_us"]
    )


def test_moentwine_ni_balancer_is_causal() -> None:
    topo = load_topology(
        ROOT / "configs/network/hierarchical_16w_4x4_mesh_rail_500.yaml"
    )
    windows = build_causal_ni_windows(
        base_mapping={0: [0], 1: [1]},
        window_stats=[
            {"window_id": 0, "expert_counts": {"0": 100, "1": 1}},
            {"window_id": 1, "expert_counts": {"0": 100, "1": 1}},
        ],
        topo=topo,
        expert_weight_bytes=1024,
        max_replicas_per_expert=2,
        shadow_slots_per_die=1,
    )

    assert windows[0]["mapping"]["0"] == [0]
    assert len(windows[0]["next_mapping"]["0"]) == 2
    assert windows[1]["mapping"]["0"] == windows[0]["next_mapping"]["0"]
    assert windows[0]["observation_semantics"] == "window_w_updates_window_w_plus_1"


def test_route_plan_records_effective_mapping_replica_counts() -> None:
    topo = load_topology(
        ROOT / "configs/network/hierarchical_16w_4x4_mesh_rail_500.yaml"
    )
    plan = build_routed_steps(
        steps=[
            {
                "step_id": 0,
                "window_id": 0,
                "src_expert_counts": {"0": {"0": 1}},
                "tokens_per_die": {"0": 1},
                "active": True,
            },
            {
                "step_id": 1,
                "window_id": 1,
                "src_expert_counts": {"0": {"0": 1}},
                "tokens_per_die": {"0": 1},
                "active": True,
            },
        ],
        static_mapping={0: [0]},
        window_mappings=[
            {"window_id": 0, "mapping": {"0": [0, 1]}},
            {"window_id": 1, "mapping": {"0": [0, 1, 2]}},
        ],
        topo=topo,
        dispatch_bytes_per_token=4,
        combine_bytes_per_token=8,
        routing_policy="nearest",
    )

    provenance = plan["metadata"]["mapping_copy_provenance"]
    assert provenance["source"] == "effective_mapping_used_for_each_step"
    assert [row["extra_replicas"] for row in provenance["per_step"]] == [1, 2]
