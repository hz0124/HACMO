#!/usr/bin/env python3
"""Small dependency-free smoke tests for SupernodeSim."""

from __future__ import annotations

import json
from pathlib import Path

from supernode_sim.api import simulate_layer
from supernode_sim.config import load_config
from supernode_sim.backends.ramulator import run_hbm_trace
from supernode_sim.backends.astra_wafer_core import (
    AstraWaferCoreCase,
    AstraWaferCoreConfig,
    AstraWaferCoreFlow,
    run_cases as run_astra_wafer_core_cases,
)
from supernode_sim.backends.wafer_core_noc import (
    WaferCoreNoCConfig,
    WaferCoreNoCTraffic,
    estimate_wafer_core_noc,
)
from supernode_sim.tracegen.moe import (
    analysis_compute_us_per_step,
    build_dispatch_matrices,
    build_window_counts,
    route_plan_to_simulator_inputs,
    solve_mapping,
)


def test_smoke_config_runs() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "deepseek_v32_attention_smoke.json")
    result = simulate_layer(config)
    assert result.total_latency_us > 0
    assert result.sram_near_bytes > 0
    assert result.hbm_bytes > 0


def test_wafer_core_noc_matches_legacy_byte_hop_formula() -> None:
    config = WaferCoreNoCConfig(
        rows=68,
        cols=68,
        base_noc_bw_tbps=5.0,
        gamma=1.3044203594054058,
        noc_bw_scale_mode="cols",
        near_sram_hops=4.0,
    )
    traffic = WaferCoreNoCTraffic(
        hbm_bytes=1024.0 * 1024.0,
        near_sram_bytes=256.0 * 1024.0,
        remote_sram_bytes=512.0 * 1024.0,
        avg_hbm_hops=12.5,
        near_sram_hops=4.0,
        avg_remote_hops=45.0,
    )
    result = estimate_wafer_core_noc(config, traffic)
    bw_bps = 5.0 * max(68 / 20.0, 1.0) * 1e12
    expected_hbm_us = (
        traffic.hbm_bytes * traffic.avg_hbm_hops / bw_bps * config.gamma * 1e6
    )
    expected_near_us = (
        traffic.near_sram_bytes * traffic.near_sram_hops / bw_bps * config.gamma * 1e6
    )
    expected_remote_us = (
        traffic.remote_sram_bytes
        * traffic.avg_remote_hops
        / bw_bps
        * config.gamma
        * 1e6
    )

    assert abs(result.hbm_us - expected_hbm_us) < 1e-12
    assert abs(result.near_sram_us - expected_near_us) < 1e-12
    assert abs(result.remote_sram_us - expected_remote_us) < 1e-12


def test_astra_wafer_core_tracegen_uses_explicit_flows(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[2]
    out = tmp_path / "astra_wafer_core"
    config = AstraWaferCoreConfig(
        astra_sim_dir=repo / "astra-sim",
        output_root=out,
        network_config=repo / "missing_network.yaml",
        system_config=repo / "missing_system.json",
        remote_memory_config=repo / "missing_memory.json",
        num_ranks=4,
        active_ranks=0,
        run_astra=False,
    )
    case = AstraWaferCoreCase(
        name="edge_to_inner",
        rows=2,
        cols=2,
        flows=(
            AstraWaferCoreFlow(src=0, dst=3, bytes=1024, tag=7),
            AstraWaferCoreFlow(src=1, dst=2, bytes=2048),
        ),
        hbm_bytes=3072.0,
        avg_hbm_hops=1.5,
    )
    rows = run_astra_wafer_core_cases(config, [case])
    summary = (
        out / "traces" / "edge_to_inner" / "wafer_core_noc.summary.json"
    ).read_text(encoding="utf-8")

    assert rows[0]["analytic_hbm_us"] > 0
    assert '"src": 0' in summary
    assert '"dst": 3' in summary
    assert '"tag": 7' in summary
    assert '"bytes": 2048' in summary
    assert '"total_flow_bytes": 3072' in summary


def test_moe_trace_plan_has_cross_wafer_traffic() -> None:
    root = Path(__file__).resolve().parents[1]
    repo = root.parent
    config = load_config(root / "configs" / "deepseek_v32_moe_astra_smoke.json")
    config.supernode.run_astra = False
    window_stats, window_src = build_window_counts(config, repo)
    mapping = solve_mapping(config, window_src)
    dispatch = build_dispatch_matrices(config, window_src, mapping)
    total = sum(b for mat in dispatch for dsts in mat.values() for b in dsts.values())
    assert window_stats
    assert len(mapping) == config.moe.num_experts
    assert total > 0


def test_roofline_compute_reuses_expert_weights_per_rank_batch() -> None:
    config = load_config(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "deepseek_v32_moe_astra_smoke.json"
    )
    config.workload.batch_size = 1024
    config.moe.num_requests = 1
    config.moe.window_size = 1
    config.moe.compute_backend = "roofline_h200"
    config.moe.hidden_size = 7168
    config.moe.expert_intermediate_size = 2048
    config.moe.h200_effective_tflops = 1979.0
    config.moe.h200_hbm_bw_tbps = 4.8
    config.moe.weight_bytes_per_elem = 1
    config.moe.activation_bytes_per_elem = 2
    mapping = {0: [0]}
    window_stats = [{"window_id": 0, "expert_counts": {"0": 1}}]
    window_src = [{"window_id": 0, "src_expert_counts": {"0": {"0": 1}}}]

    compute_us = analysis_compute_us_per_step(
        config,
        window_stats,
        mapping,
        window_mappings=[mapping],
        window_src_expert=window_src,
    )

    # 1024 tokens for one active expert should be dominated by batched GEMM
    # FLOPs, not by rereading the full expert weights 1024 times.
    assert 40.0 < compute_us < 60.0


def test_route_plan_is_consumed_without_rerouting() -> None:
    plan = {
        "metadata": {"batch_scale": 1.0},
        "steps": [
            {
                "step_id": 0,
                "window_id": 0,
                "token_idx": 1,
                "tokens_per_die": {"0": 1},
                "assignments": [
                    {"src_die": 2, "expert": 7, "dst_die": 3, "activations": 2}
                ],
                "dispatch_bytes": {"2": {"3": 100}},
                "combine_bytes": {"3": {"2": 200}},
                "rank_expert_tokens": {"3": {"7": 2}},
                "validation": {
                    "all_activations_routed": True,
                    "compute_matches_assignments": True,
                },
            }
        ],
    }

    stats, dispatch, combine, rank_experts = route_plan_to_simulator_inputs(
        plan, num_ranks=4
    )

    assert stats[0]["expert_counts"] == {"7": 2}
    assert dispatch == [{2: {3: 100}}]
    assert combine == [{3: {2: 200}}]
    assert rank_experts[0][3] == {7: 2.0}


def test_astra_tracegen_consumes_route_plan_bytes_exactly(tmp_path: Path, monkeypatch) -> None:
    # A trace-only run must also work in a fresh source checkout before build.
    original_exists = Path.exists
    monkeypatch.setattr(
        Path, "exists",
        lambda path: False if path.name.startswith("AstraSim_Analytical_") else original_exists(path),
    )
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "deepseek_v32_moe_astra_smoke.json")
    plan_path = tmp_path / "route_plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "batch_scale": 1.0,
                    "mapping_copy_provenance": {
                        "source": "effective_mapping_used_for_each_step",
                        "per_step": [
                            {
                                "step_id": 0,
                                "window_id": 0,
                                "total_copies": 258,
                                "extra_replicas": 2,
                            }
                        ],
                    },
                },
                "steps": [
                    {
                        "step_id": 0,
                        "window_id": 0,
                        "token_idx": 1,
                        "tokens_per_die": {"0": 1},
                        "pre_dispatch_segments": [
                            {
                                "name": "test_collective",
                                "messages": {"0": {"1": 50}},
                            }
                        ],
                        "assignments": [
                            {
                                "src_die": 2,
                                "expert": 7,
                                "dst_die": 3,
                                "activations": 2,
                            }
                        ],
                        "dispatch_bytes": {"2": {"3": 100}},
                        "combine_bytes": {"3": {"2": 200}},
                        "rank_expert_tokens": {"3": {"7": 2}},
                        "validation": {
                            "all_activations_routed": True,
                            "compute_matches_assignments": True,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    config.moe.route_plan_json = str(plan_path)
    config.moe.astra_trace_mode = "communication_only"
    config.supernode.output_root = str(tmp_path / "astra")
    config.supernode.run_astra = False

    result = simulate_layer(config)

    assert result.metadata["profile"]["route_plan_consumed"] is True
    assert result.metadata["profile"]["total_pre_dispatch_collective_bytes"] == 50
    assert result.metadata["profile"]["total_dispatch_bytes"] == 100
    assert result.metadata["profile"]["total_combine_bytes"] == 200
    assert result.metadata["profile"]["avg_extra_replicas"] == 2.0
    assert result.metadata["profile"]["replica_count_source"] == (
        "route_plan_effective_mapping"
    )


def test_ramulator_hbm_wrapper_requires_executable(tmp_path: Path) -> None:
    trace = tmp_path / "hbm.csv"
    trace.write_text(
        "case,op,addr,size,layer,step,batch,token,pattern\n"
        "smoke,R,0x0,128,0,0,0,0,profile\n",
        encoding="utf-8",
    )
    try:
        run_hbm_trace(
            trace, tmp_path / "out", ramulator_exec=str(tmp_path / "missing_ramulator2")
        )
    except FileNotFoundError as exc:
        assert "Ramulator2 executable not found" in str(exc)
    else:
        raise AssertionError(
            "run_hbm_trace should fail loudly without a Ramulator2 executable"
        )


if __name__ == "__main__":
    test_smoke_config_runs()
    test_moe_trace_plan_has_cross_wafer_traffic()
    print("supernode_sim smoke passed")
