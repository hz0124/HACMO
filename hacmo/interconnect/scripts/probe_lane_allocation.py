#!/usr/bin/env python3
"""Probe lane-level TP3 crossbar allocation policies.

This is a diagnostic tool. It reuses the current TP3 edge-local timing basis
but replaces integer logical-port grants with continuous lane/credit shares.
It does not overwrite the conservative TP3 final pack.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PROJECT_DIR
for p in PROJECT_DIR.parents:
    if (p / "hacmo" / "moe").exists() and (p / "supernode_sim").exists():
        REPO_DIR = p
        break
TP2_DIR = REPO_DIR / "hacmo" / "moe"

sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(PROJECT_DIR))

from run_tp3_experiment import (  # noqa: E402
    load_json,
    load_tp2_metrics,
    normalize_mapping,
    rail_yaml_for_crossbar,
    row_float,
    runtime_windows_path,
    runtime_windows_to_mappings,
    topology_label_from_stem,
    traffic_matrix_path,
    write_csv,
)
from supernode_sim.plugins.crossbar import (  # noqa: E402
    _allocation_summary,
    _apply_phase_service_times,
    _phase_crossbar_ns,
    _phase_projected_traffic,
    allocate_edge_local_ports,
    build_projected_edge_traffic,
    crossbar_comm_us_per_step,
    edge_groups,
    load_crossbar_config,
)
from supernode_sim.tracegen.moe import build_combine_matrices, build_dispatch_matrices  # noqa: E402
from supernode_sim.types import MoEConfig, SimulationConfig, SupernodeConfig, WorkloadConfig  # noqa: E402


PhaseExit = Dict[int, Dict[int, Dict[int, float]]]
ProjectedTraffic = Dict[int, Dict[int, Dict[int, Dict[str, float]]]]
ServiceMap = Dict[int, Dict[int, Dict[int, Dict[str, float]]]]


def _active_groups(projected: ProjectedTraffic) -> Iterable[Tuple[int, int, int, float, float]]:
    for wafer, wafer_traffic in projected.items():
        for edge, edge_traffic in wafer_traffic.items():
            for die, vals in edge_traffic.items():
                exit_bytes = float(vals.get("exit", 0.0))
                transit = float(vals.get("transit", 0.0))
                if exit_bytes > 0:
                    yield wafer, edge, die, exit_bytes, transit


def _alloc_lane_shares(
    policy: str,
    projected: ProjectedTraffic,
    side: int,
    beta: float,
    phase_exit: PhaseExit | None = None,
) -> Dict[int, Dict[int, Dict[int, Dict[str, float]]]]:
    """Allocate continuous lane shares inside each wafer-edge scope."""
    shares: Dict[int, Dict[int, Dict[int, Dict[str, float]]]] = {}
    for wafer, wafer_traffic in projected.items():
        shares[wafer] = {}
        for edge, edge_traffic in wafer_traffic.items():
            local_ids = list(edge_traffic.keys())
            if policy == "lane_uniform":
                score = {die: 1.0 for die in local_ids}
                active = local_ids
            elif policy == "lane_exit":
                score = {die: float(vals.get("exit", 0.0)) for die, vals in edge_traffic.items()}
                active = [die for die in local_ids if score[die] > 0]
            elif policy == "lane_score":
                score = {
                    die: float(vals.get("exit", 0.0)) + beta * float(vals.get("transit", 0.0))
                    for die, vals in edge_traffic.items()
                }
                active = [die for die in local_ids if float(edge_traffic[die].get("exit", 0.0)) > 0]
            elif policy == "lane_phase_exit":
                if phase_exit is None:
                    raise ValueError("lane_phase_exit requires phase_exit")
                phase_edge = phase_exit.get(wafer, {}).get(edge, {})
                score = {die: float(phase_edge.get(die, 0.0)) for die in local_ids}
                active = [die for die in local_ids if score[die] > 0]
            else:
                raise ValueError(f"Unknown lane policy: {policy}")

            total_score = sum(score[die] for die in active)
            shares[wafer][edge] = {}
            for die in local_ids:
                if not active or die not in active:
                    lane_share = 0.0
                elif total_score <= 0:
                    lane_share = float(side) / len(active)
                else:
                    lane_share = float(side) * score[die] / total_score
                shares[wafer][edge][die] = {
                    "lanes": lane_share,
                    "ports": lane_share,
                    "score": score.get(die, 0.0),
                }
    return shares


def _lanes_per_logical_port(cfg: dict) -> int:
    """Map the current YAML bandwidth to the RTL planning lane-count anchor."""
    per_port_bw = float(cfg["per_port_bw_gbps"])
    return max(1, int(round(per_port_bw / 50.0)))


def _alloc_random_lane_slices(
    projected: ProjectedTraffic,
    side: int,
    lanes_per_port: int,
    rng: random.Random,
) -> Dict[int, Dict[int, Dict[int, Dict[str, float]]]]:
    """Randomly grant physical lane slices inside each wafer-edge scope.

    Each logical I/O port is split into ``lanes_per_port`` equal slices. A slice
    contributes ``1 / lanes_per_port`` logical-port bandwidth.
    """
    total_slices = side * max(lanes_per_port, 1)
    shares: Dict[int, Dict[int, Dict[int, Dict[str, float]]]] = {}
    for wafer, wafer_traffic in projected.items():
        shares[wafer] = {}
        for edge, edge_traffic in wafer_traffic.items():
            local_ids = list(edge_traffic.keys())
            active = [die for die, vals in edge_traffic.items() if float(vals.get("exit", 0.0)) > 0]
            counts = {die: 0 for die in local_ids}
            if active:
                for _ in range(total_slices):
                    counts[rng.choice(active)] += 1
            shares[wafer][edge] = {}
            for die in local_ids:
                lane_share = counts[die] / max(lanes_per_port, 1)
                shares[wafer][edge][die] = {
                    "lanes": lane_share,
                    "ports": lane_share,
                    "score": float(counts[die]),
                }
    return shares


def _lane_phase_service(
    phase_exit: PhaseExit,
    lane_alloc: Dict[int, Dict[int, Dict[int, Dict[str, float]]]],
    per_port_bw_gbps: float,
) -> ServiceMap:
    out: ServiceMap = {}
    for wafer, wafer_exit in phase_exit.items():
        out[wafer] = {}
        for edge, exits in wafer_exit.items():
            out[wafer][edge] = {}
            for die, exit_bytes in exits.items():
                lanes = float(lane_alloc.get(wafer, {}).get(edge, {}).get(die, {}).get("lanes", 0.0))
                if lanes > 0 and exit_bytes > 0:
                    service_ns = float(exit_bytes) / max(lanes * per_port_bw_gbps, 1e-9)
                else:
                    service_ns = 0.0
                out[wafer][edge][die] = {
                    "ports": lanes,
                    "lanes": lanes,
                    "exit_bytes": float(exit_bytes),
                    "exit_bw_gbps": lanes * per_port_bw_gbps,
                    "service_ns": service_ns,
                }
    return out


def _policy_comm_one_window(
    policy: str,
    dispatch_mat: Dict[int, Dict[int, int]],
    combine_mat: Dict[int, Dict[int, int]],
    cfg: dict,
    beta: float,
    rho: float,
    rng: random.Random | None = None,
) -> Tuple[float, float, ProjectedTraffic, Dict[int, Dict[int, Dict[int, Dict[str, float]]]]]:
    dpw = cfg["dies_per_wafer"]
    side = cfg["side"]
    num_wafers = int(cfg.get("num_wafers", 1))
    per_port_bw = float(cfg["per_port_bw_gbps"])

    projected = build_projected_edge_traffic(dispatch_mat, combine_mat, dpw, side, num_wafers=num_wafers)
    dispatch_exit = _phase_projected_traffic(dispatch_mat, dpw, side, num_wafers=num_wafers)
    combine_exit = _phase_projected_traffic(combine_mat, dpw, side, num_wafers=num_wafers)

    if policy == "port_static":
        port_alloc = {
            wafer: {
                edge: {
                    die: {
                        "ports": 1 if float(vals.get("exit", 0.0)) > 0 else 0,
                        "exit_bw_gbps": per_port_bw if float(vals.get("exit", 0.0)) > 0 else 0.0,
                        "service_ns": 0.0,
                    }
                    for die, vals in edge_traffic.items()
                }
                for edge, edge_traffic in wafer_traffic.items()
            }
            for wafer, wafer_traffic in projected.items()
        }
        dispatch_service = _apply_phase_service_times(dispatch_exit, port_alloc, per_port_bw, "naive_crossbar", rho=0.0)
        combine_service = _apply_phase_service_times(combine_exit, port_alloc, per_port_bw, "naive_crossbar", rho=0.0)
        alloc_for_debug = port_alloc
    elif policy == "port_exit":
        port_alloc = allocate_edge_local_ports("naive_crossbar", projected, side, per_port_bw, beta=0.0, rho=0.0)
        dispatch_service = _apply_phase_service_times(dispatch_exit, port_alloc, per_port_bw, "naive_crossbar", rho=0.0)
        combine_service = _apply_phase_service_times(combine_exit, port_alloc, per_port_bw, "naive_crossbar", rho=0.0)
        alloc_for_debug = port_alloc
    elif policy == "port_full":
        port_alloc = allocate_edge_local_ports("dynamic_bw", projected, side, per_port_bw, beta=beta, rho=rho)
        dispatch_service = _apply_phase_service_times(dispatch_exit, port_alloc, per_port_bw, "dynamic_bw", rho=rho)
        combine_service = _apply_phase_service_times(combine_exit, port_alloc, per_port_bw, "dynamic_bw", rho=rho)
        alloc_for_debug = port_alloc
    elif policy in {"lane_uniform", "lane_exit", "lane_score"}:
        lane_alloc = _alloc_lane_shares(policy, projected, side, beta)
        dispatch_service = _lane_phase_service(dispatch_exit, lane_alloc, per_port_bw)
        combine_service = _lane_phase_service(combine_exit, lane_alloc, per_port_bw)
        alloc_for_debug = lane_alloc
    elif policy == "lane_random_slices":
        if rng is None:
            rng = random.Random(0)
        lane_alloc = _alloc_random_lane_slices(projected, side, _lanes_per_logical_port(cfg), rng)
        dispatch_service = _lane_phase_service(dispatch_exit, lane_alloc, per_port_bw)
        combine_service = _lane_phase_service(combine_exit, lane_alloc, per_port_bw)
        alloc_for_debug = lane_alloc
    elif policy == "lane_phase_exit":
        dispatch_alloc = _alloc_lane_shares(policy, projected, side, beta, phase_exit=dispatch_exit)
        combine_alloc = _alloc_lane_shares(policy, projected, side, beta, phase_exit=combine_exit)
        dispatch_service = _lane_phase_service(dispatch_exit, dispatch_alloc, per_port_bw)
        combine_service = _lane_phase_service(combine_exit, combine_alloc, per_port_bw)
        alloc_for_debug = dispatch_alloc
    else:
        raise ValueError(f"Unknown policy: {policy}")

    dispatch_ns = _phase_crossbar_ns(dispatch_mat, dispatch_service, cfg)
    combine_ns = _phase_crossbar_ns(combine_mat, combine_service, cfg)
    return dispatch_ns / 1000.0, combine_ns / 1000.0, projected, alloc_for_debug


def _allocation_stats(
    projected: ProjectedTraffic,
    alloc: Dict[int, Dict[int, Dict[int, Dict[str, float]]]],
    side: int,
) -> Dict[str, float]:
    active_counts: List[int] = []
    lane_values: List[float] = []
    max_exit_imbalance = 0.0
    for wafer, wafer_traffic in projected.items():
        for edge, edge_traffic in wafer_traffic.items():
            active = [die for die, vals in edge_traffic.items() if float(vals.get("exit", 0.0)) > 0]
            active_counts.append(len(active))
            exits = [float(edge_traffic[die].get("exit", 0.0)) for die in active]
            if exits and min(exits) > 0:
                max_exit_imbalance = max(max_exit_imbalance, max(exits) / min(exits))
            for die in edge_traffic:
                vals = alloc.get(wafer, {}).get(edge, {}).get(die, {})
                lane_values.append(float(vals.get("lanes", vals.get("ports", 0.0))))

    fully_active = sum(1 for c in active_counts if c == side)
    total_scopes = len(active_counts)
    nonzero_lanes = [v for v in lane_values if v > 0]
    return {
        "edge_scopes": float(total_scopes),
        "fully_active_edge_scopes": float(fully_active),
        "fully_active_fraction": fully_active / max(total_scopes, 1),
        "min_active_groups": float(min(active_counts) if active_counts else 0),
        "max_active_groups": float(max(active_counts) if active_counts else 0),
        "min_lane_or_port_grant": min(nonzero_lanes) if nonzero_lanes else 0.0,
        "max_lane_or_port_grant": max(nonzero_lanes) if nonzero_lanes else 0.0,
        "max_exit_imbalance_within_scope": max_exit_imbalance,
    }


def _build_matrices(args: argparse.Namespace, topo_art: Path, rail_yaml: Path, bs: int, cfg: dict, tp2_tm: dict):
    mapping_ours = load_json(topo_art / "mappings" / "ours_static.json")
    mapping_input_dir = topo_art / "mapping_inputs" / "runtime"
    if not mapping_input_dir.exists():
        mapping_input_dir = topo_art / "mapping_inputs" / "offline"
    if not mapping_input_dir.exists():
        mapping_input_dir = topo_art / "mapping_inputs"
    window_src = load_json(mapping_input_dir / "window_src_expert_counts.json")
    runtime_windows = load_json(runtime_windows_path(topo_art, bs))

    tm_params = tp2_tm.get("params", {}) if isinstance(tp2_tm, dict) else {}
    effective_window_size = int(tm_params.get("window_size", args.window_size))
    effective_hidden_size = int(tm_params.get("hidden_size", args.hidden_size))
    effective_dispatch_bpe = int(tm_params.get("dispatch_bytes_per_hidden_elem", args.dispatch_bytes_per_hidden_elem))
    effective_combine_bpe = int(tm_params.get("combine_bytes_per_hidden_elem", args.combine_bytes_per_hidden_elem))
    effective_batch_scale = float(tm_params.get("batch_scale", 0.0))
    if effective_batch_scale > 0:
        effective_num_requests = max(1, int(round(bs / effective_batch_scale)))
    else:
        effective_num_requests = args.num_requests
        effective_batch_scale = bs / max(effective_num_requests, 1)

    expert_mapping = normalize_mapping(mapping_ours)
    window_mappings = runtime_windows_to_mappings(runtime_windows)
    sim_cfg = SimulationConfig(
        workload=WorkloadConfig(batch_size=bs),
        moe=MoEConfig(
            num_experts=args.num_experts,
            topk=args.topk,
            num_requests=effective_num_requests,
            window_size=effective_window_size,
            hidden_size=effective_hidden_size,
            bytes_per_hidden_elem=2,
            dispatch_bytes_per_hidden_elem=effective_dispatch_bpe,
            combine_bytes_per_hidden_elem=effective_combine_bpe,
            expert_mapping=expert_mapping,
        ),
        supernode=SupernodeConfig(
            num_wafers=cfg["num_wafers"],
            dies_per_wafer=cfg["dies_per_wafer"],
            network_config=str(rail_yaml),
        ),
    )
    dispatch_mats = build_dispatch_matrices(sim_cfg, window_src, expert_mapping, window_mappings=window_mappings)
    combine_mats = build_combine_matrices(sim_cfg, dispatch_mats)
    return dispatch_mats, combine_mats, effective_batch_scale, effective_window_size


def main() -> None:
    p = argparse.ArgumentParser(description="Probe lane-level TP3 allocation policies.")
    p.add_argument(
        "--tp2-results-dir",
        default=str(TP2_DIR / "outputs" / "tp2_serving_epoch_paper_pack_20260613"),
    )
    p.add_argument("--network-yamls", nargs="+", default=[
        str(PROJECT_DIR / "configs/network/hierarchical_4w_8x8_mesh_crossbar_900.yaml"),
        str(PROJECT_DIR / "configs/network/hierarchical_16w_4x4_mesh_crossbar_500.yaml"),
    ])
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[16, 32, 64, 128])
    p.add_argument("--hidden-size", type=int, default=7168)
    p.add_argument("--dispatch-bytes-per-hidden-elem", type=int, default=1)
    p.add_argument("--combine-bytes-per-hidden-elem", type=int, default=2)
    p.add_argument("--num-experts", type=int, default=256)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--num-requests", type=int, default=64)
    p.add_argument("--window-size", type=int, default=1)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--rho", type=float, default=0.4)
    p.add_argument("--crossbar-latency-ns", type=float, default=20.0)
    p.add_argument(
        "--ratio-denominator",
        choices=["crossbar_internal_rail", "tp2_traffic_matrix"],
        default="crossbar_internal_rail",
        help=(
            "Analytical rail denominator used to calibrate lane policies. "
            "Use 'tp2_traffic_matrix' only to reproduce legacy TP3 pack scaling."
        ),
    )
    p.add_argument("--window-limit", type=int, default=0)
    p.add_argument("--random-samples", type=int, default=16)
    p.add_argument("--random-seed", type=int, default=928)
    p.add_argument(
        "--policies",
        nargs="+",
        default=[
            "port_static",
            "port_exit",
            "port_full",
            "lane_uniform",
            "lane_random_slices",
            "lane_exit",
            "lane_score",
            "lane_phase_exit",
        ],
        help="Allocation policies to evaluate.",
    )
    p.add_argument("--output-root", default=None)
    args = p.parse_args()

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root).resolve() if args.output_root else (
        PROJECT_DIR / "outputs" / f"tp3_lane_probe_{ts}"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    tp2_dir = Path(args.tp2_results_dir).resolve()
    tp2_metrics = load_tp2_metrics(tp2_dir)
    if not tp2_metrics:
        raise SystemExit(f"No TP2 metrics found in {tp2_dir}")

    valid_policies = {
        "port_static",
        "port_exit",
        "port_full",
        "lane_uniform",
        "lane_random_slices",
        "lane_exit",
        "lane_score",
        "lane_phase_exit",
    }
    policies = list(args.policies)
    invalid_policies = [policy for policy in policies if policy not in valid_policies]
    if invalid_policies:
        raise SystemExit(f"Unknown policies: {', '.join(invalid_policies)}")
    rows: List[Dict[str, object]] = []
    debug: Dict[str, object] = {}

    for crossbar_yaml_str in args.network_yamls:
        crossbar_yaml = Path(crossbar_yaml_str).resolve()
        cfg = load_crossbar_config(str(crossbar_yaml))
        cfg["crossbar_latency_ns"] = args.crossbar_latency_ns
        rail_yaml = rail_yaml_for_crossbar(crossbar_yaml)
        topo_label = topology_label_from_stem(rail_yaml.stem)
        topo_art = tp2_dir / topo_label
        if not topo_art.exists():
            print(f"[WARN] Missing topology artifacts: {topo_art}", flush=True)
            continue

        for bs in args.batch_sizes:
            bar1 = tp2_metrics.get((topo_label, "vllm", bs))
            bar2 = tp2_metrics.get((topo_label, "ours_dynamic", bs))
            if not bar1 or not bar2:
                print(f"[WARN] Missing TP2 rows for {topo_label} bpd{bs}", flush=True)
                continue

            tp2_tm_path = traffic_matrix_path(topo_art, "ours_dynamic", bs)
            tp2_tm = load_json(tp2_tm_path)
            rail_analytical_comm = (
                float(tp2_tm["summary"]["avg_step_dispatch_us"])
                + float(tp2_tm["summary"]["avg_step_combine_us"])
            )
            bar1_total = row_float(bar1, "total_us")
            bar2_compute = row_float(bar2, "compute_us")
            bar2_comm = row_float(bar2, "communication_us", "comm_us")

            dispatch_mats, combine_mats, _batch_scale, window_size = _build_matrices(
                args, topo_art, rail_yaml, bs, cfg, tp2_tm
            )
            if args.window_limit > 0:
                dispatch_mats = dispatch_mats[:args.window_limit]
                combine_mats = combine_mats[:args.window_limit]
            internal_rail = crossbar_comm_us_per_step(
                dispatch_mats,
                combine_mats,
                cfg,
                "naive_crossbar",
                batch_scale=1.0,
                window_size=window_size,
                beta=0.0,
                rho=0.0,
                crossbar_latency_ns=args.crossbar_latency_ns,
            )
            crossbar_internal_rail_comm = float(internal_rail["rail_analytical_comm_us"])
            if args.ratio_denominator == "tp2_traffic_matrix":
                ratio_denominator_comm = rail_analytical_comm
            else:
                ratio_denominator_comm = crossbar_internal_rail_comm

            print(
                f"[LaneProbe] {topo_label} bpd{bs}: {len(dispatch_mats)} windows; "
                f"denominator={args.ratio_denominator} {ratio_denominator_comm:.6f}us",
                flush=True,
            )
            for policy in policies:
                dispatch_us_values: List[float] = []
                combine_us_values: List[float] = []
                stat_accum: Dict[str, List[float]] = {}
                first_debug = None
                for dispatch_mat, combine_mat in zip(dispatch_mats, combine_mats):
                    if policy == "lane_random_slices":
                        sample_dispatch: List[float] = []
                        sample_combine: List[float] = []
                        sample_projected = None
                        sample_alloc = None
                        for sample_idx in range(max(args.random_samples, 1)):
                            rng = random.Random(
                                args.random_seed
                                + sample_idx
                                + 1009 * len(dispatch_us_values)
                                + 9176 * cfg["num_wafers"]
                                + 131 * cfg["side"]
                            )
                            d_us, c_us, projected, alloc = _policy_comm_one_window(
                                policy, dispatch_mat, combine_mat, cfg, args.beta, args.rho, rng=rng
                            )
                            sample_dispatch.append(d_us)
                            sample_combine.append(c_us)
                            sample_projected = projected
                            sample_alloc = alloc
                        dispatch_us = sum(sample_dispatch) / len(sample_dispatch)
                        combine_us = sum(sample_combine) / len(sample_combine)
                        projected = sample_projected if sample_projected is not None else {}
                        alloc = sample_alloc if sample_alloc is not None else {}
                    else:
                        dispatch_us, combine_us, projected, alloc = _policy_comm_one_window(
                            policy, dispatch_mat, combine_mat, cfg, args.beta, args.rho
                        )
                    dispatch_us_values.append(dispatch_us)
                    combine_us_values.append(combine_us)
                    stats = _allocation_stats(projected, alloc, cfg["side"])
                    for key, val in stats.items():
                        stat_accum.setdefault(key, []).append(float(val))
                    if first_debug is None:
                        first_debug = _allocation_summary(projected, alloc)

                num_windows = len(dispatch_us_values) or 1
                dispatch_us = sum(dispatch_us_values) / num_windows / max(window_size, 1)
                combine_us = sum(combine_us_values) / num_windows / max(window_size, 1)
                analytical_comm = dispatch_us + combine_us
                ratio = analytical_comm / max(ratio_denominator_comm, 1e-9)
                calibrated_comm = min(bar2_comm * ratio, bar2_comm)
                total_us = bar2_compute + calibrated_comm
                row = {
                    "topology": topo_label,
                    "batch_size": bs,
                    "policy": policy,
                    "allocation_granularity": "integer_port" if policy.startswith("port_") else "continuous_lane",
                    "analytical_dispatch_us": f"{dispatch_us:.6f}",
                    "analytical_combine_us": f"{combine_us:.6f}",
                    "analytical_comm_us": f"{analytical_comm:.6f}",
                    "tp2_traffic_matrix_rail_analytical_comm_us": f"{rail_analytical_comm:.6f}",
                    "crossbar_internal_rail_analytical_comm_us": f"{crossbar_internal_rail_comm:.6f}",
                    "ratio_denominator_basis": args.ratio_denominator,
                    "ratio_denominator_comm_us": f"{ratio_denominator_comm:.6f}",
                    "ratio_to_denominator": f"{ratio:.9f}",
                    "calibrated_comm_us": f"{calibrated_comm:.6f}",
                    "total_us": f"{total_us:.6f}",
                    "speedup_vs_naive_rail": f"{bar1_total / max(total_us, 1e-9):.6f}",
                    "delta_total_vs_port_exit_us": "",
                    "crossbar_latency_ns": f"{args.crossbar_latency_ns:.6f}",
                    "beta": f"{args.beta:.6f}",
                    "rho": f"{args.rho:.6f}",
                }
                for key, values in stat_accum.items():
                    row[f"avg_{key}"] = f"{(sum(values) / max(len(values), 1)):.6f}"
                rows.append(row)
                debug[f"{topo_label}_bpd{bs}_{policy}"] = {
                    "first_window": first_debug,
                    "policy": policy,
                }

    baseline_by_topo = {
        (r["topology"], r["batch_size"]): float(r["total_us"])
        for r in rows
        if r["policy"] == "port_exit"
    }
    for row in rows:
        base = baseline_by_topo.get((row["topology"], row["batch_size"]))
        if base:
            row["delta_total_vs_port_exit_us"] = f"{float(row['total_us']) - base:.6f}"

    fieldnames = [
        "topology", "batch_size", "policy", "allocation_granularity",
        "analytical_dispatch_us", "analytical_combine_us", "analytical_comm_us",
        "tp2_traffic_matrix_rail_analytical_comm_us",
        "crossbar_internal_rail_analytical_comm_us",
        "ratio_denominator_basis", "ratio_denominator_comm_us",
        "ratio_to_denominator", "calibrated_comm_us", "total_us",
        "speedup_vs_naive_rail", "delta_total_vs_port_exit_us",
        "crossbar_latency_ns", "beta", "rho",
        "avg_edge_scopes", "avg_fully_active_edge_scopes", "avg_fully_active_fraction",
        "avg_min_active_groups", "avg_max_active_groups",
        "avg_min_lane_or_port_grant", "avg_max_lane_or_port_grant",
        "avg_max_exit_imbalance_within_scope",
    ]
    write_csv(output_root / "tp3_lane_probe_metrics.csv", rows, fieldnames)
    (output_root / "tp3_lane_probe_debug.json").write_text(
        json.dumps(debug, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "generated_at": dt.datetime.now().isoformat(),
        "tp2_results_dir": str(tp2_dir),
        "model": "ASTRA rail-calibrated analytical edge-local crossbar lane probe",
        "ratio_denominator": args.ratio_denominator,
        "policies": {
            "port_static": "fixed one logical port per active edge group under the current edge projection",
            "port_exit": "current integer-port Crossbar+TP2 policy",
            "port_full": "current integer-port Full+TP2 policy",
            "lane_uniform": "one continuous lane per edge group, no dynamic reassignment",
            "lane_random_slices": "random physical-lane slice grants within one edge-local crossbar",
            "lane_exit": "continuous lanes assigned proportional to projected external exit demand",
            "lane_score": "continuous lanes assigned proportional to exit + beta * transit",
            "lane_phase_exit": "best-case per-phase exit-proportional lane reassignment",
        },
        "random_samples": args.random_samples,
        "random_seed": args.random_seed,
    }
    (output_root / "tp3_lane_probe_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[DONE] Lane probe results in: {output_root}", flush=True)


if __name__ == "__main__":
    main()
