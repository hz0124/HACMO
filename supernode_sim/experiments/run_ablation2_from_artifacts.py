#!/usr/bin/env python3
"""Run SupernodeSim from precomputed TP2 trace/mapping artifacts."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import statistics
from pathlib import Path

import yaml

from hacmo.moe.tp2_core.moentwine_baseline import (
    incremental_collective_accounting,
)
from supernode_sim.api import simulate_layer
from supernode_sim.config import load_config


METHOD_TO_MAPPING = {
    "vllm": "uniform",
    "moentwine": "moentwine_static",
    "ours_static": "ours_static",
    "ours_dynamic": "ours_static",
}


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run SupernodeSim using moe artifacts."
    )
    p.add_argument(
        "--base-config",
        default="supernode_sim/configs/deepseek_v32_moe_astra_smoke.json",
    )
    p.add_argument(
        "--artifact-topology-dir",
        required=True,
        help="Topology output dir, e.g. hacmo/moe/outputs/.../16w_4x4_rail_500",
    )
    p.add_argument("--network-config", required=True)
    p.add_argument(
        "--methods",
        nargs="+",
        default=["vllm", "moentwine", "ours_static", "ours_dynamic"],
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Route-plan seeds. When provided, consume route_plans/seed<seed>/<method>_bpd<batch>.json.",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=None)
    p.add_argument(
        "--batch-scope",
        choices=["global", "per_die"],
        default="global",
        help="Interpret --batch-size/--batch-sizes as total system batch or per-die batch.",
    )
    p.add_argument(
        "--num-requests",
        type=int,
        default=None,
        help="Base request count represented by artifact trace. Defaults to mapping_inputs/runtime/metadata.json:num_requests.",
    )
    p.add_argument("--num-experts", type=int, default=256)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--hidden-size", type=int, default=7168)
    p.add_argument("--window-size", type=int, default=1)
    p.add_argument("--decode-len", type=int, default=2)
    p.add_argument(
        "--exclude-leading-windows",
        type=int,
        default=0,
        help="Exclude this many leading active runtime windows from ASTRA trace generation and normalization.",
    )
    p.add_argument("--topology-label", default="")
    p.add_argument(
        "--compute-backend",
        choices=["roofline_h200", "constant_per_token"],
        default="roofline_h200",
    )
    p.add_argument("--dispatch-bytes-per-hidden-elem", type=int, default=1)
    p.add_argument("--combine-bytes-per-hidden-elem", type=int, default=2)
    p.add_argument("--expert-compute-us-per-token", type=float, default=0.1)
    p.add_argument("--expert-intermediate-size", type=int, default=2048)
    p.add_argument("--h200-effective-tflops", type=float, default=1979.0)
    p.add_argument("--h200-hbm-bw-tbps", type=float, default=4.8)
    p.add_argument("--weight-bytes-per-elem", type=int, default=1)
    p.add_argument("--activation-bytes-per-elem", type=int, default=2)
    p.add_argument(
        "--direct-p2p",
        action="store_true",
        help="Disable explicit src->edge->edge->dst routing for cross-wafer messages.",
    )
    p.add_argument(
        "--astra-trace-mode",
        choices=["full", "communication_only", "dispatch_only", "combine_only"],
        default="full",
        help="ASTRA trace dependency mode. communication_only removes expert compute; dispatch_only/combine_only isolate one collective.",
    )
    p.add_argument(
        "--communication-only",
        action="store_true",
        help="Deprecated alias for --astra-trace-mode communication_only.",
    )
    p.add_argument(
        "--astra-network-backend",
        choices=["congestion_aware", "congestion_unaware"],
        default="congestion_aware",
        help="ASTRA analytical backend. congestion_aware models queued switch-mesh links.",
    )
    p.add_argument("--run-astra", action="store_true")
    p.add_argument("--output-root", default="/tmp/supernode_sim/tp2_artifact_check")
    p.add_argument("--output-csv", default="/tmp/supernode_sim/tp2_artifact_check.csv")
    args = p.parse_args()

    artifact_dir = Path(args.artifact_topology_dir)
    base = load_config(args.base_config)
    artifact_num_requests = _artifact_num_requests(artifact_dir)
    num_requests = (
        int(args.num_requests)
        if args.num_requests is not None
        else artifact_num_requests
    )
    batch_sizes = (
        args.batch_sizes if args.batch_sizes is not None else [args.batch_size]
    )
    topology_label = args.topology_label or artifact_dir.name
    rows = []
    for bs in batch_sizes:
        run_seeds: list[int | None] = args.seeds if args.seeds is not None else [None]
        for seed in run_seeds:
            for method in args.methods:
                mapping_name = METHOD_TO_MAPPING[method]
                cfg = copy.deepcopy(base)
                cfg.backend = "astra"
                cfg.workload.decode_len = args.decode_len
                cfg.workload.hidden_size = args.hidden_size
                cfg.moe.num_experts = args.num_experts
                cfg.moe.topk = args.topk
                cfg.moe.num_requests = num_requests
                cfg.moe.window_size = args.window_size
                cfg.moe.hidden_size = args.hidden_size
                cfg.moe.compute_backend = args.compute_backend
                cfg.moe.dispatch_bytes_per_hidden_elem = (
                    args.dispatch_bytes_per_hidden_elem
                )
                cfg.moe.combine_bytes_per_hidden_elem = (
                    args.combine_bytes_per_hidden_elem
                )
                cfg.moe.expert_compute_us_per_token = args.expert_compute_us_per_token
                cfg.moe.expert_intermediate_size = args.expert_intermediate_size
                cfg.moe.h200_effective_tflops = args.h200_effective_tflops
                cfg.moe.h200_hbm_bw_tbps = args.h200_hbm_bw_tbps
                cfg.moe.weight_bytes_per_elem = args.weight_bytes_per_elem
                cfg.moe.activation_bytes_per_elem = args.activation_bytes_per_elem
                cfg.moe.route_via_edge_ports = (
                    False
                    if args.astra_network_backend == "congestion_aware"
                    else not args.direct_p2p
                )
                cfg.moe.mapping_method = f"external_{method}"
                cfg.moe.expert_mapping_json = str(
                    artifact_dir / f"mappings/{mapping_name}.json"
                )
                route_plan = (
                    artifact_dir
                    / "route_plans"
                    / f"seed{seed}"
                    / f"{method}_bpd{bs}.json"
                    if seed is not None
                    else None
                )
                if route_plan is not None:
                    if not route_plan.exists():
                        raise FileNotFoundError(f"route plan not found: {route_plan}")
                    cfg.moe.route_plan_json = str(route_plan)
                    cfg.moe.window_stats_json = ""
                    cfg.moe.window_src_expert_json = ""
                else:
                    cfg.moe.route_plan_json = ""
                    cfg.moe.window_stats_json = str(
                        artifact_dir / "mapping_inputs/runtime/window_stats.json"
                    )
                    cfg.moe.window_src_expert_json = str(
                        artifact_dir
                        / "mapping_inputs/runtime/window_src_expert_counts.json"
                    )
                cfg.moe.window_mappings_json = ""
                cfg.moe.exclude_leading_windows = max(args.exclude_leading_windows, 0)
                trace_mode = (
                    "communication_only"
                    if args.communication_only
                    else args.astra_trace_mode
                )
                cfg.moe.astra_trace_mode = trace_mode
                cfg.supernode.network_config = args.network_config
                cfg.supernode.astra_network_backend = args.astra_network_backend
                _apply_network_shape(cfg, args.network_config)
                num_ranks = cfg.supernode.num_wafers * cfg.supernode.dies_per_wafer
                batch_per_die, global_batch = _resolve_batch_scope(
                    bs, args.batch_scope, num_ranks
                )
                if method == "ours_dynamic" and route_plan is None:
                    cfg.moe.window_mappings_json = str(
                        _dynamic_windows_path(
                            artifact_dir, bs, args.batch_scope, global_batch
                        )
                    )
                cfg.workload.batch_size = global_batch
                cfg.supernode.run_astra = args.run_astra
                mode_suffix = "" if trace_mode == "full" else f"_{trace_mode}"
                batch_tag = (
                    f"bs{bs}"
                    if args.batch_scope == "global"
                    else f"bpd{bs}_gb{global_batch}"
                )
                seed_tag = f"seed{seed}/" if seed is not None else ""
                cfg.supernode.output_root = str(
                    Path(args.output_root)
                    / topology_label
                    / seed_tag
                    / f"{method}_{batch_tag}{mode_suffix}"
                )
                result = simulate_layer(cfg)
                profile = result.metadata["profile"]
                steps = (
                    _route_plan_steps(route_plan, max(args.exclude_leading_windows, 0))
                    if route_plan is not None
                    else _trace_steps(
                        artifact_dir / "mapping_inputs/runtime/window_stats.json",
                        args.decode_len,
                        exclude_leading_windows=max(args.exclude_leading_windows, 0),
                    )
                )
                analysis_compute_us = float(
                    profile.get("analysis_compute_us_per_step", 0.0)
                )
                astra_comm_us = (
                    result.communication_us / steps if args.run_astra else 0.0
                )
                astra_wall_us = (
                    result.total_latency_us / steps if args.run_astra else 0.0
                )
                # Paper-facing TP2 latency uses ASTRA-Sim for communication only
                # when expert compute has been removed from the trace. In full
                # traces, ASTRA's Comm counter includes dataflow stalls waiting for
                # peer compute readiness, so adding it to analysis compute would
                # double-count compute.
                paper_total_us = (
                    analysis_compute_us + astra_comm_us
                    if args.run_astra and trace_mode != "full"
                    else 0.0
                )
                dataflow_total_us = (
                    astra_wall_us if args.run_astra and trace_mode == "full" else 0.0
                )
                moentwine_components = {
                    "no_collective_total_us": None,
                    "conventional_tp_collective_us": None,
                    "full_her_collective_us": None,
                    "incremental_collective_us": None,
                    "full_her_total_us": None,
                    "conventional_tp_total_us": None,
                    "conventional_tp_output_root": "",
                    "full_her_output_root": "",
                    "conventional_tp_component": "",
                    "deployment_policy": "",
                    "deployment_selected_variant": "",
                    "deployment_selection_reason": "",
                    "deployment_selection_metric": "",
                    "deployment_full_her_total_us": None,
                    "deployment_no_her_total_us": None,
                }
                route_metadata = (
                    json.loads(route_plan.read_text(encoding="utf-8")).get(
                        "metadata", {}
                    )
                    if method == "moentwine" and route_plan is not None
                    else {}
                )
                collective_metadata = route_metadata.get("moentwine_collective", {})
                deployment_selection = collective_metadata.get(
                    "deployment_selection", {}
                )
                selected_deployment_variant = str(
                    deployment_selection.get("selected_variant", "full_her")
                )
                if method == "moentwine":
                    moentwine_components.update(
                        {
                            "deployment_policy": deployment_selection.get(
                                "policy", "full_her"
                            ),
                            "deployment_selected_variant": (
                                selected_deployment_variant
                            ),
                            "deployment_selection_reason": deployment_selection.get(
                                "selection_reason", "legacy_default_full_her"
                            ),
                            "deployment_selection_metric": deployment_selection.get(
                                "selection_metric", ""
                            ),
                            "deployment_full_her_total_us": deployment_selection.get(
                                "full_her_total_us"
                            ),
                            "deployment_no_her_total_us": deployment_selection.get(
                                "no_her_total_us"
                            ),
                        }
                    )
                if (
                    method == "moentwine"
                    and route_plan is not None
                    and args.run_astra
                    and trace_mode != "full"
                ):
                    conventional_component = collective_metadata.get(
                        "conventional_collective_component", "conventional_tp4"
                    )
                    if selected_deployment_variant == "full_her":
                        component_paths = collective_metadata.get(
                            "component_route_plans", {}
                        )
                        required_components = {conventional_component, "full_her"}
                        if not required_components.issubset(set(component_paths)):
                            raise ValueError(
                                "MoEntwine incremental accounting requires same-TP "
                                "conventional and full_her component route plans"
                            )
                        component_costs = {}
                        for component in sorted(required_components):
                            component_path = Path(component_paths[component])
                            if not component_path.exists():
                                raise FileNotFoundError(
                                    "MoEntwine component route plan not found: "
                                    f"{component_path}"
                                )
                            component_cfg = copy.deepcopy(cfg)
                            component_cfg.moe.route_plan_json = str(component_path)
                            component_cfg.supernode.output_root = str(
                                Path(args.output_root)
                                / topology_label
                                / seed_tag
                                / f"moentwine_{component}_{batch_tag}_collective_only"
                            )
                            component_result = simulate_layer(component_cfg)
                            component_steps = _route_plan_steps(
                                component_path, max(args.exclude_leading_windows, 0)
                            )
                            if component_steps != steps:
                                raise ValueError(
                                    f"MoEntwine {component} steps={component_steps}, expected {steps}"
                                )
                            component_costs[component] = (
                                component_result.communication_us / steps
                            )
                            output_root = str(component_result.metadata["output_root"])
                            if component == conventional_component:
                                moentwine_components[
                                    "conventional_tp_output_root"
                                ] = output_root
                            elif component == "full_her":
                                moentwine_components[
                                    "full_her_output_root"
                                ] = output_root
                        accounting = incremental_collective_accounting(
                            no_collective_total_us=paper_total_us,
                            conventional_tp_collective_us=component_costs[
                                conventional_component
                            ],
                            full_her_collective_us=component_costs["full_her"],
                        )
                        no_collective_total_us = paper_total_us
                        paper_total_us = accounting["charged_total_us"]
                        astra_comm_us += accounting["incremental_collective_us"]
                        moentwine_components.update(
                            {
                                "no_collective_total_us": no_collective_total_us,
                                "conventional_tp_collective_us": accounting[
                                    "conventional_tp_collective_us"
                                ],
                                "full_her_collective_us": accounting[
                                    "full_her_collective_us"
                                ],
                                "incremental_collective_us": accounting[
                                    "incremental_collective_us"
                                ],
                                "full_her_total_us": no_collective_total_us
                                + accounting["full_her_collective_us"],
                                "conventional_tp_total_us": no_collective_total_us
                                + accounting["conventional_tp_collective_us"],
                                "conventional_tp_component": conventional_component,
                            }
                        )
                    elif selected_deployment_variant == "no_her":
                        no_collective_total_us = paper_total_us
                        moentwine_components.update(
                            {
                                "no_collective_total_us": no_collective_total_us,
                                "conventional_tp_collective_us": 0.0,
                                "full_her_collective_us": 0.0,
                                "incremental_collective_us": 0.0,
                                "full_her_total_us": moentwine_components[
                                    "deployment_full_her_total_us"
                                ],
                                "conventional_tp_total_us": None,
                                "conventional_tp_component": conventional_component,
                            }
                        )
                    else:
                        raise ValueError(
                            "unknown MoEntwine deployment variant: "
                            f"{selected_deployment_variant}"
                        )
                row = {
                    "method": method,
                    "seed": seed if seed is not None else "",
                    "topology": topology_label,
                    "batch_size": bs,
                    "batch_scope": args.batch_scope,
                    "batch_per_die": batch_per_die,
                    "global_batch": global_batch,
                    "num_ranks": num_ranks,
                    "trace_mode": trace_mode,
                    "backend": result.backend,
                    "source": (
                        "supernode_astra" if args.run_astra else "supernode_tracegen"
                    ),
                    "astra_network_backend": args.astra_network_backend,
                    "total_us": paper_total_us,
                    "dataflow_total_us": dataflow_total_us,
                    "compute_us": analysis_compute_us,
                    "communication_us": astra_comm_us,
                    "astra_wall_us": astra_wall_us,
                    "astra_inferred_noncomm_us": (
                        result.compute_us / steps if args.run_astra else 0.0
                    ),
                    "astra_layer_total_us": result.total_latency_us,
                    "astra_layer_comm_us": result.communication_us,
                    "reported_steps": steps,
                    "excluded_leading_windows": max(args.exclude_leading_windows, 0),
                    "excluded_window_ids": ";".join(
                        str(x) for x in profile.get("excluded_window_ids", [])
                    ),
                    "analysis_compute_us_per_step": analysis_compute_us,
                    "route_plan_json": (
                        str(route_plan) if route_plan is not None else ""
                    ),
                    "route_plan_consumed": bool(
                        profile.get("route_plan_consumed", False)
                    ),
                    "total_dispatch_bytes": profile["total_dispatch_bytes"],
                    "total_pre_dispatch_collective_bytes": profile.get(
                        "total_pre_dispatch_collective_bytes", 0
                    ),
                    "total_combine_bytes": profile.get("total_combine_bytes", 0),
                    "cross_wafer_ratio": profile["cross_wafer_ratio"],
                    "avg_extra_replicas": profile.get("avg_extra_replicas", 0.0),
                    "replica_count_source": profile.get(
                        "replica_count_source", "window_mapping"
                    ),
                    "moentwine_accounting_formula": (
                        route_metadata.get("moentwine_collective", {}).get(
                            "formula",
                            "moe_no_collective + max(full_her - same_tp_conventional, 0)",
                        )
                        if method == "moentwine"
                        else ""
                    ),
                    "moentwine_conventional_tp_component": moentwine_components[
                        "conventional_tp_component"
                    ],
                    "moentwine_no_collective_total_us": moentwine_components[
                        "no_collective_total_us"
                    ],
                    "moentwine_conventional_tp_collective_us": moentwine_components[
                        "conventional_tp_collective_us"
                    ],
                    "moentwine_full_her_collective_us": moentwine_components[
                        "full_her_collective_us"
                    ],
                    "moentwine_incremental_collective_us": moentwine_components[
                        "incremental_collective_us"
                    ],
                    "moentwine_full_her_total_us": moentwine_components[
                        "full_her_total_us"
                    ],
                    "moentwine_conventional_tp_total_us": moentwine_components[
                        "conventional_tp_total_us"
                    ],
                    "moentwine_deployment_policy": moentwine_components[
                        "deployment_policy"
                    ],
                    "moentwine_deployment_selected_variant": moentwine_components[
                        "deployment_selected_variant"
                    ],
                    "moentwine_deployment_selection_reason": moentwine_components[
                        "deployment_selection_reason"
                    ],
                    "moentwine_deployment_selection_metric": moentwine_components[
                        "deployment_selection_metric"
                    ],
                    "moentwine_deployment_full_her_total_us": moentwine_components[
                        "deployment_full_her_total_us"
                    ],
                    "moentwine_deployment_no_her_total_us": moentwine_components[
                        "deployment_no_her_total_us"
                    ],
                    "moentwine_conventional_tp_output_root": moentwine_components[
                        "conventional_tp_output_root"
                    ],
                    "moentwine_full_her_output_root": moentwine_components[
                        "full_her_output_root"
                    ],
                    "output_root": result.metadata["output_root"],
                }
                rows.append(row)
                print(
                    f"{method} seed={seed} {args.batch_scope}_bs={bs} global_batch={global_batch}: backend={result.backend} "
                    f"total={row['total_us']:.3f}us astra_wall={astra_wall_us:.3f}us "
                    f"steps={steps} cross={profile['cross_wafer_ratio']:.3f}"
                )

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.seeds is not None:
        samples_out = out.with_name(f"{out.stem}_samples{out.suffix}")
        with samples_out.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        rows = _aggregate_seed_rows(rows)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out}")


def _aggregate_seed_rows(rows: list[dict]) -> list[dict]:
    metrics = [
        "total_us",
        "dataflow_total_us",
        "compute_us",
        "communication_us",
        "astra_wall_us",
        "astra_inferred_noncomm_us",
        "astra_layer_total_us",
        "astra_layer_comm_us",
        "analysis_compute_us_per_step",
        "total_dispatch_bytes",
        "total_pre_dispatch_collective_bytes",
        "total_combine_bytes",
        "cross_wafer_ratio",
        "avg_extra_replicas",
        "moentwine_no_collective_total_us",
        "moentwine_conventional_tp_collective_us",
        "moentwine_full_her_collective_us",
        "moentwine_incremental_collective_us",
        "moentwine_full_her_total_us",
        "moentwine_conventional_tp_total_us",
        "moentwine_deployment_full_her_total_us",
        "moentwine_deployment_no_her_total_us",
    ]
    grouped: dict[tuple[str, str, int], list[dict]] = {}
    for row in rows:
        key = (str(row["topology"]), str(row["method"]), int(row["batch_size"]))
        grouped.setdefault(key, []).append(row)

    out = []
    for _, samples in sorted(grouped.items()):
        row = {
            key: value
            for key, value in samples[0].items()
            if key not in metrics
            and key
            not in {
                "seed",
                "route_plan_json",
                "output_root",
                "moentwine_conventional_tp_output_root",
                "moentwine_full_her_output_root",
            }
        }
        row["seed"] = ""
        row["seed_count"] = len(samples)
        row["seeds"] = ";".join(
            str(sample["seed"])
            for sample in sorted(samples, key=lambda item: int(item["seed"]))
        )
        row["route_plan_json"] = "multiple; see samples CSV"
        row["output_root"] = "multiple; see samples CSV"
        for metric in metrics:
            values = [
                float(sample[metric])
                for sample in samples
                if sample.get(metric) not in {None, ""}
            ]
            row[metric] = statistics.fmean(values) if values else None
            row[f"{metric}_std"] = (
                statistics.stdev(values)
                if len(values) > 1
                else (0.0 if values else None)
            )
        out.append(row)
    return out


def _apply_network_shape(cfg, network_config: str) -> None:
    raw = yaml.safe_load(Path(network_config).read_text(encoding="utf-8")) or {}
    npus_count = raw.get("npus_count")
    if not npus_count or len(npus_count) < 2:
        return
    cfg.supernode.num_wafers = int(npus_count[0])
    cfg.supernode.dies_per_wafer = int(npus_count[1])


def _resolve_batch_scope(
    batch_size: int, batch_scope: str, num_ranks: int
) -> tuple[float, int]:
    if batch_scope == "global":
        return batch_size / max(num_ranks, 1), batch_size
    if batch_scope == "per_die":
        return float(batch_size), batch_size * num_ranks
    raise ValueError(f"Unsupported batch_scope: {batch_scope}")


def _artifact_num_requests(artifact_dir: Path) -> int:
    meta_path = artifact_dir / "mapping_inputs/runtime/metadata.json"
    if not meta_path.exists():
        return 64
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return 64
    return int(meta.get("num_requests") or meta.get("trace_num_requests") or 64)


def _dynamic_windows_path(
    artifact_dir: Path,
    batch_size: int,
    batch_scope: str,
    global_batch: int,
) -> Path:
    runtime_dir = artifact_dir / "runtime_windows"
    candidates = []
    if batch_scope == "per_die":
        candidates.append(runtime_dir / f"ours_dynamic_windows_bpd{batch_size}.json")
    else:
        candidates.append(runtime_dir / f"ours_dynamic_windows_bs{batch_size}.json")
        candidates.append(runtime_dir / f"ours_dynamic_windows_gb{global_batch}.json")
    candidates.append(runtime_dir / "ours_dynamic_windows.json")
    return next((p for p in candidates if p.exists()), candidates[-1])


def _trace_steps(
    window_stats_json: Path, fallback_decode_len: int, exclude_leading_windows: int = 0
) -> int:
    if not window_stats_json.exists():
        return max(fallback_decode_len, 1)
    try:
        raw = yaml.safe_load(window_stats_json.read_text(encoding="utf-8")) or []
    except Exception:
        return max(fallback_decode_len, 1)
    total = 0
    active_seen = 0
    for ws in raw:
        tokens = sum(int(v) for v in (ws.get("tokens_per_die") or {}).values())
        activations = sum(int(v) for v in (ws.get("expert_counts") or {}).values())
        if tokens <= 0 and activations <= 0:
            continue
        if active_seen < exclude_leading_windows:
            active_seen += 1
            continue
        active_seen += 1
        start = int(ws.get("start_token_idx", 1))
        end = int(ws.get("end_token_idx", start))
        total += max(end - start + 1, 1)
    return max(total, 1)


def _route_plan_steps(route_plan_json: Path, exclude_leading_steps: int = 0) -> int:
    raw = json.loads(route_plan_json.read_text(encoding="utf-8"))
    active = [step for step in raw.get("steps", []) if bool(step.get("active", True))]
    remaining = active[max(exclude_leading_steps, 0) :]
    if not remaining:
        raise ValueError(
            f"exclude_leading_windows={exclude_leading_steps} removed all route-plan steps"
        )
    return len(remaining)


if __name__ == "__main__":
    main()
