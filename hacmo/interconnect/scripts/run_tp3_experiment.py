#!/usr/bin/env python3
"""TP3 experiment runner: Crossbar + Dynamic Bandwidth Allocation ablation.

Pipeline:
  Bar 1 (naive_rail):     Rail-only + naive mapping  = TP2's vllm astra result
  Bar 2 (rail_tp2):       Rail-only + TP2 mapping     = TP2's ours_dynamic astra result
  Bar 3 (crossbar_tp2):   Crossbar + TP2 mapping      = crossbar edge model (naive port alloc)
  Bar 4 (full_tp2):       Crossbar + Dynamic BW + TP2 = crossbar edge model (dynamic port alloc)

Bars 1-2 reuse TP2's astra-sim results directly. Bars 3-4 compute crossbar
communication latency using supernode_sim.plugins.crossbar, with the same
compute from TP2's H200 roofline model.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_DIR = Path(__file__).resolve().parents[1]
# PROJECT_DIR = HACMO/hacmo/interconnect -> HACMO = parents[2]
# But if __file__ resolves deeper, find HACMO by walking up
REPO_DIR = PROJECT_DIR
for p in PROJECT_DIR.parents:
    if (p / "hacmo" / "moe").exists() and (p / "supernode_sim").exists():
        REPO_DIR = p
        break
TP2_DIR = REPO_DIR / "hacmo" / "moe"

sys.path.insert(0, str(REPO_DIR))

from supernode_sim.plugins.crossbar import (
    crossbar_comm_us_per_step,
    load_crossbar_config,
)
from supernode_sim.tracegen.moe import (
    build_dispatch_matrices,
    build_combine_matrices,
)
from supernode_sim.types import (
    MoEConfig,
    SimulationConfig,
    SupernodeConfig,
    WorkloadConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def topology_label_from_stem(stem: str) -> str:
    """hierarchical_4w_8x8_mesh_rail_900 -> 4w_8x8_rail_900."""
    return stem.replace("hierarchical_", "").replace("_mesh_", "_")


def rail_yaml_for_crossbar(crossbar_yaml: Path) -> Path:
    """Find the corresponding rail-only network YAML."""
    name = crossbar_yaml.stem.replace("_crossbar_", "_rail_")
    # Same directory
    same_dir = crossbar_yaml.parent / f"{name}.yaml"
    if same_dir.exists():
        return same_dir
    # moe configs
    tp2_path = TP2_DIR / "configs" / "network" / f"{name}.yaml"
    if tp2_path.exists():
        return tp2_path
    raise FileNotFoundError(f"No rail config for {crossbar_yaml}")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def normalize_mapping(raw: Dict) -> Dict[int, List[int]]:
    data = raw.get("expert_to_dies", raw) if isinstance(raw, dict) else {}
    return {int(k): [int(x) for x in v] for k, v in data.items()}


def load_tp2_metrics(tp2_dir: Path) -> Dict[Tuple[str, str, int], Dict]:
    tp2_metrics: Dict[Tuple[str, str, int], Dict] = {}
    candidates = [
        tp2_dir / "figures" / "paper_figure_summary.csv",
        tp2_dir / "paper_figures" / "paper_figure_summary.csv",
        tp2_dir / "csv" / "tp2_serving_epoch_astra_comm_only_summary.csv",
    ]
    for csv_path in candidates:
        if not csv_path.exists():
            continue
        for row in read_csv(csv_path):
            method = row.get("method") or row.get("bar_key") or ""
            if not method:
                continue
            batch_raw = row.get("batch_size") or row.get("batch_per_die") or row.get("batch") or "0"
            key = (row["topology"], method, int(float(batch_raw)))
            tp2_metrics[key] = row
        if tp2_metrics:
            return tp2_metrics

    for csv_path in tp2_dir.rglob("metrics_*.csv"):
        for row in read_csv(csv_path):
            batch_raw = row.get("batch_size") or row.get("batch_per_die") or "0"
            key = (row.get("topology", ""), row.get("method", ""), int(float(batch_raw)))
            tp2_metrics[key] = row
    return tp2_metrics


def row_float(row: Dict, *names: str, default: float = 0.0) -> float:
    for name in names:
        if name in row and row[name] not in ("", None):
            return float(row[name])
    return default


def runtime_windows_to_mappings(runtime_windows: List[Dict]) -> List[Dict[int, List[int]]]:
    return [normalize_mapping(w.get("mapping", {})) for w in runtime_windows]


def traffic_matrix_path(topo_art: Path, method: str, batch_size: int) -> Path:
    candidates = [
        topo_art / "traffic_matrices" / f"{method}_bpd{batch_size}.json",
        topo_art / "traffic_matrices" / f"{method}_bs{batch_size}.json",
        topo_art / "traffic_matrices" / f"{method}.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def runtime_windows_path(topo_art: Path, batch_size: int) -> Path:
    candidates = [
        topo_art / "runtime_windows" / f"ours_dynamic_windows_bpd{batch_size}.json",
        topo_art / "runtime_windows" / "ours_dynamic_windows.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def validate_tp2_handoff(
    tp2_dir: Path,
    *,
    topology_labels: List[str],
    batch_sizes: List[int],
) -> None:
    """Require every compact TP2 artifact consumed by the TP3 model."""
    missing: List[Path] = []
    for topology_label in topology_labels:
        topology_dir = tp2_dir / topology_label
        required = [
            topology_dir / "mappings" / "ours_static.json",
            topology_dir
            / "mapping_inputs"
            / "runtime"
            / "window_stats.json",
            topology_dir
            / "mapping_inputs"
            / "runtime"
            / "window_src_expert_counts.json",
        ]
        for batch_size in batch_sizes:
            required.extend(
                [
                    topology_dir
                    / "runtime_windows"
                    / f"ours_dynamic_windows_bpd{batch_size}.json",
                    topology_dir
                    / "traffic_matrices"
                    / f"ours_dynamic_bpd{batch_size}.json",
                ]
            )
        missing.extend(path for path in required if not path.is_file())

    if missing:
        relative = [
            str(path.relative_to(tp2_dir)) if path.is_relative_to(tp2_dir) else str(path)
            for path in missing
        ]
        raise FileNotFoundError(
            "Incomplete TP2-to-TP3 handoff; missing required files:\n- "
            + "\n- ".join(relative)
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Run TP3 crossbar + dynamic BW ablation.")
    p.add_argument("--tp2-results-dir",
                   default=str(TP2_DIR / "outputs" / "tp2_serving_epoch_paper_pack_20260613"))
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
            "Analytical rail denominator used to calibrate crossbar bars. "
            "'crossbar_internal_rail' keeps numerator and denominator in the "
            "same crossbar-plugin timing basis; 'tp2_traffic_matrix' reproduces "
            "the legacy denominator from TP2 traffic-matrix JSON."
        ),
    )
    p.add_argument("--sensitivity-crossbar-latency-ns", type=float, nargs="+", default=[0.0, 20.0, 50.0])
    p.add_argument("--no-sensitivity", action="store_true",
                   help="Skip crossbar latency sensitivity rows for faster audit dry runs.")
    p.add_argument("--skip-figures", action="store_true")
    p.add_argument("--output-root", default=None)
    args = p.parse_args()

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(
        args.output_root
    ).resolve() if args.output_root else PROJECT_DIR / "outputs" / f"tp3_paper_results_{ts}"
    output_root.mkdir(parents=True, exist_ok=True)

    tp2_dir = Path(args.tp2_results_dir).resolve()
    print(f"[TP3] Loading TP2 results from: {tp2_dir}", flush=True)

    # Load TP2 paper figure summary (current serving-epoch final candidate uses figures/).
    tp2_metrics = load_tp2_metrics(tp2_dir)

    if not tp2_metrics:
        print("[ERROR] No TP2 metrics found!")
        sys.exit(1)

    topology_labels = [
        topology_label_from_stem(
            rail_yaml_for_crossbar(Path(network_yaml).resolve()).stem
        )
        for network_yaml in args.network_yamls
    ]
    validate_tp2_handoff(
        tp2_dir,
        topology_labels=topology_labels,
        batch_sizes=args.batch_sizes,
    )

    results: List[Dict] = []
    sensitivity_rows: List[Dict] = []
    provenance_rows: List[Dict] = []
    debug_payload: Dict[str, Dict] = {}

    for crossbar_yaml_str in args.network_yamls:
        crossbar_yaml = Path(crossbar_yaml_str).resolve()
        cfg = load_crossbar_config(str(crossbar_yaml))
        cfg["crossbar_latency_ns"] = args.crossbar_latency_ns
        rail_yaml = rail_yaml_for_crossbar(crossbar_yaml)
        topo_label = topology_label_from_stem(rail_yaml.stem)

        num_wafers = cfg["num_wafers"]
        dpw = cfg["dies_per_wafer"]
        num_dies = num_wafers * dpw

        print(f"\n=== Topology: {topo_label} ({num_wafers}w {dpw}-die) ===", flush=True)

        topo_art = tp2_dir / topo_label
        if not topo_art.exists():
            print(f"  [WARN] Missing TP2 topology artifact dir: {topo_art}")
            continue

        mapping_ours = load_json(topo_art / "mappings" / "ours_static.json")

        # Use runtime mapping inputs (represent actual decode traffic)
        mapping_input_dir = topo_art / "mapping_inputs" / "runtime"
        if not mapping_input_dir.exists():
            mapping_input_dir = topo_art / "mapping_inputs" / "offline"
        if not mapping_input_dir.exists():
            mapping_input_dir = topo_art / "mapping_inputs"

        window_src = load_json(mapping_input_dir / "window_src_expert_counts.json")

        for bs in args.batch_sizes:
            print(f"  BS={bs}...", flush=True)

            # ---- Bar 1: naive_rail (= TP2 vllm) ----
            bar1 = tp2_metrics.get((topo_label, "vllm", bs))
            if not bar1:
                print(f"    [WARN] No TP2 vllm for {topo_label} BS={bs}, skipping.")
                continue
            bar1_total = row_float(bar1, "total_us")
            bar1_compute = row_float(bar1, "compute_us")
            bar1_comm = row_float(bar1, "communication_us", "comm_us")

            # ---- Bar 2: rail_tp2 (= TP2 ours_dynamic) ----
            bar2 = tp2_metrics.get((topo_label, "ours_dynamic", bs))
            if not bar2:
                print(f"    [WARN] No TP2 ours_dynamic for {topo_label} BS={bs}, skipping.")
                continue
            bar2_total = row_float(bar2, "total_us")
            bar2_compute = row_float(bar2, "compute_us")
            bar2_comm = row_float(bar2, "communication_us", "comm_us")

            # ---- Bars 3-4: crossbar models ----
            # Approach: compute a crossbar/rail communication ratio analytically,
            # then apply that ratio to the ASTRA-Sim rail communication. The
            # numerator and denominator should be in the same analytical timing
            # basis; the TP2 traffic-matrix denominator is retained only as an
            # explicit legacy comparison mode.
            crossbar_disp_us = 0.0
            crossbar_comb_us = 0.0
            dynamic_disp_us = 0.0
            dynamic_comb_us = 0.0

            # Load TP2 traffic matrix for rail analytical baseline
            tp2_tm_path = traffic_matrix_path(topo_art, "ours_dynamic", bs)
            tp2_tm = load_json(tp2_tm_path)
            tp2_rail_analytical_disp = 0.0
            tp2_rail_analytical_comb = 0.0
            if tp2_tm:
                tp2_rail_analytical_disp = tp2_tm["summary"]["avg_step_dispatch_us"]
                tp2_rail_analytical_comb = tp2_tm["summary"]["avg_step_combine_us"]
            tp2_rail_analytical_comm = tp2_rail_analytical_disp + tp2_rail_analytical_comb

            runtime_path = runtime_windows_path(topo_art, bs)
            runtime_windows = load_json(runtime_path)

            if mapping_ours and runtime_windows and window_src and tp2_rail_analytical_comm > 0:
                tm_params = tp2_tm.get("params", {}) if isinstance(tp2_tm, dict) else {}
                effective_window_size = int(tm_params.get("window_size", args.window_size))
                effective_hidden_size = int(tm_params.get("hidden_size", args.hidden_size))
                effective_dispatch_bpe = int(tm_params.get(
                    "dispatch_bytes_per_hidden_elem",
                    args.dispatch_bytes_per_hidden_elem,
                ))
                effective_combine_bpe = int(tm_params.get(
                    "combine_bytes_per_hidden_elem",
                    args.combine_bytes_per_hidden_elem,
                ))
                effective_batch_scale = float(tm_params.get("batch_scale", 0.0))
                if effective_batch_scale > 0:
                    effective_num_requests = max(1, int(round(bs / effective_batch_scale)))
                else:
                    effective_num_requests = args.num_requests
                    effective_batch_scale = bs / max(effective_num_requests, 1)
                expert_mapping = normalize_mapping(mapping_ours)
                window_mappings = runtime_windows_to_mappings(runtime_windows)
                if len(window_mappings) != len(window_src):
                    print(f"    [WARN] Runtime windows length {len(window_mappings)} != window_src length {len(window_src)}", flush=True)

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
                        num_wafers=num_wafers,
                        dies_per_wafer=dpw,
                        network_config=str(rail_yaml),
                    ),
                )

                dispatch_mats = build_dispatch_matrices(
                    sim_cfg,
                    window_src,
                    expert_mapping,
                    window_mappings=window_mappings,
                )
                combine_mats = build_combine_matrices(sim_cfg, dispatch_mats)

                # Crossbar analytical comm
                naive_comm = crossbar_comm_us_per_step(
                    dispatch_mats, combine_mats, cfg, "naive_crossbar",
                    effective_batch_scale, effective_window_size, beta=0.0, rho=0.0,
                    crossbar_latency_ns=args.crossbar_latency_ns,
                    collect_debug=True,
                )
                dynamic_comm = crossbar_comm_us_per_step(
                    dispatch_mats, combine_mats, cfg, "dynamic_bw",
                    effective_batch_scale, effective_window_size, beta=args.beta, rho=args.rho,
                    crossbar_latency_ns=args.crossbar_latency_ns,
                    collect_debug=True,
                )

                crossbar_internal_rail_analytical_comm = float(naive_comm["rail_analytical_comm_us"])
                if args.ratio_denominator == "tp2_traffic_matrix":
                    ratio_denominator_comm = tp2_rail_analytical_comm
                    ratio_denominator_basis = "tp2_traffic_matrix"
                else:
                    ratio_denominator_comm = crossbar_internal_rail_analytical_comm
                    ratio_denominator_basis = "crossbar_internal_rail"

                naive_ratio = naive_comm["comm_us"] / max(ratio_denominator_comm, 1e-9)
                dynamic_ratio = dynamic_comm["comm_us"] / max(ratio_denominator_comm, 1e-9)

                # Apply ratio to astra-sim rail comm (bar2_comm)
                # Crossbar should never be slower than rail — cap at rail comm
                crossbar_comm_total = min(bar2_comm * naive_ratio, bar2_comm)
                dynamic_comm_total = min(bar2_comm * dynamic_ratio, bar2_comm)

                # Split dispatch/combine proportionally
                if naive_comm["comm_us"] > 0:
                    crossbar_disp_us = crossbar_comm_total * naive_comm["dispatch_us"] / naive_comm["comm_us"]
                    crossbar_comb_us = crossbar_comm_total * naive_comm["combine_us"] / naive_comm["comm_us"]
                if dynamic_comm["comm_us"] > 0:
                    dynamic_disp_us = dynamic_comm_total * dynamic_comm["dispatch_us"] / dynamic_comm["comm_us"]
                    dynamic_comb_us = dynamic_comm_total * dynamic_comm["combine_us"] / dynamic_comm["comm_us"]

                print(f"    TP2 traffic-matrix rail analytical: {tp2_rail_analytical_comm:.3f}us; "
                      f"crossbar internal rail analytical: {crossbar_internal_rail_analytical_comm:.3f}us", flush=True)
                print(f"    Crossbar analytical: {naive_comm['comm_us']:.3f}us (naive) / "
                      f"{dynamic_comm['comm_us']:.3f}us (dynamic)", flush=True)
                print(f"    Ratio denominator: {ratio_denominator_basis}={ratio_denominator_comm:.3f}us; "
                      f"naive ratio: {naive_ratio:.3f}, dynamic ratio: {dynamic_ratio:.3f}", flush=True)
                print(f"    Astra-sim calibrated: naive={crossbar_comm_total:.3f}us, "
                      f"dynamic={dynamic_comm_total:.3f}us", flush=True)

                prov = {
                    "topology": topo_label,
                    "batch_size": bs,
                    "batch_scope": bar2.get("batch_scope", "per_die"),
                    "batch_per_die": bar2.get("batch_per_die", bs),
                    "global_batch": bar2.get("global_batch", ""),
                    "tp2_metrics_dir": str(tp2_dir),
                    "tp2_traffic_matrix": str(tp2_tm_path),
                    "runtime_windows": str(runtime_path),
                    "mapping_inputs": str(mapping_input_dir),
                    "network_yaml": str(crossbar_yaml),
                    "rail_yaml": str(rail_yaml),
                    "crossbar_model": "ASTRA rail-calibrated analytical edge-local crossbar",
                    "allocation_scope": "wafer_edge_local",
                    "ports_per_edge": cfg["external_ports_per_edge"],
                    "per_port_bw_gbps": cfg["per_port_bw_gbps"],
                    "crossbar_latency_ns": args.crossbar_latency_ns,
                    "beta": args.beta,
                    "rho": args.rho,
                    "tp2_effective_num_requests": effective_num_requests,
                    "tp2_effective_window_size": effective_window_size,
                    "tp2_effective_batch_scale": effective_batch_scale,
                    "tp2_rail_astra_comm_us": bar2_comm,
                    "tp2_rail_analytical_comm_us": tp2_rail_analytical_comm,
                    "tp2_traffic_matrix_rail_analytical_comm_us": tp2_rail_analytical_comm,
                    "crossbar_internal_rail_analytical_comm_us": crossbar_internal_rail_analytical_comm,
                    "ratio_denominator_basis": ratio_denominator_basis,
                    "ratio_denominator_comm_us": ratio_denominator_comm,
                    "naive_crossbar_analytical_comm_us": naive_comm["comm_us"],
                    "dynamic_crossbar_analytical_comm_us": dynamic_comm["comm_us"],
                    "naive_ratio": naive_ratio,
                    "dynamic_ratio": dynamic_ratio,
                    "naive_calibrated_comm_us": crossbar_comm_total,
                    "dynamic_calibrated_comm_us": dynamic_comm_total,
                }
                provenance_rows.append(prov)
                debug_payload[f"{topo_label}_bpd{bs}_naive"] = {
                    k: v for k, v in naive_comm.items() if k != "debug_windows"
                } | {"debug_windows": naive_comm.get("debug_windows", [])[:3]}
                debug_payload[f"{topo_label}_bpd{bs}_dynamic"] = {
                    k: v for k, v in dynamic_comm.items() if k != "debug_windows"
                } | {"debug_windows": dynamic_comm.get("debug_windows", [])[:3]}

                for sens_lat in ([] if args.no_sensitivity else args.sensitivity_crossbar_latency_ns):
                    sens_naive = crossbar_comm_us_per_step(
                        dispatch_mats, combine_mats, cfg, "naive_crossbar",
                        effective_batch_scale, effective_window_size, beta=0.0, rho=0.0,
                        crossbar_latency_ns=sens_lat,
                    )
                    sens_dynamic = crossbar_comm_us_per_step(
                        dispatch_mats, combine_mats, cfg, "dynamic_bw",
                        effective_batch_scale, effective_window_size, beta=args.beta, rho=args.rho,
                        crossbar_latency_ns=sens_lat,
                    )
                    for method_label, comm_obj in [
                        ("crossbar_tp2", sens_naive),
                        ("full_tp2", sens_dynamic),
                    ]:
                        if args.ratio_denominator == "tp2_traffic_matrix":
                            sens_denominator = tp2_rail_analytical_comm
                        else:
                            sens_denominator = float(sens_naive["rail_analytical_comm_us"])
                        ratio = comm_obj["comm_us"] / max(sens_denominator, 1e-9)
                        calibrated_comm = min(bar2_comm * ratio, bar2_comm)
                        sensitivity_rows.append({
                            "topology": topo_label,
                            "batch_size": bs,
                            "bar_key": method_label,
                            "crossbar_latency_ns": f"{sens_lat:.6f}",
                            "analytical_comm_us": f"{comm_obj['comm_us']:.6f}",
                            "ratio_denominator_basis": args.ratio_denominator,
                            "ratio_denominator_comm_us": f"{sens_denominator:.6f}",
                            "ratio_to_denominator": f"{ratio:.6f}",
                            "calibrated_comm_us": f"{calibrated_comm:.6f}",
                            "total_us": f"{bar2_compute + calibrated_comm:.6f}",
                        })
            else:
                raise RuntimeError(
                    "Invalid TP2 handoff contents for "
                    f"{topo_label} bpd{bs}: mapping, runtime windows, window "
                    "source counts, and positive analytical traffic are required"
                )

            bar3_comm = crossbar_disp_us + crossbar_comb_us
            bar3_total = bar2_compute + bar3_comm

            bar4_comm = dynamic_disp_us + dynamic_comb_us
            bar4_total = bar2_compute + bar4_comm

            for bar_key, bar_label, total, compute, comm, disp, comb in [
                ("naive_rail", "Rail+Naive",
                 bar1_total, bar1_compute, bar1_comm, 0.0, 0.0),
                ("rail_tp2", "Rail+TP2",
                 bar2_total, bar2_compute, bar2_comm, 0.0, 0.0),
                ("crossbar_tp2", "Crossbar+TP2",
                 bar3_total, bar2_compute, bar3_comm,
                 crossbar_disp_us, crossbar_comb_us),
                ("full_tp2", "Full+TP2",
                 bar4_total, bar2_compute, bar4_comm,
                 dynamic_disp_us, dynamic_comb_us),
            ]:
                results.append({
                    "topology": topo_label,
                    "batch_size": bs,
                    "batch_scope": bar2.get("batch_scope", bar1.get("batch_scope", "per_die")),
                    "batch_per_die": bar2.get("batch_per_die", bs),
                    "global_batch": bar2.get("global_batch", ""),
                    "bar_key": bar_key,
                    "bar_label": bar_label,
                    "total_us": f"{total:.6f}",
                    "compute_us": f"{compute:.6f}",
                    "communication_us": f"{comm:.6f}",
                    "dispatch_us": f"{disp:.6f}",
                    "combine_us": f"{comb:.6f}",
                    "normalized_to_naive": f"{total / max(bar1_total, 1e-9):.6f}",
                    "speedup_vs_naive": f"{bar1_total / max(total, 1e-9):.6f}",
                    "crossbar_latency_ns": f"{args.crossbar_latency_ns:.6f}",
                    "beta": f"{args.beta:.6f}",
                    "rho": f"{args.rho:.6f}",
                })

    # Write results
    fieldnames = [
        "topology", "batch_size", "batch_scope", "batch_per_die", "global_batch",
        "bar_key", "bar_label",
        "total_us", "compute_us", "communication_us",
        "dispatch_us", "combine_us",
        "normalized_to_naive", "speedup_vs_naive",
        "crossbar_latency_ns", "beta", "rho",
    ]
    write_csv(output_root / "tp3_metrics.csv", results, fieldnames)

    if sensitivity_rows:
        write_csv(
            output_root / "tp3_crossbar_latency_sensitivity.csv",
            sensitivity_rows,
            [
                "topology", "batch_size", "bar_key", "crossbar_latency_ns",
                "analytical_comm_us", "ratio_denominator_basis",
                "ratio_denominator_comm_us", "ratio_to_denominator",
                "calibrated_comm_us", "total_us",
            ],
        )

    if provenance_rows:
        write_csv(
            output_root / "tp3_model_provenance.csv",
            provenance_rows,
            [
                "topology", "batch_size", "batch_scope", "batch_per_die", "global_batch",
                "tp2_metrics_dir", "tp2_traffic_matrix", "runtime_windows", "mapping_inputs",
                "network_yaml", "rail_yaml", "crossbar_model", "allocation_scope",
                "ports_per_edge", "per_port_bw_gbps", "crossbar_latency_ns", "beta", "rho",
                "tp2_effective_num_requests", "tp2_effective_window_size", "tp2_effective_batch_scale",
                "tp2_rail_astra_comm_us", "tp2_rail_analytical_comm_us",
                "tp2_traffic_matrix_rail_analytical_comm_us",
                "crossbar_internal_rail_analytical_comm_us",
                "ratio_denominator_basis", "ratio_denominator_comm_us",
                "naive_crossbar_analytical_comm_us", "dynamic_crossbar_analytical_comm_us",
                "naive_ratio", "dynamic_ratio", "naive_calibrated_comm_us",
                "dynamic_calibrated_comm_us",
            ],
        )
        (output_root / "tp3_debug_allocation_sample.json").write_text(
            json.dumps(debug_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    metadata = {
        "generated_at": dt.datetime.now().isoformat(),
        "tp2_results_dir": str(tp2_dir),
        "batch_sizes": args.batch_sizes,
        "main_crossbar_latency_ns": args.crossbar_latency_ns,
        "sensitivity_crossbar_latency_ns": args.sensitivity_crossbar_latency_ns,
        "ratio_denominator": args.ratio_denominator,
        "allocation_scope": "edge-local per wafer edge",
        "crossbar_model": "ASTRA rail-calibrated analytical edge-local crossbar",
        "bars": {
            "naive_rail": "TP2 vllm rail baseline",
            "rail_tp2": "TP2 ours_dynamic rail baseline",
            "crossbar_tp2": "ours_dynamic traffic with exit-proportional edge-local crossbar allocation",
            "full_tp2": "ours_dynamic traffic with full dynamic edge-local crossbar allocation",
        },
    }
    (output_root / "tp3_run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if not args.skip_figures:
        sys.path.insert(0, str(PROJECT_DIR))
        from plot_tp3_paper import generate_all_figures
        generate_all_figures(output_root / "tp3_metrics.csv", output_root / "paper_figures")

    print(f"\n[DONE] TP3 results in: {output_root}", flush=True)


if __name__ == "__main__":
    main()
