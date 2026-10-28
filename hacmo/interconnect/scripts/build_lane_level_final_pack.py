#!/usr/bin/env python3
"""Build the paper-facing TP3 lane-level final pack.

This script packages the basis-aligned lane probe into the standard TP3
four-bar format:

  1. Rail+Naive       = TP2 vLLM rail row from the conservative TP3 pack
  2. Rail+TP2         = TP2 ours_dynamic rail row from the conservative TP3 pack
  3. Crossbar+TP2     = edge-local crossbar with whole-port exit allocation
  4. Full+TP2         = lane-level dynamic allocation, default lane_exit

The lane probe remains copied as diagnostics so the final pack preserves the
policy comparison that motivated the paper-facing choice.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PROJECT_DIR
for p in PROJECT_DIR.parents:
    if (p / "hacmo" / "moe").exists() and (p / "supernode_sim").exists():
        REPO_DIR = p
        break

sys.path.insert(0, str(PROJECT_DIR))
from plot_tp3_paper import generate_all_figures  # noqa: E402


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def f(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    val = row.get(key, "")
    return float(val) if val not in ("", None) else default


def by_metric_key(rows: Iterable[Dict[str, str]]) -> Dict[Tuple[str, str, int], Dict[str, str]]:
    out = {}
    for row in rows:
        out[(row["topology"], row["bar_key"], int(float(row["batch_size"])))] = row
    return out


def by_lane_key(rows: Iterable[Dict[str, str]]) -> Dict[Tuple[str, str, int], Dict[str, str]]:
    out = {}
    for row in rows:
        out[(row["topology"], row["policy"], int(float(row["batch_size"])))] = row
    return out


def _rail_row(
    src: Dict[str, str],
    naive_total: float,
    allocation_policy: str,
    allocation_granularity: str,
    paper_facing_model: str,
) -> Dict[str, object]:
    total = f(src, "total_us")
    return {
        "topology": src["topology"],
        "batch_size": int(float(src["batch_size"])),
        "batch_scope": src.get("batch_scope", "per_die"),
        "batch_per_die": src.get("batch_per_die", src["batch_size"]),
        "global_batch": src.get("global_batch", ""),
        "bar_key": src["bar_key"],
        "bar_label": src.get("bar_label", ""),
        "total_us": f"{total:.6f}",
        "compute_us": f"{f(src, 'compute_us'):.6f}",
        "communication_us": f"{f(src, 'communication_us'):.6f}",
        "dispatch_us": f"{f(src, 'dispatch_us'):.6f}",
        "combine_us": f"{f(src, 'combine_us'):.6f}",
        "normalized_to_naive": f"{total / max(naive_total, 1e-9):.6f}",
        "speedup_vs_naive": f"{naive_total / max(total, 1e-9):.6f}",
        "crossbar_latency_ns": src.get("crossbar_latency_ns", ""),
        "beta": src.get("beta", ""),
        "rho": src.get("rho", ""),
        "allocation_policy": allocation_policy,
        "allocation_granularity": allocation_granularity,
        "paper_facing_model": paper_facing_model,
    }


def _lane_bar_row(
    lane: Dict[str, str],
    rail_tp2: Dict[str, str],
    bar_key: str,
    bar_label: str,
    naive_total: float,
    paper_facing_model: str,
) -> Dict[str, object]:
    analytical_comm = f(lane, "analytical_comm_us")
    calibrated_comm = f(lane, "calibrated_comm_us")
    dispatch = calibrated_comm * f(lane, "analytical_dispatch_us") / max(analytical_comm, 1e-9)
    combine = calibrated_comm * f(lane, "analytical_combine_us") / max(analytical_comm, 1e-9)
    compute = f(rail_tp2, "compute_us")
    total = compute + calibrated_comm
    return {
        "topology": lane["topology"],
        "batch_size": int(float(lane["batch_size"])),
        "batch_scope": rail_tp2.get("batch_scope", "per_die"),
        "batch_per_die": rail_tp2.get("batch_per_die", lane["batch_size"]),
        "global_batch": rail_tp2.get("global_batch", ""),
        "bar_key": bar_key,
        "bar_label": bar_label,
        "total_us": f"{total:.6f}",
        "compute_us": f"{compute:.6f}",
        "communication_us": f"{calibrated_comm:.6f}",
        "dispatch_us": f"{dispatch:.6f}",
        "combine_us": f"{combine:.6f}",
        "normalized_to_naive": f"{total / max(naive_total, 1e-9):.6f}",
        "speedup_vs_naive": f"{naive_total / max(total, 1e-9):.6f}",
        "crossbar_latency_ns": lane.get("crossbar_latency_ns", ""),
        "beta": lane.get("beta", ""),
        "rho": lane.get("rho", ""),
        "allocation_policy": lane["policy"],
        "allocation_granularity": lane["allocation_granularity"],
        "paper_facing_model": paper_facing_model,
    }


def _diagnostic_rows(lane_rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[str, int], Dict[str, str]] = {}
    for row in lane_rows:
        if row["policy"] == "lane_exit":
            grouped[(row["topology"], int(float(row["batch_size"])))] = row

    out: List[Dict[str, object]] = []
    for row in lane_rows:
        key = (row["topology"], int(float(row["batch_size"])))
        full = grouped.get(key)
        full_total = f(full, "total_us") if full else 0.0
        copied = dict(row)
        copied["delta_total_vs_lane_exit_us"] = (
            f"{f(row, 'total_us') - full_total:.6f}" if full else ""
        )
        out.append(copied)
    return out


def _provenance_rows(
    lane_rows: List[Dict[str, str]],
    whole_prov_rows: List[Dict[str, str]],
    whole_pack: Path,
    lane_probe_dir: Path,
    tp2_results_dir: str,
    crossbar_policy: str,
    full_policy: str,
) -> List[Dict[str, object]]:
    prov_by_topo = {
        (row["topology"], int(float(row["batch_size"]))): row
        for row in whole_prov_rows
    }
    lane_by_key = by_lane_key(lane_rows)
    topo_batches = sorted({(r["topology"], int(float(r["batch_size"]))) for r in lane_rows})
    policies = sorted({r["policy"] for r in lane_rows})
    out: List[Dict[str, object]] = []

    for topology, batch in topo_batches:
        base = prov_by_topo.get((topology, batch), {})
        row: Dict[str, object] = {
            "topology": topology,
            "batch_size": batch,
            "tp2_results_dir": tp2_results_dir,
            "whole_port_pack": str(whole_pack),
            "lane_probe_dir": str(lane_probe_dir),
            "crossbar_model": "ASTRA rail-calibrated analytical edge-local crossbar",
            "paper_facing_model": "lane-level dynamic crossbar allocation",
            "allocation_scope": "wafer_edge_local",
            "crossbar_policy": crossbar_policy,
            "full_policy": full_policy,
            "ratio_denominator_basis": "",
            "ratio_denominator_comm_us": "",
            "crossbar_internal_rail_analytical_comm_us": "",
            "tp2_traffic_matrix_rail_analytical_comm_us": "",
            "crossbar_latency_ns": "",
            "beta": "",
            "rho": "",
            "ports_per_edge": base.get("ports_per_edge", ""),
            "per_port_bw_gbps": base.get("per_port_bw_gbps", ""),
            "global_batch": base.get("global_batch", ""),
            "batch_per_die": base.get("batch_per_die", batch),
            "note": (
                "Full+TP2 uses continuous lane/credit-slice grants proportional "
                "to projected edge-local external exit demand. No cross-edge "
                "or cross-wafer bandwidth borrowing is modeled."
            ),
        }
        for policy in policies:
            cur = lane_by_key.get((topology, policy, batch))
            if not cur:
                continue
            if not row["ratio_denominator_basis"]:
                row["ratio_denominator_basis"] = cur.get("ratio_denominator_basis", "")
                row["ratio_denominator_comm_us"] = cur.get("ratio_denominator_comm_us", "")
                row["crossbar_internal_rail_analytical_comm_us"] = cur.get(
                    "crossbar_internal_rail_analytical_comm_us", ""
                )
                row["tp2_traffic_matrix_rail_analytical_comm_us"] = cur.get(
                    "tp2_traffic_matrix_rail_analytical_comm_us", ""
                )
                row["crossbar_latency_ns"] = cur.get("crossbar_latency_ns", "")
                row["beta"] = cur.get("beta", "")
                row["rho"] = cur.get("rho", "")
            prefix = policy
            row[f"{prefix}_analytical_comm_us"] = cur.get("analytical_comm_us", "")
            row[f"{prefix}_calibrated_comm_us"] = cur.get("calibrated_comm_us", "")
            row[f"{prefix}_total_us"] = cur.get("total_us", "")
            row[f"{prefix}_speedup_vs_naive_rail"] = cur.get("speedup_vs_naive_rail", "")
        out.append(row)

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build TP3 lane-level final pack.")
    parser.add_argument(
        "--whole-port-pack",
        default=str(PROJECT_DIR / "outputs" / "tp3_edge_local_basis_aligned_multibatch_20260614"),
    )
    parser.add_argument(
        "--lane-probe-dir",
        default=str(PROJECT_DIR / "outputs" / "tp3_lane_probe_basis_aligned_multibatch_20260614"),
    )
    parser.add_argument("--crossbar-policy", default="port_exit")
    parser.add_argument("--full-policy", default="lane_exit")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root).resolve() if args.output_root else (
        PROJECT_DIR / "outputs" / f"tp3_lane_level_final_candidate_{ts}"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    whole_pack = Path(args.whole_port_pack).resolve()
    lane_probe_dir = Path(args.lane_probe_dir).resolve()
    whole_metrics = read_csv(whole_pack / "tp3_metrics.csv")
    whole_prov = read_csv(whole_pack / "tp3_model_provenance.csv")
    lane_rows = read_csv(lane_probe_dir / "tp3_lane_probe_metrics.csv")
    whole_by_key = by_metric_key(whole_metrics)
    lane_by_key = by_lane_key(lane_rows)

    result_rows: List[Dict[str, object]] = []
    topo_batches = sorted({(r["topology"], int(float(r["batch_size"]))) for r in lane_rows})
    for topology, batch in topo_batches:
        naive = whole_by_key[(topology, "naive_rail", batch)]
        rail_tp2 = whole_by_key[(topology, "rail_tp2", batch)]
        naive_total = f(naive, "total_us")
        result_rows.append(_rail_row(naive, naive_total, "vllm_rail", "rail", "tp2_rail_baseline"))
        result_rows.append(_rail_row(rail_tp2, naive_total, "ours_dynamic_rail", "rail", "tp2_rail_baseline"))
        result_rows.append(_lane_bar_row(
            lane_by_key[(topology, args.crossbar_policy, batch)],
            rail_tp2,
            "crossbar_tp2",
            "Crossbar+TP2",
            naive_total,
            "edge_local_crossbar_whole_port_baseline",
        ))
        result_rows.append(_lane_bar_row(
            lane_by_key[(topology, args.full_policy, batch)],
            rail_tp2,
            "full_tp2",
            "Full+TP2",
            naive_total,
            "lane_level_dynamic_crossbar_allocation",
        ))

    metric_fields = [
        "topology", "batch_size", "batch_scope", "batch_per_die", "global_batch",
        "bar_key", "bar_label",
        "total_us", "compute_us", "communication_us", "dispatch_us", "combine_us",
        "normalized_to_naive", "speedup_vs_naive",
        "crossbar_latency_ns", "beta", "rho",
        "allocation_policy", "allocation_granularity", "paper_facing_model",
    ]
    write_csv(output_root / "tp3_metrics.csv", result_rows, metric_fields)

    diag_rows = _diagnostic_rows(lane_rows)
    diag_fields = list(diag_rows[0].keys()) if diag_rows else []
    if diag_rows:
        write_csv(output_root / "tp3_lane_policy_diagnostics.csv", diag_rows, diag_fields)

    tp2_results_dir = ""
    if whole_prov:
        tp2_results_dir = whole_prov[0].get("tp2_metrics_dir", "")
    prov_rows = _provenance_rows(
        lane_rows,
        whole_prov,
        whole_pack,
        lane_probe_dir,
        tp2_results_dir,
        args.crossbar_policy,
        args.full_policy,
    )
    prov_fields = sorted({key for row in prov_rows for key in row.keys()})
    preferred = [
        "topology", "batch_size", "tp2_results_dir", "whole_port_pack", "lane_probe_dir",
        "crossbar_model", "paper_facing_model", "allocation_scope",
        "crossbar_policy", "full_policy", "ratio_denominator_basis",
        "ratio_denominator_comm_us", "crossbar_internal_rail_analytical_comm_us",
        "tp2_traffic_matrix_rail_analytical_comm_us", "crossbar_latency_ns",
        "beta", "rho", "ports_per_edge", "per_port_bw_gbps",
        "batch_per_die", "global_batch", "note",
    ]
    prov_fields = preferred + [key for key in prov_fields if key not in preferred]
    write_csv(output_root / "tp3_model_provenance.csv", prov_rows, prov_fields)

    diagnostics_dir = output_root / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    for src in [
        lane_probe_dir / "tp3_lane_probe_metrics.csv",
        lane_probe_dir / "tp3_lane_probe_metadata.json",
        lane_probe_dir / "tp3_lane_probe_debug.json",
        whole_pack / "tp3_metrics.csv",
        whole_pack / "tp3_model_provenance.csv",
        whole_pack / "tp3_crossbar_latency_sensitivity.csv",
    ]:
        if src.exists():
            shutil.copy2(src, diagnostics_dir / src.name)

    metadata = {
        "generated_at": dt.datetime.now().isoformat(),
        "whole_port_pack": str(whole_pack),
        "lane_probe_dir": str(lane_probe_dir),
        "output_root": str(output_root),
        "crossbar_policy": args.crossbar_policy,
        "full_policy": args.full_policy,
        "model": "ASTRA rail-calibrated analytical edge-local crossbar with lane-level dynamic allocation",
        "bar_semantics": {
            "naive_rail": "TP2 vllm rail baseline",
            "rail_tp2": "TP2 ours_dynamic rail baseline",
            "crossbar_tp2": "TP2 ours_dynamic traffic with edge-local whole-port exit allocation",
            "full_tp2": "TP2 ours_dynamic traffic with lane-level dynamic allocation",
        },
        "paper_facing_warning": (
            "Use this pack only with manuscript text that explicitly defines "
            "lane-level or credit-slice dynamic allocation inside each edge-local crossbar."
        ),
    }
    (output_root / "tp3_run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if not args.skip_figures:
        generate_all_figures(output_root / "tp3_metrics.csv", output_root / "paper_figures")

    print(f"[DONE] TP3 lane-level final pack in: {output_root}", flush=True)


if __name__ == "__main__":
    main()
