#!/usr/bin/env python3
"""Validate a HACMO DSE output pack."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

DSE_DIR = Path(__file__).resolve().parent


def main() -> None:
    args = parse_args()
    out = Path(args.output_root) if args.output_root else latest_output()
    errors: list[str] = []
    rows = read_csv(out / "raw_results.csv")
    invalid = read_csv(out / "invalid_cases.csv")
    pareto = read_csv(out / "summary_pareto.csv")
    pack_kind = args.pack_kind
    if pack_kind == "auto":
        pack_kind = detect_pack_kind(rows)

    required = common_required_files()
    if pack_kind == "main":
        required.extend(main_required_files())
    elif pack_kind == "edge_sensitivity":
        required.extend(edge_sensitivity_required_files())
    else:
        errors.append(f"unknown pack kind: {pack_kind}")

    for rel in required:
        if not (out / rel).exists():
            errors.append(f"missing {rel}")

    validate_rows(rows, invalid, pareto, pack_kind, errors)
    validate_metadata(out, rows, invalid, errors)

    best_t = max(rows, key=lambda r: float(r["throughput_tps"])) if rows else {}
    summary = {
        "output_root": str(out.resolve()),
        "pack_kind": pack_kind,
        "status": "ok" if not errors else "error",
        "errors": errors,
        "valid_cases": len(rows),
        "invalid_cases": len(invalid),
        "pareto_cases": sorted({r["case_id"] for r in pareto}),
        "best_throughput_case": best_t.get("case_id", ""),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


def common_required_files() -> list[str]:
    return [
        "raw_results.csv",
        "summary_pareto.csv",
        "invalid_cases.csv",
        "dse_config_dump.json",
        "dse_analysis.md",
        "plot_data/plot_data_pareto_summary.csv",
    ]


def main_required_files() -> list[str]:
    return [
        "figures/fig_dse_sram_compute.png",
        "figures/fig_dse_sram_compute.pdf",
        "figures/fig_dse_core_hbm.png",
        "figures/fig_dse_core_hbm.pdf",
        "figures/fig_dse_wafer_scale.png",
        "figures/fig_dse_wafer_scale.pdf",
        "figures/fig_dse_core_hbm_frontier_bars.png",
        "figures/fig_dse_core_hbm_frontier_bars.pdf",
        "plot_data/plot_data_sram_compute.csv",
        "plot_data/plot_data_core_hbm.csv",
        "plot_data/plot_data_wafer_scale.csv",
        "plot_data/plot_data_core_hbm_frontier_bars.csv",
    ]


def edge_sensitivity_required_files() -> list[str]:
    return [
        "figures/fig_dse_edge_sensitivity.png",
        "figures/fig_dse_edge_sensitivity.pdf",
        "figures/fig_dse_edge_sensitivity_crossbar.png",
        "figures/fig_dse_edge_sensitivity_crossbar.pdf",
        "figures/fig_dse_edge_sensitivity_bandwidth_slices.png",
        "figures/fig_dse_edge_sensitivity_bandwidth_slices.pdf",
        "figures/fig_dse_edge_sensitivity_latency_slices.png",
        "figures/fig_dse_edge_sensitivity_latency_slices.pdf",
        "figures/fig_dse_edge_sensitivity_throughput.png",
        "figures/fig_dse_edge_sensitivity_throughput.pdf",
        "figures/fig_dse_edge_sensitivity_comm_share.png",
        "figures/fig_dse_edge_sensitivity_comm_share.pdf",
        "figures/fig_dse_edge_sensitivity_comm_bars.png",
        "figures/fig_dse_edge_sensitivity_comm_bars.pdf",
        "figures/fig_dse_edge_sensitivity_tpot_bars.png",
        "figures/fig_dse_edge_sensitivity_tpot_bars.pdf",
        "plot_data/plot_data_edge_sensitivity.csv",
        "plot_data/plot_data_edge_sensitivity_comm_bars.csv",
        "plot_data/plot_data_edge_sensitivity_tpot_bars.csv",
    ]


def detect_pack_kind(rows: list[dict[str, str]]) -> str:
    modes = {r.get("mode", "") for r in rows}
    if modes == {"edge_sensitivity"}:
        return "edge_sensitivity"
    return "main"


def validate_rows(
    rows: list[dict[str, str]],
    invalid: list[dict[str, str]],
    pareto: list[dict[str, str]],
    pack_kind: str,
    errors: list[str],
) -> None:
    if not rows:
        errors.append("raw_results.csv has no rows")
        return

    common_columns = [
        "throughput_tps",
        "normalized_throughput",
        "sparse_hit_rate",
        "moe_comm_us",
        "tpot_ms",
        "wafer_area_headroom",
        "bottleneck",
    ]
    for col in common_columns:
        if col not in rows[0]:
            errors.append(f"raw_results missing column {col}")

    if pack_kind == "main":
        if len(rows) != 24:
            errors.append(f"expected 24 valid rows, got {len(rows)}")
        if len(invalid) != 11:
            errors.append(f"expected 11 invalid rows, got {len(invalid)}")
        modes = {r.get("mode", "") for r in rows}
        expected_modes = {"sram_compute", "core_hbm", "wafer_scale", "edge_bandwidth"}
        if modes != expected_modes:
            errors.append(f"main pack modes mismatch: got {sorted(modes)}")
    elif pack_kind == "edge_sensitivity":
        if len(rows) != 63:
            errors.append(f"expected 63 valid rows, got {len(rows)}")
        if len(invalid) != 0:
            errors.append(f"expected 0 invalid rows, got {len(invalid)}")
        modes = {r.get("mode", "") for r in rows}
        if modes != {"edge_sensitivity"}:
            errors.append(f"edge-sensitivity pack modes mismatch: got {sorted(modes)}")
        for col in [
            "communication_us",
            "crossbar_comm_us",
            "rail_analytical_comm_us",
            "switch_latency_ns",
            "per_port_bw_gbps",
        ]:
            if col not in rows[0]:
                errors.append(f"edge-sensitivity raw_results missing column {col}")

    pareto_ids = {r["case_id"] for r in pareto}
    recomputed = {r["case_id"] for r in pareto_front(rows)}
    if pareto_ids != recomputed:
        errors.append(f"pareto mismatch: csv={sorted(pareto_ids)} recomputed={sorted(recomputed)}")


def validate_metadata(out: Path, rows: list[dict[str, str]], invalid: list[dict[str, str]], errors: list[str]) -> None:
    meta_path = out / "dse_config_dump.json"
    if not meta_path.exists():
        return
    meta = json.loads((out / "dse_config_dump.json").read_text(encoding="utf-8"))
    if meta.get("num_valid_cases") != len(rows):
        errors.append("dse_config_dump num_valid_cases mismatch")
    if meta.get("num_invalid_cases") != len(invalid):
        errors.append("dse_config_dump num_invalid_cases mismatch")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate HACMO DSE outputs.")
    p.add_argument("--output-root", default="")
    p.add_argument(
        "--pack-kind",
        choices=["auto", "main", "edge_sensitivity"],
        default="auto",
        help="Expected output-pack shape. Auto-detects from raw_results.csv by default.",
    )
    return p.parse_args()


def latest_output() -> Path:
    outs = sorted((DSE_DIR / "outputs").glob("dse_20*"))
    if not outs:
        raise FileNotFoundError("No hacmo/DSE/outputs/dse_* directory found.")
    return outs[-1]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def pareto_front(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out = []
    for row in rows:
        t = float(row["throughput_tps"])
        h = float(row.get("wafer_area_headroom", 0.0))
        dominated = False
        for other in rows:
            if other is row:
                continue
            ot = float(other["throughput_tps"])
            oh = float(other.get("wafer_area_headroom", 0.0))
            if ot >= t and oh >= h and (ot > t or oh > h):
                dominated = True
                break
        if not dominated:
            out.append(row)
    return out


if __name__ == "__main__":
    main()
