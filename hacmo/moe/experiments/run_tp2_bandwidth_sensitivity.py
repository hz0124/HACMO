#!/usr/bin/env python3
"""Run SupernodeSim/ASTRA bandwidth sensitivity for TP2 artifacts."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml


METHODS = ["vllm", "moentwine", "ours_static", "ours_dynamic"]
METHOD_LABELS = {
    "vllm": "vLLM",
    "moentwine": "MoEnTwine",
    "ours_static": "Ours static",
    "ours_dynamic": "Ours dynamic",
}
METHOD_COLORS = {
    "vllm": "#7f8c8d",
    "moentwine": "#2e86de",
    "ours_static": "#e67e22",
    "ours_dynamic": "#c0392b",
}


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def variant_specs(
    base_switch_bw: float,
    base_mesh_bw: float,
    base_switch_lat: float,
    base_mesh_lat: float,
    include_latency: bool,
) -> list[tuple[str, float, float, float, float]]:
    specs = [
        ("base", base_switch_bw, base_mesh_bw, base_switch_lat, base_mesh_lat),
        ("inter_half", base_switch_bw * 0.5, base_mesh_bw, base_switch_lat, base_mesh_lat),
        ("inter_tenth", base_switch_bw * 0.1, base_mesh_bw, base_switch_lat, base_mesh_lat),
        ("mesh_5tb", base_switch_bw, 5000.0, base_switch_lat, base_mesh_lat),
        ("inter_half_mesh_5tb", base_switch_bw * 0.5, 5000.0, base_switch_lat, base_mesh_lat),
        ("inter_tenth_mesh_5tb", base_switch_bw * 0.1, 5000.0, base_switch_lat, base_mesh_lat),
    ]
    if include_latency:
        specs.extend([
            ("lat_200_20", base_switch_bw, base_mesh_bw, 200.0, 20.0),
            ("lat_200_20_inter_tenth", base_switch_bw * 0.1, base_mesh_bw, 200.0, 20.0),
            ("lat_50_5", base_switch_bw, base_mesh_bw, 50.0, 5.0),
            ("lat_50_5_inter_tenth", base_switch_bw * 0.1, base_mesh_bw, 50.0, 5.0),
            ("lat_20_2", base_switch_bw, base_mesh_bw, 20.0, 2.0),
            ("lat_20_2_inter_tenth", base_switch_bw * 0.1, base_mesh_bw, 20.0, 2.0),
        ])
    return specs


def write_variant_config(
    src: Path,
    dst: Path,
    switch_bw: float,
    mesh_bw: float,
    switch_latency: float,
    mesh_latency: float,
) -> None:
    raw = yaml.safe_load(src.read_text(encoding="utf-8")) or {}
    bandwidth = list(raw.get("bandwidth", [switch_bw, mesh_bw]))
    bandwidth[0] = float(switch_bw)
    bandwidth[1] = float(mesh_bw)
    raw["bandwidth"] = bandwidth
    latency = list(raw.get("latency", [switch_latency, mesh_latency]))
    latency[0] = float(switch_latency)
    latency[1] = float(mesh_latency)
    raw["latency"] = latency
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


def run_variant(
    *,
    repo_root: Path,
    artifact_topology_dir: Path,
    network_config: Path,
    output_root: Path,
    output_csv: Path,
    methods: list[str],
    batch_size: int,
    num_requests: int,
    window_size: int,
    decode_len: int,
    astra_network_backend: str,
) -> None:
    cmd = [
        sys.executable,
        "supernode_sim/experiments/run_ablation2_from_artifacts.py",
        "--artifact-topology-dir", str(artifact_topology_dir),
        "--network-config", str(network_config),
        "--methods", *methods,
        "--batch-sizes", str(batch_size),
        "--num-requests", str(num_requests),
        "--window-size", str(window_size),
        "--decode-len", str(decode_len),
        "--astra-network-backend", astra_network_backend,
        "--run-astra",
        "--output-root", str(output_root),
        "--output-csv", str(output_csv),
    ]
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(repo_root)
    subprocess.run(cmd, cwd=str(repo_root), env=env, check=True)


def plot_sensitivity(rows: list[dict], output_dir: Path, batch_size: int) -> None:
    topologies = list(dict.fromkeys(r["topology"] for r in rows))
    for topo in topologies:
        sub = [r for r in rows if r["topology"] == topo]
        variants = list(dict.fromkeys(r["variant"] for r in sub))
        methods = [m for m in METHODS[1:] if any(r["method"] == m for r in sub)]
        if not methods:
            continue
        fig, ax = plt.subplots(figsize=(8.5, 3.8))
        x = list(range(len(variants)))
        width = min(0.8 / max(len(methods), 1), 0.28)
        for i, method in enumerate(methods):
            vals = []
            for variant in variants:
                base_row = next((r for r in sub if r["variant"] == variant and r["method"] == "vllm"), None)
                cur_row = next((r for r in sub if r["variant"] == variant and r["method"] == method), None)
                if base_row is None or cur_row is None:
                    vals.append(0.0)
                    continue
                base = float(base_row["total_us"])
                cur = float(cur_row["total_us"])
                vals.append(base / max(cur, 1e-9))
            offs = [v + (i - (len(methods) - 1) / 2) * width for v in x]
            ax.bar(offs, vals, width=width, label=METHOD_LABELS[method], color=METHOD_COLORS[method])
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(f"{topo} bandwidth sensitivity, batch {batch_size}")
        ax.set_ylabel("Speedup over vLLM")
        ax.set_xticks(x)
        ax.set_xticklabels(variants, rotation=15, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / f"bandwidth_sensitivity_speedup_{topo}_bs{batch_size}.png", dpi=240)
        plt.close(fig)


def plot_absolute_latency(rows: list[dict], output_dir: Path, batch_size: int) -> None:
    topologies = list(dict.fromkeys(r["topology"] for r in rows))
    for topo in topologies:
        sub = [r for r in rows if r["topology"] == topo]
        variants = list(dict.fromkeys(r["variant"] for r in sub))
        methods = [m for m in METHODS if any(r["method"] == m for r in sub)]
        fig, ax = plt.subplots(figsize=(9.2, 4.0))
        x = list(range(len(variants)))
        width = min(0.8 / max(len(methods), 1), 0.28)
        for i, method in enumerate(methods):
            vals = []
            for variant in variants:
                row = next((r for r in sub if r["variant"] == variant and r["method"] == method), None)
                vals.append(float(row["total_us"]) if row else 0.0)
            offs = [v + (i - (len(methods) - 1) / 2) * width for v in x]
            ax.bar(offs, vals, width=width, label=METHOD_LABELS[method], color=METHOD_COLORS[method])
        ax.set_title(f"{topo} absolute latency sensitivity, batch {batch_size}")
        ax.set_ylabel("MoE latency (us/layer)")
        ax.set_xticks(x)
        ax.set_xticklabels(variants, rotation=15, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8, ncol=4)
        fig.tight_layout()
        fig.savefig(output_dir / f"bandwidth_sensitivity_absolute_{topo}_bs{batch_size}.png", dpi=240)
        plt.close(fig)


def plot_single_method_absolute(rows: list[dict], output_dir: Path, batch_size: int, method: str) -> None:
    topologies = list(dict.fromkeys(r["topology"] for r in rows))
    fig, axes = plt.subplots(1, len(topologies), figsize=(5.6 * len(topologies), 3.6), sharey=False)
    if len(topologies) == 1:
        axes = [axes]
    for ax, topo in zip(axes, topologies):
        sub = [r for r in rows if r["topology"] == topo and r["method"] == method]
        variants = list(dict.fromkeys(r["variant"] for r in sub))
        vals = [float(next(r["total_us"] for r in sub if r["variant"] == v)) for v in variants]
        ax.bar(range(len(variants)), vals, color=METHOD_COLORS.get(method, "#666666"))
        ax.set_title(topo)
        ax.set_xlabel("Network setting")
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels(variants, rotation=20, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel(f"{METHOD_LABELS.get(method, method)} latency (us/layer)")
    fig.suptitle(f"Absolute latency under bandwidth/latency variants, batch {batch_size}", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / f"bandwidth_sensitivity_single_{method}_absolute_bs{batch_size}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Run TP2 SupernodeSim/ASTRA bandwidth sensitivity.")
    p.add_argument("--artifact-root", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--topologies", nargs="+", default=["4w_8x8_rail_900", "16w_4x4_rail_500"])
    p.add_argument("--network-configs", nargs="+", required=True)
    p.add_argument("--methods", nargs="+", default=METHODS)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-requests", type=int, default=64)
    p.add_argument("--window-size", type=int, default=1)
    p.add_argument("--decode-len", type=int, default=2)
    p.add_argument("--include-latency-variants", action="store_true")
    p.add_argument("--astra-network-backend", choices=["congestion_aware", "congestion_unaware"],
                   default="congestion_aware")
    p.add_argument("--plot-only", action="store_true",
                   help="Read bandwidth_sensitivity_metrics.csv from output-root and regenerate plots without rerunning ASTRA.")
    p.add_argument("--single-method-absolute", default="ours_static",
                   help="Method used for the single-method absolute latency sensitivity plot.")
    args = p.parse_args()

    repo_root = Path.cwd()
    artifact_root = Path(args.artifact_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    configs_dir = output_root / "network_configs"
    runs_dir = output_root / "runs"
    plots_dir = output_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []

    if args.plot_only:
        metrics_csv = output_root / "bandwidth_sensitivity_metrics.csv"
        all_rows = read_csv(metrics_csv)
        plot_sensitivity(all_rows, plots_dir, args.batch_size)
        plot_absolute_latency(all_rows, plots_dir, args.batch_size)
        if any(r["method"] == args.single_method_absolute for r in all_rows):
            plot_single_method_absolute(all_rows, plots_dir, args.batch_size, args.single_method_absolute)
        print(f"Regenerated bandwidth sensitivity plots from {metrics_csv}")
        return

    for topology, cfg_str in zip(args.topologies, args.network_configs):
        src_cfg = Path(cfg_str)
        raw = yaml.safe_load(src_cfg.read_text(encoding="utf-8")) or {}
        base_switch, base_mesh = [float(x) for x in raw.get("bandwidth", [500.0, 2000.0])[:2]]
        base_switch_lat, base_mesh_lat = [float(x) for x in raw.get("latency", [2000.0, 500.0])[:2]]
        for variant, switch_bw, mesh_bw, switch_lat, mesh_lat in variant_specs(
            base_switch, base_mesh, base_switch_lat, base_mesh_lat, args.include_latency_variants
        ):
            variant_cfg = configs_dir / f"{topology}_{variant}.yaml"
            write_variant_config(src_cfg, variant_cfg, switch_bw, mesh_bw, switch_lat, mesh_lat)
            variant_csv = output_root / f"metrics_{topology}_{variant}_bs{args.batch_size}.csv"
            run_variant(
                repo_root=repo_root,
                artifact_topology_dir=artifact_root / topology,
                network_config=variant_cfg,
                output_root=runs_dir / variant,
                output_csv=variant_csv,
                methods=args.methods,
                batch_size=args.batch_size,
                num_requests=args.num_requests,
                window_size=args.window_size,
                decode_len=args.decode_len,
                astra_network_backend=args.astra_network_backend,
            )
            for row in read_csv(variant_csv):
                row["variant"] = variant
                row["switch_bw_gbps"] = switch_bw
                row["mesh_bw_gbps"] = mesh_bw
                row["switch_latency_ns"] = switch_lat
                row["mesh_latency_ns"] = mesh_lat
                all_rows.append(row)

    write_csv(output_root / "bandwidth_sensitivity_metrics.csv", all_rows)
    plot_sensitivity(all_rows, plots_dir, args.batch_size)
    plot_absolute_latency(all_rows, plots_dir, args.batch_size)
    if args.single_method_absolute in args.methods:
        plot_single_method_absolute(all_rows, plots_dir, args.batch_size, args.single_method_absolute)
    print(f"Wrote bandwidth sensitivity to {output_root}")


if __name__ == "__main__":
    main()
