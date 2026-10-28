#!/usr/bin/env python3
"""Generate visual inspection figures for TP1 rebuttal outputs."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-tp1")

import matplotlib.pyplot as plt
import numpy as np


METHOD_LABELS = {
    "G": "GPU+vLLM",
    "B": "vLLM+HSN",
    "C": "Naive Buffer",
    "S": "Static",
    "O": "Ours",
}

METHOD_COLORS = {
    "G": "#6f6f6f",
    "B": "#a0a0a0",
    "C": "#b8b8b8",
    "S": "#4c78a8",
    "O": "#f58518",
}


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_index(out_dir: Path, figures: Iterable[tuple[str, str]]) -> None:
    lines = ["# TP1 Rebuttal Visual Report", ""]
    for title, rel in figures:
        lines.append(f"## {title}")
        lines.append("")
        lines.append(f"![{title}]({rel})")
        lines.append("")
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def plot_hbm_service(service_json: Path, out_dir: Path) -> list[tuple[str, str]]:
    data = json.loads(service_json.read_text(encoding="utf-8"))
    rows = data.get("rows", [])
    figs: list[tuple[str, str]] = []
    if not rows:
        return figs

    contexts = sorted({int(r["context_len"]) for r in rows})
    patterns = ["dense", "profile", "page_grouped", "random"]
    colors = {"dense": "#2ca02c", "profile": "#4c78a8", "page_grouped": "#f58518", "random": "#d62728"}

    fig, axes = plt.subplots(1, len(contexts), figsize=(5.2 * len(contexts), 4.2), sharey=True)
    if len(contexts) == 1:
        axes = [axes]
    for ax, ctx in zip(axes, contexts):
        sub = [r for r in rows if int(r["context_len"]) == ctx]
        for pat in patterns:
            pts = sorted([r for r in sub if r["pattern"] == pat], key=lambda x: int(x["batch_size"]))
            if not pts:
                continue
            ax.plot(
                [int(x["batch_size"]) for x in pts],
                [float(x["effective_gbps"]) for x in pts],
                marker="o",
                linewidth=2,
                color=colors.get(pat),
                label=pat,
            )
        ax.set_xscale("log", base=2)
        ax.set_title(f"Context {ctx // 1024}k")
        ax.set_xlabel("Batch")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Effective HBM bandwidth (GB/s)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=min(4, len(labels)), loc="upper center", bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    path = out_dir / "hbm_effective_bandwidth.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    figs.append(("HBM Effective Bandwidth", path.name))

    fig, axes = plt.subplots(1, len(contexts), figsize=(5.2 * len(contexts), 4.2), sharey=True)
    if len(contexts) == 1:
        axes = [axes]
    for ax, ctx in zip(axes, contexts):
        sub = [r for r in rows if int(r["context_len"]) == ctx]
        for pat in patterns:
            pts = sorted([r for r in sub if r["pattern"] == pat], key=lambda x: int(x["batch_size"]))
            if not pts:
                continue
            ax.plot(
                [int(x["batch_size"]) for x in pts],
                [float(x["amplification_est"]) for x in pts],
                marker="s",
                linewidth=2,
                color=colors.get(pat),
                label=pat,
            )
        ax.set_xscale("log", base=2)
        ax.set_title(f"Context {ctx // 1024}k")
        ax.set_xlabel("Batch")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Estimated DRAM/requested-byte amplification")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=min(4, len(labels)), loc="upper center", bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    path = out_dir / "hbm_amplification.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    figs.append(("HBM Amplification", path.name))
    return figs


def plot_mapping(mapping_json: Path, out_dir: Path) -> list[tuple[str, str]]:
    data = json.loads(mapping_json.read_text(encoding="utf-8"))
    profiles = data.get("profiles", {})
    figs: list[tuple[str, str]] = []
    if not profiles:
        return figs

    shapes = list(profiles.keys())
    fig, axes = plt.subplots(len(shapes), 1, figsize=(8.5, 3.2 * len(shapes)), squeeze=False)
    for ax, shape in zip(axes[:, 0], shapes):
        prof = profiles[shape]
        x = np.arange(len(prof["row_compute_weights"]))
        ax.plot(x, prof["row_compute_weights"], linewidth=2, label="compute weight", color="#4c78a8")
        ax.plot(x, prof["row_cache_weights"], linewidth=2, label="KV buffer weight", color="#f58518")
        ax.set_title(f"Static mapping row profile {shape}")
        ax.set_xlabel("Row")
        ax.set_ylabel("Share")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    path = out_dir / "static_mapping_row_profiles.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    figs.append(("Static Mapping Row Profiles", path.name))

    terms = ["avg_hbm_hops_scale", "avg_remote_hops_scale", "remote_fraction_scale", "tail_penalty_scale"]
    x = np.arange(len(terms))
    width = 0.8 / max(len(shapes), 1)
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    for i, shape in enumerate(shapes):
        vals = [float(profiles[shape]["derived"][t]) for t in terms]
        ax.bar(x + (i - (len(shapes) - 1) / 2) * width, vals, width=width, label=shape)
    ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
    ax.set_xticks(x, ["HBM hop", "Remote hop", "Remote frac", "Tail"])
    ax.set_ylabel("Scale vs baseline")
    ax.set_title("Static mapping derived scales")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = out_dir / "static_mapping_scales.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    figs.append(("Static Mapping Scales", path.name))
    return figs


def plot_ablation(final_root: Path, out_dir: Path) -> list[tuple[str, str]]:
    rows = _read_csv(final_root / "csv" / "ablation1_tp1.csv")
    figs: list[tuple[str, str]] = []
    if not rows:
        return figs

    key_ctx = 32768
    key_array = (20, 20)
    sub = [
        r
        for r in rows
        if int(r["context_len"]) == key_ctx and int(r["array_rows"]) == key_array[0] and int(r["array_cols"]) == key_array[1]
    ]
    labels = []
    for r in sub:
        if r["batch_label"] not in labels:
            labels.append(r["batch_label"])
    methods = ["G", "B", "C", "S", "O"]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.0), sharex=True)

    width = 0.15
    for i, method in enumerate(methods):
        vals = []
        for label in labels:
            match = [r for r in sub if r["batch_label"] == label and r["method"] == method]
            vals.append(float(match[0]["T_attn_total_us"]) if match else 0.0)
        axes[0].bar(x + (i - 2) * width, vals, width=width, label=METHOD_LABELS[method], color=METHOD_COLORS[method])
    axes[0].set_ylabel("Single-layer latency (us)")
    axes[0].set_title("Ablation latency overview (ctx=32k, array=20x20)")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend(ncol=3)

    for method in ["B", "C", "S", "O"]:
        comp, select, qkv, cp = [], [], [], []
        for label in labels:
            match = [r for r in sub if r["batch_label"] == label and r["method"] == method]
            if not match:
                comp.append(0.0); select.append(0.0); qkv.append(0.0); cp.append(0.0)
                continue
            r = match[0]
            comp.append(float(r["T_comp_us"]))
            select.append(float(r["T_select_mem_us"]))
            qkv.append(float(r["T_qkv_mem_us"]))
            cp.append(float(r["T_cp_comm_us"]))
        if method == "O":
            bottom = np.zeros(len(labels))
            axes[1].bar(x, comp, label="compute", color="#72b7b2")
            bottom += np.array(comp)
            axes[1].bar(x, select, bottom=bottom, label="select mem", color="#54a24b")
            bottom += np.array(select)
            axes[1].bar(x, qkv, bottom=bottom, label="KV mem", color="#f58518")
            bottom += np.array(qkv)
            axes[1].bar(x, cp, bottom=bottom, label="CP comm", color="#e45756")
    axes[1].set_ylabel("Ours breakdown (us)")
    axes[1].set_xticks(x, labels)
    axes[1].set_xlabel("Batch group")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend(ncol=4)
    fig.tight_layout()
    path = out_dir / "ablation_latency_and_ours_breakdown.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    figs.append(("Ablation Latency And Ours Breakdown", path.name))

    heat_rows = []
    ylabels = []
    for ctx in sorted({int(r["context_len"]) for r in rows}):
        for arr in sorted({(int(r["array_rows"]), int(r["array_cols"])) for r in rows if int(r["context_len"]) == ctx}):
            row_vals = []
            for label in labels:
                o = [r for r in rows if int(r["context_len"]) == ctx and int(r["array_rows"]) == arr[0] and int(r["array_cols"]) == arr[1] and r["batch_label"] == label and r["method"] == "O"]
                b = [r for r in rows if int(r["context_len"]) == ctx and int(r["array_rows"]) == arr[0] and int(r["array_cols"]) == arr[1] and r["batch_label"] == label and r["method"] == "B"]
                row_vals.append(float(b[0]["T_attn_total_us"]) / max(float(o[0]["T_attn_total_us"]), 1e-9) if o and b else np.nan)
            heat_rows.append(row_vals)
            ylabels.append(f"ctx{ctx//1024}k {arr[0]}x{arr[1]}")
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    im = ax.imshow(np.array(heat_rows), aspect="auto", cmap="YlOrRd")
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_yticks(np.arange(len(ylabels)), ylabels)
    ax.set_title("Ours speedup over vLLM+hierarchical supernode")
    for i in range(len(ylabels)):
        for j in range(len(labels)):
            val = heat_rows[i][j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}x", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Speedup")
    fig.tight_layout()
    path = out_dir / "ours_vs_vllm_speedup_heatmap.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    figs.append(("Ours Vs vLLM Speedup Heatmap", path.name))
    return figs


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot visual TP1 rebuttal report.")
    parser.add_argument("--pipeline_root", required=True)
    parser.add_argument("--final_root", required=True)
    parser.add_argument("--output_dir", default="")
    args = parser.parse_args()

    pipeline_root = Path(args.pipeline_root).resolve()
    final_root = Path(args.final_root).resolve()
    out_dir = Path(args.output_dir).resolve() if args.output_dir else final_root / "visual_report"
    out_dir.mkdir(parents=True, exist_ok=True)

    figures: list[tuple[str, str]] = []
    mapping = pipeline_root / "core_static_mapping_profile.json"
    service = pipeline_root / "hbm_service_curve" / "hbm_service_curve.json"
    if mapping.exists():
        figures.extend(plot_mapping(mapping, out_dir))
    if service.exists():
        figures.extend(plot_hbm_service(service, out_dir))
    if (final_root / "csv" / "ablation1_tp1.csv").exists():
        figures.extend(plot_ablation(final_root, out_dir))
    _write_index(out_dir, figures)
    print(out_dir)


if __name__ == "__main__":
    main()
