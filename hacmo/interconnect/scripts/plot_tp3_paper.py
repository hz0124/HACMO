#!/usr/bin/env python3
"""Generate paper-facing TP3 figures: absolute latency, normalized, and breakdown.

Style matches moe/plot_tp2_paper_figures.py for visual consistency.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BAR_ORDER = ["naive_rail", "rail_tp2", "crossbar_tp2", "full_tp2"]
BAR_LABELS = {
    "naive_rail": "Rail+Naive",
    "rail_tp2": "Rail+TP2",
    "crossbar_tp2": "Crossbar+TP2",
    "full_tp2": "Full+TP2",
}
BAR_COLORS = {
    "naive_rail": "#7f8c8d",
    "rail_tp2": "#c0392b",
    "crossbar_tp2": "#2e86de",
    "full_tp2": "#27ae60",
}


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _topology_rows(rows: List[Dict], topology: str) -> List[Dict]:
    return [r for r in rows if r["topology"] == topology and r["bar_key"] in BAR_ORDER]


def _batches(rows: List[Dict]) -> List[int]:
    return sorted({int(r["batch_size"]) for r in rows})


def _get(rows: List[Dict], bar_key: str, batch: int, key: str) -> float:
    for r in rows:
        if r["bar_key"] == bar_key and int(r["batch_size"]) == batch:
            v = r.get(key, "")
            return float(v) if v else 0.0
    return 0.0


def plot_absolute(rows: List[Dict], topology: str, out_dir: Path) -> None:
    sub = _topology_rows(rows, topology)
    batches = _batches(sub)
    x = list(range(len(batches)))
    width = 0.18
    fig, ax = plt.subplots(figsize=(7.2, 3.6))

    for i, bar_key in enumerate(BAR_ORDER):
        vals = [_get(sub, bar_key, bs, "total_us") for bs in batches]
        offs = [v + (i - 1.5) * width for v in x]
        ax.bar(offs, vals, width=width, label=BAR_LABELS[bar_key],
               color=BAR_COLORS[bar_key], edgecolor="black", linewidth=0.3)

    ax.set_title(f"Single-layer MoE latency - {topology}")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Latency (us)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in batches])
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=7, ncol=4)
    fig.tight_layout()
    fig.savefig(out_dir / f"paper_absolute_latency_{topology}.png", dpi=300)
    fig.savefig(out_dir / f"paper_absolute_latency_{topology}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_breakdown(rows: List[Dict], topology: str, out_dir: Path) -> None:
    sub = _topology_rows(rows, topology)
    batches = _batches(sub)
    x = list(range(len(batches)))
    width = 0.18
    fig, ax = plt.subplots(figsize=(7.8, 3.8))

    for i, bar_key in enumerate(BAR_ORDER):
        comm = [_get(sub, bar_key, bs, "communication_us") for bs in batches]
        comp = [_get(sub, bar_key, bs, "compute_us") for bs in batches]
        offs = [v + (i - 1.5) * width for v in x]
        ax.bar(offs, comm, width=width, color=BAR_COLORS[bar_key], alpha=0.78,
               label=BAR_LABELS[bar_key], edgecolor="black", linewidth=0.3)
        ax.bar(offs, comp, width=width, bottom=comm, color=BAR_COLORS[bar_key], alpha=0.30)

    ax.set_title(f"Latency breakdown - {topology}")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Latency (us)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in batches])
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=7, ncol=4)
    ax.text(
        0.99, 0.98,
        "solid: communication\nlight: compute",
        transform=ax.transAxes, ha="right", va="top", fontsize=7,
    )
    fig.tight_layout()
    fig.savefig(out_dir / f"paper_latency_breakdown_{topology}.png", dpi=300)
    fig.savefig(out_dir / f"paper_latency_breakdown_{topology}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_normalized(rows: List[Dict], topology: str, out_dir: Path) -> None:
    sub = _topology_rows(rows, topology)
    batches = _batches(sub)
    x = list(range(len(batches)))
    width = 0.18
    fig, ax = plt.subplots(figsize=(7.2, 3.6))

    for i, bar_key in enumerate(BAR_ORDER):
        vals = []
        for bs in batches:
            base = _get(sub, "naive_rail", bs, "total_us")
            cur = _get(sub, bar_key, bs, "total_us")
            vals.append(cur / max(base, 1e-9))
        offs = [v + (i - 1.5) * width for v in x]
        ax.bar(offs, vals, width=width, label=BAR_LABELS[bar_key],
               color=BAR_COLORS[bar_key], edgecolor="black", linewidth=0.3)

    # Annotate final bar with speedup percentage
    for j, bs in enumerate(batches):
        base = _get(sub, "naive_rail", bs, "total_us")
        final = _get(sub, "full_tp2", bs, "total_us")
        if base > 0:
            drop_pct = (base - final) / base * 100.0
            y = final / max(base, 1e-9)
            ax.text(x[j] + 1.5 * width, y + 0.03, f"-{drop_pct:.1f}%",
                    ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title(f"Normalized MoE latency - {topology}")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Normalized to Rail+Naive")
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in batches])
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=7, ncol=4)
    fig.tight_layout()
    fig.savefig(out_dir / f"paper_normalized_latency_{topology}.png", dpi=300)
    fig.savefig(out_dir / f"paper_normalized_latency_{topology}.pdf", bbox_inches="tight")
    plt.close(fig)


def generate_all_figures(metrics_csv: Path, out_dir: Path) -> None:
    rows = read_csv(metrics_csv)
    out_dir.mkdir(parents=True, exist_ok=True)

    for topology in sorted({r["topology"] for r in rows}):
        plot_absolute(rows, topology, out_dir)
        plot_breakdown(rows, topology, out_dir)
        plot_normalized(rows, topology, out_dir)

    # Write summary CSV
    summary_rows = []
    for topology in sorted({r["topology"] for r in rows}):
        sub = _topology_rows(rows, topology)
        for bs in _batches(sub):
            base = _get(sub, "naive_rail", bs, "total_us")
            for bar_key in BAR_ORDER:
                total = _get(sub, bar_key, bs, "total_us")
                summary_rows.append({
                    "topology": topology,
                    "batch_size": bs,
                    "bar_key": bar_key,
                    "bar_label": BAR_LABELS[bar_key],
                    "total_us": f"{total:.6f}",
                    "compute_us": f"{_get(sub, bar_key, bs, 'compute_us'):.6f}",
                    "communication_us": f"{_get(sub, bar_key, bs, 'communication_us'):.6f}",
                    "normalized_to_naive": f"{total / max(base, 1e-9):.6f}",
                    "speedup_vs_naive": f"{base / max(total, 1e-9):.6f}",
                })

    fieldnames = ["topology", "batch_size", "bar_key", "bar_label", "total_us",
                  "compute_us", "communication_us", "normalized_to_naive", "speedup_vs_naive"]
    csv_path = out_dir / "paper_figure_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(summary_rows)

    print(f"Wrote TP3 paper figures to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    generate_all_figures(Path(args.metrics_csv), Path(args.output_dir))


if __name__ == "__main__":
    main()
