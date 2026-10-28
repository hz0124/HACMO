#!/usr/bin/env python3
"""Generate the TP2 motivation overhead-breakdown figure.

This is the serving-epoch replacement for the older
motivation/challenge23_supernode_breakdown figure.  It uses the current TP2
serving-epoch paper pack and the current TP3 lane-level final pack, then
reconstructs the same empirical decomposition:

    Full+TP2 communication
  + (Rail+Naive communication - Rail+TP2 communication)
  + (Rail+TP2 communication - Full+TP2 communication)
  = Rail+Naive communication.

The first increment is the communication overhead removed by TP2 placement
under the rail topology.  The second increment is the fixed edge-I/O overhead
removed by TP3 lane-level allocation; it is included because the legacy
motivation figure explains Challenge-2/3 together.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
TP2_DIR = SCRIPT_DIR.parent
HACMO_DIR = TP2_DIR.parent

DEFAULT_TP2_SUMMARY_CSV = (
    TP2_DIR
    / "outputs"
    / "tp2_serving_epoch_paper_pack_20260831"
    / "paper_figures"
    / "paper_figure_summary.csv"
)
DEFAULT_TP3_METRICS_CSV = (
    HACMO_DIR
    / "interconnect"
    / "outputs"
    / "tp3_lane_level_multibatch_final_candidate_20260614"
    / "tp3_metrics.csv"
)
DEFAULT_OUTPUT_DIR = (
    TP2_DIR
    / "motivation"
    / "tp2_serving_epoch_overhead_breakdown_20260617"
)

TOPOLOGY_ORDER = ["4w_8x8_rail_900", "16w_4x4_rail_500"]
TOPOLOGY_TITLES = {
    "4w_8x8_rail_900": "4W 8x8",
    "16w_4x4_rail_500": "16W 4x4",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def tp2_row(rows: list[dict[str, str]], topology: str, batch: int, method: str) -> dict[str, str]:
    for row in rows:
        if row["topology"] == topology and int(float(row["batch_size"])) == batch and row["method"] == method:
            return row
    raise KeyError(f"missing TP2 row: topology={topology} batch={batch} method={method}")


def tp3_row(rows: list[dict[str, str]], topology: str, batch: int, bar_key: str) -> dict[str, str]:
    for row in rows:
        if row["topology"] == topology and int(float(row["batch_size"])) == batch and row["bar_key"] == bar_key:
            return row
    raise KeyError(f"missing TP3 row: topology={topology} batch={batch} bar_key={bar_key}")


def build_rows(tp2_rows: list[dict[str, str]], tp3_rows: list[dict[str, str]], topologies: list[str], batches: list[int]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for topology in topologies:
        for batch in batches:
            naive = tp2_row(tp2_rows, topology, batch, "vllm")
            rail_tp2 = tp2_row(tp2_rows, topology, batch, "ours_dynamic")
            full_tp2 = tp3_row(tp3_rows, topology, batch, "full_tp2")

            naive_comm = float(naive["communication_us"])
            rail_tp2_comm = float(rail_tp2["communication_us"])
            full_tp2_comm = float(full_tp2["communication_us"])

            if full_tp2_comm > rail_tp2_comm + 1e-6:
                raise ValueError(f"Full+TP2 communication exceeds Rail+TP2 for {topology} b{batch}")
            if rail_tp2_comm > naive_comm + 1e-6:
                raise ValueError(f"Rail+TP2 communication exceeds Rail+Naive for {topology} b{batch}")

            c2_overhead = max(naive_comm - rail_tp2_comm, 0.0)
            c3_overhead = max(rail_tp2_comm - full_tp2_comm, 0.0)
            full_comm = full_tp2_comm + c2_overhead + c3_overhead
            out.append(
                {
                    "topology": topology,
                    "batch_size": batch,
                    "source_mode": "serving_epoch_paper_results",
                    "ideal_latency_us": f"{full_tp2_comm:.6f}",
                    "challenge2_latency_us": f"{full_tp2_comm + c2_overhead:.6f}",
                    "full_latency_us": f"{full_comm:.6f}",
                    "challenge2_overhead_us": f"{c2_overhead:.6f}",
                    "challenge3_overhead_us": f"{c3_overhead:.6f}",
                    "full_overhead_us": f"{c2_overhead + c3_overhead:.6f}",
                    "challenge2_overhead_pct_vs_ideal": f"{100.0 * c2_overhead / max(full_tp2_comm, 1e-9):.6f}",
                    "challenge3_overhead_pct_vs_challenge2": f"{100.0 * c3_overhead / max(full_tp2_comm + c2_overhead, 1e-9):.6f}",
                    "full_overhead_pct_vs_ideal": f"{100.0 * (c2_overhead + c3_overhead) / max(full_tp2_comm, 1e-9):.6f}",
                    "ideal_share_of_full_pct": f"{100.0 * full_tp2_comm / max(full_comm, 1e-9):.6f}",
                    "challenge2_share_of_full_pct": f"{100.0 * c2_overhead / max(full_comm, 1e-9):.6f}",
                    "challenge3_share_of_full_pct": f"{100.0 * c3_overhead / max(full_comm, 1e-9):.6f}",
                    "naive_rail_comm_us": f"{naive_comm:.6f}",
                    "rail_tp2_comm_us": f"{rail_tp2_comm:.6f}",
                    "full_tp2_comm_us": f"{full_tp2_comm:.6f}",
                    "tp2_comm_reduction_us": f"{c2_overhead:.6f}",
                    "tp3_comm_reduction_us": f"{c3_overhead:.6f}",
                }
            )
    return out


def write_ppt_csv(path: Path, rows: list[dict[str, object]]) -> None:
    ppt_rows = [
        {
            "topology": row["topology"],
            "batch_size": row["batch_size"],
            "optimized_lower_bound_us": row["ideal_latency_us"],
            "tp2_rail_placement_overhead_us": row["challenge2_overhead_us"],
            "tp3_fixed_io_overhead_us": row["challenge3_overhead_us"],
            "rail_naive_comm_us": row["full_latency_us"],
            "tp2_overhead_share_pct": row["challenge2_share_of_full_pct"],
            "tp3_overhead_share_pct": row["challenge3_share_of_full_pct"],
        }
        for row in rows
    ]
    write_csv(path, ppt_rows)


def plot_breakdown(rows: list[dict[str, object]], output_dir: Path) -> None:
    colors = {
        "ideal": "#b0bec5",
        "c2": "#4e79a7",
        "c3": "#f28e2b",
    }
    fig, axes = plt.subplots(
        len(TOPOLOGY_ORDER),
        2,
        figsize=(11.5, 6.8),
        gridspec_kw={"width_ratios": [1.58, 1.0], "wspace": 0.28, "hspace": 0.38},
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.075, right=0.965, top=0.90, bottom=0.10)

    legend_handles = None
    legend_labels = None
    for row_idx, topology in enumerate(TOPOLOGY_ORDER):
        topo_rows = [r for r in rows if r["topology"] == topology]
        batch_sizes = [int(r["batch_size"]) for r in topo_rows]
        ideal = np.array([float(r["ideal_latency_us"]) for r in topo_rows])
        c2 = np.array([float(r["challenge2_overhead_us"]) for r in topo_rows])
        c3 = np.array([float(r["challenge3_overhead_us"]) for r in topo_rows])
        total = ideal + c2 + c3
        c3_pct = np.array([float(r["challenge3_share_of_full_pct"]) for r in topo_rows])

        x = np.arange(len(batch_sizes))
        ax = axes[row_idx, 0]
        ax.bar(
            x,
            ideal,
            width=0.58,
            color=colors["ideal"],
            edgecolor="#263238",
            linewidth=0.5,
            label="Optimized lower bound (Full+TP2)",
        )
        ax.bar(
            x,
            c2,
            width=0.58,
            bottom=ideal,
            color=colors["c2"],
            edgecolor="#263238",
            linewidth=0.5,
            label="TP2 rail/placement overhead",
        )
        ax.bar(
            x,
            c3,
            width=0.58,
            bottom=ideal + c2,
            color=colors["c3"],
            edgecolor="#263238",
            linewidth=0.5,
            hatch="///",
            label="Fixed edge-I/O overhead",
        )
        for idx, val in enumerate(total):
            ax.text(idx, val + max(total) * 0.024, f"{val:.1f}", ha="center", va="bottom", fontsize=8)
            if c2[idx] > max(total) * 0.055:
                ax.text(idx, ideal[idx] + c2[idx] / 2.0, f"+{c2[idx]:.1f}", ha="center", va="center", fontsize=8, color="white")
        ax.set_xticks(x)
        ax.set_xticklabels([str(bs) for bs in batch_sizes])
        ax.set_xlabel("Batch per die")
        ax.set_ylabel("Communication latency (us/layer)")
        ax.set_title(f"{TOPOLOGY_TITLES[topology]} overhead breakdown", fontsize=12)
        ax.grid(axis="y", alpha=0.22)
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

        ax_zoom = axes[row_idx, 1]
        ax_zoom.bar(x, c3, width=0.54, color=colors["c3"], edgecolor="#263238", linewidth=0.5, hatch="///")
        for idx, val in enumerate(c3):
            ax_zoom.text(idx, val + max(c3) * 0.055, f"{val:.1f}", ha="center", va="bottom", fontsize=8)
        ax_zoom.set_xticks(x)
        ax_zoom.set_xticklabels([str(bs) for bs in batch_sizes])
        ax_zoom.set_xlabel("Batch per die")
        ax_zoom.set_ylabel("Fixed edge-I/O overhead (us/layer)")
        ax_zoom.set_title(f"{TOPOLOGY_TITLES[topology]} fixed-I/O zoom", fontsize=12)
        ax_zoom.grid(axis="y", alpha=0.22)
        ax_zoom.set_ylim(0.0, max(c3) * 1.33 if max(c3) > 0 else 1.0)
        ax_pct = ax_zoom.twinx()
        ax_pct.plot(x, c3_pct, color="#8a4b08", marker="o", linewidth=1.45)
        ax_pct.set_ylabel("Share of Rail+Naive comm. (%)")
        ax_pct.set_ylim(0.0, max(c3_pct) * 1.42 if max(c3_pct) > 0 else 1.0)
        ax_pct.tick_params(axis="y", colors="#8a4b08")
        ax_pct.yaxis.label.set_color("#8a4b08")

    if legend_handles is not None:
        fig.legend(legend_handles, legend_labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.985), fontsize=9)
    fig.text(
        0.985,
        0.018,
        "Serving-epoch paper results: Full+TP2 + (Rail+Naive - Rail+TP2) + (Rail+TP2 - Full+TP2) = Rail+Naive communication.",
        ha="right",
        va="bottom",
        fontsize=7.2,
        color="#607d8b",
    )
    fig.savefig(output_dir / "tp2_serving_epoch_overhead_breakdown.png", dpi=360, bbox_inches="tight")
    fig.savefig(output_dir / "tp2_serving_epoch_overhead_breakdown.pdf", bbox_inches="tight")
    plt.close(fig)


def write_readme(output_dir: Path, args: argparse.Namespace) -> None:
    lines = [
        "# TP2 Serving-Epoch Overhead Breakdown",
        "",
        "This is the serving-epoch replacement for the older `challenge23_supernode_breakdown` motivation figure.",
        "",
        "## Model",
        "",
        "- `ideal_latency_us`: current TP3 `Full+TP2` communication latency, used as the optimized communication lower bound.",
        "- `challenge2_overhead_us`: `Rail+Naive - Rail+TP2`, the communication overhead removed by TP2 expert placement under the same rail topology.",
        "- `challenge3_overhead_us`: `Rail+TP2 - Full+TP2`, the fixed edge-I/O overhead removed by TP3 lane-level dynamic allocation.",
        "- Therefore `ideal_latency_us + challenge2_overhead_us + challenge3_overhead_us = Rail+Naive` communication latency.",
        "",
        "## Files",
        "",
        "- `tp2_serving_epoch_overhead_breakdown.png` / `.pdf`: paper/PPT figure.",
        "- `tp2_serving_epoch_overhead_breakdown.csv`: full numeric table.",
        "- `tp2_serving_epoch_overhead_breakdown_ppt.csv`: compact reproduction table.",
        "",
        "## Sources",
        "",
        f"- TP2 summary CSV: `{args.tp2_summary_csv}`.",
        f"- TP3 metrics CSV: `{args.tp3_metrics_csv}`.",
        "",
        "Scope: communication-only ASTRA routed service from the current serving-epoch paper pack; H200 expert compute is intentionally excluded from this motivation breakdown.",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TP2 serving-epoch overhead-breakdown motivation figure.")
    parser.add_argument("--tp2-summary-csv", type=Path, default=DEFAULT_TP2_SUMMARY_CSV)
    parser.add_argument("--tp3-metrics-csv", type=Path, default=DEFAULT_TP3_METRICS_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--topologies", nargs="+", default=TOPOLOGY_ORDER)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[16, 32, 64, 128])
    args = parser.parse_args()

    tp2_rows = read_csv(args.tp2_summary_csv)
    tp3_rows = read_csv(args.tp3_metrics_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows(tp2_rows, tp3_rows, args.topologies, args.batch_sizes)
    write_csv(args.output_dir / "tp2_serving_epoch_overhead_breakdown.csv", rows)
    write_ppt_csv(args.output_dir / "tp2_serving_epoch_overhead_breakdown_ppt.csv", rows)
    plot_breakdown(rows, args.output_dir)
    write_readme(args.output_dir, args)
    print(f"Wrote TP2 serving-epoch overhead breakdown to {args.output_dir}")


if __name__ == "__main__":
    main()
