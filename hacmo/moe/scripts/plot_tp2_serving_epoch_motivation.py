#!/usr/bin/env python3
"""Generate a TP2 serving-epoch motivation figure from the current paper pack.

The figure has two rows, one per topology, and two columns:
  - absolute single-layer MoE latency across batch sizes;
  - dynamic latency saved over the static baseline.

The data source is the current serving-epoch paper pack so the figure matches
the rebuttal-era semantics and can be cited directly in the response draft.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
TP2_DIR = SCRIPT_DIR.parent

DEFAULT_SUMMARY_CSV = (
    TP2_DIR
    / "outputs"
    / "tp2_serving_epoch_paper_pack_20260831"
    / "paper_figures"
    / "paper_figure_summary.csv"
)
DEFAULT_OUTPUT_DIR = (
    TP2_DIR
    / "motivation"
    / "tp2_serving_epoch_motivation_20260617"
)

METHODS = ["vllm", "moentwine", "ours_static", "ours_dynamic"]
METHOD_LABELS = {
    "vllm": "vLLM",
    "moentwine": "MoEntwine",
    "ours_static": "HACMO static",
    "ours_dynamic": "HACMO dynamic",
}
METHOD_COLORS = {
    "vllm": "#7f8c8d",
    "moentwine": "#2e86de",
    "ours_static": "#e67e22",
    "ours_dynamic": "#c0392b",
}
TOPOLOGY_ORDER = ["4w_8x8_rail_900", "16w_4x4_rail_500"]
TOPOLOGY_TITLES = {
    "4w_8x8_rail_900": "4w 8x8",
    "16w_4x4_rail_500": "16w 4x4",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_table(rows: list[dict[str, str]]) -> dict[str, dict[int, dict[str, float]]]:
    table: dict[str, dict[int, dict[str, float]]] = {}
    for r in rows:
        topo = r["topology"]
        if topo not in TOPOLOGY_ORDER:
            continue
        method = r["method"]
        if method not in METHODS:
            continue
        batch = int(r["batch_size"])
        table.setdefault(topo, {}).setdefault(batch, {})[method] = float(r["total_us"])
    return table


def plot_absolute(ax: plt.Axes, table: dict[int, dict[str, float]], topo: str) -> None:
    batches = sorted(table)
    x = list(range(len(batches)))
    width = 0.18
    for idx, method in enumerate(METHODS):
        vals = [table[bs][method] for bs in batches]
        offs = [v + (idx - 1.5) * width for v in x]
        ax.bar(
            offs,
            vals,
            width=width,
            color=METHOD_COLORS[method],
            edgecolor="#263238",
            linewidth=0.45,
            label=METHOD_LABELS[method],
        )
    ax.set_title(f"({TOPOLOGY_TITLES[topo]}) Absolute latency", fontsize=12.5)
    ax.set_ylabel("Latency (us)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in batches])
    ax.grid(axis="y", alpha=0.22)
    ax.set_ylim(0.0, max(max(table[bs][m] for m in METHODS) for bs in batches) * 1.15)


def plot_gain(ax: plt.Axes, table: dict[int, dict[str, float]], topo: str) -> None:
    batches = sorted(table)
    gain = [table[bs]["ours_static"] - table[bs]["ours_dynamic"] for bs in batches]
    pct = [
        100.0 * (table[bs]["ours_static"] - table[bs]["ours_dynamic"]) / max(table[bs]["ours_static"], 1e-9)
        for bs in batches
    ]
    x = list(range(len(batches)))
    bars = ax.bar(x, gain, width=0.58, color="#27ae60", edgecolor="#1f6f43", linewidth=0.45)
    for bar, p in zip(bars, pct):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + max(max(gain) * 0.035, 0.15),
            f"{p:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9.3,
        )
    ax.set_title(f"({TOPOLOGY_TITLES[topo]}) Dynamic latency saved over static", fontsize=12.5)
    ax.set_xlabel("Batch per die")
    ax.set_ylabel("Latency saved (us)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in batches])
    ax.grid(axis="y", alpha=0.22)
    ax.set_ylim(0.0, max(gain) * 1.22 if max(gain) > 0 else 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TP2 serving-epoch motivation figure.")
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    rows = read_rows(args.summary_csv)
    table = make_table(rows)
    missing = [topo for topo in TOPOLOGY_ORDER if topo not in table]
    if missing:
        raise ValueError(f"Missing topologies in summary CSV: {missing}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11.6, 6.7),
        sharex="col",
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.11, top=0.89, wspace=0.16, hspace=0.28)

    for row_idx, topo in enumerate(TOPOLOGY_ORDER):
        plot_absolute(axes[row_idx, 0], table[topo], topo)
        plot_gain(axes[row_idx, 1], table[topo], topo)
        if row_idx == 1:
            axes[row_idx, 0].set_xlabel("Batch per die")
            axes[row_idx, 1].set_xlabel("Batch per die")
        else:
            axes[row_idx, 0].set_xlabel("")
            axes[row_idx, 1].set_xlabel("")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.985))

    fig.savefig(args.output_dir / "tp2_serving_epoch_motivation.png", dpi=320, bbox_inches="tight")
    fig.savefig(args.output_dir / "tp2_serving_epoch_motivation.pdf", bbox_inches="tight")
    plt.close(fig)

    summary_rows: list[dict[str, object]] = []
    for topo in TOPOLOGY_ORDER:
        for batch in sorted(table[topo]):
            vllm = table[topo][batch]["vllm"]
            moe = table[topo][batch]["moentwine"]
            static = table[topo][batch]["ours_static"]
            dynamic = table[topo][batch]["ours_dynamic"]
            summary_rows.append(
                {
                    "topology": topo,
                    "batch_size": batch,
                    "vllm_us": f"{vllm:.6f}",
                    "moentwine_us": f"{moe:.6f}",
                    "static_us": f"{static:.6f}",
                    "dynamic_us": f"{dynamic:.6f}",
                    "dynamic_minus_static_us": f"{static - dynamic:.6f}",
                    "dynamic_speedup_over_static": f"{static / max(dynamic, 1e-9):.6f}",
                    "dynamic_reduction_vs_static_pct": f"{100.0 * (static - dynamic) / max(static, 1e-9):.6f}",
                    "static_speedup_over_vllm": f"{vllm / max(static, 1e-9):.6f}",
                    "dynamic_speedup_over_vllm": f"{vllm / max(dynamic, 1e-9):.6f}",
                }
            )

    write_rows(args.output_dir / "tp2_serving_epoch_motivation.csv", summary_rows)

    readme = "\n".join(
        [
            "# TP2 Serving-Epoch Motivation",
            "",
            "Generated from the current TP2 serving-epoch paper pack.",
            "",
            "- Scope: batch-per-die 16/32/64/128 under serving-epoch placement refresh.",
            "- Left panels: absolute single-layer MoE latency for vLLM, MoEntwine, HACMO static, and HACMO dynamic.",
            "- Right panels: latency saved by HACMO dynamic over HACMO static.",
            "- Data source: `paper_figures/paper_figure_summary.csv` from `tp2_serving_epoch_paper_pack_20260831`.",
            "",
        ]
    )
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
