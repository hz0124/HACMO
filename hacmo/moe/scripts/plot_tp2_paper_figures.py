#!/usr/bin/env python3
"""Generate paper-facing TP2 ASTRA-backed figures."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def read_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        with path.open(encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _topology_rows(rows: list[dict], topology: str) -> list[dict]:
    return [r for r in rows if r["topology"] == topology and r["method"] in METHODS]


def _batches(rows: list[dict]) -> list[int]:
    return sorted({int(r["batch_size"]) for r in rows})


def _batch_axis_label(rows: list[dict]) -> str:
    scopes = {r.get("batch_scope", "global") for r in rows}
    if scopes == {"per_die"}:
        return "Batch per die"
    if scopes == {"global"}:
        return "Global batch size"
    return "Batch size"


def _get(rows: list[dict], method: str, batch: int, key: str) -> float:
    for r in rows:
        if r["method"] == method and int(r["batch_size"]) == batch:
            return _metric(r, key)
    return 0.0


def _metric(row: dict, key: str) -> float:
    aliases = {
        "total_us": ("total_us", "astra_total_us"),
        "compute_us": ("compute_us", "astra_compute_us"),
        "communication_us": ("communication_us", "astra_communication_us"),
    }
    for candidate in aliases.get(key, (key,)):
        value = row.get(candidate)
        if value not in (None, ""):
            return float(value)
    raise KeyError(f"Missing metric {key!r}; available columns: {sorted(row.keys())}")


def _metric_source(rows: list[dict]) -> str:
    if not rows:
        return "unknown"
    keys = rows[0].keys()
    if "total_us" in keys:
        return "total_us"
    if "astra_total_us" in keys:
        return "astra_total_us"
    return "unknown"


def plot_absolute(rows: list[dict], topology: str, out_dir: Path) -> None:
    sub = _topology_rows(rows, topology)
    batches = _batches(sub)
    x = list(range(len(batches)))
    width = 0.18
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for i, method in enumerate(METHODS):
        vals = [_get(sub, method, bs, "total_us") for bs in batches]
        offs = [v + (i - 1.5) * width for v in x]
        ax.bar(offs, vals, width=width, label=METHOD_LABELS[method], color=METHOD_COLORS[method])
    ax.set_title(f"Single-layer MoE latency - {topology}")
    ax.set_xlabel(_batch_axis_label(sub))
    ax.set_ylabel("Latency (us)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in batches])
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=4)
    fig.tight_layout()
    fig.savefig(out_dir / f"paper_absolute_latency_{topology}.png", dpi=300)
    fig.savefig(out_dir / f"paper_absolute_latency_{topology}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_breakdown(rows: list[dict], topology: str, out_dir: Path) -> None:
    sub = _topology_rows(rows, topology)
    batches = _batches(sub)
    x = list(range(len(batches)))
    width = 0.18
    fig, ax = plt.subplots(figsize=(7.8, 3.8))
    for i, method in enumerate(METHODS):
        comm = [_get(sub, method, bs, "communication_us") for bs in batches]
        comp = [_get(sub, method, bs, "compute_us") for bs in batches]
        offs = [v + (i - 1.5) * width for v in x]
        ax.bar(offs, comm, width=width, color=METHOD_COLORS[method], alpha=0.78, label=METHOD_LABELS[method])
        ax.bar(offs, comp, width=width, bottom=comm, color=METHOD_COLORS[method], alpha=0.30)
    ax.set_title(f"Latency breakdown - {topology}")
    ax.set_xlabel(_batch_axis_label(sub))
    ax.set_ylabel("Latency (us)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in batches])
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=4)
    ax.text(
        0.99,
        0.98,
        "solid: ASTRA comm\nlight: H200 compute",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(out_dir / f"paper_latency_breakdown_{topology}.png", dpi=300)
    fig.savefig(out_dir / f"paper_latency_breakdown_{topology}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_normalized(rows: list[dict], topology: str, out_dir: Path) -> None:
    sub = _topology_rows(rows, topology)
    batches = _batches(sub)
    x = list(range(len(batches)))
    width = 0.18
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for i, method in enumerate(METHODS):
        vals = []
        for bs in batches:
            base = _get(sub, "vllm", bs, "total_us")
            cur = _get(sub, method, bs, "total_us")
            vals.append(cur / max(base, 1e-9))
        offs = [v + (i - 1.5) * width for v in x]
        ax.bar(offs, vals, width=width, label=METHOD_LABELS[method], color=METHOD_COLORS[method])
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title(f"Normalized MoE latency - {topology}")
    ax.set_xlabel(_batch_axis_label(sub))
    ax.set_ylabel("Normalized to vLLM")
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in batches])
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=4)
    fig.tight_layout()
    fig.savefig(out_dir / f"paper_normalized_latency_{topology}.png", dpi=300)
    fig.savefig(out_dir / f"paper_normalized_latency_{topology}.pdf", bbox_inches="tight")
    plt.close(fig)


def make_summary(rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    metric_source = _metric_source(rows)
    for topology in sorted({r["topology"] for r in rows}):
        sub = _topology_rows(rows, topology)
        for bs in _batches(sub):
            base = _get(sub, "vllm", bs, "total_us")
            for method in METHODS:
                total = _get(sub, method, bs, "total_us")
                out.append({
                    "topology": topology,
                    "batch_size": bs,
                    "batch_scope": next((r.get("batch_scope", "global") for r in sub if int(r["batch_size"]) == bs), "global"),
                    "batch_per_die": next((r.get("batch_per_die", "") for r in sub if int(r["batch_size"]) == bs), ""),
                    "global_batch": next((r.get("global_batch", "") for r in sub if int(r["batch_size"]) == bs), ""),
                    "method": method,
                    "total_us": f"{total:.6f}",
                    "compute_us": f"{_get(sub, method, bs, 'compute_us'):.6f}",
                    "communication_us": f"{_get(sub, method, bs, 'communication_us'):.6f}",
                    "normalized_to_vllm": f"{total / max(base, 1e-9):.6f}",
                    "speedup_over_vllm": f"{base / max(total, 1e-9):.6f}",
                    "metric_source": metric_source,
                })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot paper-facing TP2 figures from ASTRA-backed metrics.")
    parser.add_argument("--metrics-csv", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    rows = read_rows([Path(p) for p in args.metrics_csv])
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for topology in sorted({r["topology"] for r in rows}):
        plot_absolute(rows, topology, out_dir)
        plot_breakdown(rows, topology, out_dir)
        plot_normalized(rows, topology, out_dir)
    write_csv(out_dir / "paper_figure_summary.csv", make_summary(rows))
    print(f"Wrote paper TP2 figures to {out_dir}")


if __name__ == "__main__":
    main()
