#!/usr/bin/env python3
"""Rebuild the Figure 9--12 audit plots and paper-level claim summary."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Claim:
    figure: int
    claim_id: str
    description: str
    formula: str
    samples: int
    computed_value: float
    unit: str
    paper_value: float
    tolerance: float

    @property
    def absolute_difference(self) -> float:
        return abs(self.computed_value - self.paper_value)

    @property
    def status(self) -> str:
        return "PASS" if self.absolute_difference <= self.tolerance else "FAIL"


BASELINE_ORDER = ["dgx", "nvl72", "ls_vllm", "moentwine", "waferllm", "hacmo"]
BASELINE_LABELS = {
    "dgx": "vLLM+DGX",
    "nvl72": "vLLM+NVL72",
    "ls_vllm": "vLLM+LS",
    "moentwine": "MoEnTwine",
    "waferllm": "WaferLLM+LS",
    "hacmo": "HACMO",
}
COLORS = {
    "dgx": "#f4a261",
    "nvl72": "#e9869f",
    "ls_vllm": "#8ecae6",
    "moentwine": "#b7b7b7",
    "waferllm": "#f6c177",
    "hacmo": "#944ca2",
}
HATCHES = {
    "dgx": "///",
    "nvl72": "xx",
    "ls_vllm": "---",
    "moentwine": "xx",
    "waferllm": "\\\\",
    "hacmo": "",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp1-csv", required=True)
    parser.add_argument("--tp2-csv", required=True)
    parser.add_argument("--tp3-csv", required=True)
    parser.add_argument("--overall-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def grouped(rows: Iterable[dict[str, str]], keys: tuple[str, ...]) -> dict[tuple[str, ...], list[dict[str, str]]]:
    out: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        out[tuple(row[key] for key in keys)].append(row)
    return dict(out)


def one(rows: Iterable[dict[str, str]], field: str, value: str) -> dict[str, str]:
    matches = [row for row in rows if row[field] == value]
    if len(matches) != 1:
        raise ValueError(f"expected one row with {field}={value}, found {len(matches)}")
    return matches[0]


def ratio_pairs(
    rows: list[dict[str, str]],
    keys: tuple[str, ...],
    method_field: str,
    baseline: str,
    candidate: str,
    value_field: str,
) -> list[float]:
    ratios: list[float] = []
    for key, group in grouped(rows, keys).items():
        base = float(one(group, method_field, baseline)[value_field])
        current = float(one(group, method_field, candidate)[value_field])
        if base <= 0 or current <= 0:
            raise ValueError(f"non-positive metric for {key}")
        ratios.append(current / base)
    return ratios


def compute_claims(
    tp1: list[dict[str, str]],
    tp2: list[dict[str, str]],
    tp3: list[dict[str, str]],
    overall: list[dict[str, str]],
) -> list[Claim]:
    tp1_static = [float(row["S_us"]) / float(row["C_us"]) for row in tp1]
    tp1_full = [float(row["O_us"]) / float(row["C_us"]) for row in tp1]
    if len(tp1) != 16:
        raise ValueError(f"expected 16 TP1 context/array/batch groups, found {len(tp1)}")
    tp1_speedups = [1.0 / value for value in tp1_full]

    tp2_keys = ("topology", "batch_size")
    tp2_dynamic = ratio_pairs(tp2, tp2_keys, "method", "vllm", "ours_dynamic", "total_us")

    tp3_keys = ("topology", "batch_size")
    tp3_full_naive = ratio_pairs(tp3, tp3_keys, "bar_key", "naive_rail", "full_tp2", "total_us")
    tp3_full_rail = ratio_pairs(tp3, tp3_keys, "bar_key", "rail_tp2", "full_tp2", "total_us")

    model_rows = [row for row in overall if row["scenario"] == "model_compare"]
    vs_dgx: list[float] = []
    vs_ls: list[float] = []
    for _, group in grouped(model_rows, ("model_key",)).items():
        hacmo = float(one(group, "baseline_key", "hacmo")["throughput_tps"])
        vs_dgx.append(hacmo / float(one(group, "baseline_key", "dgx")["throughput_tps"]))
        vs_ls.append(hacmo / float(one(group, "baseline_key", "ls_vllm")["throughput_tps"]))

    return [
        Claim(9, "overall_vs_dgx_max", "Maximum HACMO throughput speedup over vLLM+DGX", "max(HACMO_TPS / DGX_TPS) over models", len(vs_dgx), max(vs_dgx), "x", 8.90, 0.445),
        Claim(9, "overall_vs_ls_max", "Maximum HACMO software speedup on the same layered supernode", "max(HACMO_TPS / vLLM+LS_TPS) over models", len(vs_ls), max(vs_ls), "x", 2.03, 0.1015),
        Claim(10, "tp1_full_reduction_max", "Maximum sparse-attention latency reduction from the full executor", "max((1 - O/C) * 100) over 16 groups", len(tp1_full), max((1.0 - value) * 100.0 for value in tp1_full), "%", 63.5, 3.175),
        Claim(10, "tp1_static_reduction_avg", "Average sparse-attention latency reduction from static mapping", "(1 - mean(S/C)) * 100 over 16 groups", len(tp1_static), (1.0 - mean(tp1_static)) * 100.0, "%", 18.9, 0.945),
        Claim(10, "tp1_full_reduction_avg", "Average sparse-attention latency reduction from the full executor", "(1 - mean(O/C)) * 100 over 16 groups", len(tp1_full), (1.0 - mean(tp1_full)) * 100.0, "%", 53.9, 2.695),
        Claim(10, "tp1_full_speedup_avg", "Average sparse-attention speedup over the normalized naive baseline", "mean(C/O) over 16 groups", len(tp1_speedups), mean(tp1_speedups), "x", 2.21, 0.1105),
        Claim(11, "tp2_full_reduction_max", "Maximum MoE latency reduction from static+dynamic mapping", "max((1 - dynamic/vLLM) * 100)", len(tp2_dynamic), max((1.0 - value) * 100.0 for value in tp2_dynamic), "%", 44.9, 2.245),
        Claim(11, "tp2_full_reduction_avg", "Average MoE latency reduction from static+dynamic mapping", "mean((1 - dynamic/vLLM) * 100)", len(tp2_dynamic), mean((1.0 - value) * 100.0 for value in tp2_dynamic), "%", 33.4, 1.67),
        Claim(12, "tp3_full_reduction_max", "Maximum MoE latency reduction of the full co-design", "max((1 - Full/Rail+Naive) * 100)", len(tp3_full_naive), max((1.0 - value) * 100.0 for value in tp3_full_naive), "%", 49.3, 2.465),
        Claim(12, "tp3_full_reduction_avg", "Average MoE latency reduction of the full co-design", "mean((1 - Full/Rail+Naive) * 100)", len(tp3_full_naive), mean((1.0 - value) * 100.0 for value in tp3_full_naive), "%", 41.1, 2.055),
        Claim(12, "tp3_network_reduction_max", "Maximum additional reduction from crossbar and bandwidth allocation", "max((1 - Full/Rail+TP2) * 100)", len(tp3_full_rail), max((1.0 - value) * 100.0 for value in tp3_full_rail), "%", 19.0, 0.95),
        Claim(12, "tp3_network_reduction_avg", "Average additional reduction from crossbar and bandwidth allocation", "mean((1 - Full/Rail+TP2) * 100)", len(tp3_full_rail), mean((1.0 - value) * 100.0 for value in tp3_full_rail), "%", 11.1, 0.555),
    ]


def write_claims(claims: list[Claim], output_dir: Path) -> None:
    rows = []
    for claim in claims:
        row = asdict(claim)
        row["absolute_difference"] = claim.absolute_difference
        row["status"] = claim.status
        rows.append(row)

    csv_path = output_dir / "paper_claims.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "paper_claims.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    for claim in claims:
        print(
            f"{claim.status}: Figure {claim.figure} {claim.claim_id}: "
            f"computed={claim.computed_value:.6f}{claim.unit}, "
            f"paper={claim.paper_value:g}{claim.unit}, "
            f"difference={claim.absolute_difference:.6f}{claim.unit}"
        )


def style_axis(ax: plt.Axes, ylabel: str, title: str) -> None:
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)


def grouped_bars(
    ax: plt.Axes,
    categories: list[str],
    series: list[str],
    values: dict[tuple[str, str], float],
    labels: dict[str, str],
    colors: dict[str, str],
    hatches: dict[str, str] | None = None,
) -> None:
    x = np.arange(len(categories), dtype=float)
    width = 0.82 / max(len(series), 1)
    for index, item in enumerate(series):
        y = [values[(category, item)] for category in categories]
        ax.bar(
            x + (index - (len(series) - 1) / 2.0) * width,
            y,
            width=width,
            label=labels[item],
            color=colors[item],
            edgecolor="black",
            linewidth=0.7,
            hatch=(hatches or {}).get(item, ""),
        )
    ax.set_xticks(x, categories)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def render_figure9(rows: list[dict[str, str]], output_dir: Path) -> None:
    fig = plt.figure(figsize=(15.5, 7.5))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.05])
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    axes.append(fig.add_subplot(grid[1, :2]))
    axes.append(fig.add_subplot(grid[1, 2]))

    model_rows = [row for row in rows if row["scenario"] == "model_compare"]
    models = [item for item in ["deepseek_v32", "qwen3_235b_a22b", "llama4_maverick"] if any(row["model_key"] == item for row in model_rows)]
    model_labels = {row["model_key"]: row["model"] for row in model_rows}
    tps_values: dict[tuple[str, str], float] = {}
    for model, group in grouped(model_rows, ("model_key",)).items():
        ref = float(one(group, "baseline_key", "dgx")["throughput_tps"])
        for row in group:
            tps_values[(model[0], row["baseline_key"])] = float(row["throughput_tps"]) / ref
    grouped_bars(axes[0], models, BASELINE_ORDER, tps_values, BASELINE_LABELS, COLORS, HATCHES)
    axes[0].set_xticklabels([model_labels[item] for item in models], rotation=10, ha="right")
    style_axis(axes[0], "Normalized decode TPS", "(a) Throughput on different models")

    tpot_rows = [row for row in rows if row["scenario"] == "tpot"]
    tpot_values: dict[tuple[str, str], float] = {}
    for model, group in grouped(tpot_rows, ("model_key",)).items():
        ref = float(one(group, "baseline_key", "dgx")["tpot_ms"])
        for row in group:
            tpot_values[(model[0], row["baseline_key"])] = float(row["tpot_ms"]) / ref
    grouped_bars(axes[1], models, BASELINE_ORDER, tpot_values, BASELINE_LABELS, COLORS, HATCHES)
    axes[1].set_xticklabels([model_labels[item] for item in models], rotation=10, ha="right")
    style_axis(axes[1], "Normalized TPOT", "(b) TPOT on different models")

    context_rows = [row for row in rows if row["scenario"] == "context_tps"]
    contexts = sorted({row["context_len"] for row in context_rows}, key=int)
    context_values: dict[tuple[str, str], float] = {}
    for context, group in grouped(context_rows, ("context_len",)).items():
        ref = float(one(group, "baseline_key", "dgx")["throughput_tps"])
        for row in group:
            context_values[(context[0], row["baseline_key"])] = float(row["throughput_tps"]) / ref
    grouped_bars(axes[2], contexts, BASELINE_ORDER, context_values, BASELINE_LABELS, COLORS, HATCHES)
    axes[2].set_xticklabels([f"{int(item) // 1024}K" for item in contexts])
    style_axis(axes[2], "Normalized decode TPS", "(c) TPS on different context lengths")

    topology_rows = [row for row in rows if row["scenario"] == "topology"]
    topology_methods = ["ls_vllm", "moentwine", "waferllm", "hacmo"]
    topology_keys = sorted(
        {(row["topology"], row["batch_per_worker"]) for row in topology_rows},
        key=lambda item: (item[0], int(item[1])),
    )
    topology_categories = [f"{topology}|B{batch}" for topology, batch in topology_keys]
    topology_values: dict[tuple[str, str], float] = {}
    for (topology, batch), group in grouped(topology_rows, ("topology", "batch_per_worker")).items():
        ref = float(one(group, "baseline_key", "ls_vllm")["throughput_tps"])
        category = f"{topology}|B{batch}"
        for row in group:
            if row["baseline_key"] in topology_methods:
                topology_values[(category, row["baseline_key"])] = float(row["throughput_tps"]) / ref
    grouped_bars(axes[3], topology_categories, topology_methods, topology_values, BASELINE_LABELS, COLORS, HATCHES)
    axes[3].set_xticklabels([item.replace("ls_", "").replace("|", "\n") for item in topology_categories])
    style_axis(axes[3], "Normalized decode TPS", "(d) Throughput across supernode scales")

    breakdown = [row for row in rows if row["scenario"] == "breakdown"]
    breakdown.sort(key=lambda row: BASELINE_ORDER.index(row["baseline_key"]))
    x = np.arange(len(breakdown))
    attention = [float(row["attention_matrix_stack_ms"]) + float(row["kv_attention_stack_ms"]) for row in breakdown]
    moe = [float(row["moe_compute_stack_ms"]) + float(row["moe_comm_stack_ms"]) for row in breakdown]
    axes[4].bar(x, attention, color="#f4a261", edgecolor="black", label="Attention")
    axes[4].bar(x, moe, bottom=attention, color="#e9869f", edgecolor="black", label="MoE")
    axes[4].set_xticks(x, [BASELINE_LABELS[row["baseline_key"]] for row in breakdown], rotation=20, ha="right")
    axes[4].legend(fontsize=8)
    style_axis(axes[4], "TPOT (ms)", "(e) DeepSeek-V3.2 latency breakdown")

    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes[:4]:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig, output_dir, "figure9_overall")


def render_figure10(rows: list[dict[str, str]], output_dir: Path) -> None:
    methods = ["B", "C", "S", "O"]
    labels = {"B": "vLLM+LS", "C": "Naive Buffer", "S": "Static Mapping", "O": "Static+Dynamic"}
    colors = {"B": "#e9869f", "C": "#8ecae6", "S": "#d6b5dd", "O": "#944ca2"}
    hatches = {"B": "xx", "C": "---", "S": "..", "O": ""}
    batch_labels = ["16", "32", "64", "128+CP2"]
    arrays = ["20x20", "68x68"]
    fig, axes = plt.subplots(2, 1, figsize=(14.5, 7.2), sharey=True)
    for ax, context in zip(axes, ["8k", "16k"]):
        categories = [f"{array}|{batch}" for array in arrays for batch in batch_labels]
        values: dict[tuple[str, str], float] = {}
        for array in arrays:
            for batch in batch_labels:
                group = [row for row in rows if row["context"] == context and row["array"] == array and row["batch_label"] == batch]
                if len(group) != 1:
                    raise ValueError(f"expected one TP1 row for {context}/{array}/{batch}")
                row = group[0]
                base = float(row["C_us"])
                category = f"{array}|{batch}"
                for method in methods:
                    values[(category, method)] = float(row[f"{method}_us"]) / base
        grouped_bars(ax, categories, methods, values, labels, colors, hatches)
        ax.set_xticklabels([item.split("|")[1].replace("+", "\n") for item in categories])
        ax.axvline(3.5, color="black", linestyle="--", linewidth=0.9)
        ax.text(1.5, ax.get_ylim()[1] * 0.94, "20x20 core array", ha="center", va="top")
        ax.text(5.5, ax.get_ylim()[1] * 0.94, "68x68 core array", ha="center", va="top")
        style_axis(ax, "Normalized sparse-attention latency", f"Context length = {context.upper()}")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    for ax in axes:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    fig.legend(handles, legend_labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig, output_dir, "figure10_sparse_attention")


def render_two_topology_figure(
    rows: list[dict[str, str]],
    output_dir: Path,
    stem: str,
    method_field: str,
    methods: list[str],
    baseline: str,
    labels: dict[str, str],
    colors: dict[str, str],
    hatches: dict[str, str],
) -> None:
    topologies = [item for item in ["16w_4x4_rail_500", "4w_8x8_rail_900"] if any(row["topology"] == item for row in rows)]
    batches = sorted({row["batch_size"] for row in rows}, key=int)
    categories = [f"{topology}|B{batch}" for topology in topologies for batch in batches]
    values: dict[tuple[str, str], float] = {}
    for (topology, batch), group in grouped(rows, ("topology", "batch_size")).items():
        base = float(one(group, method_field, baseline)["total_us"])
        category = f"{topology}|B{batch}"
        for row in group:
            if row[method_field] in methods:
                values[(category, row[method_field])] = float(row["total_us"]) / base

    fig, ax = plt.subplots(figsize=(13.5, 4.2))
    grouped_bars(ax, categories, methods, values, labels, colors, hatches)
    ax.set_xticklabels([item.split("|")[1] for item in categories])
    ax.set_ylim(0.0, 1.12)
    if len(topologies) == 2:
        split = len(batches) - 0.5
        ax.axvline(split, color="black", linestyle="--", linewidth=0.9)
        ax.text((len(batches) - 1) / 2, 1.075, "16x(4x4) system", ha="center", va="center")
        ax.text(len(batches) + (len(batches) - 1) / 2, 1.075, "4x(8x8) system", ha="center", va="center")
    style_axis(ax, "Normalized MoE latency", "")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=len(methods), frameon=False)
    fig.tight_layout()
    save_figure(fig, output_dir, stem)


def render_figure11(rows: list[dict[str, str]], output_dir: Path) -> None:
    methods = ["vllm", "moentwine", "ours_static", "ours_dynamic"]
    render_two_topology_figure(
        rows,
        output_dir,
        "figure11_moe_mapping",
        "method",
        methods,
        "vllm",
        {"vllm": "Naive mapping", "moentwine": "MoEnTwine", "ours_static": "Static Mapping", "ours_dynamic": "Static+Dynamic"},
        {"vllm": "#f4a261", "moentwine": "#e9869f", "ours_static": "#8ecae6", "ours_dynamic": "#944ca2"},
        {"vllm": "///", "moentwine": "xx", "ours_static": "---", "ours_dynamic": ""},
    )


def render_figure12(rows: list[dict[str, str]], output_dir: Path) -> None:
    methods = ["naive_rail", "rail_tp2", "crossbar_tp2", "full_tp2"]
    render_two_topology_figure(
        rows,
        output_dir,
        "figure12_network",
        "bar_key",
        methods,
        "naive_rail",
        {"naive_rail": "Rail+Naive", "rail_tp2": "Rail+Mapping", "crossbar_tp2": "Crossbar", "full_tp2": "Crossbar+BW Allocation"},
        {"naive_rail": "#f4a261", "rail_tp2": "#e9869f", "crossbar_tp2": "#8ecae6", "full_tp2": "#944ca2"},
        {"naive_rail": "///", "rail_tp2": "xx", "crossbar_tp2": "---", "full_tp2": ""},
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tp1 = read_csv(Path(args.tp1_csv))
    tp2 = read_csv(Path(args.tp2_csv))
    tp3 = read_csv(Path(args.tp3_csv))
    overall = read_csv(Path(args.overall_csv))

    claims = compute_claims(tp1, tp2, tp3, overall)
    write_claims(claims, output_dir)
    render_figure9(overall, output_dir)
    render_figure10(tp1, output_dir)
    render_figure11(tp2, output_dir)
    render_figure12(tp3, output_dir)

    failures = [claim.claim_id for claim in claims if claim.status != "PASS"]
    if failures:
        raise SystemExit("paper claim mismatch: " + ", ".join(failures))
    print(f"PASS: Figure 9--12 audit artifacts written to {output_dir}")


if __name__ == "__main__":
    main()
