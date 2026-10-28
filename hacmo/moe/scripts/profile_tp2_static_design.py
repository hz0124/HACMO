#!/usr/bin/env python3
"""Profile TP2 static/dynamic MoE mappings and generate diagnostic plots."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parents[1]
CORE_DIR = PROJECT_DIR / "tp2_core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from utils_topology import load_topology


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
def read_csv(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def traffic_matrix_path(
    artifact_topo_dir: Path, method: str, bs: int, seed: int
) -> Path:
    return (
        artifact_topo_dir
        / "traffic_matrices"
        / f"seed{seed}"
        / f"{method}_bpd{bs}.json"
    )


def route_plan_path(
    artifact_topo_dir: Path, method: str, bs: int, seed: int
) -> Path:
    return (
        artifact_topo_dir
        / "route_plans"
        / f"seed{seed}"
        / f"{method}_bpd{bs}.json"
    )


def add_path_load(topo, loads: dict, src: int, dst: int, nbytes: int) -> None:
    for edge in topo.get_path(src, dst):
        loads[edge] += nbytes


def max_by_kind(loads: dict, kind: str) -> int:
    vals = [v for edge, v in loads.items() if edge[0] == kind]
    return max(vals) if vals else 0


def sum_by_kind(loads: dict, kind: str) -> int:
    return sum(v for edge, v in loads.items() if edge[0] == kind)


def util(max_bytes: float, phase_us: float, bw_gbps: float) -> float:
    if phase_us <= 0 or bw_gbps <= 0:
        return 0.0
    return max_bytes / (phase_us * 1000.0 * bw_gbps)


def phase_summary(
    artifact_topo_dir: Path, method: str, bs: int, seeds: List[int]
) -> dict:
    summaries = [
        load_json(traffic_matrix_path(artifact_topo_dir, method, bs, seed)).get(
            "summary", {}
        )
        for seed in seeds
    ]
    keys = {
        "avg_step_dispatch_us",
        "avg_step_compute_us",
        "avg_step_combine_us",
        "avg_step_total_us",
    }
    return {
        key: sum(float(summary.get(key, 0.0)) for summary in summaries)
        / max(len(summaries), 1)
        for key in keys
    }


def astra_metrics(astra_root: Path, topology: str) -> dict:
    path = astra_root / f"metrics_{topology}.csv"
    if not path.exists():
        path = (
            astra_root
            / topology
            / "supernode_astra_congestion_aware"
            / f"metrics_{topology}.csv"
        )
    if path.exists():
        rows = read_csv(path)
    else:
        topo_short = "4w" if topology.startswith("4w_") else "16w" if topology.startswith("16w_") else topology
        path = astra_root / "csv" / f"tp2_serving_epoch_astra_{topo_short}_comm_only.csv"
        rows = read_csv(path) if path.exists() else []
    return {
        (r["method"], int(float(r["batch_size"]))): r
        for r in rows
    }


def profile_method(
    *,
    artifact_topo_dir: Path,
    astra_rows: dict,
    topology: str,
    network_config: Path,
    method: str,
    batch_size: int,
    seeds: List[int],
) -> tuple[dict, List[List[float]]]:
    topo = load_topology(network_config)
    dpw = topo.cfg.dies_per_wafer
    num_wafers = topo.cfg.num_wafers

    total_steps = 0
    dispatch_payload = 0
    combine_payload = 0
    dispatch_cross = 0
    dispatch_same = 0
    combine_cross = 0
    combine_same = 0
    dispatch_mesh_link_bytes = 0
    dispatch_switch_link_bytes = 0
    combine_mesh_link_bytes = 0
    combine_switch_link_bytes = 0
    max_dispatch_mesh_per_step = 0.0
    max_dispatch_switch_per_step = 0.0
    max_combine_mesh_per_step = 0.0
    max_combine_switch_per_step = 0.0
    max_rank_recv_per_step = 0.0
    dispatch_message_count = 0.0
    dispatch_cross_message_count = 0.0
    dispatch_same_message_count = 0.0
    max_rank_send_messages_per_step = 0.0
    max_rank_recv_messages_per_step = 0.0
    wafer_matrix = [[0.0 for _ in range(num_wafers)] for _ in range(num_wafers)]

    for seed in seeds:
        route_plan = load_json(
            route_plan_path(artifact_topo_dir, method, batch_size, seed)
        )
        for step in route_plan.get("steps", []):
            if not bool(step.get("active", True)):
                continue
            total_steps += 1
            dispatch_loads = defaultdict(int)
            combine_loads = defaultdict(int)
            rank_recv = defaultdict(int)
            rank_send_messages = defaultdict(int)
            rank_recv_messages = defaultdict(int)

            for src_str, dsts in step.get("dispatch_bytes", {}).items():
                src = int(src_str)
                for dst_str, raw_bytes in dsts.items():
                    dst = int(dst_str)
                    nbytes = int(raw_bytes)
                    if nbytes <= 0:
                        continue
                    src_w = src // dpw
                    dst_w = dst // dpw
                    dispatch_payload += nbytes
                    wafer_matrix[src_w][dst_w] += nbytes
                    rank_recv[dst] += nbytes
                    dispatch_message_count += 1.0
                    rank_send_messages[src] += 1
                    rank_recv_messages[dst] += 1
                    if src_w == dst_w:
                        dispatch_same += nbytes
                        dispatch_same_message_count += 1.0
                    else:
                        dispatch_cross += nbytes
                        dispatch_cross_message_count += 1.0
                    add_path_load(topo, dispatch_loads, src, dst, nbytes)

            for src_str, dsts in step.get("combine_bytes", {}).items():
                src = int(src_str)
                for dst_str, raw_bytes in dsts.items():
                    dst = int(dst_str)
                    nbytes = int(raw_bytes)
                    if nbytes <= 0:
                        continue
                    combine_payload += nbytes
                    if src // dpw == dst // dpw:
                        combine_same += nbytes
                    else:
                        combine_cross += nbytes
                    add_path_load(topo, combine_loads, src, dst, nbytes)

            dispatch_mesh_link_bytes += sum_by_kind(dispatch_loads, "mesh")
            dispatch_switch_link_bytes += sum_by_kind(dispatch_loads, "switch")
            combine_mesh_link_bytes += sum_by_kind(combine_loads, "mesh")
            combine_switch_link_bytes += sum_by_kind(combine_loads, "switch")
            max_dispatch_mesh_per_step = max(
                max_dispatch_mesh_per_step, max_by_kind(dispatch_loads, "mesh")
            )
            max_dispatch_switch_per_step = max(
                max_dispatch_switch_per_step, max_by_kind(dispatch_loads, "switch")
            )
            max_combine_mesh_per_step = max(
                max_combine_mesh_per_step, max_by_kind(combine_loads, "mesh")
            )
            max_combine_switch_per_step = max(
                max_combine_switch_per_step, max_by_kind(combine_loads, "switch")
            )
            if rank_recv:
                max_rank_recv_per_step = max(
                    max_rank_recv_per_step, max(rank_recv.values())
                )
            if rank_send_messages:
                max_rank_send_messages_per_step = max(
                    max_rank_send_messages_per_step, max(rank_send_messages.values())
                )
            if rank_recv_messages:
                max_rank_recv_messages_per_step = max(
                    max_rank_recv_messages_per_step, max(rank_recv_messages.values())
                )

    total_steps = max(total_steps, 1)
    wafer_matrix = [
        [value / total_steps for value in row]
        for row in wafer_matrix
    ]
    phases = phase_summary(artifact_topo_dir, method, batch_size, seeds)
    dispatch_us = float(phases.get("avg_step_dispatch_us", 0.0))
    combine_us = float(phases.get("avg_step_combine_us", 0.0))
    compute_us = float(phases.get("avg_step_compute_us", 0.0))
    analytical_total_us = float(phases.get("avg_step_total_us", 0.0))
    astra = astra_rows.get((method, batch_size), {})
    row = {
        "topology": topology,
        "method": method,
        "batch_size": batch_size,
        "dispatch_payload_bytes_per_step": dispatch_payload / total_steps,
        "combine_payload_bytes_per_step": combine_payload / total_steps,
        "dispatch_cross_bytes_per_step": dispatch_cross / total_steps,
        "dispatch_same_bytes_per_step": dispatch_same / total_steps,
        "combine_cross_bytes_per_step": combine_cross / total_steps,
        "combine_same_bytes_per_step": combine_same / total_steps,
        "dispatch_cross_ratio": dispatch_cross / max(dispatch_payload, 1),
        "combine_cross_ratio": combine_cross / max(combine_payload, 1),
        "dispatch_mesh_link_bytes_per_step": dispatch_mesh_link_bytes / total_steps,
        "dispatch_switch_link_bytes_per_step": dispatch_switch_link_bytes / total_steps,
        "combine_mesh_link_bytes_per_step": combine_mesh_link_bytes / total_steps,
        "combine_switch_link_bytes_per_step": combine_switch_link_bytes / total_steps,
        "max_dispatch_mesh_link_bytes_per_step": max_dispatch_mesh_per_step,
        "max_dispatch_switch_link_bytes_per_step": max_dispatch_switch_per_step,
        "max_combine_mesh_link_bytes_per_step": max_combine_mesh_per_step,
        "max_combine_switch_link_bytes_per_step": max_combine_switch_per_step,
        "max_rank_recv_bytes_per_step": max_rank_recv_per_step,
        "dispatch_message_count_per_step": dispatch_message_count / total_steps,
        "dispatch_cross_message_count_per_step": dispatch_cross_message_count / total_steps,
        "dispatch_same_message_count_per_step": dispatch_same_message_count / total_steps,
        "max_rank_send_messages_per_step": max_rank_send_messages_per_step,
        "max_rank_recv_messages_per_step": max_rank_recv_messages_per_step,
        "dispatch_mesh_util": util(max_dispatch_mesh_per_step, dispatch_us, topo.cfg.mesh_bw_gbps),
        "dispatch_switch_util": util(max_dispatch_switch_per_step, dispatch_us, topo.cfg.switch_bw_gbps),
        "combine_mesh_util": util(max_combine_mesh_per_step, combine_us, topo.cfg.mesh_bw_gbps),
        "combine_switch_util": util(max_combine_switch_per_step, combine_us, topo.cfg.switch_bw_gbps),
        "analytical_dispatch_us": dispatch_us,
        "analytical_compute_us": compute_us,
        "analytical_combine_us": combine_us,
        "analytical_total_us": analytical_total_us,
        "astra_total_us": float(astra.get("total_us", 0.0) or 0.0),
        "astra_compute_us": float(astra.get("compute_us", 0.0) or 0.0),
        "astra_communication_us": float(astra.get("communication_us", 0.0) or 0.0),
    }
    return row, wafer_matrix


def trace_profile(artifact_topo_dir: Path, network_config: Path, output_dir: Path, topk: int) -> dict:
    topo = load_topology(network_config)
    dpw = topo.cfg.dies_per_wafer
    num_wafers = topo.cfg.num_wafers
    window_src = load_json(artifact_topo_dir / "mapping_inputs/runtime/window_src_expert_counts.json")
    expert_counts = Counter()
    wafer_expert = defaultdict(Counter)
    src_tokens = Counter()
    for wdata in window_src:
        for src_str, e_counts in wdata["src_expert_counts"].items():
            src = int(src_str)
            sw = src // dpw
            src_tokens[sw] += 1
            for e_str, c in e_counts.items():
                e = int(e_str)
                cnt = int(c)
                expert_counts[e] += cnt
                wafer_expert[sw][e] += cnt

    total = sum(expert_counts.values()) or 1
    top = [e for e, _ in expert_counts.most_common(topk)]
    matrix = [[wafer_expert[w][e] for e in top] for w in range(num_wafers)]
    affinities = []
    for e, cnt in expert_counts.items():
        by_w = [wafer_expert[w][e] for w in range(num_wafers)]
        affinities.append(max(by_w) / max(cnt, 1))
    top10_share = sum(c for _, c in expert_counts.most_common(10)) / total
    values = list(expert_counts.values())
    mean = total / max(len(values), 1)
    cv = (math.sqrt(sum((x - mean) ** 2 for x in values) / max(len(values), 1)) / mean) if mean else 0.0

    plot_heatmap(
        matrix,
        xlabels=[str(e) for e in top],
        ylabels=[str(i) for i in range(num_wafers)],
        title=f"{artifact_topo_dir.name}: source-wafer demand for top experts",
        cbar_label="activations",
        path=output_dir / f"trace_heatmap_{artifact_topo_dir.name}.png",
    )
    plot_expert_popularity(expert_counts, output_dir / f"expert_popularity_{artifact_topo_dir.name}.png")
    return {
        "topology": artifact_topo_dir.name,
        "num_active_experts": len(expert_counts),
        "total_expert_activations": total,
        "top1_share": expert_counts.most_common(1)[0][1] / total if expert_counts else 0.0,
        "top10_share": top10_share,
        "expert_count_cv": cv,
        "weighted_avg_source_wafer_affinity": sum(
            expert_counts[e] * (max(wafer_expert[w][e] for w in range(num_wafers)) / max(expert_counts[e], 1))
            for e in expert_counts
        ) / total,
    }


def plot_heatmap(matrix: List[List[float]], xlabels: List[str], ylabels: List[str], title: str, cbar_label: str, path: Path) -> None:
    fig_w = max(6.0, min(14.0, len(xlabels) * 0.35))
    fig_h = max(3.0, len(ylabels) * 0.35 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Expert ID")
    ax.set_ylabel("Source wafer")
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_expert_popularity(counts: Counter, path: Path) -> None:
    vals = [c for _, c in counts.most_common()]
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.plot(range(1, len(vals) + 1), vals, marker=".", linewidth=1.2)
    ax.set_xlabel("Expert rank")
    ax.set_ylabel("Activations")
    ax.set_title("Runtime expert popularity")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_cross_ratio(rows: List[dict], output_dir: Path) -> None:
    topologies = sorted({r["topology"] for r in rows})
    fig, axes = plt.subplots(1, len(topologies), figsize=(5.2 * len(topologies), 3.4), sharey=True)
    if len(topologies) == 1:
        axes = [axes]
    for ax, topo in zip(axes, topologies):
        for method in METHODS:
            sub = sorted([r for r in rows if r["topology"] == topo and r["method"] == method], key=lambda x: int(x["batch_size"]))
            ax.plot(
                [int(r["batch_size"]) for r in sub],
                [float(r["dispatch_cross_ratio"]) for r in sub],
                marker="o",
                label=METHOD_LABELS[method],
                color=METHOD_COLORS[method],
            )
        ax.set_title(topo)
        ax.set_xlabel("Batch size")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Dispatch cross-wafer ratio")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "cross_wafer_ratio_vs_batch.png", dpi=240)
    plt.close(fig)


def plot_cross_bytes(rows: List[dict], output_dir: Path) -> None:
    topologies = sorted({r["topology"] for r in rows})
    fig, axes = plt.subplots(1, len(topologies), figsize=(5.2 * len(topologies), 3.4), sharey=False)
    if len(topologies) == 1:
        axes = [axes]
    for ax, topo in zip(axes, topologies):
        for method in METHODS:
            sub = sorted([r for r in rows if r["topology"] == topo and r["method"] == method], key=lambda x: int(x["batch_size"]))
            ax.plot(
                [int(r["batch_size"]) for r in sub],
                [float(r["dispatch_cross_bytes_per_step"]) / 1e6 for r in sub],
                marker="o",
                label=METHOD_LABELS[method],
                color=METHOD_COLORS[method],
            )
        ax.set_title(topo)
        ax.set_xlabel("Batch size")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Cross-wafer dispatch MB/step")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "cross_wafer_bytes_vs_batch.png", dpi=240)
    plt.close(fig)


def plot_link_util(rows: List[dict], output_dir: Path, batch_size: int) -> None:
    selected = [r for r in rows if int(r["batch_size"]) == batch_size]
    topologies = sorted({r["topology"] for r in selected})
    util_keys = [
        ("dispatch_switch_util", "Dispatch switch"),
        ("combine_switch_util", "Combine switch"),
        ("dispatch_mesh_util", "Dispatch mesh"),
        ("combine_mesh_util", "Combine mesh"),
    ]
    fig, axes = plt.subplots(len(topologies), 1, figsize=(9.5, 3.4 * len(topologies)), sharey=True)
    if len(topologies) == 1:
        axes = [axes]
    for ax, topo in zip(axes, topologies):
        topo_rows = {r["method"]: r for r in selected if r["topology"] == topo}
        x = list(range(len(util_keys)))
        width = 0.18
        for i, method in enumerate(METHODS):
            vals = [float(topo_rows.get(method, {}).get(k, 0.0)) for k, _ in util_keys]
            offs = [v + (i - 1.5) * width for v in x]
            ax.bar(offs, vals, width=width, label=METHOD_LABELS[method], color=METHOD_COLORS[method])
        ax.set_title(f"{topo}, batch {batch_size}")
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in util_keys], rotation=15, ha="right")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Worst-link utilization")
    axes[-1].legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout()
    fig.savefig(output_dir / f"link_utilization_bs{batch_size}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_message_pressure(rows: List[dict], output_dir: Path, batch_size: int) -> None:
    selected = [r for r in rows if int(r["batch_size"]) == batch_size]
    topologies = sorted({r["topology"] for r in selected})
    keys = [
        ("dispatch_message_count_per_step", "Dispatch messages"),
        ("dispatch_cross_message_count_per_step", "Cross-wafer messages"),
        ("max_rank_send_messages_per_step", "Max rank sends"),
        ("max_rank_recv_messages_per_step", "Max rank receives"),
    ]
    fig, axes = plt.subplots(len(topologies), 1, figsize=(9.5, 3.4 * len(topologies)), sharey=False)
    if len(topologies) == 1:
        axes = [axes]
    for ax, topo in zip(axes, topologies):
        topo_rows = {r["method"]: r for r in selected if r["topology"] == topo}
        x = list(range(len(keys)))
        width = 0.18
        for i, method in enumerate(METHODS):
            vals = [float(topo_rows.get(method, {}).get(k, 0.0)) for k, _ in keys]
            offs = [v + (i - 1.5) * width for v in x]
            ax.bar(offs, vals, width=width, label=METHOD_LABELS[method], color=METHOD_COLORS[method])
        ax.set_title(f"{topo}, batch {batch_size}")
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in keys], rotation=15, ha="right")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Count per MoE step")
    axes[-1].legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout()
    fig.savefig(output_dir / f"message_pressure_bs{batch_size}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_latency_breakdown(rows: List[dict], output_dir: Path, batch_size: int) -> None:
    selected = [r for r in rows if int(r["batch_size"]) == batch_size]
    topologies = sorted({r["topology"] for r in selected})
    fig, axes = plt.subplots(1, len(topologies), figsize=(5.4 * len(topologies), 3.7), sharey=True)
    if len(topologies) == 1:
        axes = [axes]
    for ax, topo in zip(axes, topologies):
        topo_rows = [r for r in selected if r["topology"] == topo]
        xs = list(range(len(METHODS)))
        comm = [float(next((r for r in topo_rows if r["method"] == m), {}).get("astra_communication_us", 0.0)) for m in METHODS]
        comp = [float(next((r for r in topo_rows if r["method"] == m), {}).get("astra_compute_us", 0.0)) for m in METHODS]
        ax.bar(xs, comm, color="#74b9ff", label="ASTRA comm")
        ax.bar(xs, comp, bottom=comm, color="#fdcb6e", label="ASTRA non-comm")
        ax.set_xticks(xs)
        ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS], rotation=20, ha="right")
        ax.set_title(f"{topo}, batch {batch_size}")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Latency per MoE step (us)")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / f"latency_breakdown_bs{batch_size}.png", dpi=240)
    plt.close(fig)


def plot_wafer_matrices(matrices: dict, output_dir: Path, batch_size: int) -> None:
    for (topo, method, bs), matrix in matrices.items():
        if int(bs) != batch_size:
            continue
        plot_heatmap(
            matrix,
            xlabels=[str(i) for i in range(len(matrix))],
            ylabels=[str(i) for i in range(len(matrix))],
            title=f"{topo} {METHOD_LABELS[method]} dispatch wafer matrix, batch {batch_size}",
            cbar_label="bytes/step",
            path=output_dir / f"wafer_dispatch_matrix_{topo}_{method}_bs{batch_size}.png",
        )


def write_notes(rows: List[dict], trace_rows: List[dict], output_dir: Path, batch_size: int) -> None:
    lines = [
        "# TP2 Static Mapping Profile Notes",
        "",
        "This directory profiles the current TP2 artifacts with SupernodeSim/ASTRA results.",
        "Traffic metrics are computed from the same seeded route plans consumed by analytical and ASTRA evaluation.",
        "",
        f"## Batch {batch_size} Snapshot",
        "",
    ]
    for topo in sorted({r["topology"] for r in rows}):
        sub = {r["method"]: r for r in rows if r["topology"] == topo and int(r["batch_size"]) == batch_size}
        base = sub.get("vllm")
        if not base:
            continue
        lines.append(f"### {topo}")
        base_cross = float(base["dispatch_cross_bytes_per_step"])
        base_total = float(base["astra_total_us"])
        for method in METHODS:
            r = sub.get(method)
            if not r:
                continue
            cross_reduction = 1.0 - float(r["dispatch_cross_bytes_per_step"]) / max(base_cross, 1.0)
            speedup = base_total / max(float(r["astra_total_us"]), 1e-9)
            lines.append(
                f"- {METHOD_LABELS[method]}: cross-dispatch reduction vs vLLM "
                f"{cross_reduction * 100:.1f}%, ASTRA speedup {speedup:.3f}x, "
                f"switch util dispatch/combine "
                f"{float(r['dispatch_switch_util']) * 100:.1f}%/"
                f"{float(r['combine_switch_util']) * 100:.1f}%, "
                f"dispatch messages {float(r['dispatch_message_count_per_step']):.1f}/step."
            )
        lines.append("")
    lines.extend([
        "## Trace Hotness",
        "",
    ])
    for r in trace_rows:
        lines.append(
            f"- {r['topology']}: active experts {r['num_active_experts']}, "
            f"top1 share {float(r['top1_share']) * 100:.1f}%, "
            f"top10 share {float(r['top10_share']) * 100:.1f}%, "
            f"weighted source-wafer affinity {float(r['weighted_avg_source_wafer_affinity']) * 100:.1f}%."
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- Static mapping can reduce cross-wafer bytes only when the runtime trace has hot experts and source-wafer affinity.",
        "- If latency does not improve proportionally, inspect switch utilization and ASTRA communication time; fixed hop latency and synchronization can dominate over byte volume.",
        "- Bandwidth sensitivity should amplify methods that reduce cross-wafer traffic when switch bandwidth is reduced or mesh bandwidth is increased.",
        "",
    ])
    (output_dir / "PROFILE_NOTES.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Profile TP2 mappings and plot diagnostics.")
    p.add_argument("--artifact-root", required=True)
    p.add_argument("--astra-root", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--topologies", nargs="+", default=["4w_8x8_rail_900", "16w_4x4_rail_500"])
    p.add_argument("--network-configs", nargs="+", required=True)
    p.add_argument("--methods", nargs="+", default=METHODS)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[16, 32, 64, 128])
    p.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    p.add_argument("--profile-batch-size", type=int, default=64)
    p.add_argument("--trace-topk", type=int, default=32)
    args = p.parse_args()

    artifact_root = Path(args.artifact_root)
    astra_root = Path(args.astra_root)
    output_dir = Path(args.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    network_configs = {topo: Path(cfg) for topo, cfg in zip(args.topologies, args.network_configs)}

    rows = []
    matrices = {}
    trace_rows = []
    for topology in args.topologies:
        artifact_topo_dir = artifact_root / topology
        astra = astra_metrics(astra_root, topology)
        trace_rows.append(trace_profile(artifact_topo_dir, network_configs[topology], plot_dir, args.trace_topk))
        for bs in args.batch_sizes:
            for method in args.methods:
                row, matrix = profile_method(
                    artifact_topo_dir=artifact_topo_dir,
                    astra_rows=astra,
                    topology=topology,
                    network_config=network_configs[topology],
                    method=method,
                    batch_size=bs,
                    seeds=args.seeds,
                )
                rows.append(row)
                matrices[(topology, method, bs)] = matrix

    write_csv(output_dir / "profile_metrics.csv", rows)
    write_csv(output_dir / "trace_summary.csv", trace_rows)
    plot_cross_ratio(rows, plot_dir)
    plot_cross_bytes(rows, plot_dir)
    plot_link_util(rows, plot_dir, args.profile_batch_size)
    plot_message_pressure(rows, plot_dir, args.profile_batch_size)
    plot_latency_breakdown(rows, plot_dir, args.profile_batch_size)
    plot_wafer_matrices(matrices, plot_dir, args.profile_batch_size)
    write_notes(rows, trace_rows, output_dir, args.profile_batch_size)
    print(f"Wrote TP2 profile to {output_dir}")


if __name__ == "__main__":
    main()
