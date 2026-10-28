#!/usr/bin/env python3
"""Plot DSE dual-axis figures."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def plot_all(rows: list[dict[str, Any]], pareto_rows: list[dict[str, Any]], out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    panel_dir = out_dir / "plot_data"
    fig_dir.mkdir(parents=True, exist_ok=True)
    panel_dir.mkdir(parents=True, exist_ok=True)
    plot_panel(select_sram(rows), fig_dir / "fig_dse_sram_compute", "(a) SRAM / compute trade-off", right_key="sparse_hit_rate", right_label="Sparse hit rate")
    write_csv(panel_dir / "plot_data_sram_compute.csv", select_sram(rows))
    plot_panel(select_core_hbm(rows), fig_dir / "fig_dse_core_hbm", "(b) Core array / HBM trade-off", right_key="wafer_area_utilization", right_label="Area utilization")
    write_csv(panel_dir / "plot_data_core_hbm.csv", select_core_hbm(rows))
    plot_panel(select_wafer(rows), fig_dir / "fig_dse_wafer_scale", "(c) Wafer scale", right_key="moe_cross_wafer_ratio", right_label="Cross-wafer ratio")
    write_csv(panel_dir / "plot_data_wafer_scale.csv", select_wafer(rows))
    batch_rows = select_batch(rows)
    if batch_rows:
        plot_panel(batch_rows, fig_dir / "fig_dse_batch_sweep", "(d) Per-die batch sweep", right_key="sparse_hit_rate", right_label="Sparse hit rate")
        write_csv(panel_dir / "plot_data_batch_sweep.csv", batch_rows)
    core_frontier_rows = select_core_hbm_frontier(rows)
    plot_core_hbm_frontier_bars(
        core_frontier_rows,
        fig_dir / "fig_dse_core_hbm_frontier_bars",
        "Supplementary: area-constrained core / HBM frontier",
    )
    write_csv(panel_dir / "plot_data_core_hbm_frontier_bars.csv", core_frontier_rows)
    edge_rows = with_edge_comm_metrics(select_edge_bandwidth(rows))
    plot_edge_bandwidth(edge_rows, fig_dir / "fig_dse_edge_bandwidth", "Diagnostic: edge / rack bandwidth search")
    write_csv(panel_dir / "plot_data_edge_bandwidth.csv", edge_rows)
    diag_rows = with_edge_sensitivity_metrics(select_edge_sensitivity(rows))
    if diag_rows:
        plot_edge_sensitivity(diag_rows, fig_dir / "fig_dse_edge_sensitivity", "(e) Edge bandwidth / latency sensitivity")
        plot_edge_sensitivity_heatmap(
            diag_rows,
            fig_dir / "fig_dse_edge_sensitivity_crossbar",
            value_key="crossbar_comm_us",
            title="(f) Raw crossbar service sensitivity",
            cbar_label="Crossbar comm us/layer",
        )
        plot_edge_sensitivity_bandwidth_slices(
            diag_rows,
            fig_dir / "fig_dse_edge_sensitivity_bandwidth_slices",
            value_key="communication_us",
            title="(g) Communication vs bandwidth at fixed switch latency",
        )
        plot_edge_sensitivity_latency_slices(
            diag_rows,
            fig_dir / "fig_dse_edge_sensitivity_latency_slices",
            value_key="communication_us",
            title="(h) Communication vs switch latency at fixed bandwidth",
        )
        plot_edge_sensitivity_heatmap(
            diag_rows,
            fig_dir / "fig_dse_edge_sensitivity_throughput",
            value_key="throughput_tps",
            title="(i) End-to-end throughput sensitivity",
            cbar_label="Decode TPS",
            fmt=".0f",
        )
        plot_edge_sensitivity_heatmap(
            diag_rows,
            fig_dir / "fig_dse_edge_sensitivity_comm_share",
            value_key="comm_tpot_share_percent",
            title="(j) Communication share of TPOT",
            cbar_label="MoE communication / TPOT (%)",
            fmt=".1f",
        )
        plot_edge_sensitivity_comm_bars(
            diag_rows,
            fig_dir / "fig_dse_edge_sensitivity_comm_bars",
            "(k) Edge sensitivity: communication bars",
        )
        plot_edge_sensitivity_tpot_bars(
            diag_rows,
            fig_dir / "fig_dse_edge_sensitivity_tpot_bars",
            "(l) Edge sensitivity: TPOT component bars",
        )
        write_csv(panel_dir / "plot_data_edge_sensitivity.csv", diag_rows)
        write_csv(panel_dir / "plot_data_edge_sensitivity_comm_bars.csv", select_edge_bar_slices(diag_rows))
        write_csv(panel_dir / "plot_data_edge_sensitivity_tpot_bars.csv", select_edge_bar_slices(diag_rows))
    write_csv(panel_dir / "plot_data_pareto_summary.csv", select_pareto_summary(rows, pareto_rows))


def select_sram(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = ["sram_sram_small_compute_heavy", "sram_balanced_low", "sram_balanced_mid", "sram_sram_large", "sram_sram_max"]
    by_id = {r["case_id"]: r for r in rows if r["mode"] == "sram_compute"}
    return [by_id[k] for k in order if k in by_id]


def select_batch(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = [r for r in rows if r["mode"] == "batch_sweep"]
    return sorted(candidates, key=lambda r: int(r["batch_size_per_die"]))


def select_core_hbm(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = [r for r in rows if r["mode"] == "core_hbm"]
    if not candidates:
        return []
    picks: list[dict[str, Any]] = []
    for cid in ["core_hbm_a16_h2", "core_hbm_a16_h4", "core_hbm_a24_h2", "core_hbm_a32_h2", "core_hbm_a32_h4"]:
        row = next((r for r in candidates if r["case_id"] == cid), None)
        if row:
            picks.append(row)
    best_tps = max(candidates, key=lambda r: float(r["throughput_tps"]))
    best_util = max(candidates, key=lambda r: float(r["wafer_area_utilization"]))
    for row in [best_tps, best_util]:
        if row["case_id"] not in {p["case_id"] for p in picks}:
            picks.append(row)
    return picks[:7]


def select_core_hbm_frontier(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = {r["case_id"]: r for r in rows if r["mode"] == "core_hbm"}
    order = [
        "core_hbm_a16_h2",
        "core_hbm_a24_h2",
        "core_hbm_a24_h4",
        "core_hbm_a32_h2",
        "core_hbm_a32_h4",
        "core_hbm_a32_h8",
    ]
    picked = [candidates[cid] for cid in order if cid in candidates]
    if not picked:
        return select_core_hbm(rows)
    return picked


def select_wafer(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted([r for r in rows if r["mode"] == "wafer_scale"], key=lambda r: (int(r["wafer_side"]), int(r["num_wafers"])))


def select_edge_bandwidth(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = [r for r in rows if r["mode"] == "edge_bandwidth"]
    if not candidates:
        return []
    picks: list[dict[str, Any]] = []
    for cid in ["edge_bw_ext100", "edge_bw_ext250", "edge_bw_base", "edge_bw_ext700", "edge_bw_ext800", "edge_bw_ext900"]:
        row = next((r for r in candidates if r["case_id"] == cid), None)
        if row:
            picks.append(row)
    best = max(candidates, key=lambda r: float(r["throughput_tps"]))
    if best["case_id"] not in {r["case_id"] for r in picks}:
        picks.append(best)
    return picks


def select_edge_sensitivity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = [r for r in rows if r["mode"] == "edge_sensitivity"]
    return sorted(
        candidates,
        key=lambda r: (float(r.get("per_port_bw_gbps", 0.0)), float(r.get("switch_latency_ns", 0.0))),
    )


def select_edge_bar_slices(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bw_rows = select_edge_sensitivity_bandwidth_slice(rows, 10.0, [1.0, 5.0, 20.0, 100.0, 500.0])
    lat_rows = select_edge_sensitivity_latency_slice(rows, 1.0, [10.0, 50.0, 250.0, 1000.0, 2000.0])
    picked: list[dict[str, Any]] = []
    for row in bw_rows + lat_rows:
        if row["case_id"] not in {r["case_id"] for r in picked}:
            picked.append(row)
    return picked


def select_edge_sensitivity_bandwidth_slice(rows: list[dict[str, Any]], latency: float, bandwidths: list[float]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for bw in bandwidths:
        row = next(
            (
                r
                for r in rows
                if float(r.get("per_port_bw_gbps", 0.0)) == float(bw)
                and float(r.get("switch_latency_ns", 0.0)) == float(latency)
            ),
            None,
        )
        if row is not None:
            out.append(row)
    return out


def select_edge_sensitivity_latency_slice(rows: list[dict[str, Any]], bandwidth: float, latencies: list[float]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for lat in latencies:
        row = next(
            (
                r
                for r in rows
                if float(r.get("per_port_bw_gbps", 0.0)) == float(bandwidth)
                and float(r.get("switch_latency_ns", 0.0)) == float(lat)
            ),
            None,
        )
        if row is not None:
            out.append(row)
    return out


def with_edge_comm_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    base = next((r for r in rows if r["case_id"] == "edge_bw_base"), rows[0])
    base_comm = float(base["moe_comm_us"])
    out: list[dict[str, Any]] = []
    for row in rows:
        comm = max(float(row["moe_comm_us"]), 1e-12)
        copied = dict(row)
        copied["edge_comm_speedup_vs_base"] = round(base_comm / comm, 6)
        copied["edge_comm_reduction_percent_vs_base"] = round((1.0 - comm / max(base_comm, 1e-12)) * 100.0, 6)
        out.append(copied)
    return out


def with_edge_sensitivity_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        copied = dict(row)
        moe_layer_us = max(float(copied.get("moe_layer_us", 0.0)), 1e-12)
        moe_stack_ms = max(float(copied.get("moe_stack_ms", 0.0)), 0.0)
        tpot_ms = max(float(copied.get("tpot_ms", 0.0)), 1e-12)
        comm_us = float(copied.get("communication_us", copied.get("moe_comm_us", 0.0)))
        inferred_moe_layers = moe_stack_ms * 1000.0 / moe_layer_us
        copied["moe_comm_share_percent"] = round(100.0 * comm_us / moe_layer_us, 6)
        copied["comm_tpot_share_percent"] = round(100.0 * comm_us * inferred_moe_layers / (tpot_ms * 1000.0), 6)
        out.append(copied)
    return out


def select_pareto_summary(rows: list[dict[str, Any]], pareto_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    picks: list[dict[str, Any]] = []
    all_rows = list(rows)
    roles = [
        min(all_rows, key=lambda r: abs(float(r["sram_per_core_mb"]) - 2.0) + abs(float(r["compute_per_core_tflops"]) - 0.512)),
        max(all_rows, key=lambda r: float(r["throughput_tps"])),
        max(all_rows, key=lambda r: float(r.get("wafer_area_headroom", 0.0))),
        max(all_rows, key=lambda r: float(r["compute_per_core_tflops"])),
        max(all_rows, key=lambda r: float(r["sram_per_core_mb"])),
        max(all_rows, key=lambda r: int(r["num_cores"])),
        max(all_rows, key=lambda r: int(r["hbm_stacks_per_die"])),
        max(all_rows, key=lambda r: int(r["wafer_side"])),
    ]
    edge_best = max([r for r in all_rows if r["mode"] == "edge_bandwidth"], key=lambda r: float(r["throughput_tps"]), default=max(all_rows, key=lambda r: float(r["throughput_tps"])))
    roles.append(edge_best)
    for row in roles + pareto_rows:
        if row["case_id"] not in {p["case_id"] for p in picks}:
            picks.append(row)
        if len(picks) >= 7:
            break
    return picks


def plot_panel(rows: list[dict[str, Any]], prefix: Path, title: str, right_key: str, right_label: str) -> None:
    if not rows:
        return
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )
    labels = [str(r["label"]) for r in rows]
    x = list(range(len(rows)))
    tps = [float(r["normalized_throughput"]) for r in rows]
    right = [float(r.get(right_key, 0.0)) for r in rows]
    right_max_idx = right.index(max(right)) if right else 0
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    bars = ax.bar(x, tps, color="#4C78A8", width=0.62, label="Norm. TPS")
    ax.set_ylabel("Normalized throughput")
    ax.set_xticks(x, labels, rotation=22, ha="right")
    ax.set_ylim(0, max(tps) * 1.25)
    ax.grid(axis="y", alpha=0.25)
    ax2 = ax.twinx()
    if right_key in {"sparse_hit_rate", "wafer_area_utilization", "moe_cross_wafer_ratio", "edge_bandwidth_index"}:
        ax2.plot(x, right, color="#D62728", marker="o", linewidth=2.0, label=right_label)
        ax2.set_ylabel(right_label)
        ax2.set_ylim(0, max(right) * 1.25 if max(right) > 0 else 1.0)
    else:
        ax2.plot(x, right, color="#D62728", marker="o", linewidth=2.0, label=right_label)
        ax2.set_ylabel(right_label)
        ax2.set_ylim(0, max(right) * 1.25 if max(right) > 0 else 1.0)
    for idx, row in enumerate(rows):
        if row.get("is_pareto") == "1" or idx in {tps.index(max(tps)), right_max_idx}:
            ax.text(idx, tps[idx] + max(tps) * 0.035, f"{tps[idx]:.2f}x", ha="center", va="bottom", fontsize=7)
    handles = [bars, ax2.lines[0]]
    ax.legend(handles, [h.get_label() for h in handles], loc="upper right", frameon=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(prefix.with_suffix(".png"), dpi=220)
    fig.savefig(prefix.with_suffix(".pdf"))
    plt.close(fig)


def plot_edge_bandwidth(rows: list[dict[str, Any]], prefix: Path, title: str) -> None:
    if not rows:
        return
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )
    labels = [str(r["label"]) for r in rows]
    x = list(range(len(rows)))
    comm = [float(r.get("communication_us", r.get("moe_comm_us", 0.0))) for r in rows]
    bw_index = [float(r.get("edge_bandwidth_index", 0.0)) for r in rows]
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    bars = ax.bar(x, comm, color="#4C78A8", width=0.62, label="MoE comm (us/layer)")
    ax.set_ylabel("MoE communication (us/layer)")
    ax.set_xticks(x, labels, rotation=22, ha="right")
    min_comm = min(comm)
    max_comm = max(comm)
    pad = max((max_comm - min_comm) * 0.45, 0.01)
    ax.set_ylim(max(0.0, min_comm - pad), max_comm + pad)
    ax.grid(axis="y", alpha=0.25)
    ax2 = ax.twinx()
    ax2.plot(x, bw_index, color="#D62728", marker="o", linewidth=2.0, label="Bandwidth index")
    ax2.set_ylabel("Bandwidth index")
    ax2.set_ylim(0, max(bw_index) * 1.25 if max(bw_index) > 0 else 1.0)
    best_idx = comm.index(min(comm))
    for idx in {0, best_idx}:
        ax.text(idx, comm[idx] + pad * 0.08, f"{comm[idx]:.3f}us", ha="center", va="bottom", fontsize=7)
    handles = [bars, ax2.lines[0]]
    ax.legend(handles, [h.get_label() for h in handles], loc="upper right", frameon=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(prefix.with_suffix(".png"), dpi=220)
    fig.savefig(prefix.with_suffix(".pdf"))
    plt.close(fig)


def plot_core_hbm_frontier_bars(rows: list[dict[str, Any]], prefix: Path, title: str) -> None:
    if not rows:
        return
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )
    labels = [str(r["label"]) for r in rows]
    x = list(range(len(rows)))
    attn = [float(r.get("attention_stack_ms", 0.0)) for r in rows]
    moe_layers = [_infer_moe_layers(r) for r in rows]
    moe_compute = [float(r.get("moe_compute_us", 0.0)) * layers / 1000.0 for r, layers in zip(rows, moe_layers)]
    moe_comm = [
        float(r.get("communication_us", r.get("moe_comm_us", 0.0))) * layers / 1000.0
        for r, layers in zip(rows, moe_layers)
    ]
    area_util = [float(r.get("wafer_area_utilization", 0.0)) for r in rows]
    tps = [float(r.get("throughput_tps", 0.0)) / 1000.0 for r in rows]
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    bottom = [0.0 for _ in rows]
    series = [
        (attn, "#4C78A8", "Attention stack"),
        (moe_compute, "#F58518", "MoE compute"),
        (moe_comm, "#54A24B", "MoE comm"),
    ]
    for vals, color, label in series:
        ax.bar(x, vals, bottom=bottom, color=color, width=0.62, label=label)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ymax = max(bottom) if bottom else 1.0
    for idx, row in enumerate(rows):
        ax.text(
            idx,
            bottom[idx] + ymax * 0.025,
            f"{float(row.get('tpot_ms', 0.0)):.1f}ms\n{tps[idx]:.0f}k TPS",
            ha="center",
            va="bottom",
            fontsize=7,
            linespacing=1.1,
        )
    ax.set_ylabel("Decode TPOT (ms)")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylim(0, ymax * 1.28)
    ax.grid(axis="y", alpha=0.22)
    ax.set_title(title)
    ax2 = ax.twinx()
    ax2.plot(x, area_util, color="#D62728", marker="o", linewidth=2.0, label="Area utilization")
    ax2.axhline(1.0, color="#D62728", linestyle="--", linewidth=1.0, alpha=0.55, label="Area budget")
    ax2.set_ylabel("Wafer area utilization")
    ax2.set_ylim(0, 1.05)
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="upper right", frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(prefix.with_suffix(".png"), dpi=220)
    fig.savefig(prefix.with_suffix(".pdf"))
    plt.close(fig)


def plot_edge_sensitivity(rows: list[dict[str, Any]], prefix: Path, title: str) -> None:
    plot_edge_sensitivity_heatmap(rows, prefix, value_key="communication_us", title=title, cbar_label="Communication us/layer")


def plot_edge_sensitivity_comm_bars(rows: list[dict[str, Any]], prefix: Path, title: str) -> None:
    if not rows:
        return
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )
    bw_rows = select_edge_sensitivity_bandwidth_slice(rows, 10.0, [1.0, 5.0, 20.0, 100.0, 500.0])
    lat_rows = select_edge_sensitivity_latency_slice(rows, 1.0, [10.0, 50.0, 250.0, 1000.0, 2000.0])
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.4), constrained_layout=True)
    _plot_grouped_edge_bars(
        axes[0],
        bw_rows,
        x_key="per_port_bw_gbps",
        x_label="Per-port external bandwidth (GB/s)",
        title="(a) Fixed 10 ns switch latency",
        y_label="MoE comm / layer (us)",
    )
    _plot_grouped_edge_bars(
        axes[1],
        lat_rows,
        x_key="switch_latency_ns",
        x_label="Switch latency (ns)",
        title="(b) Fixed 1 GB/s external bandwidth",
        y_label="MoE comm / layer (us)",
    )
    fig.suptitle(title)
    fig.savefig(prefix.with_suffix(".png"), dpi=220)
    fig.savefig(prefix.with_suffix(".pdf"))
    plt.close(fig)


def plot_edge_sensitivity_tpot_bars(rows: list[dict[str, Any]], prefix: Path, title: str) -> None:
    if not rows:
        return
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )
    bw_rows = select_edge_sensitivity_bandwidth_slice(rows, 10.0, [1.0, 5.0, 20.0, 100.0, 500.0])
    lat_rows = select_edge_sensitivity_latency_slice(rows, 1.0, [10.0, 50.0, 250.0, 1000.0, 2000.0])
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.9), constrained_layout=True)
    _plot_stacked_tpot_bars(
        axes[0],
        bw_rows,
        x_key="per_port_bw_gbps",
        x_label="Per-port external bandwidth (GB/s)",
        title="(a) TPOT decomposition at 10 ns switch latency",
    )
    _plot_stacked_tpot_bars(
        axes[1],
        lat_rows,
        x_key="switch_latency_ns",
        x_label="Switch latency (ns)",
        title="(b) TPOT decomposition at 1 GB/s external bandwidth",
    )
    fig.suptitle(title)
    fig.savefig(prefix.with_suffix(".png"), dpi=220)
    fig.savefig(prefix.with_suffix(".pdf"))
    plt.close(fig)


def _plot_grouped_edge_bars(
    ax,
    rows: list[dict[str, Any]],
    *,
    x_key: str,
    x_label: str,
    title: str,
    y_label: str,
) -> None:
    if not rows:
        ax.set_axis_off()
        return
    labels = [str(int(float(r[x_key]))) if float(r[x_key]).is_integer() else f"{float(r[x_key]):g}" for r in rows]
    x = list(range(len(rows)))
    series = [
        ("communication_us", "#4C78A8", "ASTRA-calibrated"),
        ("crossbar_comm_us", "#F58518", "Raw crossbar"),
        ("rail_analytical_comm_us", "#54A24B", "Rail analytical"),
    ]
    width = 0.24
    offsets = [-(width * (len(series) - 1)) / 2.0 + i * width for i in range(len(series))]
    ymax = 0.0
    for idx, (key, color, label) in enumerate(series):
        vals = [float(r.get(key, 0.0)) for r in rows]
        ymax = max(ymax, max(vals) if vals else 0.0)
        ax.bar([v + offsets[idx] for v in x], vals, width=width, color=color, label=label)
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.22)
    ax.set_ylim(0, ymax * 1.22 if ymax > 0 else 1.0)
    ax.legend(loc="upper left", frameon=False)


def _plot_stacked_tpot_bars(
    ax,
    rows: list[dict[str, Any]],
    *,
    x_key: str,
    x_label: str,
    title: str,
) -> None:
    if not rows:
        ax.set_axis_off()
        return
    labels = [str(int(float(r[x_key]))) if float(r[x_key]).is_integer() else f"{float(r[x_key]):g}" for r in rows]
    x = list(range(len(rows)))
    attn = [float(r.get("attention_stack_ms", 0.0)) for r in rows]
    moe_layers = [_infer_moe_layers(r) for r in rows]
    moe_compute = [float(r.get("moe_compute_us", 0.0)) * layers / 1000.0 for r, layers in zip(rows, moe_layers)]
    moe_comm = [float(r.get("communication_us", 0.0)) * layers / 1000.0 for r, layers in zip(rows, moe_layers)]
    series = [
        (attn, "#4C78A8", "Attention stack"),
        (moe_compute, "#F58518", "MoE compute"),
        (moe_comm, "#54A24B", "MoE comm"),
    ]
    bottom = [0.0 for _ in rows]
    ymax = 0.0
    for vals, color, label in series:
        ax.bar(x, vals, width=0.62, bottom=bottom, color=color, label=label)
        bottom = [b + v for b, v in zip(bottom, vals)]
        ymax = max(ymax, max(bottom) if bottom else 0.0)
    for idx, row in enumerate(rows):
        ax.text(idx, bottom[idx] + ymax * 0.02, f"{float(row.get('tpot_ms', 0.0)):.2f}ms", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Decode TPOT (ms)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.22)
    ax.set_ylim(0, ymax * 1.28 if ymax > 0 else 1.0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False)


def _infer_moe_layers(row: dict[str, Any]) -> float:
    moe_layer_us = float(row.get("moe_layer_us", 0.0))
    moe_stack_ms = float(row.get("moe_stack_ms", 0.0))
    if moe_layer_us > 0 and moe_stack_ms > 0:
        return moe_stack_ms * 1000.0 / moe_layer_us
    return 58.0


def plot_edge_sensitivity_heatmap(
    rows: list[dict[str, Any]],
    prefix: Path,
    value_key: str,
    title: str,
    cbar_label: str,
    fmt: str = ".1f",
) -> None:
    if not rows:
        return
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 8,
        }
    )
    bw_values = sorted({float(r.get("per_port_bw_gbps", 0.0)) for r in rows})
    lat_values = sorted({float(r.get("switch_latency_ns", 0.0)) for r in rows})
    grid = [[None for _ in bw_values] for _ in lat_values]
    by_key = {
        (float(r.get("per_port_bw_gbps", 0.0)), float(r.get("switch_latency_ns", 0.0))): r
        for r in rows
    }
    for yi, lat in enumerate(lat_values):
        for xi, bw in enumerate(bw_values):
            row = by_key.get((bw, lat))
            if row is None:
                continue
            grid[yi][xi] = float(row.get(value_key, row.get("communication_us", row.get("moe_comm_us", 0.0))))
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    import numpy as np
    arr = np.array(grid, dtype=float)
    im = ax.imshow(arr, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(bw_values)), [f"{int(bw)}" if bw.is_integer() else f"{bw:g}" for bw in bw_values], rotation=20)
    ax.set_yticks(range(len(lat_values)), [f"{int(lat)}" if lat.is_integer() else f"{lat:g}" for lat in lat_values])
    ax.set_xlabel("Per-port external bandwidth (GB/s)")
    ax.set_ylabel("Switch latency (ns)")
    ax.set_title(title)
    for yi, lat in enumerate(lat_values):
        for xi, bw in enumerate(bw_values):
            val = grid[yi][xi]
            if val is None:
                continue
            ax.text(
                xi,
                yi,
                f"{val:{fmt}}",
                ha="center",
                va="center",
                fontsize=6,
                color="white" if val < np.nanmean(arr) else "black",
            )
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(prefix.with_suffix(".png"), dpi=220)
    fig.savefig(prefix.with_suffix(".pdf"))
    plt.close(fig)


def plot_edge_sensitivity_bandwidth_slices(rows: list[dict[str, Any]], prefix: Path, value_key: str, title: str) -> None:
    if not rows:
        return
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )
    bw_values = sorted({float(r.get("per_port_bw_gbps", 0.0)) for r in rows})
    slices = [10.0, 100.0, 500.0, 2000.0]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for lat in slices:
        ys = [float(next(r.get(value_key, 0.0) for r in rows if float(r.get("per_port_bw_gbps", 0.0)) == bw and float(r.get("switch_latency_ns", 0.0)) == lat)) for bw in bw_values]
        ax.plot(bw_values, ys, marker="o", linewidth=1.8, label=f"{int(lat)} ns")
    ax.set_xscale("log")
    ax.set_xlabel("Per-port external bandwidth (GB/s, log scale)")
    ax.set_ylabel("Communication us/layer")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(prefix.with_suffix(".png"), dpi=220)
    fig.savefig(prefix.with_suffix(".pdf"))
    plt.close(fig)


def plot_edge_sensitivity_latency_slices(rows: list[dict[str, Any]], prefix: Path, value_key: str, title: str) -> None:
    if not rows:
        return
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )
    lat_values = sorted({float(r.get("switch_latency_ns", 0.0)) for r in rows})
    slices = [1.0, 10.0, 100.0, 500.0]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for bw in slices:
        ys = [float(next(r.get(value_key, 0.0) for r in rows if float(r.get("per_port_bw_gbps", 0.0)) == bw and float(r.get("switch_latency_ns", 0.0)) == lat)) for lat in lat_values]
        ax.plot(lat_values, ys, marker="o", linewidth=1.8, label=f"{int(bw)} GB/s")
    ax.set_xlabel("Switch latency (ns)")
    ax.set_ylabel("Communication us/layer")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(prefix.with_suffix(".png"), dpi=220)
    fig.savefig(prefix.with_suffix(".pdf"))
    plt.close(fig)
