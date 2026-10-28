#!/usr/bin/env python3
"""Generate the large-font Section 8.4 hardware-analysis figure.

This script is a reproducible large-font companion to
``plot_sec84_hardware_summary.py``.  It keeps the original two-panel structure:
cluster power on the left and a wafer-area pie chart on the right.  Outputs are
written as ``fig_sec84_hardware_analysis_zh_largefont.{png,pdf}`` by default.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import plot_sec84_hardware_summary as base  # noqa: E402


def configure_large_style() -> None:
    base.setup_fonts("zh")
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 11.5,
            "axes.labelsize": 12.0,
            "axes.titlesize": 12.0,
            "xtick.labelsize": 12.0,
            "ytick.labelsize": 12.0,
            "legend.fontsize": 11.5,
            "axes.unicode_minus": False,
            "figure.dpi": 180,
            "savefig.dpi": 360,
        }
    )


def save_large(fig: plt.Figure, output_dir: Path, stem: str, suffix: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    base.apply_cjk_font(fig)
    for ext in ("png", "pdf"):
        path = output_dir / f"{stem}{suffix}.{ext}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.035)
        print(f"[Save] {path}")
    plt.close(fig)


def add_aligned_subfigure_captions(
    fig: plt.Figure,
    captions: list[tuple[plt.Axes, str]],
    *,
    pad: float = 0.014,
    fontsize: float = 12.0,
) -> None:
    base.apply_cjk_font(fig)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    caption_y = max(
        min(
            ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted()).y0
            for ax, _ in captions
        )
        - pad,
        0.01,
    )
    for ax, caption in captions:
        bbox = ax.get_position()
        fig.text(
            bbox.x0 + bbox.width / 2.0,
            caption_y,
            caption,
            ha="center",
            va="top",
            fontsize=fontsize,
        )
    base.apply_cjk_font(fig)


def plot_large(output_dir: Path, output_suffix: str) -> None:
    cluster_rows = base.build_cluster_power_rows()
    area_rows = base.build_area_rows()

    wafers = np.array([int(r["total_wafers"]) for r in cluster_rows])
    optical_low = np.array([float(r["optical_power_kw_14w"]) for r in cluster_rows])
    optical_high = np.array([float(r["optical_power_kw_17w"]) for r in cluster_rows])
    dac = np.array([float(r["passive_dac_power_kw"]) for r in cluster_rows])
    endpoints = np.array([int(r["optical_endpoints"]) for r in cluster_rows])

    fig, (ax0, ax1) = plt.subplots(
        1,
        2,
        figsize=(7.35, 3.25),
        gridspec_kw={"width_ratios": [1.16, 1.0], "wspace": 0.32},
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.35, top=0.94)

    x = np.arange(len(wafers))
    bar = ax0.bar(
        x,
        optical_high,
        width=0.58,
        color="#D95F02",
        alpha=0.90,
        label="光模块 17 W/端口",
    )
    low = ax0.scatter(
        x,
        optical_low,
        marker="_",
        s=100,
        linewidths=2.0,
        color="#8C2D04",
        label="14 W 下界",
        zorder=3,
    )
    line, = ax0.plot(
        x,
        dac,
        marker="o",
        linewidth=1.7,
        markersize=4.0,
        color="#4C78A8",
        label="被动铜缆参考",
    )
    ax0.set_xticks(x)
    ax0.set_xticklabels([str(v) for v in wafers])
    ax0.set_xlabel("部署中的总晶圆数")
    ax0.set_ylabel("跨机柜端口功耗 (kW)")
    ax0.set_ylim(0, max(optical_high) * 1.16)
    ax0.grid(axis="y", color="#dddddd", linewidth=0.65)
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)
    ax0.legend(handles=[bar, low, line], frameon=False, loc="upper left", handlelength=1.6, borderaxespad=0.2)

    ax2 = ax0.twinx()
    ax2.plot(x, endpoints, linestyle="--", linewidth=1.35, color="#666666", alpha=0.72)
    ax2.set_ylabel("光互联端口数")
    ax2.tick_params(axis="y", labelsize=12.0)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.set_ylim(0, max(endpoints) * 1.16)

    base_rows = area_rows[:3]
    values = [float(r["area_mm2"]) for r in base_rows]
    colors = ["#8DA0CB", "#D62728", "#BDBDBD"]
    wedges, _ = ax1.pie(
        values,
        colors=colors,
        explode=[0.0, 0.10, 0.0],
        startangle=100,
        counterclock=False,
        wedgeprops=dict(linewidth=0.8, edgecolor="white"),
    )

    hacmo = base_rows[1]
    ax1.text(
        0.0,
        0.0,
        "HACMO crossbar\n4×FRED3 L1\n"
        f"{float(hacmo['area_mm2']):,.0f} mm²\n"
        f"{float(hacmo['share_of_wafer_pct']):.1f}%",
        ha="center",
        va="center",
        fontsize=8.8,
        weight="bold",
        color="#A50F15",
        linespacing=1.15,
    )

    legend_labels = [
        f"NPU+HBM+I/O: {float(base_rows[0]['area_mm2']):,.0f} mm² ({float(base_rows[0]['share_of_wafer_pct']):.1f}%)",
        f"crossbar: {float(base_rows[1]['area_mm2']):,.0f} mm² ({float(base_rows[1]['share_of_wafer_pct']):.1f}%)",
        f"剩余面积: {float(base_rows[2]['area_mm2']):,.0f} mm² ({float(base_rows[2]['share_of_wafer_pct']):.1f}%)",
    ]
    ax1.legend(
        wedges,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.20),
        frameon=False,
        fontsize=12,
        handlelength=1.1,
        handletextpad=0.45,
        labelspacing=0.32,
    )

    add_aligned_subfigure_captions(
        fig,
        [
            (ax0, "(a) 集群功耗：4片晶圆/机柜"),
            (ax1, "(b) 晶圆上面积预算（对齐FRED配置）"),
        ],
    )
    save_large(fig, output_dir, "fig_sec84_hardware_analysis_zh", output_suffix)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the large-font Section 8.4 hardware figure.")
    parser.add_argument("--output-dir", type=Path, default=base.OUT_DIR)
    parser.add_argument("--output-suffix", default="_largefont")
    args = parser.parse_args()

    configure_large_style()
    plot_large(args.output_dir, args.output_suffix)


if __name__ == "__main__":
    main()
