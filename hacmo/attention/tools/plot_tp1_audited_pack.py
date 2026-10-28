#!/usr/bin/env python3
"""Build paper-readable summaries for an audited TP1 result pack."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-tp1-audited")

import matplotlib
import matplotlib.pyplot as plt


SIMULATOR_METHODS = ("B", "C", "S", "O")
PLOT_METHODS = ("G", *SIMULATOR_METHODS)
METHOD_LABELS = {
    "G": "G: GPU+vLLM (H200)",
    "B": "B: Uniform, no cache",
    "C": "C: Uniform cache",
    "S": "S: Non-uniform mapping",
    "O": "O: + Causal prefetch",
}
METHOD_COLORS = {
    "G": "#4B5563",
    "B": "#6B7280",
    "C": "#D4A72C",
    "S": "#2F6B9A",
    "O": "#C4473A",
}
METHOD_MARKERS = {"G": "P", "B": "o", "C": "s", "S": "^", "O": "D"}
BAR_COLORS = {
    "G": "#595959",
    "B": "#7F7F7F",
    "C": "#A0A0A0",
    "S": "#4C78A8",
    "O": "#F58518",
}
BAR_HATCHES = {"G": "xx", "B": "", "C": "..", "S": "///", "O": "\\\\\\"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_key_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip()
    return values


def _load_gpu_baseline(
    profile_path: Path,
    context_sweep_path: Path,
    env_info_path: Path,
) -> tuple[dict[tuple[str, int], float], list[dict[str, object]], dict[str, object]]:
    required_batches = {16, 32, 64}
    measurements: dict[int, float] = {}
    for row in _read_csv(profile_path):
        if row.get("module") != "sparse_attention_mla":
            continue
        if int(row["s_q"]) != 1 or int(row["s_kv"]) != 32768:
            continue
        if int(row["topk"]) != 2048:
            continue
        batch = int(row["batch_size"])
        if batch not in required_batches:
            continue
        if batch in measurements:
            raise ValueError(f"duplicate GPU measurement for batch {batch}")
        measurements[batch] = float(row["e2e_time_us"])
    if set(measurements) != required_batches:
        raise ValueError(
            f"GPU profile lacks batches {sorted(required_batches - set(measurements))}"
        )
    if any(value <= 0.0 for value in measurements.values()):
        raise ValueError(f"GPU latency must be positive: {measurements}")

    context_measurements: dict[int, float] = {}
    for row in _read_csv(context_sweep_path):
        if int(row["batch_size"]) != 32 or int(row["s_q"]) != 1:
            continue
        if int(row["topk"]) != 2048:
            continue
        context = int(row["s_kv"])
        if context not in {8192, 16384, 32768}:
            continue
        if context in context_measurements:
            raise ValueError(f"duplicate GPU context measurement for {context}")
        context_measurements[context] = float(row["e2e_time_us"])
    if set(context_measurements) != {8192, 16384, 32768}:
        raise ValueError("GPU context sweep must contain 8k, 16k, and 32k at batch 32")

    reference_us = context_measurements[32768]
    target_context_delta_pct = {
        context: 100.0 * (latency / reference_us - 1.0)
        for context, latency in context_measurements.items()
        if context != 32768
    }
    max_abs_delta_pct = max(abs(value) for value in target_context_delta_pct.values())
    if max_abs_delta_pct > 5.0:
        raise ValueError(
            "fixed-topk GPU context reuse is not supported by the measured sweep: "
            f"max delta is {max_abs_delta_pct:.2f}%"
        )

    env_info = _read_key_values(env_info_path)
    gpu_model = env_info.get("gpu", "")
    if "H200" not in gpu_model:
        raise ValueError(f"expected an H200 GPU profile, found {gpu_model!r}")

    by_group: dict[tuple[str, int], float] = {}
    rows: list[dict[str, object]] = []
    for context in ("8k", "16k"):
        for batch in sorted(required_batches):
            latency_us = measurements[batch]
            by_group[(context, batch)] = latency_us
            rows.append(
                {
                    "context": context,
                    "batch": batch,
                    "G_us": latency_us,
                    "gpu_model": gpu_model,
                    "measurement_scope": "isolated vLLM FlashMLA sparse-attention kernel",
                    "measurement_s_kv": 32768,
                    "topk": 2048,
                    "s_q": 1,
                    "context_policy": "reuse direct fixed-topk batch measurement",
                    "target_context_validation_delta_pct": target_context_delta_pct[
                        8192 if context == "8k" else 16384
                    ],
                }
            )
    validation = {
        "gpu_model": gpu_model,
        "direct_batch_measurements_us": measurements,
        "context_sweep_batch": 32,
        "context_sweep_us": context_measurements,
        "max_abs_8k_16k_delta_vs_32k_pct": max_abs_delta_pct,
        "context_policy": (
            "The selected work is fixed at topk=2048. Direct batch measurements at "
            "s_kv=32768 are reused for 8k/16k; a separate measured context sweep "
            "bounds the target-context difference below 5%."
        ),
    }
    return by_group, rows, validation


def _load_groups(
    metrics_path: Path,
    gpu_baseline: dict[tuple[str, int], float] | None = None,
) -> list[dict[str, object]]:
    raw_rows = _read_csv(metrics_path)
    grouped: dict[tuple[str, int, int], dict[str, float]] = defaultdict(dict)
    for row in raw_rows:
        key = (
            row["context_group"],
            int(row["array_rows"]),
            int(row["batch_size"]),
        )
        method = row["method"]
        if method not in SIMULATOR_METHODS:
            continue
        if method in grouped[key]:
            raise ValueError(f"duplicate method {method} for group {key}")
        grouped[key][method] = float(row["mean_exposed_us"])

    expected = set(SIMULATOR_METHODS)
    summary: list[dict[str, object]] = []
    context_order = {"8k": 0, "16k": 1}
    for (context, side, batch), values in sorted(
        grouped.items(),
        key=lambda item: (
            context_order.get(item[0][0], 99),
            item[0][1],
            item[0][2],
        ),
    ):
        if set(values) != expected:
            raise ValueError(
                f"incomplete method block for {(context, side, batch)}: {sorted(values)}"
            )
        c_us = values["C"]
        s_us = values["S"]
        o_us = values["O"]
        row: dict[str, object] = {
            "context": context,
            "array": f"{side}x{side}",
            "batch": batch,
            "B_us": values["B"],
            "C_us": c_us,
            "S_us": s_us,
            "O_us": o_us,
            "S_vs_C_latency_reduction_pct": 100.0 * (1.0 - s_us / c_us),
            "O_vs_S_latency_reduction_pct": 100.0 * (1.0 - o_us / s_us),
            "O_vs_C_speedup": c_us / o_us,
        }
        if gpu_baseline is not None:
            key = (context, batch)
            if key not in gpu_baseline:
                raise ValueError(f"GPU baseline lacks group {key}")
            g_us = gpu_baseline[key]
            row = {
                "context": context,
                "array": f"{side}x{side}",
                "batch": batch,
                "G_us": g_us,
                **{key: value for key, value in row.items() if key not in {"context", "array", "batch"}},
                "O_vs_G_speedup": g_us / o_us,
            }
        summary.append(row)
    if len(summary) != 12:
        raise ValueError(f"expected 12 experiment groups, found {len(summary)}")
    return summary


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.7,
            "lines.markersize": 4.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _summary_methods(summary: list[dict[str, object]]) -> tuple[str, ...]:
    return PLOT_METHODS if summary and "G_us" in summary[0] else SIMULATOR_METHODS


def _plot_latency(summary: list[dict[str, object]], fig_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.1), constrained_layout=True)
    panels = (("8k", "20x20"), ("8k", "68x68"), ("16k", "20x20"), ("16k", "68x68"))
    panel_labels = ("(a)", "(b)", "(c)", "(d)")
    for ax, (context, array), panel_label in zip(axes.flat, panels, panel_labels):
        rows = [
            row
            for row in summary
            if row["context"] == context and row["array"] == array
        ]
        batches = [int(row["batch"]) for row in rows]
        for method in _summary_methods(summary):
            ax.plot(
                batches,
                [float(row[f"{method}_us"]) for row in rows],
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                label=METHOD_LABELS[method],
            )
        ax.set_title(f"{panel_label} Context {context}, array {array}", loc="left", pad=4)
        ax.set_xticks(batches)
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Mean exposed latency (us/window)")
        ax.grid(axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="outside upper center", frameon=False)
    outputs = [fig_dir / "tp1_latency_us.png", fig_dir / "tp1_latency_us.pdf"]
    fig.savefig(outputs[0], dpi=300, bbox_inches="tight")
    fig.savefig(outputs[1], bbox_inches="tight")
    plt.close(fig)
    return outputs


def _plot_incremental_reduction(
    summary: list[dict[str, object]], fig_dir: Path
) -> list[Path]:
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.0), constrained_layout=True)
    panels = (("8k", "20x20"), ("8k", "68x68"), ("16k", "20x20"), ("16k", "68x68"))
    panel_labels = ("(a)", "(b)", "(c)", "(d)")
    bar_width = 0.34
    for ax, (context, array), panel_label in zip(axes.flat, panels, panel_labels):
        rows = [
            row
            for row in summary
            if row["context"] == context and row["array"] == array
        ]
        positions = list(range(len(rows)))
        s_values = [float(row["S_vs_C_latency_reduction_pct"]) for row in rows]
        o_values = [float(row["O_vs_S_latency_reduction_pct"]) for row in rows]
        ax.bar(
            [x - bar_width / 2 for x in positions],
            s_values,
            width=bar_width,
            color=METHOD_COLORS["S"],
            edgecolor="#1F2937",
            linewidth=0.5,
            hatch="///",
            label="S vs C",
        )
        ax.bar(
            [x + bar_width / 2 for x in positions],
            o_values,
            width=bar_width,
            color=METHOD_COLORS["O"],
            edgecolor="#1F2937",
            linewidth=0.5,
            hatch="\\\\\\",
            label="O vs S",
        )
        ax.set_title(f"{panel_label} Context {context}, array {array}", loc="left", pad=4)
        ax.set_xticks(positions, [str(row["batch"]) for row in rows])
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Latency reduction (%)")
        ax.set_ylim(0.0, 55.0)
        ax.grid(axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="outside upper center", frameon=False)
    outputs = [
        fig_dir / "tp1_incremental_reduction.png",
        fig_dir / "tp1_incremental_reduction.pdf",
    ]
    fig.savefig(outputs[0], dpi=300, bbox_inches="tight")
    fig.savefig(outputs[1], bbox_inches="tight")
    plt.close(fig)
    return outputs


def _plot_legacy_style_bars(
    summary: list[dict[str, object]], fig_dir: Path
) -> list[Path]:
    outputs: list[Path] = []
    context_lengths = {"8k": 8192, "16k": 16384}
    methods = _summary_methods(summary)
    bar_width = 0.15 if len(methods) == 5 else 0.19
    for context in ("8k", "16k"):
        for array in ("20x20", "68x68"):
            rows = [
                row
                for row in summary
                if row["context"] == context and row["array"] == array
            ]
            positions = list(range(len(rows)))
            fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
            for method_index, method in enumerate(methods):
                offset = (method_index - (len(methods) - 1) / 2.0) * bar_width
                ax.bar(
                    [x + offset for x in positions],
                    [float(row[f"{method}_us"]) for row in rows],
                    width=bar_width,
                    color=BAR_COLORS[method],
                    edgecolor="#374151",
                    linewidth=0.45,
                    hatch=BAR_HATCHES[method],
                    label=METHOD_LABELS[method],
                )
            ax.set_xticks(positions, [str(row["batch"]) for row in rows])
            ax.set_xlabel("Batch size")
            ax.set_ylabel("Mean exposed latency (us/window)")
            ax.set_title(f"TP1 ablation (ctx={context}, array={array})")
            ax.grid(axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.8)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(ncol=2, frameon=False, loc="upper left")
            stem = f"ablation_ctx{context_lengths[context]}_array{array}"
            png_path = fig_dir / f"{stem}.png"
            pdf_path = fig_dir / f"{stem}.pdf"
            fig.savefig(png_path, dpi=300, bbox_inches="tight")
            fig.savefig(pdf_path, bbox_inches="tight")
            plt.close(fig)
            outputs.extend((png_path, pdf_path))
    return outputs


def _write_latex(path: Path, summary: list[dict[str, object]]) -> None:
    has_gpu = "G_us" in summary[0]
    column_spec = "ll" + "r" * (9 if has_gpu else 7)
    lines = [
        "% Generated by hacmo/attention/tools/plot_tp1_audited_pack.py",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{TP1 trace-driven mean exposed latency. All latency values are in $\\mu$s per sampled window.}",
        "\\label{tab:tp1-audited-latency}",
        "\\resizebox{\\textwidth}{!}{%",
        f"\\begin{{tabular}}{{{column_spec}}}",
        "\\toprule",
        (
            "Context & Array & Batch & G & B & C & S & O & S vs. C $\\downarrow$ & "
            "O vs. S $\\downarrow$ & O vs. G speedup \\\\"
            if has_gpu
            else "Context & Array & Batch & B & C & S & O & S vs. C $\\downarrow$ & O vs. S $\\downarrow$ \\\\"
        ),
        "\\midrule",
    ]
    for row in summary:
        prefix = f"{row['context']} & {row['array']} & {row['batch']} & "
        if has_gpu:
            prefix += f"{float(row['G_us']):.3f} & "
        lines.append(
            prefix
            + f"{float(row['B_us']):.3f} & {float(row['C_us']):.3f} & "
            + f"{float(row['S_us']):.3f} & {float(row['O_us']):.3f} & "
            + f"{float(row['S_vs_C_latency_reduction_pct']):.2f}\\% & "
            + f"{float(row['O_vs_S_latency_reduction_pct']):.2f}\\%"
            + (f" & {float(row['O_vs_G_speedup']):.2f}$\\times$" if has_gpu else "")
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table*}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report(
    pack_root: Path,
    summary: list[dict[str, object]],
    validation: dict[str, object],
    gpu_validation: dict[str, object] | None,
) -> Path:
    has_gpu = gpu_validation is not None
    s_reductions = [float(row["S_vs_C_latency_reduction_pct"]) for row in summary]
    o_reductions = [float(row["O_vs_S_latency_reduction_pct"]) for row in summary]
    o_speedups = [float(row["O_vs_C_speedup"]) for row in summary]
    gpu_speedups = [float(row["O_vs_G_speedup"]) for row in summary] if has_gpu else []
    lines = [
        "# TP1 ASTRA-Sim + Ramulator Result Pack",
        "",
        "Status: current paper-facing pack. This pack supersedes the preserved 20260612 paper-facing pack.",
        "",
        "## Scope",
        "",
        "- 48 cases: 8k/16k contexts, 20x20/68x68 arrays, batches 16/32/64, and methods B/C/S/O.",
        "- Each case times 3 layers x 2 decode steps; all 61 layers replay the same cache/LRU state transitions.",
        "- Values below are mean exposed selected-KV core-attention latency in us per sampled window.",
        "- CP communication, the DSA Indexer, and common QKV/output projections are outside this pack.",
        *(
            [
                "- G is an isolated H200 vLLM/FlashMLA sparse-attention kernel baseline at the same topk=2048 scope.",
                "- G is measured on the native GPU and therefore does not run through ASTRA-Sim or Ramulator.",
            ]
            if has_gpu
            else []
        ),
        "",
        "## Main Result",
        "",
        f"- S reduces latency relative to C in all 12 groups: {min(s_reductions):.2f}% to {max(s_reductions):.2f}%.",
        f"- O further reduces latency relative to S in all 12 groups: {min(o_reductions):.2f}% to {max(o_reductions):.2f}%.",
        f"- O achieves {min(o_speedups):.2f}x to {max(o_speedups):.2f}x speedup over C.",
        *(
            [
                f"- O achieves {min(gpu_speedups):.2f}x to {max(gpu_speedups):.2f}x speedup over the measured H200 GPU kernel baseline."
            ]
            if has_gpu
            else []
        ),
        (
            "- No confidence intervals are shown: B/C/S/O are deterministic six-window means; G retains the CUDA-event mean but not the per-iteration samples."
            if has_gpu
            else "- No confidence intervals are shown: each point is a deterministic mean over six complete trace windows, not an independent repeated-run distribution."
        ),
        "",
        "## Exact Values",
        "",
        (
            "| Context | Array | Batch | G (us) | B (us) | C (us) | S (us) | O (us) | S vs C reduction | O vs S reduction | O vs G speedup |"
            if has_gpu
            else "| Context | Array | Batch | B (us) | C (us) | S (us) | O (us) | S vs C reduction | O vs S reduction |"
        ),
        (
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
            if has_gpu
            else "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        ),
    ]
    for row in summary:
        gpu_prefix = f"{float(row['G_us']):.3f} | " if has_gpu else ""
        gpu_suffix = f" | {float(row['O_vs_G_speedup']):.2f}x" if has_gpu else ""
        lines.append(
            f"| {row['context']} | {row['array']} | {row['batch']} | "
            f"{gpu_prefix}"
            f"{float(row['B_us']):.3f} | {float(row['C_us']):.3f} | "
            f"{float(row['S_us']):.3f} | {float(row['O_us']):.3f} | "
            f"{float(row['S_vs_C_latency_reduction_pct']):.2f}% | "
            f"{float(row['O_vs_S_latency_reduction_pct']):.2f}%{gpu_suffix} |"
        )
    lines.extend(
        [
            "",
            "## Simulator And Validation Contract",
            "",
            "- Runtime HBM backend: request-aware Ramulator2 HBM3 with full drain.",
            "- Runtime NoC backend: congestion-aware ASTRA-Sim 2D Mesh with all physical cores represented.",
            "- Canonical input: per-core KV access routing trace produced before residency/source classification.",
            "- S/O use the same frozen offline mapping; O adds only causal layer prefetch.",
            "- Paper Eq. (1) is used only for offline candidate scoring; runtime simulator latency does not use gamma.",
            *(
                [
                    "- GPU context handling: direct batch measurements at s_kv=32768 are reused because selected work is fixed at topk=2048.",
                    f"- The measured batch-32 context sweep bounds 8k/16k deviation from 32k at {float(gpu_validation['max_abs_8k_16k_delta_vs_32k_pct']):.2f}%.",
                ]
                if has_gpu
                else []
            ),
            f"- Validator: status={validation['status']}, cases={validation['case_records_checked']}, window rows={validation['window_rows_checked']}, core rows={validation['core_rows_checked']}, errors={len(validation['errors'])}, warnings={len(validation['warnings'])}.",
            "",
            "## Files",
            "",
            "- `csv/tp1_audited_metrics.csv`: original 48-case aggregate metrics.",
            "- `csv/tp1_audited_window_metrics.csv`: 288 sampled-window rows.",
            "- `csv/tp1_audited_core_metrics.csv`: 120,576 per-core rows.",
            "- `summary/tp1_latency_summary.csv`: compact exact values and incremental effects.",
            *(
                [
                    "- `summary/tp1_gpu_measured_baseline.csv`: measured H200 batch values, scope, and context-reuse provenance."
                ]
                if has_gpu
                else []
            ),
            "- `summary/tp1_latency_summary.tex`: paper-ready companion table source.",
            "- `summary/plot_manifest.json`: summary-input and output hashes.",
            "- `figs/tp1_latency_us.{png,pdf}`: absolute latency overview.",
            "- `figs/tp1_incremental_reduction.{png,pdf}`: C-to-S and S-to-O reductions.",
            "- `figs/ablation_ctx*_array*.{png,pdf}`: old-pack-style grouped bars generated without rerunning simulation.",
            "- `metadata.json`: trace, input, binary, configuration, and implementation provenance.",
            "- `validation.json`: generator validation record.",
        ]
    )
    path = pack_root / "RESULTS.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack-root", required=True, type=Path)
    parser.add_argument(
        "--gpu-profile-csv",
        type=Path,
        help="vLLM kernel profile CSV used to add measured G values",
    )
    parser.add_argument(
        "--gpu-context-sweep-csv",
        type=Path,
        help="FlashMLA context sweep used to validate fixed-topk context reuse",
    )
    parser.add_argument(
        "--gpu-env-info",
        type=Path,
        help="structured GPU profiling environment information",
    )
    args = parser.parse_args()

    pack_root = args.pack_root.resolve()
    metrics_path = pack_root / "csv" / "tp1_audited_metrics.csv"
    metadata_path = pack_root / "metadata.json"
    validation_path = pack_root / "validation.json"
    for required in (metrics_path, metadata_path, validation_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("status") != "ok":
        raise ValueError(f"refusing to plot invalid pack: {validation}")

    gpu_inputs = (
        args.gpu_profile_csv,
        args.gpu_context_sweep_csv,
        args.gpu_env_info,
    )
    if any(path is not None for path in gpu_inputs) and not all(
        path is not None for path in gpu_inputs
    ):
        raise ValueError("all three --gpu-* inputs are required to add G")

    gpu_baseline: dict[tuple[str, int], float] | None = None
    gpu_rows: list[dict[str, object]] = []
    gpu_validation: dict[str, object] | None = None
    resolved_gpu_inputs: tuple[Path, Path, Path] | None = None
    if (
        args.gpu_profile_csv is not None
        and args.gpu_context_sweep_csv is not None
        and args.gpu_env_info is not None
    ):
        resolved_gpu_inputs = (
            args.gpu_profile_csv.resolve(),
            args.gpu_context_sweep_csv.resolve(),
            args.gpu_env_info.resolve(),
        )
        for required in resolved_gpu_inputs:
            if not required.is_file():
                raise FileNotFoundError(required)
        gpu_baseline, gpu_rows, gpu_validation = _load_gpu_baseline(
            *resolved_gpu_inputs
        )

    summary = _load_groups(metrics_path, gpu_baseline)
    summary_dir = pack_root / "summary"
    fig_dir = pack_root / "figs"
    summary_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = summary_dir / "tp1_latency_summary.csv"
    _write_csv(summary_csv, summary)
    gpu_csv: Path | None = None
    if gpu_rows:
        gpu_csv = summary_dir / "tp1_gpu_measured_baseline.csv"
        _write_csv(gpu_csv, gpu_rows)
    latex_path = summary_dir / "tp1_latency_summary.tex"
    _write_latex(latex_path, summary)
    _configure_matplotlib()
    figures = (
        _plot_latency(summary, fig_dir)
        + _plot_incremental_reduction(summary, fig_dir)
        + _plot_legacy_style_bars(summary, fig_dir)
    )
    report_path = _write_report(pack_root, summary, validation, gpu_validation)

    outputs = [summary_csv, *([gpu_csv] if gpu_csv is not None else []), latex_path, *figures, report_path]
    manifest_inputs = {
        str(metrics_path): _sha256(metrics_path),
        str(metadata_path): _sha256(metadata_path),
        str(validation_path): _sha256(validation_path),
    }
    if resolved_gpu_inputs is not None:
        manifest_inputs.update(
            {str(path): _sha256(path) for path in resolved_gpu_inputs}
        )
    manifest = {
        "generator": str(Path(__file__).resolve()),
        "generator_sha256": _sha256(Path(__file__).resolve()),
        "python_version": sys.version,
        "matplotlib_version": matplotlib.__version__,
        "inputs": manifest_inputs,
        "outputs": {str(path): _sha256(path) for path in outputs},
        "publication_note": (
            "B/C/S/O are deterministic six-window means. G is a retained CUDA-event "
            "mean without per-iteration samples. No inferential error bars are claimed."
            if gpu_validation is not None
            else "Deterministic six-window means; no inferential error bars."
        ),
        "gpu_baseline": gpu_validation,
    }
    manifest_path = pack_root / "summary" / "plot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[TP1] wrote audited summary artifacts under {pack_root}")


if __name__ == "__main__":
    main()
