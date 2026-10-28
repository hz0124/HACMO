#!/usr/bin/env python3
"""Build a simulator-backed TP1 critical-path motivation breakdown."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-tp1-motivation")

import matplotlib
import matplotlib.pyplot as plt


CONTEXTS = ("8k", "16k")
BATCH_LABELS = ("16", "32", "64", "128+CP2")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty CSV: {path}")
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


def _close(left: float, right: float, tolerance: float = 1e-8) -> bool:
    return abs(left - right) <= tolerance * max(abs(left), abs(right), 1.0)


def _build_rows(
    aggregate_rows: list[dict[str, str]],
    core_rows: list[dict[str, str]],
    hybrid_rows: list[dict[str, str]],
    *,
    method: str,
    array_side: int,
) -> list[dict[str, object]]:
    aggregates: dict[tuple[str, int, str], dict[str, str]] = {}
    for row in aggregate_rows:
        if int(row["array_rows"]) != array_side or row["method"] != method:
            continue
        key = (row["context_group"], int(row["batch_size"]), row["method"])
        if key in aggregates:
            raise ValueError(f"duplicate aggregate row: {key}")
        aggregates[key] = row

    cores: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in core_rows:
        if int(row["array_rows"]) != array_side or row["method"] != method:
            continue
        cores[(row["context_group"], int(row["batch_size"]), row["method"])].append(row)

    hybrid: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in hybrid_rows:
        if int(row["array_rows"]) != array_side or row["method"] != method:
            continue
        key = (row["context"], row["batch_label"], row["method"])
        if key in hybrid:
            raise ValueError(f"duplicate hybrid row: {key}")
        hybrid[key] = row

    output: list[dict[str, object]] = []
    for context in CONTEXTS:
        for batch_label in BATCH_LABELS:
            hybrid_key = (context, batch_label, method)
            if hybrid_key not in hybrid:
                raise ValueError(f"hybrid pack lacks required row: {hybrid_key}")
            hybrid_row = hybrid[hybrid_key]
            source_batch = int(hybrid_row["source_batch_size"])
            aggregate_key = (context, source_batch, method)
            if aggregate_key not in aggregates or aggregate_key not in cores:
                raise ValueError(f"source pack lacks required case: {aggregate_key}")
            aggregate = aggregates[aggregate_key]
            sampled_windows = int(aggregate["sampled_windows"])
            if sampled_windows <= 0:
                raise ValueError(f"invalid sampled window count: {aggregate_key}")

            critical = min(
                cores[aggregate_key],
                key=lambda row: (-float(row["post_phase_finish_us"]), int(row["core_id"])),
            )
            sequence_finish_us = float(critical["post_phase_finish_us"])
            sequence_compute_us = float(critical["compute_busy_us"])
            sequence_residual_us = float(critical["exposed_noncompute_us"])
            simulated_total_us = float(aggregate["mean_exposed_us"])
            simulated_compute_us = sequence_compute_us / sampled_windows
            simulated_wait_tail_us = sequence_residual_us / sampled_windows
            critical_finish_us = sequence_finish_us / sampled_windows
            hybrid_simulated_us = float(hybrid_row["simulated_core_attention_us"])
            analytical_cp_us = float(hybrid_row["analytical_cp_us"])
            total_us = float(hybrid_row["total_us"])

            if not _close(
                sequence_compute_us + sequence_residual_us, sequence_finish_us
            ):
                raise ValueError(f"critical-core components do not conserve: {aggregate_key}")
            if not _close(critical_finish_us, simulated_total_us):
                raise ValueError(f"critical-core finish does not match aggregate: {aggregate_key}")
            if not _close(simulated_total_us, hybrid_simulated_us):
                raise ValueError(f"source/hybrid simulated latency mismatch: {hybrid_key}")
            if not _close(simulated_compute_us + simulated_wait_tail_us, simulated_total_us):
                raise ValueError(f"per-window components do not conserve: {aggregate_key}")
            if not _close(simulated_total_us + analytical_cp_us, total_us):
                raise ValueError(f"hybrid total does not conserve: {hybrid_key}")

            output.append(
                {
                    "context": context,
                    "array": f"{array_side}x{array_side}",
                    "method": method,
                    "batch_label": batch_label,
                    "global_batch": int(hybrid_row["global_batch"]),
                    "cp_degree": int(hybrid_row["cp_degree"]),
                    "simulated_batch_per_die": int(
                        hybrid_row["simulated_batch_per_die"]
                    ),
                    "source_batch_size": source_batch,
                    "sampled_windows": sampled_windows,
                    "critical_core_id": int(critical["core_id"]),
                    "critical_core_row": int(critical["row"]),
                    "critical_core_col": int(critical["col"]),
                    "simulated_compute_busy_us": simulated_compute_us,
                    "simulated_exposed_wait_tail_us": simulated_wait_tail_us,
                    "simulated_core_attention_us": simulated_total_us,
                    "analytical_cp_us": analytical_cp_us,
                    "total_us": total_us,
                    "result_class": hybrid_row["result_class"],
                    "ramulator_backend": aggregate["ramulator_backend"],
                    "noc_backend": aggregate["noc_backend"],
                    "breakdown_contract": "critical_core_conserved_per_window",
                }
            )
    return output


def _validate_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    errors: list[str] = []
    expected = {(context, batch) for context in CONTEXTS for batch in BATCH_LABELS}
    observed = {(str(row["context"]), str(row["batch_label"])) for row in rows}
    if observed != expected:
        errors.append(f"scenario coverage mismatch: {sorted(observed)}")
    for row in rows:
        simulated = float(row["simulated_core_attention_us"])
        compute = float(row["simulated_compute_busy_us"])
        residual = float(row["simulated_exposed_wait_tail_us"])
        cp_us = float(row["analytical_cp_us"])
        total = float(row["total_us"])
        if not _close(compute + residual, simulated):
            errors.append(f"simulated component mismatch: {row}")
        if not _close(simulated + cp_us, total):
            errors.append(f"total component mismatch: {row}")
        if row["batch_label"] == "128+CP2":
            if cp_us <= 0.0 or row["result_class"] != "hybrid_simulator_plus_analytical_cp":
                errors.append(f"invalid hybrid row: {row}")
        elif cp_us != 0.0 or row["result_class"] != "simulator_only":
            errors.append(f"invalid direct row: {row}")
    if errors:
        raise ValueError("; ".join(errors[:5]))
    return {
        "status": "ok",
        "rows_checked": len(rows),
        "groups_checked": len(observed),
        "component_conservation_checked": True,
        "source_hybrid_identity_checked": True,
        "contexts": list(CONTEXTS),
        "batch_labels": list(BATCH_LABELS),
        "errors": [],
    }


def _plot(rows: list[dict[str, object]], output_dir: Path) -> list[Path]:
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
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(7.15, 3.25), sharey=True)
    for axis, context, panel in zip(axes, CONTEXTS, ("(a)", "(b)")):
        selected = [row for row in rows if row["context"] == context]
        order = {label: index for index, label in enumerate(BATCH_LABELS)}
        selected.sort(key=lambda row: order[str(row["batch_label"])])
        x = list(range(len(selected)))
        compute = [float(row["simulated_compute_busy_us"]) for row in selected]
        wait_tail = [
            float(row["simulated_exposed_wait_tail_us"]) for row in selected
        ]
        cp_us = [float(row["analytical_cp_us"]) for row in selected]
        simulated = [left + right for left, right in zip(compute, wait_tail)]
        totals = [left + right for left, right in zip(simulated, cp_us)]
        axis.bar(
            x,
            compute,
            width=0.64,
            color="#4C78A8",
            edgecolor="#374151",
            linewidth=0.5,
            label="Compute busy (sim.)",
        )
        axis.bar(
            x,
            wait_tail,
            width=0.64,
            bottom=compute,
            color="#F2CF5B",
            edgecolor="#374151",
            linewidth=0.5,
            hatch="//",
            label="Exposed wait + tail (sim.)",
        )
        axis.bar(
            x,
            cp_us,
            width=0.64,
            bottom=simulated,
            color="#B8B8B8",
            edgecolor="#374151",
            linewidth=0.5,
            hatch="xx",
            label="CP2 communication (analytical)",
        )
        for position, total in zip(x, totals):
            axis.annotate(
                f"{total:.1f}",
                (position, total),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.2,
            )
        axis.set_xticks(x, ("16", "32", "64", "128+CP2\n(hybrid)"))
        axis.set_xlabel("Batch configuration")
        axis.set_title(f"{panel} Context {context}, 68x68 C baseline", loc="left")
        axis.grid(axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Latency (us/window)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, ncol=3, loc="outside upper center", frameon=False)
    figure.subplots_adjust(top=0.78, bottom=0.23, left=0.10, right=0.99, wspace=0.18)
    png = output_dir / "tp1_core_attention_motivation.png"
    pdf = output_dir / "tp1_core_attention_motivation.pdf"
    figure.savefig(png, dpi=300, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return [png, pdf]


def _write_readme(output_dir: Path, rows: list[dict[str, object]]) -> Path:
    direct_rows = [row for row in rows if row["batch_label"] != "128+CP2"]
    wait_shares = [
        100.0
        * float(row["simulated_exposed_wait_tail_us"])
        / float(row["simulated_core_attention_us"])
        for row in direct_rows
    ]
    lines = [
        "# TP1 Simulator-Backed Core-Attention Motivation",
        "",
        "This artifact replaces the legacy analytical motivation breakdown with a conserved critical-path decomposition from the audited TP1 pipeline.",
        "",
        "- Scope: selected-KV core attention; method C; array 68x68; contexts 8k/16k.",
        "- Batches 16/32/64: direct Ramulator2 + ASTRA-Sim results.",
        "- 128+CP2: per-die batch-64 simulation plus separately shown analytical CP2 communication.",
        "- Simulated decomposition: critical-core compute busy plus exposed data-readiness/NoC/imbalance tail.",
        "- HBM and NoC are not split into additive bars because their events overlap in the dependency simulation.",
        "- Values are deterministic six-window means; no error bars or significance claim is made.",
        f"- Main observation: exposed wait/tail is {min(wait_shares):.1f}% to {max(wait_shares):.1f}% of direct simulated latency.",
        "",
        "## Exact Values",
        "",
        "| Context | Batch | Compute (us) | Exposed wait + tail (us) | CP2 (us) | Total (us) |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['context']} | {row['batch_label']} | "
            f"{float(row['simulated_compute_busy_us']):.3f} | "
            f"{float(row['simulated_exposed_wait_tail_us']):.3f} | "
            f"{float(row['analytical_cp_us']):.3f} | "
            f"{float(row['total_us']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Reproduction",
            "",
            "```bash",
            "bash hacmo/attention/scripts/run_tp1_simulator_motivation.sh",
            "```",
        ]
    )
    path = output_dir / "README.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pack", required=True, type=Path)
    parser.add_argument("--hybrid-pack", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--method", default="C")
    parser.add_argument("--array-side", type=int, default=68)
    return parser


def build_artifact(args: argparse.Namespace) -> Path:
    source = args.source_pack.resolve()
    hybrid = args.hybrid_pack.resolve()
    output = args.output_dir.resolve()
    source_metrics = source / "csv" / "tp1_audited_metrics.csv"
    source_cores = source / "csv" / "tp1_audited_core_metrics.csv"
    source_metadata = source / "metadata.json"
    source_validation = source / "validation.json"
    hybrid_metrics = hybrid / "csv" / "tp1_hybrid_cp2_metrics.csv"
    hybrid_metadata = hybrid / "metadata.json"
    hybrid_validation = hybrid / "validation.json"
    inputs = (
        source_metrics,
        source_cores,
        source_metadata,
        source_validation,
        hybrid_metrics,
        hybrid_metadata,
        hybrid_validation,
    )
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (source_validation, hybrid_validation):
        validation = json.loads(path.read_text(encoding="utf-8"))
        if validation.get("status") != "ok":
            raise ValueError(f"input pack is not validated: {path}")

    rows = _build_rows(
        _read_csv(source_metrics),
        _read_csv(source_cores),
        _read_csv(hybrid_metrics),
        method=args.method,
        array_side=args.array_side,
    )
    validation = _validate_rows(rows)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "tp1_core_attention_motivation.csv"
    _write_csv(csv_path, rows)
    figures = _plot(rows, output)
    readme = _write_readme(output, rows)
    metadata = {
        "status": "final_support",
        "scope": "TP1 selected-KV core-attention critical-path motivation",
        "method": args.method,
        "array": f"{args.array_side}x{args.array_side}",
        "contexts": list(CONTEXTS),
        "batch_labels": list(BATCH_LABELS),
        "breakdown": {
            "simulated_compute": "compute busy time of the core defining sequence exposed latency, divided by sampled windows",
            "simulated_wait_tail": "same critical core's exposed non-compute residual; includes overlapped data readiness, NoC, imbalance, and tail effects",
            "analytical_cp": "separate CP2 component copied from the hybrid final pack",
            "conservation": "simulated_compute + simulated_wait_tail + analytical_cp = total",
        },
        "non_additive_diagnostic_excluded": "astra_comm_us overlaps other events and is not a conserved latency component",
        "source_pack": str(source),
        "hybrid_pack": str(hybrid),
        "gpu_baseline_included": False,
        "error_bars": "none; deterministic six-window simulator means",
    }
    metadata_path = output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    validation_path = output / "validation.json"
    validation_path.write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")

    generator = Path(__file__).resolve()
    wrapper = generator.parents[1] / "scripts" / "run_tp1_simulator_motivation.sh"
    generated = [csv_path, *figures, readme, metadata_path, validation_path]
    manifest = {
        "generator": str(generator),
        "generator_sha256": _sha256(generator),
        "wrapper": str(wrapper),
        "wrapper_sha256": _sha256(wrapper) if wrapper.is_file() else None,
        "python_version": sys.version,
        "matplotlib_version": matplotlib.__version__,
        "inputs": {str(path): _sha256(path) for path in inputs},
        "outputs": {str(path): _sha256(path) for path in generated},
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return output


def main() -> None:
    output = build_artifact(_parser().parse_args())
    print(f"[TP1] simulator-backed motivation: {output}")


if __name__ == "__main__":
    main()
