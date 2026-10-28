#!/usr/bin/env python3
"""Build the final TP1 B/C/S/O pack with an analytical CP extension."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-tp1-hybrid-cp")

import matplotlib
import matplotlib.pyplot as plt


METHODS = ("B", "C", "S", "O")
METHOD_LABELS = {
    "B": "B: Uniform, no cache",
    "C": "C: Uniform cache",
    "S": "S: Non-uniform mapping",
    "O": "O: + Causal prefetch",
}
METHOD_COLORS = {
    "B": "#7F7F7F",
    "C": "#A0A0A0",
    "S": "#4C78A8",
    "O": "#F58518",
}
METHOD_HATCHES = {"B": "", "C": "..", "S": "///", "O": "\\\\\\"}
CONTEXTS = ("8k", "16k")
ARRAY_SIDES = (20, 68)


@dataclass(frozen=True)
class BatchSpec:
    label: str
    global_batch: int
    cp_degree: int
    simulated_batch_per_die: int


BATCH_SPECS = (
    BatchSpec("16", 16, 1, 16),
    BatchSpec("32", 32, 1, 32),
    BatchSpec("64", 64, 1, 64),
    BatchSpec("128+CP2", 128, 2, 64),
)


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


def analytical_cp_components(
    *,
    simulated_batch_per_die: int,
    cp_degree: int,
    topk: int,
    vector_dim: int,
    element_bytes: int,
    bandwidth_tbps: float,
    hops: float,
) -> tuple[int, float]:
    if simulated_batch_per_die <= 0 or cp_degree <= 0:
        raise ValueError("batch and CP degree must be positive")
    if topk <= 0 or vector_dim <= 0 or element_bytes <= 0:
        raise ValueError("CP tensor dimensions must be positive")
    if bandwidth_tbps <= 0.0 or hops < 0.0:
        raise ValueError("CP bandwidth must be positive and hops non-negative")
    if cp_degree == 1:
        return 0, 0.0
    cp_bytes = int(
        element_bytes
        * simulated_batch_per_die
        * vector_dim
        * topk
        * (cp_degree - 1)
        / cp_degree
    )
    latency_us = cp_bytes * hops / (bandwidth_tbps * 1e12) * 1e6
    return cp_bytes, latency_us


def _source_index(metrics_path: Path) -> dict[tuple[str, int, int, str], dict[str, str]]:
    indexed: dict[tuple[str, int, int, str], dict[str, str]] = {}
    for row in _read_csv(metrics_path):
        method = row["method"]
        if method not in METHODS:
            continue
        key = (
            row["context_group"],
            int(row["array_rows"]),
            int(row["batch_size"]),
            method,
        )
        if key in indexed:
            raise ValueError(f"duplicate source metric row: {key}")
        indexed[key] = row
    return indexed


def _build_rows(
    source: dict[tuple[str, int, int, str], dict[str, str]],
    *,
    topk: int,
    vector_dim: int,
    element_bytes: int,
    cp_bandwidth_tbps: float,
    cp2_hops: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    long_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for context in CONTEXTS:
        for side in ARRAY_SIDES:
            for spec in BATCH_SPECS:
                hops = cp2_hops if spec.cp_degree == 2 else 0.0
                cp_bytes, cp_us = analytical_cp_components(
                    simulated_batch_per_die=spec.simulated_batch_per_die,
                    cp_degree=spec.cp_degree,
                    topk=topk,
                    vector_dim=vector_dim,
                    element_bytes=element_bytes,
                    bandwidth_tbps=cp_bandwidth_tbps,
                    hops=hops,
                )
                result_class = (
                    "simulator_only"
                    if spec.cp_degree == 1
                    else "hybrid_simulator_plus_analytical_cp"
                )
                totals: dict[str, float] = {}
                simulated: dict[str, float] = {}
                for method in METHODS:
                    key = (context, side, spec.simulated_batch_per_die, method)
                    if key not in source:
                        raise ValueError(f"source pack lacks required row {key}")
                    source_row = source[key]
                    simulated_us = float(source_row["mean_exposed_us"])
                    if simulated_us <= 0.0:
                        raise ValueError(f"source latency must be positive for {key}")
                    total_us = simulated_us + cp_us
                    simulated[method] = simulated_us
                    totals[method] = total_us
                    long_rows.append(
                        {
                            "context": context,
                            "array_rows": side,
                            "array_cols": side,
                            "batch_label": spec.label,
                            "global_batch": spec.global_batch,
                            "cp_degree": spec.cp_degree,
                            "simulated_batch_per_die": spec.simulated_batch_per_die,
                            "method": method,
                            "simulated_core_attention_us": simulated_us,
                            "analytical_cp_us": cp_us,
                            "total_us": total_us,
                            "analytical_cp_bytes": cp_bytes,
                            "analytical_cp_hops": hops,
                            "analytical_cp_bandwidth_tbps": cp_bandwidth_tbps,
                            "result_class": result_class,
                            "hbm_backend": source_row["ramulator_backend"],
                            "noc_backend": source_row["noc_backend"],
                            "source_batch_size": spec.simulated_batch_per_die,
                        }
                    )
                summary_rows.append(
                    {
                        "context": context,
                        "array": f"{side}x{side}",
                        "batch_label": spec.label,
                        "global_batch": spec.global_batch,
                        "cp_degree": spec.cp_degree,
                        "simulated_batch_per_die": spec.simulated_batch_per_die,
                        "result_class": result_class,
                        "analytical_cp_us": cp_us,
                        "B_simulated_us": simulated["B"],
                        "C_simulated_us": simulated["C"],
                        "S_simulated_us": simulated["S"],
                        "O_simulated_us": simulated["O"],
                        "B_us": totals["B"],
                        "C_us": totals["C"],
                        "S_us": totals["S"],
                        "O_us": totals["O"],
                        "S_vs_C_latency_reduction_pct": 100.0
                        * (1.0 - totals["S"] / totals["C"]),
                        "O_vs_S_latency_reduction_pct": 100.0
                        * (1.0 - totals["O"] / totals["S"]),
                        "O_vs_C_speedup": totals["C"] / totals["O"],
                    }
                )
    return long_rows, summary_rows


def _validate_rows(
    long_rows: list[dict[str, object]], summary_rows: list[dict[str, object]]
) -> dict[str, object]:
    errors: list[str] = []
    if len(long_rows) != 64:
        errors.append(f"expected 64 long rows, found {len(long_rows)}")
    if len(summary_rows) != 16:
        errors.append(f"expected 16 summary rows, found {len(summary_rows)}")
    grouped: dict[tuple[object, ...], set[str]] = defaultdict(set)
    for row in long_rows:
        grouped[(row["context"], row["array_rows"], row["batch_label"])].add(
            str(row["method"])
        )
        simulated_us = float(row["simulated_core_attention_us"])
        cp_us = float(row["analytical_cp_us"])
        total_us = float(row["total_us"])
        if abs(total_us - simulated_us - cp_us) > 1e-9:
            errors.append(f"component mismatch for {row}")
        if row["batch_label"] == "128+CP2":
            if row["result_class"] != "hybrid_simulator_plus_analytical_cp":
                errors.append("128+CP2 is not labeled hybrid")
            if row["source_batch_size"] != 64 or cp_us <= 0.0:
                errors.append("128+CP2 does not reuse batch 64 with positive CP")
        elif cp_us != 0.0 or row["result_class"] != "simulator_only":
            errors.append(f"direct batch contains an analytical component: {row}")
    for key, methods in grouped.items():
        if methods != set(METHODS):
            errors.append(f"incomplete method block {key}: {sorted(methods)}")
    if any(str(row["method"]) == "G" for row in long_rows):
        errors.append("GPU method G must not appear in this pack")
    if errors:
        raise ValueError("; ".join(errors[:5]))
    return {
        "status": "ok",
        "long_rows_checked": len(long_rows),
        "summary_rows_checked": len(summary_rows),
        "groups_checked": len(grouped),
        "methods": list(METHODS),
        "gpu_method_present": False,
        "component_identity_checked": True,
        "cp2_reuse_checked": True,
        "errors": [],
    }


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
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _panel_rows(
    summary_rows: list[dict[str, object]], context: str, array: str
) -> list[dict[str, object]]:
    selected = [
        row
        for row in summary_rows
        if row["context"] == context and row["array"] == array
    ]
    order = {spec.label: index for index, spec in enumerate(BATCH_SPECS)}
    return sorted(selected, key=lambda row: order[str(row["batch_label"])])


def _draw_bars(ax: object, rows: list[dict[str, object]]) -> None:
    positions = list(range(len(rows)))
    width = 0.19
    for method_index, method in enumerate(METHODS):
        offset = (method_index - 1.5) * width
        ax.bar(
            [position + offset for position in positions],
            [float(row[f"{method}_us"]) for row in rows],
            width=width,
            color=METHOD_COLORS[method],
            edgecolor="#374151",
            linewidth=0.45,
            hatch=METHOD_HATCHES[method],
            label=METHOD_LABELS[method],
        )
    tick_labels = [
        "128+CP2\n(hybrid)" if row["batch_label"] == "128+CP2" else str(row["batch_label"])
        for row in rows
    ]
    ax.set_xticks(positions, tick_labels)
    ax.grid(axis="y", color="#D1D5DB", linewidth=0.6, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)


def _plot_figures(
    summary_rows: list[dict[str, object]], fig_dir: Path
) -> list[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    context_lengths = {"8k": 8192, "16k": 16384}
    for context in CONTEXTS:
        for side in ARRAY_SIDES:
            rows = _panel_rows(summary_rows, context, f"{side}x{side}")
            fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
            _draw_bars(ax, rows)
            ax.set_xlabel("Batch configuration")
            ax.set_ylabel("Latency (us/window, lower is better)")
            ax.set_title(f"TP1 ablation (ctx={context}, array={side}x{side})")
            ax.legend(ncol=2, frameon=False, loc="upper left")
            stem = f"ablation_ctx{context_lengths[context]}_array{side}x{side}"
            png_path = fig_dir / f"{stem}.png"
            pdf_path = fig_dir / f"{stem}.pdf"
            fig.savefig(png_path, dpi=300, bbox_inches="tight")
            fig.savefig(pdf_path, bbox_inches="tight")
            plt.close(fig)
            outputs.extend((png_path, pdf_path))

    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.2), constrained_layout=True)
    panels = (("8k", 20), ("8k", 68), ("16k", 20), ("16k", 68))
    for ax, (context, side), label in zip(
        axes.flat, panels, ("(a)", "(b)", "(c)", "(d)")
    ):
        _draw_bars(ax, _panel_rows(summary_rows, context, f"{side}x{side}"))
        ax.set_title(f"{label} Context {context}, array {side}x{side}", loc="left")
        ax.set_xlabel("Batch configuration")
        ax.set_ylabel("Latency (us/window)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="outside upper center", frameon=False)
    overview_png = fig_dir / "tp1_hybrid_cp2_ablation.png"
    overview_pdf = fig_dir / "tp1_hybrid_cp2_ablation.pdf"
    fig.savefig(overview_png, dpi=300, bbox_inches="tight")
    fig.savefig(overview_pdf, bbox_inches="tight")
    plt.close(fig)
    outputs.extend((overview_png, overview_pdf))
    return outputs


def _write_report(
    output_root: Path,
    source_root: Path,
    summary_rows: list[dict[str, object]],
    cp2_us: float,
) -> Path:
    s_reductions = [float(row["S_vs_C_latency_reduction_pct"]) for row in summary_rows]
    o_reductions = [float(row["O_vs_S_latency_reduction_pct"]) for row in summary_rows]
    lines = [
        "# TP1 ASTRA-Sim + Ramulator Hybrid CP2 Final Pack",
        "",
        "Status: final derived hybrid pack. GPU+vLLM is intentionally excluded.",
        "",
        "## Scope",
        "",
        "- Methods: B/C/S/O.",
        "- Batch configurations: 16, 32, 64, and 128+CP2.",
        "- Contexts: 8k/16k; arrays: 20x20/68x68.",
        "- Batches 16/32/64 are direct selected-KV core-attention simulator results.",
        "- 128+CP2 reuses the direct per-die batch-64 simulator result and adds analytical CP2 communication.",
        "- DSA Indexer, common projections, and GPU baseline are outside this pack.",
        "",
        "## Hybrid Boundary",
        "",
        f"- Source simulator pack: `{source_root}`.",
        "- Core attention: request-aware Ramulator2 HBM3 plus congestion-aware ASTRA-Sim Mesh2D.",
        f"- CP2 analytical latency: {cp2_us:.6f} us/window, added identically to B/C/S/O.",
        "- 128+CP2 is not described as fully CP-simulated because CP traffic is not present in the routed trace.",
        "",
        "## Main Result",
        "",
        f"- S vs C latency reduction across all groups: {min(s_reductions):.2f}% to {max(s_reductions):.2f}%.",
        f"- O vs S latency reduction across all groups: {min(o_reductions):.2f}% to {max(o_reductions):.2f}%.",
        "- No error bars are claimed; source simulator points are deterministic six-window means.",
        "",
        "## Exact Values",
        "",
        "| Context | Array | Batch | B (us) | C (us) | S (us) | O (us) | CP (us) | Result class |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['context']} | {row['array']} | {row['batch_label']} | "
            f"{float(row['B_us']):.3f} | {float(row['C_us']):.3f} | "
            f"{float(row['S_us']):.3f} | {float(row['O_us']):.3f} | "
            f"{float(row['analytical_cp_us']):.3f} | {row['result_class']} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `csv/tp1_hybrid_cp2_metrics.csv`: long-form simulator/CP/total components.",
            "- `csv/tp1_hybrid_cp2_summary.csv`: compact exact B/C/S/O totals.",
            "- `figs/ablation_ctx*_array*.{png,pdf}`: individual grouped bars.",
            "- `figs/tp1_hybrid_cp2_ablation.{png,pdf}`: four-panel overview.",
            "- `metadata.json`: source simulator and analytical CP provenance.",
            "- `validation.json`: conservation and method-completeness checks.",
            "- `manifest.json`: generator, input, and output SHA-256 hashes.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "bash hacmo/attention/scripts/run_tp1_hybrid_cp_pack.sh",
            "```",
        ]
    )
    path = output_root / "RESULTS.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pack", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--cp-bandwidth-tbps", type=float, default=5.0)
    parser.add_argument("--cp2-hops", type=float, default=1.0)
    parser.add_argument("--cp-vector-dim", type=int, default=576)
    parser.add_argument("--cp-element-bytes", type=int, default=2)
    return parser


def build_pack(args: argparse.Namespace) -> Path:
    source_root = args.source_pack.resolve()
    output_root = args.output_root.resolve()
    if source_root == output_root:
        raise ValueError("source and output pack must be different directories")
    metrics_path = source_root / "csv" / "tp1_audited_metrics.csv"
    source_metadata_path = source_root / "metadata.json"
    source_validation_path = source_root / "validation.json"
    for required in (metrics_path, source_metadata_path, source_validation_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    source_validation = json.loads(source_validation_path.read_text(encoding="utf-8"))
    if source_validation.get("status") != "ok":
        raise ValueError("source simulator pack is not validated")
    if source_metadata.get("scope") != "non-CP TP1 selected-KV core attention":
        raise ValueError(f"unexpected source scope: {source_metadata.get('scope')}")
    topk = int(source_metadata["hardware"]["topk"])

    long_rows, summary_rows = _build_rows(
        _source_index(metrics_path),
        topk=topk,
        vector_dim=args.cp_vector_dim,
        element_bytes=args.cp_element_bytes,
        cp_bandwidth_tbps=args.cp_bandwidth_tbps,
        cp2_hops=args.cp2_hops,
    )
    validation = _validate_rows(long_rows, summary_rows)
    cp2_bytes, cp2_us = analytical_cp_components(
        simulated_batch_per_die=64,
        cp_degree=2,
        topk=topk,
        vector_dim=args.cp_vector_dim,
        element_bytes=args.cp_element_bytes,
        bandwidth_tbps=args.cp_bandwidth_tbps,
        hops=args.cp2_hops,
    )

    csv_dir = output_root / "csv"
    fig_dir = output_root / "figs"
    csv_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    long_csv = csv_dir / "tp1_hybrid_cp2_metrics.csv"
    summary_csv = csv_dir / "tp1_hybrid_cp2_summary.csv"
    _write_csv(long_csv, long_rows)
    _write_csv(summary_csv, summary_rows)
    _configure_matplotlib()
    figures = _plot_figures(summary_rows, fig_dir)

    generator_path = Path(__file__).resolve()
    wrapper_path = generator_path.parents[1] / "scripts" / "run_tp1_hybrid_cp_pack.sh"
    metadata = {
        "status": "final",
        "result_class": "hybrid_simulator_plus_analytical_cp",
        "scope": "TP1 selected-KV core attention; analytical CP only for 128+CP2",
        "methods": list(METHODS),
        "gpu_baseline_included": False,
        "batch_specs": [asdict(spec) for spec in BATCH_SPECS],
        "source_pack": {
            "path": str(source_root),
            "metrics_sha256": _sha256(metrics_path),
            "metadata_sha256": _sha256(source_metadata_path),
            "validation_sha256": _sha256(source_validation_path),
            "validation_status": source_validation["status"],
            "scope": source_metadata["scope"],
            "runtime_backends": source_metadata["runtime_backends"],
        },
        "cp_model": {
            "classification": "analytical component outside ASTRA-Sim/Ramulator",
            "formula": "element_bytes * per_die_batch * vector_dim * topk * (cp_degree - 1) / cp_degree",
            "latency_formula": "cp_bytes * hops / bandwidth_bytes_per_second",
            "bandwidth_tbps": args.cp_bandwidth_tbps,
            "cp2_hops": args.cp2_hops,
            "vector_dim": args.cp_vector_dim,
            "element_bytes": args.cp_element_bytes,
            "topk": topk,
            "cp2_bytes": cp2_bytes,
            "cp2_latency_us": cp2_us,
        },
        "limitations": [
            "128+CP2 preserves the old experiment convention global_batch/cp_degree = per-die simulated batch.",
            "CP2 traffic is not emitted into the routed trace and is not fully simulated.",
            "Indexer and common QKV/output projections are excluded.",
        ],
    }
    metadata_path = output_root / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    validation_path = output_root / "validation.json"
    validation_path.write_text(
        json.dumps({**validation, "source_validation_status": "ok"}, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path = _write_report(output_root, source_root, summary_rows, cp2_us)

    outputs = [long_csv, summary_csv, *figures, metadata_path, validation_path, report_path]
    input_hashes = {
        str(metrics_path): _sha256(metrics_path),
        str(source_metadata_path): _sha256(source_metadata_path),
        str(source_validation_path): _sha256(source_validation_path),
    }
    manifest = {
        "generator": str(generator_path),
        "generator_sha256": _sha256(generator_path),
        "wrapper": str(wrapper_path),
        "wrapper_sha256": _sha256(wrapper_path) if wrapper_path.is_file() else None,
        "python_version": sys.version,
        "matplotlib_version": matplotlib.__version__,
        "inputs": input_hashes,
        "outputs": {str(path): _sha256(path) for path in outputs},
        "row_counts": {"long": len(long_rows), "summary": len(summary_rows)},
        "publication_note": (
            "16/32/64 are simulator-only. 128+CP2 is simulated per-die batch 64 "
            "plus an analytical CP2 component retained in separate columns."
        ),
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return output_root


def main() -> None:
    output = build_pack(_parser().parse_args())
    print(f"[TP1] hybrid CP2 final pack: {output}")


if __name__ == "__main__":
    main()
