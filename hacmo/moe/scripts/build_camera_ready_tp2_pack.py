#!/usr/bin/env python3
"""Build a compact camera-ready TP2 result and TP3 handoff package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
TRAFFIC_MATRIX_BUILDER = PROJECT_DIR / "tp2_core" / "build_window_traffic_matrix.py"
NETWORK_CONFIG_DIR = PROJECT_DIR / "configs" / "network"

CASE_NAME = "high_drift_uniform_pool_law_math_elem_human_aging"
METHODS = ("vllm", "moentwine", "ours_static", "ours_dynamic")
DEFAULT_TOPOLOGIES = ("4w_8x8_rail_900", "16w_4x4_rail_500")
DEFAULT_BATCH_SIZES = (16, 32, 64, 128)
DEFAULT_SEED = 17

ROUTE_PLAN_STORAGE_NOTE = (
    "consumed_at_build_time; embedded traffic matrix is the TP3 input"
)

# Model parameters of the paper-facing TP2 protocol. They must match the
# serving-epoch runner, which is what produced the authority tables.
TRAFFIC_MATRIX_MODEL = {
    "window_size": 8,
    "num_experts": 256,
    "topk": 8,
    "hidden_size": 7168,
    "bytes_per_hidden_elem": 2,
    "dispatch_bytes_per_hidden_elem": 1,
    "combine_bytes_per_hidden_elem": 2,
    "expert_intermediate_size": 2048,
    "h200_effective_tflops": 1979.0,
    "h200_hbm_bw_tbps": 4.8,
    "weight_bytes_per_elem": 1,
}


def network_config_for(topology: str) -> Path:
    """Map a topology label back to the network config the runner consumed."""
    stem = "hierarchical_" + topology.replace("_rail_", "_mesh_rail_")
    path = NETWORK_CONFIG_DIR / f"{stem}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Missing network config for {topology}: {path}")
    return path


def scrub_absolute_paths(value):
    """Replace absolute filesystem paths with None, leaving everything else.

    Solver output records where its inputs sat on the machine that ran it.
    That provenance is meaningless once the pack moves, and shipping it is how
    a release ends up naming directories only its authors have.
    """
    if isinstance(value, dict):
        return {key: scrub_absolute_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [scrub_absolute_paths(item) for item in value]
    if isinstance(value, str) and value.startswith("/"):
        return None
    return value


def copy_mapping(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Missing TP2 mapping for the handoff: {source}")
    mapping = json.loads(source.read_text(encoding="utf-8"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(scrub_absolute_paths(mapping), indent=1) + "\n", encoding="utf-8"
    )


def scrub_route_plan_provenance(matrix: dict) -> dict:
    """Drop the build host's route plan path from a generated traffic matrix.

    TP3 consumes the embedded matrix, never the route plan, so shipping the
    path would only leak a packaging directory the way the AE v2 binaries did.
    """
    params = dict(matrix.get("params", {}))
    params["route_plan_json"] = None
    params["route_plan_storage"] = ROUTE_PLAN_STORAGE_NOTE
    return {**matrix, "params": params}


def default_traffic_matrix_builder(
    route_plan: Path,
    destination: Path,
    *,
    network_config: Path,
    batch_per_die: int,
    num_dies: int,
) -> None:
    model = TRAFFIC_MATRIX_MODEL
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(TRAFFIC_MATRIX_BUILDER),
        "--network-config", str(network_config),
        "--route-plan-json", str(route_plan),
        "--window-size", str(model["window_size"]),
        "--batch-size", str(batch_per_die),
        "--base-num-requests", str(batch_per_die),
        "--num-experts", str(model["num_experts"]),
        "--num-ranks", str(num_dies),
        "--topk", str(model["topk"]),
        "--hidden-size", str(model["hidden_size"]),
        "--bytes-per-hidden-elem", str(model["bytes_per_hidden_elem"]),
        "--dispatch-bytes-per-hidden-elem",
        str(model["dispatch_bytes_per_hidden_elem"]),
        "--combine-bytes-per-hidden-elem",
        str(model["combine_bytes_per_hidden_elem"]),
        "--compute-backend", "roofline_h200",
        "--expert-intermediate-size", str(model["expert_intermediate_size"]),
        "--h200-effective-tflops", str(model["h200_effective_tflops"]),
        "--h200-hbm-bw-tbps", str(model["h200_hbm_bw_tbps"]),
        "--weight-bytes-per-elem", str(model["weight_bytes_per_elem"]),
        "--output-json", str(destination),
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing TP2 run table: {path}")
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_root(work_root: Path, topology: str, batch_size: int) -> Path:
    return work_root / "runs" / topology / f"bpd{batch_size}"


def artifact_root(work_root: Path, topology: str, batch_size: int) -> Path:
    return run_root(work_root, topology, batch_size) / CASE_NAME / topology


def comparison_by_method(path: Path) -> dict[str, dict[str, str]]:
    rows = read_csv(path)
    by_method = {row["method"]: row for row in rows}
    missing = [method for method in METHODS if method not in by_method]
    if missing:
        raise RuntimeError(f"{path}: missing methods {missing}")
    return by_method


def copy_required(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Missing compact TP3 handoff input: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def build_tp3_traffic_matrix(
    source_artifact: Path,
    destination: Path,
    *,
    topology: str,
    batch_size: int,
    seed: int,
    builder,
) -> None:
    """Derive the TP3 handoff traffic matrix from the cell's route plan.

    The diagnostic runner that produces the paper-facing cells emits route
    plans but no traffic matrix, so the pack builds one here rather than
    copying a file that never existed.
    """
    route_plan = (
        source_artifact
        / "route_plans"
        / f"seed{seed}"
        / f"ours_dynamic_bpd{batch_size}.json"
    )
    if not route_plan.is_file():
        raise FileNotFoundError(
            f"Missing TP2 ours_dynamic route plan for the TP3 handoff: {route_plan}"
        )
    metadata = json.loads(route_plan.read_text(encoding="utf-8")).get("metadata", {})
    global_batch = int(metadata.get("global_batch", 0))
    if global_batch <= 0 or global_batch % batch_size:
        raise RuntimeError(
            f"{route_plan}: cannot derive num_dies from global_batch={global_batch}"
        )
    builder(
        route_plan,
        destination,
        network_config=network_config_for(topology),
        batch_per_die=batch_size,
        num_dies=global_batch // batch_size,
    )
    if destination.is_file():
        matrix = json.loads(destination.read_text(encoding="utf-8"))
        destination.write_text(
            json.dumps(scrub_route_plan_provenance(matrix), indent=1) + "\n",
            encoding="utf-8",
        )


def build_pack(
    *,
    work_root: Path,
    output_root: Path,
    topologies: list[str],
    batch_sizes: list[int],
    tp3_mapping_input_basis_bpd: int = 64,
    seed: int = DEFAULT_SEED,
    traffic_matrix_builder=default_traffic_matrix_builder,
) -> dict[str, object]:
    work_root = work_root.resolve()
    output_root = output_root.resolve()
    if tp3_mapping_input_basis_bpd not in batch_sizes:
        raise ValueError("TP3 mapping-input basis BPD must be one of the built batches")

    detailed_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    variant_rows: list[dict[str, object]] = []

    for topology in topologies:
        for batch_size in batch_sizes:
            main_root = run_root(work_root, topology, batch_size)
            main = comparison_by_method(
                main_root / CASE_NAME / "astra_tail16.csv"
            )
            selected_moentwine = main["moentwine"]
            selected_variant = "full_her"
            no_her_total: float | None = None
            full_her_total = float(main["moentwine"]["total_us"])

            if topology.startswith("16w_"):
                no_her_path = (
                    work_root
                    / "variants"
                    / topology
                    / f"bpd{batch_size}"
                    / "no_her"
                    / CASE_NAME
                    / "astra_tail16.csv"
                )
                no_her_rows = read_csv(no_her_path)
                no_her_by_method = {row["method"]: row for row in no_her_rows}
                if "moentwine" not in no_her_by_method:
                    raise RuntimeError(f"{no_her_path}: missing moentwine row")
                no_her_row = no_her_by_method["moentwine"]
                no_her_total = float(no_her_row["total_us"])
                if no_her_total < full_her_total:
                    selected_moentwine = no_her_row
                    selected_variant = "no_her"
                variant_rows.append(
                    {
                        "topology": topology,
                        "batch_per_die": batch_size,
                        "full_her_total_us": f"{full_her_total:.6f}",
                        "no_her_total_us": f"{no_her_total:.6f}",
                        "selected_variant": selected_variant,
                        "selection_metric": "astra_tail16_total_us",
                    }
                )

            selected_rows = dict(main)
            selected_rows["moentwine"] = selected_moentwine
            vllm_total = float(selected_rows["vllm"]["total_us"])
            for method in METHODS:
                source = selected_rows[method]
                total = float(source["total_us"])
                compute = float(source["compute_us"])
                communication = float(source["communication_us"])
                deployment_variant = selected_variant if method == "moentwine" else ""
                detailed_rows.append(
                    {
                        "source_label": f"{topology}_bpd{batch_size}",
                        "source_root": str(
                            run_root(Path("."), topology, batch_size)
                        ),
                        "case": CASE_NAME,
                        "replay": "tail16",
                        "topology": topology,
                        "batch_size": batch_size,
                        "batch_scope": "per_die",
                        "batch_per_die": batch_size,
                        "global_batch": 256 * batch_size,
                        "method": method,
                        "total_us": source["total_us"],
                        "compute_us": source["compute_us"],
                        "communication_us": source["communication_us"],
                        "speedup_over_vllm": vllm_total / total,
                        "dynamic_speedup_over_static": source.get(
                            "dynamic_speedup_over_static", ""
                        ),
                        "route_plan_consumed": source.get("route_plan_consumed", ""),
                        "moentwine_deployment_policy": (
                            "camera_ready_astra_tail16_cost_gate"
                            if method == "moentwine" and topology.startswith("16w_")
                            else ""
                        ),
                        "moentwine_deployment_selected_variant": deployment_variant,
                        "moentwine_deployment_full_her_total_us": (
                            full_her_total if method == "moentwine" else ""
                        ),
                        "moentwine_deployment_no_her_total_us": (
                            no_her_total
                            if method == "moentwine" and no_her_total is not None
                            else ""
                        ),
                        "moentwine_deployment_selection_metric": (
                            "astra_tail16_total_us"
                            if method == "moentwine" and topology.startswith("16w_")
                            else ""
                        ),
                        "moentwine_deployment_selection_reason": (
                            f"astra_{selected_variant}_not_slower"
                            if method == "moentwine" and topology.startswith("16w_")
                            else ""
                        ),
                    }
                )
                summary_rows.append(
                    {
                        "topology": topology,
                        "batch_size": batch_size,
                        "batch_scope": "per_die",
                        "batch_per_die": batch_size,
                        "global_batch": 256 * batch_size,
                        "method": method,
                        "total_us": f"{total:.6f}",
                        "compute_us": f"{compute:.6f}",
                        "communication_us": f"{communication:.6f}",
                        "normalized_to_vllm": f"{total / vllm_total:.6f}",
                        "speedup_over_vllm": f"{vllm_total / total:.6f}",
                        "metric_source": "total_us",
                        "moentwine_deployment_selected_variant": deployment_variant,
                    }
                )

        basis_artifact = artifact_root(
            work_root, topology, tp3_mapping_input_basis_bpd
        )
        handoff_topology = output_root / topology
        # ours_static drives TP3; uniform drives the Overall GPU replay.
        for mapping_name in ("ours_static.json", "uniform.json"):
            copy_mapping(
                basis_artifact / "mappings" / mapping_name,
                handoff_topology / "mappings" / mapping_name,
            )
        for filename in ("metadata.json", "window_stats.json", "window_src_expert_counts.json"):
            source = basis_artifact / "mapping_inputs" / "runtime" / filename
            if filename == "metadata.json" and not source.is_file():
                continue
            copy_required(
                source,
                handoff_topology / "mapping_inputs" / "runtime" / filename,
            )
        for batch_size in batch_sizes:
            source_artifact = artifact_root(work_root, topology, batch_size)
            copy_required(
                source_artifact
                / "runtime_windows"
                / f"ours_dynamic_windows_bpd{batch_size}.json",
                handoff_topology
                / "runtime_windows"
                / f"ours_dynamic_windows_bpd{batch_size}.json",
            )
            build_tp3_traffic_matrix(
                source_artifact,
                handoff_topology
                / "traffic_matrices"
                / f"ours_dynamic_bpd{batch_size}.json",
                topology=topology,
                batch_size=batch_size,
                seed=seed,
                builder=traffic_matrix_builder,
            )

    detailed_fields = list(detailed_rows[0]) if detailed_rows else []
    summary_fields = list(summary_rows[0]) if summary_rows else []
    write_csv(
        output_root / "csv" / "tp2_high_drift_astra_comm_only_metrics.csv",
        detailed_rows,
        detailed_fields,
    )
    write_csv(
        output_root / "csv" / "tp2_high_drift_astra_comm_only_summary.csv",
        summary_rows,
        summary_fields,
    )
    write_csv(
        output_root / "paper_figures" / "paper_figure_summary.csv",
        summary_rows,
        summary_fields,
    )
    if variant_rows:
        write_csv(
            output_root / "csv" / "moentwine_16w_full_vs_no_her_variants.csv",
            variant_rows,
            list(variant_rows[0]),
        )

    generated_files = sorted(
        path.relative_to(output_root)
        for path in output_root.rglob("*")
        if path.is_file() and path.name != "candidate_manifest.json"
    )
    manifest = {
        "version": "camera-ready-v1",
        "pack_kind": "camera_ready_tp2_with_compact_tp3_handoff",
        "topologies": topologies,
        "batch_sizes": batch_sizes,
        "tp3_mapping_input_basis_bpd": tp3_mapping_input_basis_bpd,
        "moentwine_16w_selection_metric": "astra_tail16_total_us",
        "contains_route_plans": False,
        "files": {
            str(path): sha256_file(output_root / path) for path in generated_files
        },
    }
    (output_root / "candidate_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--topologies", nargs="+", default=list(DEFAULT_TOPOLOGIES))
    parser.add_argument(
        "--batch-sizes", nargs="+", type=int, default=list(DEFAULT_BATCH_SIZES)
    )
    parser.add_argument("--tp3-mapping-input-basis-bpd", type=int, default=64)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    build_pack(
        work_root=args.work_root,
        output_root=args.output_root,
        topologies=args.topologies,
        batch_sizes=args.batch_sizes,
        tp3_mapping_input_basis_bpd=args.tp3_mapping_input_basis_bpd,
        seed=args.seed,
    )
    print(f"[DONE] Camera-ready TP2 pack: {args.output_root.resolve()}")


if __name__ == "__main__":
    main()
