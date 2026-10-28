#!/usr/bin/env python3
"""Run TP2 static-solver objective-weight sensitivity.

This script evaluates HACMO's static mapper only.  It deliberately avoids
running vLLM or MoEntwine so that baseline policies remain frozen while we
study the effect of the static objective weights.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
CORE_DIR = PROJECT_DIR / "tp2_core"
DATA_TOOLS_DIR = PROJECT_DIR / "data_tools"

PREPARE_HF = DATA_TOOLS_DIR / "prepare_hf_moe_selection_traces.py"
BUILD_INPUTS = CORE_DIR / "build_mapping_inputs.py"
SOLVE_STATIC = CORE_DIR / "solve_static_mapping.py"
BUILD_TRAFFIC = CORE_DIR / "build_window_traffic_matrix.py"

DEFAULT_ALPHA = 1.0
DEFAULT_GAMMA = 0.5
DEFAULT_DELTA = 2.0
DEFAULT_LAMBDA = 1.0

NUM_EXPERTS = 256
TOPK = 8
DECODE_LEN = 128
WINDOW_SIZE = 8
BATCH_PER_DIE = 64
LOOKUP_WINDOWS = 2
EXCLUDE_LEADING_WINDOWS = 2
HIDDEN_SIZE = 7168
EXPERT_INTERMEDIATE_SIZE = 2048
H200_EFFECTIVE_TFLOPS = 1979.0
H200_HBM_BW_TBPS = 4.8
WEIGHT_BYTES_PER_ELEM = 1
BYTES_PER_HIDDEN_ELEM = 2
DISPATCH_BYTES_PER_HIDDEN_ELEM = 1
COMBINE_BYTES_PER_HIDDEN_ELEM = 2
STATIC_MAX_REPLICAS = 4
TOPK_HOT_EXPERTS = 32


def _trace_base() -> Path:
    return (
        PROJECT_DIR
        / "data"
        / "raw_traces"
        / "hf_moe_expert_selection_trace"
        / "cognitivecomputations"
        / "DeepSeek-R1-AWQ"
    )


def _settings() -> list[dict[str, Any]]:
    raw: list[tuple[str, float, float, float, float]] = [
        ("default", DEFAULT_ALPHA, DEFAULT_GAMMA, DEFAULT_DELTA, DEFAULT_LAMBDA),
    ]
    for gamma in [0.0, 0.5, 2.0]:
        raw.append((f"gamma_{gamma:g}", DEFAULT_ALPHA, gamma, DEFAULT_DELTA, DEFAULT_LAMBDA))
    for delta in [0.0, 0.5, 2.0]:
        raw.append((f"delta_{delta:g}", DEFAULT_ALPHA, DEFAULT_GAMMA, delta, DEFAULT_LAMBDA))
    for lambda_ in [0.0, 0.5, 2.0]:
        raw.append((f"lambda_{lambda_:g}", DEFAULT_ALPHA, DEFAULT_GAMMA, DEFAULT_DELTA, lambda_))
    return [
        {"setting": name, "alpha": alpha, "gamma": gamma, "delta": delta, "lambda": lambda_}
        for name, alpha, gamma, delta, lambda_ in raw
    ]


def _topologies() -> list[Path]:
    return [
        PROJECT_DIR / "configs" / "network" / "hierarchical_4w_8x8_mesh_rail_900.yaml",
        PROJECT_DIR / "configs" / "network" / "hierarchical_16w_4x4_mesh_rail_500.yaml",
    ]


def _topology_label(config_path: Path) -> str:
    stem = config_path.stem
    return stem.replace("hierarchical_", "").replace("_mesh_rail_", "_rail_")


def _num_dies(config_path: Path) -> int:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    npus_count = raw.get("npus_count", [16, 16])
    return int(npus_count[0]) * int(npus_count[1])


def _path_label(path: Path, fallback: str) -> str:
    parts = [p for p in path.parts if p not in {"/", ""}]
    if len(parts) >= 2:
        return "__".join(parts[-2:]).replace(" ", "_")
    if parts:
        return parts[-1].replace(" ", "_")
    return fallback


def _run(cmd: list[Any], *, dry_run: bool, force: bool, outputs: list[Path] | None = None) -> None:
    outputs = outputs or []
    if outputs and not force and all(p.exists() for p in outputs):
        print(f"[skip] {outputs[0]}")
        return
    printable = " ".join(str(x) for x in cmd[:10])
    if len(cmd) > 10:
        printable += " ..."
    print(f"[cmd] {printable}")
    if dry_run:
        return
    subprocess.run([str(x) for x in cmd], check=True, cwd=REPO_ROOT)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_h_vector(input_dir: Path) -> list[float]:
    raw = _read_json(input_dir / "h.json")
    return [float(raw.get(str(e), raw.get(e, 0.0))) for e in range(NUM_EXPERTS)]


def _runtime_signature(input_dir: Path) -> list[float]:
    windows = _read_json(input_dir / "window_stats.json")
    vec = [0.0] * NUM_EXPERTS
    used = 0
    for ws in windows:
        if used >= LOOKUP_WINDOWS:
            break
        if not ws.get("active", True):
            continue
        for e_str, cnt in ws.get("expert_counts", {}).items():
            e = int(e_str)
            if 0 <= e < NUM_EXPERTS:
                vec[e] += float(cnt)
        used += 1
    total = sum(vec)
    if total > 0:
        vec = [v / total for v in vec]
    return vec


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (na * nb)


def _prepare_normalized_traces(
    output_root: Path,
    *,
    dry_run: bool,
    force: bool,
) -> tuple[Path, list[tuple[str, Path, Path]]]:
    base = _trace_base()
    runtime_root = base / "livecodebench" / "execution"
    offline_root = base / "mmlu" / "abstract_algebra"
    library_roots = [
        base / "mmlu" / "abstract_algebra",
        base / "Chinese-SimpleQA" / "中华文化",
        base / "HuggingFaceH4" / "aime_2024",
        base / "mmlu_ZH_CN" / "high_school_mathematics",
    ]

    cache_root = output_root / "cache" / "normalized"
    runtime_split = cache_root / "runtime_cross_dataset"
    _run(
        [
            sys.executable,
            PREPARE_HF,
            "--trace-root",
            offline_root,
            "--runtime-trace-root",
            runtime_root,
            "--layers",
            "23",
            "--max-decode-steps",
            DECODE_LEN,
            "--runtime-max-decode-steps",
            DECODE_LEN,
            "--output-root",
            runtime_split,
        ],
        dry_run=dry_run,
        force=force,
        outputs=[
            runtime_split / "offline_normalized_traces" / "metadata.json",
            runtime_split / "runtime_normalized_traces" / "metadata.json",
        ],
    )

    library: list[tuple[str, Path, Path]] = []
    for idx, root in enumerate(library_roots):
        label = f"{idx:02d}_{_path_label(root, f'entry{idx}')}"
        split = cache_root / "static_trace_library" / label
        _run(
            [
                sys.executable,
                PREPARE_HF,
                "--trace-root",
                root,
                "--runtime-trace-root",
                root,
                "--layers",
                "23",
                "--max-decode-steps",
                DECODE_LEN,
                "--runtime-max-decode-steps",
                DECODE_LEN,
                "--output-root",
                split,
            ],
            dry_run=dry_run,
            force=force,
            outputs=[split / "offline_normalized_traces" / "metadata.json"],
        )
        library.append((label, root, split / "offline_normalized_traces"))
    return runtime_split / "runtime_normalized_traces", library


def _build_mapping_inputs(
    output_root: Path,
    runtime_norm: Path,
    library: list[tuple[str, Path, Path]],
    *,
    dry_run: bool,
    force: bool,
) -> dict[str, dict[str, Any]]:
    cache_root = output_root / "cache" / "mapping_inputs"
    out: dict[str, dict[str, Any]] = {}
    for topo_cfg in _topologies():
        label = _topology_label(topo_cfg)
        num_dies = _num_dies(topo_cfg)
        topo_cache = cache_root / label
        runtime_input = topo_cache / "runtime"
        _run(
            [
                sys.executable,
                BUILD_INPUTS,
                "--normalized-trace-dir",
                runtime_norm,
                "--num-dies",
                num_dies,
                "--num-experts",
                NUM_EXPERTS,
                "--decode-len",
                DECODE_LEN,
                "--window-size",
                WINDOW_SIZE,
                "--source-mode",
                "replicate_all_dies",
                "--output-dir",
                runtime_input,
            ],
            dry_run=dry_run,
            force=force,
            outputs=[runtime_input / "window_src_expert_counts.json", runtime_input / "window_stats.json"],
        )
        entries: list[dict[str, Any]] = []
        for entry_label, root, norm_dir in library:
            input_dir = topo_cache / "static_trace_library" / entry_label / "mapping_inputs"
            _run(
                [
                    sys.executable,
                    BUILD_INPUTS,
                    "--normalized-trace-dir",
                    norm_dir,
                    "--num-dies",
                    num_dies,
                    "--num-experts",
                    NUM_EXPERTS,
                    "--decode-len",
                    DECODE_LEN,
                    "--window-size",
                    WINDOW_SIZE,
                    "--source-mode",
                    "replicate_all_dies",
                    "--output-dir",
                    input_dir,
                ],
                dry_run=dry_run,
                force=force,
                outputs=[input_dir / "h.json", input_dir / "p_ue.json", input_dir / "s_u.json"],
            )
            entries.append({"label": entry_label, "trace_root": root, "input_dir": input_dir})
        out[label] = {
            "topology_cfg": topo_cfg,
            "num_dies": num_dies,
            "runtime_input_dir": runtime_input,
            "library_entries": entries,
        }
    return out


def _runtime_request_count(runtime_norm: Path) -> int:
    meta = _read_json(runtime_norm / "metadata.json")
    return int(meta.get("num_requests", 1))


def _solve_library_entry(
    setting: dict[str, Any],
    topo_info: dict[str, Any],
    entry: dict[str, Any],
    run_dir: Path,
    *,
    dry_run: bool,
    force: bool,
) -> Path:
    mapping_json = run_dir / "static_trace_library" / entry["label"] / "ours_static.json"
    _run(
        [
            sys.executable,
            SOLVE_STATIC,
            "--method",
            "ours_static",
            "--network-config",
            topo_info["topology_cfg"],
            "--num-experts",
            NUM_EXPERTS,
            "--p-ue-json",
            entry["input_dir"] / "p_ue.json",
            "--h-json",
            entry["input_dir"] / "h.json",
            "--s-u-json",
            entry["input_dir"] / "s_u.json",
            "--alpha",
            setting["alpha"],
            "--gamma",
            setting["gamma"],
            "--delta",
            setting["delta"],
            "--lambda",
            setting["lambda"],
            "--topk-hot-experts-for-ilp",
            TOPK_HOT_EXPERTS,
            "--placement-source-model",
            "source_agnostic",
            "--max-replicas",
            STATIC_MAX_REPLICAS,
            "--output-json",
            mapping_json,
        ],
        dry_run=dry_run,
        force=force,
        outputs=[mapping_json],
    )
    return mapping_json


def _select_mapping(
    topo_info: dict[str, Any],
    candidate_paths: dict[str, Path],
    run_dir: Path,
) -> dict[str, Any]:
    runtime_vec = _runtime_signature(topo_info["runtime_input_dir"])
    candidates: list[dict[str, Any]] = []
    for entry in topo_info["library_entries"]:
        score = _cosine(runtime_vec, _load_h_vector(entry["input_dir"]))
        candidates.append(
            {
                "label": entry["label"],
                "trace_root": str(entry["trace_root"]),
                "mapping_json": str(candidate_paths[entry["label"]]),
                "mapping_input_dir": str(entry["input_dir"]),
                "similarity": round(score, 8),
            }
        )
    selected = max(candidates, key=lambda r: float(r["similarity"]))
    selected_mapping_path = Path(selected["mapping_json"])
    final_mapping = run_dir / "mappings" / "ours_static.json"
    final_mapping.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(selected_mapping_path, final_mapping)
    selection = {
        "selection_policy": "cosine_similarity_on_early_runtime_expert_distribution",
        "lookup_windows": LOOKUP_WINDOWS,
        "selected_label": selected["label"],
        "selected_trace_root": selected["trace_root"],
        "selected_similarity": selected["similarity"],
        "candidates": candidates,
    }
    _write_json(run_dir / "static_trace_library" / "selection.json", selection)
    return selection


def _build_traffic(
    topo_info: dict[str, Any],
    run_dir: Path,
    base_num_requests: int,
    *,
    dry_run: bool,
    force: bool,
) -> Path:
    traffic_json = run_dir / "traffic_matrices" / f"ours_static_bs{BATCH_PER_DIE}.json"
    _run(
        [
            sys.executable,
            BUILD_TRAFFIC,
            "--network-config",
            topo_info["topology_cfg"],
            "--mapping-json",
            run_dir / "mappings" / "ours_static.json",
            "--window-src-expert-json",
            topo_info["runtime_input_dir"] / "window_src_expert_counts.json",
            "--window-stats-json",
            topo_info["runtime_input_dir"] / "window_stats.json",
            "--window-size",
            WINDOW_SIZE,
            "--batch-size",
            BATCH_PER_DIE,
            "--base-num-requests",
            base_num_requests,
            "--num-experts",
            NUM_EXPERTS,
            "--num-ranks",
            topo_info["num_dies"],
            "--topk",
            TOPK,
            "--hidden-size",
            HIDDEN_SIZE,
            "--bytes-per-hidden-elem",
            BYTES_PER_HIDDEN_ELEM,
            "--dispatch-bytes-per-hidden-elem",
            DISPATCH_BYTES_PER_HIDDEN_ELEM,
            "--combine-bytes-per-hidden-elem",
            COMBINE_BYTES_PER_HIDDEN_ELEM,
            "--compute-backend",
            "roofline_h200",
            "--expert-intermediate-size",
            EXPERT_INTERMEDIATE_SIZE,
            "--h200-effective-tflops",
            H200_EFFECTIVE_TFLOPS,
            "--h200-hbm-bw-tbps",
            H200_HBM_BW_TBPS,
            "--weight-bytes-per-elem",
            WEIGHT_BYTES_PER_ELEM,
            "--exclude-leading-windows",
            EXCLUDE_LEADING_WINDOWS,
            "--output-json",
            traffic_json,
        ],
        dry_run=dry_run,
        force=force,
        outputs=[traffic_json],
    )
    return traffic_json


def _mapping_stats(mapping_json: Path) -> dict[str, Any]:
    mapping = _read_json(mapping_json)
    stats = mapping.get("metadata", {}).get("solver_stats", {})
    return {
        "objective_final": stats.get("objective_final", ""),
        "phi_comm": stats.get("phi_comm", ""),
        "phi_cong": stats.get("phi_cong", ""),
        "phi_load": stats.get("phi_load", ""),
        "extra_replicas": stats.get("extra_replicas", ""),
    }


def _row_from_result(
    setting: dict[str, Any],
    topology: str,
    selection: dict[str, Any],
    traffic_json: Path,
    mapping_json: Path,
) -> dict[str, Any]:
    traffic = _read_json(traffic_json)
    summary = traffic["summary"]
    row = {
        "setting": setting["setting"],
        "alpha": setting["alpha"],
        "gamma": setting["gamma"],
        "delta": setting["delta"],
        "lambda": setting["lambda"],
        "topology": topology,
        "method": "ours_static",
        "batch_per_die": BATCH_PER_DIE,
        "source": "analytical_phase_wall",
        "total_us": summary["avg_step_total_us"],
        "dispatch_us": summary["avg_step_dispatch_us"],
        "compute_us": summary["avg_step_compute_us"],
        "combine_us": summary["avg_step_combine_us"],
        "cross_wafer_ratio": summary["cross_wafer_ratio"],
        "avg_hops": summary["avg_hops"],
        "max_rank_recv_bytes": summary.get("max_rank_recv_bytes", ""),
        "max_edge_port_bytes": summary.get("max_edge_port_bytes", ""),
        "actual_dispatch_bytes": summary.get("actual_dispatch_bytes", ""),
        "actual_combine_bytes": summary.get("actual_combine_bytes", ""),
        "num_windows": summary.get("num_windows", ""),
        "excluded_leading_windows": summary.get("excluded_leading_windows", ""),
        "excluded_window_ids": ";".join(str(x) for x in summary.get("excluded_window_ids", [])),
        "evaluated_window_ids": ";".join(str(x) for x in summary.get("evaluated_window_ids", [])),
        "selected_label": selection.get("selected_label", ""),
        "selected_similarity": selection.get("selected_similarity", ""),
        "selected_trace_root": selection.get("selected_trace_root", ""),
    }
    row.update(_mapping_stats(mapping_json))
    return row


def _add_default_deltas(rows: list[dict[str, Any]]) -> None:
    default_by_topology: dict[str, float] = {}
    for row in rows:
        if row["setting"] == "default":
            default_by_topology[row["topology"]] = float(row["total_us"])
    for row in rows:
        base = default_by_topology.get(row["topology"])
        total = float(row["total_us"])
        if base:
            row["normalized_vs_default"] = f"{total / base:.6f}"
            row["delta_vs_default_us"] = f"{total - base:.6f}"
        else:
            row["normalized_vs_default"] = ""
            row["delta_vs_default_us"] = ""


def _compact_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keep = [
        "setting",
        "gamma",
        "delta",
        "lambda",
        "topology",
        "total_us",
        "normalized_vs_default",
        "delta_vs_default_us",
        "dispatch_us",
        "compute_us",
        "combine_us",
        "cross_wafer_ratio",
        "selected_label",
    ]
    return [{k: row.get(k, "") for k in keep} for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TP2 static solver objective-weight sensitivity.")
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_DIR / "outputs" / "tp2_static_solver_weight_sweep_20260609"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Regenerate cached files and per-setting results.")
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    runtime_norm, library = _prepare_normalized_traces(output_root, dry_run=args.dry_run, force=args.force)
    topo_inputs = _build_mapping_inputs(output_root, runtime_norm, library, dry_run=args.dry_run, force=args.force)
    if args.dry_run:
        print("[dry-run] no summary written")
        return

    base_num_requests = _runtime_request_count(runtime_norm)
    settings = _settings()
    rows: list[dict[str, Any]] = []
    for setting in settings:
        for topology, topo_info in topo_inputs.items():
            run_dir = output_root / "runs" / setting["setting"] / topology
            candidate_paths: dict[str, Path] = {}
            for entry in topo_info["library_entries"]:
                candidate_paths[entry["label"]] = _solve_library_entry(
                    setting,
                    topo_info,
                    entry,
                    run_dir,
                    dry_run=args.dry_run,
                    force=args.force,
                )
            selection = _select_mapping(topo_info, candidate_paths, run_dir)
            traffic_json = _build_traffic(
                topo_info,
                run_dir,
                base_num_requests,
                dry_run=args.dry_run,
                force=args.force,
            )
            rows.append(
                _row_from_result(
                    setting,
                    topology,
                    selection,
                    traffic_json,
                    run_dir / "mappings" / "ours_static.json",
                )
            )

    _add_default_deltas(rows)
    fieldnames = [
        "setting",
        "alpha",
        "gamma",
        "delta",
        "lambda",
        "topology",
        "method",
        "batch_per_die",
        "source",
        "total_us",
        "normalized_vs_default",
        "delta_vs_default_us",
        "dispatch_us",
        "compute_us",
        "combine_us",
        "cross_wafer_ratio",
        "avg_hops",
        "max_rank_recv_bytes",
        "max_edge_port_bytes",
        "actual_dispatch_bytes",
        "actual_combine_bytes",
        "num_windows",
        "excluded_leading_windows",
        "excluded_window_ids",
        "evaluated_window_ids",
        "selected_label",
        "selected_similarity",
        "selected_trace_root",
        "objective_final",
        "phi_comm",
        "phi_cong",
        "phi_load",
        "extra_replicas",
    ]
    _write_csv(output_root / "csv" / "tp2_static_solver_weight_sweep_summary.csv", rows, fieldnames)
    _write_csv(output_root / "csv" / "tp2_static_solver_weight_sweep_compact.csv", _compact_rows(rows), list(_compact_rows(rows)[0]))
    _write_json(
        output_root / "sweep_metadata.json",
        {
            "purpose": "TP2 static-solver one-factor-at-a-time objective-weight sensitivity",
            "baseline_policy": "not evaluated; this sweep calls only ours_static static mapping and traffic evaluation",
            "default": {
                "alpha": DEFAULT_ALPHA,
                "gamma": DEFAULT_GAMMA,
                "delta": DEFAULT_DELTA,
                "lambda": DEFAULT_LAMBDA,
            },
            "settings": settings,
            "trace_setting": {
                "runtime_trace_root": str(_trace_base() / "livecodebench" / "execution"),
                "static_trace_library_roots": [str(root) for _, root, _ in library],
                "hf_trace_layers": ["23"],
                "decode_len": DECODE_LEN,
                "window_size": WINDOW_SIZE,
                "static_lookup_windows": LOOKUP_WINDOWS,
                "exclude_leading_windows": EXCLUDE_LEADING_WINDOWS,
                "batch_per_die": BATCH_PER_DIE,
                "base_num_requests": base_num_requests,
            },
        },
    )
    print(output_root / "csv" / "tp2_static_solver_weight_sweep_summary.csv")


if __name__ == "__main__":
    main()
