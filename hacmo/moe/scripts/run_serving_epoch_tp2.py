#!/usr/bin/env python3
"""Run TP2 under explicit serving-epoch placement-refresh semantics.

The experiment has three distinct trace roles:

1. A historical Static Mapping Library.  Each entry is solved offline from a
   typical workload profile.
2. Runtime epoch t telemetry.  This is used only to select one library entry.
3. Runtime epoch t+1 evaluation.  This is the only runtime segment used for
   reported latency.

No mapping is selected from epoch t+1, and no full static solver is rerun on the
serving critical path.
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Iterable

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
CORE_DIR = PROJECT_DIR / "tp2_core"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from build_step_inputs import (
    aggregate_steps_to_windows,
    build_continuous_step_inputs,
    build_rolling_continuous_step_inputs,
    build_sampled_step_inputs,
)
from build_window_traffic_matrix import analyze_routed_steps
from compute_model import H200RooflineConfig
from moentwine_baseline import (
    build_causal_ni_windows,
    build_conventional_tp_collective_segments,
    build_er_source_domains,
    build_moentwine_collective_segments,
    incremental_collective_accounting,
    valid_attention_tp_degrees,
)
from route_steps import build_routed_steps
from utils_topology import load_topology

METHOD_TO_SOLVER = {
    "vllm": "uniform",
    "moentwine": "moentwine_static",
    "ours_static": "ours_static",
    "ours_dynamic": "ours_static",
}
RUNTIME_WORKLOAD_MODES = (
    "fixed_request_batch",
    "continuous_request_slots",
    "rolling_continuous_request_slots",
)

TRACE_SOURCE_MODES = [
    "spread_requests",
    "replicate_all_dies",
    "empirical_per_die_sampling",
]
MOENTWINE_TP_SELECTION_POLICIES = (
    "fixed",
    "topology_predeclared",
)
MOENTWINE_DEPLOYMENT_POLICIES = (
    "full_her",
    "no_her",
    "cost_gated_her",
)
MOENTWINE_TOPOLOGY_PREDECLARED_TP = {
    (16, 4, 4): {
        "attention_tp_degree": 16,
        "attention_tp_shape": [4, 4],
        "selection_reason": "16w_4x4_mesh_uses_full_wafer_tp16",
    }
}


def source_mode_semantics(source_mode: str) -> str:
    if source_mode in {"replicate_all_dies", "empirical_per_die_sampling"}:
        return "empirical_per_die_sampling"
    return "spread_requests_to_target_dies"


def sampling_contract(source_mode: str) -> str:
    if source_mode_semantics(source_mode) == "empirical_per_die_sampling":
        return (
            "The finite trace segment is an empirical request-distribution sample. "
            "For each fixed seed and batch size, every target die independently samples "
            "batch_per_die requests with replacement. The emitted step traffic is already "
            "fully expanded and uses batch_scale=1; this does not require one unique raw trace file "
            "per global in-flight request."
        )
    return "Each normalized trace request is assigned to one target-system source die."


def runtime_sampling_contract(workload_mode: str, source_mode: str) -> str:
    if workload_mode == "fixed_request_batch":
        return sampling_contract(source_mode)
    if workload_mode == "continuous_request_slots":
        return (
            "Each target die has batch_per_die fixed serving slots. Slots start "
            "from complete epoch-t requests at balanced, seeded decode ages; when "
            "an initial request completes, a seeded complete epoch-t+1 request "
            "occupies the same slot on the next decode step. The global active "
            "batch is constant and all methods consume the same emitted steps."
        )
    if workload_mode == "rolling_continuous_request_slots":
        return (
            "Each target die has batch_per_die fixed serving slots. Slots start "
            "from complete epoch-t requests at balanced, seeded decode ages. "
            "Every completed request is replaced by a seeded complete epoch-t+1 "
            "request on the next step, and replacement requests may complete and "
            "roll over again within the serving horizon. The global active batch "
            "is constant and all methods consume the same emitted steps."
        )
    raise ValueError(f"unsupported runtime workload mode: {workload_mode}")


def base_placement_source_mode(include_epoch_t_base_entry: bool) -> str:
    if include_epoch_t_base_entry:
        return "epoch_t_telemetry_plus_static_library"
    return "static_mapping_library_only"


def ours_static_reference_scope(include_epoch_t_base_entry: bool) -> str:
    if include_epoch_t_base_entry:
        return (
            "selected from an epoch_t-derived Epoch Base Placement candidate and "
            "the Static Mapping Library using epoch_t telemetry only"
        )
    return "selected from the Static Mapping Library using epoch_t telemetry only"


def dynamic_windows_path_for_batch(topo_dir: Path, batch_per_die: int) -> Path:
    return (
        topo_dir / "runtime_windows" / f"ours_dynamic_windows_bpd{batch_per_die}.json"
    )


def moentwine_conventional_component(attention_tp_degree: int) -> str:
    return f"conventional_tp{attention_tp_degree}"


def select_moentwine_deployment(
    *,
    policy: str,
    full_her_total_us: float | None = None,
    no_her_total_us: float | None = None,
) -> dict[str, object]:
    """Select whether the MoEntwine baseline deploys full HER for this run."""
    if policy not in MOENTWINE_DEPLOYMENT_POLICIES:
        raise ValueError(f"unknown MoEntwine deployment policy: {policy}")
    if policy == "full_her":
        selected = "full_her"
        reason = "policy_forces_full_her"
    elif policy == "no_her":
        selected = "no_her"
        reason = "policy_forces_no_her"
    else:
        if full_her_total_us is None or no_her_total_us is None:
            raise ValueError("cost_gated_her requires both candidate totals")
        if full_her_total_us <= no_her_total_us:
            selected = "full_her"
            reason = "analytical_full_her_not_slower_than_no_her"
        else:
            selected = "no_her"
            reason = "analytical_no_her_lower_than_full_her"
    return {
        "policy": policy,
        "selected_variant": selected,
        "selection_reason": reason,
        "selection_metric": "analytical_step_level_die_dag_total_us",
        "full_her_total_us": (
            float(full_her_total_us) if full_her_total_us is not None else None
        ),
        "no_her_total_us": (
            float(no_her_total_us) if no_her_total_us is not None else None
        ),
    }


def roofline_config_from_args(args: argparse.Namespace) -> H200RooflineConfig:
    return H200RooflineConfig(
        hidden_size=args.hidden_size,
        expert_intermediate_size=args.expert_intermediate_size,
        effective_tflops=args.h200_effective_tflops,
        hbm_bw_tbps=args.h200_hbm_bw_tbps,
        weight_bytes_per_elem=args.weight_bytes_per_elem,
        activation_bytes_per_elem=args.bytes_per_hidden_elem,
    )


def analytical_route_total_us(
    route_plan: dict,
    *,
    topo,
    num_ranks: int,
    args: argparse.Namespace,
) -> float:
    summary = analyze_routed_steps(
        route_plan,
        topo=topo,
        roofline_cfg=roofline_config_from_args(args),
        num_ranks=num_ranks,
    )["summary"]
    return float(summary["avg_step_total_us"])


def analytical_route_collective_us(
    route_plan: dict,
    *,
    topo,
    num_ranks: int,
    args: argparse.Namespace,
) -> float:
    summary = analyze_routed_steps(
        route_plan,
        topo=topo,
        roofline_cfg=roofline_config_from_args(args),
        num_ranks=num_ranks,
    )["summary"]
    return float(summary.get("avg_step_collective_us", 0.0))


def segment_total_bytes(segments_by_step: list[list[dict]]) -> int:
    return sum(
        int(nbytes)
        for segments in segments_by_step
        for segment in segments
        for dsts in segment["messages"].values()
        for nbytes in dsts.values()
    )


def validate_moentwine_attention_tp(
    *,
    query_heads: int,
    num_dies: int,
    attention_tp_degree: int,
    attention_tp_shape: list[int],
    mesh_rows: int | None = None,
    mesh_cols: int | None = None,
) -> list[int]:
    shape = [int(value) for value in attention_tp_shape]
    if (
        len(shape) != 2
        or shape[0] <= 0
        or shape[1] <= 0
        or math.prod(shape) != attention_tp_degree
    ):
        raise ValueError(
            "MoEntwine attention TP shape must have two positive dimensions whose "
            "product equals the TP degree"
        )
    valid_degrees = valid_attention_tp_degrees(query_heads, num_dies)
    if attention_tp_degree not in valid_degrees:
        raise ValueError(
            f"MoEntwine TP={attention_tp_degree} is not supported by "
            f"query_heads={query_heads}, num_dies={num_dies}; valid={valid_degrees}"
        )
    if mesh_rows is not None and mesh_cols is not None:
        if mesh_rows % shape[0] != 0 or mesh_cols % shape[1] != 0:
            raise ValueError(
                f"MoEntwine TP shape {shape} must divide mesh=({mesh_rows},{mesh_cols})"
            )
    return valid_degrees


def resolve_moentwine_attention_tp(
    *, args: argparse.Namespace, topo
) -> dict[str, object]:
    requested_degree = int(args.moentwine_attention_tp_degree)
    requested_shape = [int(value) for value in args.moentwine_attention_tp_shape]
    policy = args.moentwine_attention_tp_selection
    if policy == "fixed":
        selected = {
            "attention_tp_degree": requested_degree,
            "attention_tp_shape": requested_shape,
            "selection_reason": "fixed_cli_or_default",
        }
    elif policy == "topology_predeclared":
        topology_key = (
            topo.cfg.num_wafers,
            topo.cfg.mesh_rows,
            topo.cfg.mesh_cols,
        )
        selected = dict(
            MOENTWINE_TOPOLOGY_PREDECLARED_TP.get(
                topology_key,
                {
                    "attention_tp_degree": requested_degree,
                    "attention_tp_shape": requested_shape,
                    "selection_reason": "topology_predeclared_fallback_to_cli_or_default",
                },
            )
        )
    else:
        raise ValueError(f"unknown MoEntwine TP selection policy: {policy}")

    selected["attention_tp_candidates"] = validate_moentwine_attention_tp(
        query_heads=args.moentwine_query_heads,
        num_dies=topo.total_dies,
        attention_tp_degree=int(selected["attention_tp_degree"]),
        attention_tp_shape=list(selected["attention_tp_shape"]),
        mesh_rows=topo.cfg.mesh_rows,
        mesh_cols=topo.cfg.mesh_cols,
    )
    selected["attention_tp_selection_policy"] = policy
    selected["batch_agnostic"] = True
    selected["uses_runtime_trace_for_selection"] = False
    return selected


def runtime_windows_path(
    topo_dir: Path,
    *,
    method: str,
    seed: int,
    batch_per_die: int,
) -> Path:
    return (
        topo_dir
        / "runtime_windows"
        / f"seed{seed}"
        / f"{method}_windows_bpd{batch_per_die}.json"
    )


def run_cmd(cmd: list[object], cwd: Path | None = None) -> None:
    print("[cmd]", " ".join(str(x) for x in cmd[:10]), "..." if len(cmd) > 10 else "")
    subprocess.run([str(x) for x in cmd], check=True, cwd=str(cwd) if cwd else None)


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def topology_label(config_path: Path) -> str:
    stem = config_path.stem
    return stem.replace("hierarchical_", "").replace("_mesh_rail_", "_rail_")


def path_label(path: Path, fallback: str) -> str:
    parts = [p for p in path.parts if p not in {"/", ""}]
    if len(parts) >= 2:
        return "__".join(parts[-2:]).replace(" ", "_")
    if parts:
        return parts[-1].replace(" ", "_")
    return fallback


def request_sort_key(path: Path) -> tuple[str, int | str]:
    try:
        return (str(path.parent), int(path.stem))
    except ValueError:
        return (str(path.parent), path.stem)


def iter_json_files(root: Path) -> list[Path]:
    return sorted(root.glob("**/*.json"), key=request_sort_key)


def selected_layers(values: list[str], all_layers: bool) -> set[str] | None:
    return None if all_layers else {str(v) for v in values}


def step_to_layer_ids(step_obj: dict, layers: set[str] | None) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for layer, rows in sorted(
        step_obj.items(), key=lambda x: int(x[0]) if str(x[0]).isdigit() else str(x[0])
    ):
        layer_key = str(layer)
        if layers is not None and layer_key not in layers:
            continue
        if not isinstance(rows, list) or not rows:
            continue
        expert_ids: list[int] = []
        for row in rows:
            if isinstance(row, list):
                expert_ids.extend(int(e) for e in row)
        if expert_ids:
            out[layer_key] = expert_ids
    return out


def filter_usable_trace_files(
    files: Iterable[Path], *, layers: set[str] | None
) -> tuple[list[Path], list[Path]]:
    usable: list[Path] = []
    skipped: list[Path] = []
    for path in files:
        raw = json.loads(path.read_text(encoding="utf-8"))
        has_selected_decode = isinstance(raw, list) and any(
            isinstance(step_obj, dict) and bool(step_to_layer_ids(step_obj, layers))
            for step_obj in raw[1:]
        )
        (usable if has_selected_decode else skipped).append(path)
    return usable, skipped


def write_normalized_requests(
    out_dir: Path,
    files: Iterable[Path],
    *,
    layers: set[str] | None,
    max_decode_steps: int | None,
    role: str,
    source_root: Path,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    source_files: list[str] = []
    for path in files:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            continue
        decode = []
        for step_idx, step_obj in enumerate(raw[1:], start=1):
            if max_decode_steps is not None and len(decode) >= max_decode_steps:
                break
            if not isinstance(step_obj, dict):
                continue
            layer_to_ids = step_to_layer_ids(step_obj, layers)
            if not layer_to_ids:
                continue
            decode.append(
                {"token_idx": int(step_idx), "layer_to_expert_ids": layer_to_ids}
            )
        if not decode:
            continue
        obj = {
            "request_id": f"hf_{path.parent.name}_{path.stem}",
            "source_file": str(path),
            "prefill": {"layer_to_expert_ids": {}},
            "decode": decode,
        }
        (out_dir / f"hf_req{count:05d}.json").write_text(
            json.dumps(obj, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        source_files.append(str(path))
        count += 1
    write_json(
        out_dir / "metadata.json",
        {
            "trace_source": "hf_moe_expert_selection_trace",
            "source_root": str(source_root),
            "role": role,
            "num_requests": count,
            "selected_layers": sorted(layers) if layers is not None else "all",
            "max_decode_steps": max_decode_steps,
            "source_files": source_files,
        },
    )
    return count


def build_mapping_inputs(
    *,
    normalized_dir: Path,
    input_dir: Path,
    num_dies: int,
    num_experts: int,
    decode_len: int,
    window_size: int,
    source_mode: str,
) -> dict:
    run_cmd(
        [
            sys.executable,
            CORE_DIR / "build_mapping_inputs.py",
            "--normalized-trace-dir",
            normalized_dir,
            "--num-dies",
            num_dies,
            "--num-experts",
            num_experts,
            "--decode-len",
            decode_len,
            "--window-size",
            window_size,
            "--source-mode",
            source_mode,
            "--output-dir",
            input_dir,
        ]
    )
    return json.loads((input_dir / "metadata.json").read_text(encoding="utf-8"))


def load_normalized_requests(normalized_dir: Path) -> list[dict]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(normalized_dir.glob("hf_req*.json"), key=request_sort_key)
    ]


def build_sampled_runtime_inputs(
    *,
    normalized_dir: Path,
    initial_normalized_dir: Path | None,
    output_dir: Path,
    num_dies: int,
    batch_per_die: int,
    num_experts: int,
    decode_len: int,
    window_size: int,
    seed: int,
    workload_mode: str,
    runtime_serving_steps: int | None = None,
) -> dict:
    replacement_requests = load_normalized_requests(normalized_dir)
    if workload_mode == "fixed_request_batch":
        sampled = build_sampled_step_inputs(
            req_payloads=replacement_requests,
            num_dies=num_dies,
            batch_per_die=batch_per_die,
            decode_len=decode_len,
            window_size=window_size,
            seed=seed,
        )
    elif workload_mode == "continuous_request_slots":
        if initial_normalized_dir is None:
            raise ValueError(
                "continuous_request_slots requires the epoch-t normalized request pool"
            )
        sampled = build_continuous_step_inputs(
            initial_req_payloads=load_normalized_requests(initial_normalized_dir),
            replacement_req_payloads=replacement_requests,
            num_dies=num_dies,
            batch_per_die=batch_per_die,
            decode_len=decode_len,
            window_size=window_size,
            seed=seed,
            initial_domain="epoch_t",
            replacement_domain="epoch_t1",
        )
    elif workload_mode == "rolling_continuous_request_slots":
        if initial_normalized_dir is None:
            raise ValueError(
                "rolling_continuous_request_slots requires the epoch-t normalized request pool"
            )
        sampled = build_rolling_continuous_step_inputs(
            initial_req_payloads=load_normalized_requests(initial_normalized_dir),
            replacement_req_payloads=replacement_requests,
            num_dies=num_dies,
            batch_per_die=batch_per_die,
            request_decode_len=decode_len,
            serving_steps=runtime_serving_steps or decode_len,
            window_size=window_size,
            seed=seed,
            initial_domain="epoch_t",
            replacement_domain="epoch_t1",
        )
    else:
        raise ValueError(f"unknown runtime workload mode: {workload_mode}")
    window_stats, window_src_expert = aggregate_steps_to_windows(
        sampled["steps"], window_size=window_size
    )
    write_json(output_dir / "step_inputs.json", sampled)
    write_json(output_dir / "window_stats.json", window_stats)
    write_json(output_dir / "window_src_expert_counts.json", window_src_expert)
    write_json(output_dir / "metadata.json", sampled["metadata"])
    write_json(
        output_dir / "trace_statistics.json",
        sampled_window_statistics(window_stats, num_experts=num_experts),
    )
    hashed_files = [
        "step_inputs.json",
        "window_stats.json",
        "window_src_expert_counts.json",
        "metadata.json",
        "trace_statistics.json",
    ]
    write_json(
        output_dir / "input_hashes.json",
        {filename: sha256_file(output_dir / filename) for filename in hashed_files},
    )
    return sampled


def write_moentwine_primary_mapping(
    *,
    output_json: Path,
    num_experts: int,
    num_dies: int,
    query_heads: int,
    attention_tp_degree: int,
    attention_tp_shape: list[int],
    attention_tp_selection_policy: str = "fixed",
    attention_tp_selection_reason: str = "fixed_cli_or_default",
) -> dict:
    attention_tp_shape = [int(value) for value in attention_tp_shape]
    valid_degrees = validate_moentwine_attention_tp(
        query_heads=query_heads,
        num_dies=num_dies,
        attention_tp_degree=attention_tp_degree,
        attention_tp_shape=attention_tp_shape,
    )
    mapping = {expert: [expert % num_dies] for expert in range(num_experts)}
    obj = {
        "method": "moentwine",
        "expert_to_dies": {str(expert): dies for expert, dies in mapping.items()},
        "metadata": {
            "primary_placement": "deterministic_uniform_independent_of_expert_popularity",
            "er_her_model": "entwined_tp_groups_with_compact_full_token_domains",
            "query_head_tp_assumption": query_heads,
            "attention_tp_degree": attention_tp_degree,
            "attention_tp_shape": list(attention_tp_shape),
            "attention_tp_candidates": valid_degrees,
            "attention_tp_selection_policy": attention_tp_selection_policy,
            "attention_tp_selection_reason": attention_tp_selection_reason,
            "attention_tp_selection_uses_runtime_trace": False,
            "ni_balancer": "causal_runtime_shadow_replicas",
            "uses_hacmo_objective": False,
        },
    }
    write_json(output_json, obj)
    return obj


def load_h_vector(input_dir: Path, num_experts: int) -> list[float]:
    raw = json.loads((input_dir / "h.json").read_text(encoding="utf-8"))
    return [float(raw.get(str(e), raw.get(e, 0.0))) for e in range(num_experts)]


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("cosine vectors must have the same length")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (na * nb)


def popularity_order(vector: list[float]) -> list[int]:
    return sorted(range(len(vector)), key=lambda expert: (-vector[expert], expert))


def popularity_aligned_cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("popularity vectors must have the same length")
    a_sorted = [a[expert] for expert in popularity_order(a)]
    b_sorted = [b[expert] for expert in popularity_order(b)]
    return cosine(a_sorted, b_sorted)


def distribution_shift_metrics(
    epoch_t: list[float], epoch_t1: list[float], *, topks: tuple[int, ...] = (16, 32)
) -> dict[str, object]:
    if len(epoch_t) != len(epoch_t1):
        raise ValueError("epoch distributions must have the same length")
    epoch_t_order = popularity_order(epoch_t)
    epoch_t1_order = popularity_order(epoch_t1)
    overlaps = {
        f"top{k}_overlap": len(set(epoch_t_order[:k]) & set(epoch_t1_order[:k]))
        for k in topks
        if 0 < k <= len(epoch_t)
    }
    return {
        "expert_id_aligned_cosine": cosine(epoch_t, epoch_t1),
        "popularity_rank_aligned_cosine": popularity_aligned_cosine(epoch_t, epoch_t1),
        "epoch_t_hottest_expert": epoch_t_order[0],
        "epoch_t1_hottest_expert": epoch_t1_order[0],
        **overlaps,
    }


def sampled_window_statistics(
    windows: list[dict], *, num_experts: int
) -> dict[str, object]:
    if not windows:
        raise ValueError("sampled runtime input has no windows")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")

    vectors = [
        [
            float(window.get("expert_counts", {}).get(str(expert), 0))
            for expert in range(num_experts)
        ]
        for window in windows
    ]
    adjacent = [
        {
            "pair": f"{index}->{index + 1}",
            **distribution_shift_metrics(vectors[index], vectors[index + 1]),
        }
        for index in range(len(vectors) - 1)
    ]
    metric_names = [
        "expert_id_aligned_cosine",
        "popularity_rank_aligned_cosine",
        "top16_overlap",
        "top32_overlap",
    ]
    adjacent_summary = {
        f"{metric}_mean": statistics.fmean(
            float(row[metric]) for row in adjacent
        )
        for metric in metric_names
        if adjacent and metric in adjacent[0]
    }
    adjacent_summary.update(
        {
            f"{metric}_min": min(float(row[metric]) for row in adjacent)
            for metric in metric_names
            if adjacent and metric in adjacent[0]
        }
    )
    return {
        "first_to_last": distribution_shift_metrics(vectors[0], vectors[-1]),
        "adjacent_windows": adjacent,
        "adjacent_summary": adjacent_summary,
        "tokens_per_window": [
            sum(int(value) for value in window.get("tokens_per_die", {}).values())
            for window in windows
        ],
    }


def transfer_mapping_by_popularity(
    library_mapping: dict[int | str, list[int]],
    library_h: list[float],
    telemetry_h: list[float],
) -> tuple[dict[int, list[int]], list[dict[str, int | float]]]:
    if len(library_h) != len(telemetry_h):
        raise ValueError(
            "library and telemetry popularity vectors must have the same length"
        )

    normalized_mapping = {
        int(expert): list(dies) for expert, dies in library_mapping.items()
    }
    library_order = popularity_order(library_h)
    telemetry_order = popularity_order(telemetry_h)
    missing = [expert for expert in library_order if expert not in normalized_mapping]
    if missing:
        raise ValueError(f"library mapping is missing experts: {missing[:8]}")

    transferred: dict[int, list[int]] = {}
    permutation: list[dict[str, int | float]] = []
    for rank, (library_expert, runtime_expert) in enumerate(
        zip(library_order, telemetry_order)
    ):
        transferred[runtime_expert] = list(normalized_mapping[library_expert])
        permutation.append(
            {
                "rank": rank,
                "library_expert": library_expert,
                "runtime_expert": runtime_expert,
                "library_frequency": library_h[library_expert],
                "runtime_frequency": telemetry_h[runtime_expert],
            }
        )
    return transferred, permutation


def select_request_segments(
    files: list[Path],
    *,
    epoch_t_start: int,
    epoch_t_requests: int,
    epoch_t1_requests: int | None,
    epoch_t1_files: list[Path] | None = None,
    epoch_t1_start: int = 0,
) -> tuple[list[Path], list[Path]]:
    if epoch_t_start < 0 or epoch_t1_start < 0:
        raise ValueError("epoch starts must be non-negative")
    if epoch_t_requests <= 0:
        raise ValueError("epoch_t_requests must be positive")
    start = epoch_t_start
    mid = start + epoch_t_requests
    epoch_t = files[start:mid]
    if epoch_t1_files is None:
        evaluation_pool = files
        evaluation_start = mid
    else:
        evaluation_pool = epoch_t1_files
        evaluation_start = epoch_t1_start
    evaluation_end = (
        len(evaluation_pool)
        if epoch_t1_requests is None
        else evaluation_start + epoch_t1_requests
    )
    epoch_t1 = evaluation_pool[evaluation_start:evaluation_end]
    if len(epoch_t) != epoch_t_requests:
        raise ValueError(
            f"epoch_t has {len(epoch_t)} files, expected {epoch_t_requests}"
        )
    if not epoch_t1:
        raise ValueError("epoch_t1 evaluation segment is empty")
    overlap = {p.resolve() for p in epoch_t} & {p.resolve() for p in epoch_t1}
    if overlap:
        raise ValueError(
            f"epoch_t and epoch_t1 overlap: {sorted(str(p) for p in overlap)[:3]}"
        )
    return epoch_t, epoch_t1


def solve_mapping(
    *,
    method: str,
    topo_cfg: Path,
    input_dir: Path,
    output_json: Path,
    args: argparse.Namespace,
    max_replicas: int,
) -> None:
    cmd: list[object] = [
        sys.executable,
        CORE_DIR / "solve_static_mapping.py",
        "--method",
        method,
        "--network-config",
        topo_cfg,
        "--num-experts",
        args.num_experts,
        "--p-ue-json",
        input_dir / "p_ue.json",
        "--h-json",
        input_dir / "h.json",
        "--s-u-json",
        input_dir / "s_u.json",
        "--alpha",
        args.alpha,
        "--gamma",
        args.gamma,
        "--delta",
        args.delta,
        "--lambda",
        args.lambda_,
        "--topk-hot-experts-for-ilp",
        args.topk_hot_experts_for_ilp,
        "--placement-source-model",
        args.placement_source_model,
        "--max-replicas",
        max_replicas,
        "--expert-weight-bytes",
        3
        * args.hidden_size
        * args.expert_intermediate_size
        * args.weight_bytes_per_elem,
        "--die-memory-capacity-gib",
        args.die_memory_capacity_gib,
        "--die-compute-capacity",
        args.die_compute_capacity,
        "--refinement-max-passes",
        args.refinement_max_passes,
        "--output-json",
        output_json,
    ]
    run_cmd(cmd)


def run_runtime_scheduler(
    *,
    topo_cfg: Path,
    static_mapping_json: Path,
    runtime_input_dir: Path,
    output_json: Path,
    batch_per_die: int,
    args: argparse.Namespace,
) -> None:
    run_cmd(
        [
            sys.executable,
            CORE_DIR / "runtime_scheduler.py",
            "--network-config",
            topo_cfg,
            "--static-mapping-json",
            static_mapping_json,
            "--window-stats-json",
            runtime_input_dir / "window_stats.json",
            "--window-src-expert-json",
            runtime_input_dir / "window_src_expert_counts.json",
            "--step-inputs-json",
            runtime_input_dir / "step_inputs.json",
            "--ewma-decay",
            args.ewma_decay,
            "--hot-percentile",
            args.hot_percentile,
            "--bytes-per-token",
            args.hidden_size * args.combine_bytes_per_hidden_elem,
            "--max-replicas-per-expert",
            args.max_replicas_per_expert,
            "--alpha",
            args.alpha,
            "--gamma",
            args.gamma,
            "--delta",
            args.delta,
            "--lambda",
            args.lambda_,
            "--critical-load-weight",
            args.critical_load_weight,
            "--runtime-compute-batch-scale",
            1.0,
            "--dispatch-bytes-per-token",
            args.hidden_size * args.dispatch_bytes_per_hidden_elem,
            "--route-communication-weight",
            args.route_communication_weight,
            "--route-compute-load-weight",
            args.route_compute_load_weight,
            "--route-estimation-decisions-per-source-expert",
            args.route_estimation_decisions_per_source_expert,
            "--hidden-size",
            args.hidden_size,
            "--expert-intermediate-size",
            args.expert_intermediate_size,
            "--h200-effective-tflops",
            args.h200_effective_tflops,
            "--h200-hbm-bw-tbps",
            args.h200_hbm_bw_tbps,
            "--weight-bytes-per-elem",
            args.weight_bytes_per_elem,
            "--activation-bytes-per-elem",
            args.bytes_per_hidden_elem,
            "--max-hot-experts-per-window",
            args.max_hot_experts_per_window,
            "--max-candidate-dies-per-expert",
            args.max_candidate_dies_per_expert,
            "--max-candidate-operations-per-window",
            args.max_candidate_operations_per_window,
            "--attention-migration-slack-us-per-window",
            args.attention_migration_slack_us_per_window,
            "--output-json",
            output_json,
        ]
    )


def run_analytical_method(
    *,
    method: str,
    topo_cfg: Path,
    route_plan_json: Path,
    batch_per_die: int,
    num_dies: int,
    seed: int,
    output_json: Path,
    args: argparse.Namespace,
) -> dict:
    cmd: list[object] = [
        sys.executable,
        CORE_DIR / "build_window_traffic_matrix.py",
        "--network-config",
        topo_cfg,
        "--route-plan-json",
        route_plan_json,
        "--window-size",
        args.window_size,
        "--batch-size",
        batch_per_die,
        "--base-num-requests",
        batch_per_die,
        "--num-experts",
        args.num_experts,
        "--num-ranks",
        num_dies,
        "--topk",
        args.topk,
        "--hidden-size",
        args.hidden_size,
        "--bytes-per-hidden-elem",
        args.bytes_per_hidden_elem,
        "--dispatch-bytes-per-hidden-elem",
        args.dispatch_bytes_per_hidden_elem,
        "--combine-bytes-per-hidden-elem",
        args.combine_bytes_per_hidden_elem,
        "--compute-backend",
        "roofline_h200",
        "--expert-intermediate-size",
        args.expert_intermediate_size,
        "--h200-effective-tflops",
        args.h200_effective_tflops,
        "--h200-hbm-bw-tbps",
        args.h200_hbm_bw_tbps,
        "--weight-bytes-per-elem",
        args.weight_bytes_per_elem,
        "--output-json",
        output_json,
    ]
    run_cmd(cmd)
    raw = json.loads(output_json.read_text(encoding="utf-8"))
    summary = raw["summary"]
    return {
        "topology": topology_label(topo_cfg),
        "method": method,
        "seed": seed,
        "batch_per_die": batch_per_die,
        "global_batch": batch_per_die * num_dies,
        "total_us": summary["avg_step_total_us"],
        "collective_us": summary.get("avg_step_collective_us", 0.0),
        "dispatch_us": summary["avg_step_dispatch_us"],
        "compute_us": summary["avg_step_compute_us"],
        "combine_us": summary["avg_step_combine_us"],
        "cross_wafer_ratio": summary["cross_wafer_ratio"],
        "avg_hops": summary["avg_hops"],
        "max_rank_recv_bytes": summary.get("max_rank_recv_bytes", 0),
        "max_edge_port_bytes": summary.get(
            "max_edge_port_bytes", summary.get("max_edge_exit_bytes", 0)
        ),
        "metric_source": summary.get("metric_source", "step_level_die_dag"),
        "serial_upper_us": summary.get(
            "avg_step_serial_upper_us", summary["avg_step_total_us"]
        ),
        "route_plan_json": str(route_plan_json),
    }


def build_collective_only_route_plan(
    route_plan: dict,
    *,
    segments_by_step: list[list[dict]],
    component: str,
) -> dict:
    """Keep routed provenance while isolating one attention collective."""
    if len(segments_by_step) != len(route_plan.get("steps", [])):
        raise ValueError("collective segments must have one entry per routed step")
    component_plan = copy.deepcopy(route_plan)
    for step, segments in zip(component_plan["steps"], segments_by_step):
        step["pre_dispatch_segments"] = segments
        step["dispatch_bytes"] = {}
        step["combine_bytes"] = {}
    component_plan["metadata"].update(
        {
            "artifact": "tp2_collective_only_route_plan",
            "collective_component": component,
            "moe_dispatch_and_combine_disabled": True,
        }
    )
    return component_plan


def run_astra_for_topology(
    *,
    topo_cfg: Path,
    topo_dir: Path,
    batch_sizes: list[int],
    args: argparse.Namespace,
) -> Path:
    output_root = topo_dir / "supernode_astra_congestion_aware"
    output_csv = output_root / f"metrics_{topology_label(topo_cfg)}.csv"
    run_cmd(
        [
            sys.executable,
            "-m",
            "supernode_sim.experiments.run_ablation2_from_artifacts",
            "--artifact-topology-dir",
            topo_dir,
            "--network-config",
            topo_cfg,
            "--methods",
            *args.methods,
            "--batch-sizes",
            *batch_sizes,
            "--batch-scope",
            "per_die",
            "--seeds",
            *args.seeds,
            "--window-size",
            args.window_size,
            "--decode-len",
            args.runtime_serving_steps or args.decode_len,
            "--num-experts",
            args.num_experts,
            "--topk",
            args.topk,
            "--hidden-size",
            args.hidden_size,
            "--compute-backend",
            "roofline_h200",
            "--dispatch-bytes-per-hidden-elem",
            args.dispatch_bytes_per_hidden_elem,
            "--combine-bytes-per-hidden-elem",
            args.combine_bytes_per_hidden_elem,
            "--expert-intermediate-size",
            args.expert_intermediate_size,
            "--h200-effective-tflops",
            args.h200_effective_tflops,
            "--h200-hbm-bw-tbps",
            args.h200_hbm_bw_tbps,
            "--weight-bytes-per-elem",
            args.weight_bytes_per_elem,
            "--activation-bytes-per-elem",
            args.bytes_per_hidden_elem,
            "--astra-trace-mode",
            "communication_only",
            "--astra-network-backend",
            "congestion_aware",
            "--run-astra",
            "--output-root",
            output_root,
            "--output-csv",
            output_csv,
        ],
        cwd=REPO_ROOT,
    )
    return output_csv


def run_topology(
    args: argparse.Namespace, topo_cfg: Path, output_root: Path
) -> list[dict[str, object]]:
    topo = load_topology(topo_cfg)
    num_dies = topo.total_dies
    label = topology_label(topo_cfg)
    topo_dir = output_root / label
    topo_dir.mkdir(parents=True, exist_ok=True)
    moentwine_tp = (
        resolve_moentwine_attention_tp(args=args, topo=topo)
        if "moentwine" in args.methods
        else None
    )
    moentwine_tp_degree = (
        int(moentwine_tp["attention_tp_degree"]) if moentwine_tp is not None else None
    )
    moentwine_tp_shape = (
        list(moentwine_tp["attention_tp_shape"])
        if moentwine_tp is not None
        else None
    )
    moentwine_conventional_component_name = (
        moentwine_conventional_component(moentwine_tp_degree)
        if moentwine_tp_degree is not None
        else None
    )

    epoch_t_root = Path(args.runtime_hf_trace_root)
    epoch_t1_root = (
        Path(args.epoch_t1_hf_trace_root)
        if args.epoch_t1_hf_trace_root is not None
        else epoch_t_root
    )
    layers = selected_layers(args.hf_trace_layers, args.hf_trace_all_layers)
    runtime_candidates = iter_json_files(epoch_t_root)
    runtime_files, runtime_skipped = filter_usable_trace_files(
        runtime_candidates, layers=layers
    )
    epoch_t1_candidates = (
        iter_json_files(epoch_t1_root)
        if args.epoch_t1_hf_trace_root is not None
        else None
    )
    if epoch_t1_candidates is not None:
        epoch_t1_pool, epoch_t1_skipped = filter_usable_trace_files(
            epoch_t1_candidates, layers=layers
        )
    else:
        epoch_t1_pool = None
        epoch_t1_skipped = []
    epoch_t_files, epoch_t1_files = select_request_segments(
        runtime_files,
        epoch_t_start=args.epoch_t_start,
        epoch_t_requests=args.epoch_t_requests,
        epoch_t1_requests=args.epoch_t1_requests,
        epoch_t1_files=epoch_t1_pool,
        epoch_t1_start=args.epoch_t1_start,
    )
    epoch_t_norm = topo_dir / "serving_epoch" / "epoch_t_telemetry" / "normalized"
    epoch_t1_norm = topo_dir / "serving_epoch" / "epoch_t1_eval" / "normalized"
    epoch_t_count = write_normalized_requests(
        epoch_t_norm,
        epoch_t_files,
        layers=layers,
        max_decode_steps=args.decode_len,
        role="epoch_t_telemetry",
        source_root=epoch_t_root,
    )
    epoch_t1_count = write_normalized_requests(
        epoch_t1_norm,
        epoch_t1_files,
        layers=layers,
        max_decode_steps=args.decode_len,
        role="epoch_t1_eval",
        source_root=epoch_t1_root,
    )
    if epoch_t_count != len(epoch_t_files):
        raise RuntimeError(
            f"epoch_t normalized {epoch_t_count} requests from "
            f"{len(epoch_t_files)} prevalidated files"
        )
    if epoch_t1_count != len(epoch_t1_files):
        raise RuntimeError(
            f"epoch_t1 normalized {epoch_t1_count} requests from "
            f"{len(epoch_t1_files)} prevalidated files"
        )
    epoch_t_input = topo_dir / "serving_epoch" / "epoch_t_telemetry" / "mapping_inputs"
    runtime_input = topo_dir / "mapping_inputs" / "runtime"
    epoch_t_meta = build_mapping_inputs(
        normalized_dir=epoch_t_norm,
        input_dir=epoch_t_input,
        num_dies=num_dies,
        num_experts=args.num_experts,
        decode_len=args.decode_len,
        window_size=args.window_size,
        source_mode=args.trace_source_mode,
    )
    runtime_meta = build_mapping_inputs(
        normalized_dir=epoch_t1_norm,
        input_dir=runtime_input,
        num_dies=num_dies,
        num_experts=args.num_experts,
        decode_len=args.decode_len,
        window_size=args.window_size,
        source_mode=args.trace_source_mode,
    )

    library_rows: list[dict[str, object]] = []
    telemetry_vec = load_h_vector(epoch_t_input, args.num_experts)
    epoch_t1_vec = load_h_vector(runtime_input, args.num_experts)
    epoch_shift = distribution_shift_metrics(telemetry_vec, epoch_t1_vec)
    library_root = topo_dir / "static_mapping_library"
    library_root.mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, object]] = []
    if args.include_epoch_t_base_entry:
        entry_dir = library_root / "epoch_t_telemetry_base"
        mapping_json = entry_dir / "ours_static.json"
        solve_mapping(
            method="ours_static",
            topo_cfg=topo_cfg,
            input_dir=epoch_t_input,
            output_json=mapping_json,
            args=args,
            max_replicas=args.static_max_replicas,
        )
        base_h = load_h_vector(epoch_t_input, args.num_experts)
        row = {
            "entry_id": "epoch_t_telemetry_base",
            "source_role": "epoch_t_telemetry_base",
            "trace_root": str(Path(args.runtime_hf_trace_root)),
            "num_requests": epoch_t_count,
            "mapping_json": str(mapping_json),
            "mapping_input_dir": str(epoch_t_input),
            "similarity_to_epoch_t": round(
                popularity_aligned_cosine(telemetry_vec, base_h), 8
            ),
            "similarity_metric": "cosine_on_hot_to_cold_frequency_ranks",
        }
        candidates.append(row)
        library_rows.append(row)
    for idx, root_s in enumerate(args.library_hf_trace_roots):
        root = Path(root_s)
        library_candidates = iter_json_files(root)
        files, library_skipped = filter_usable_trace_files(
            library_candidates, layers=layers
        )
        if args.library_max_requests is not None:
            files = files[: args.library_max_requests]
        if not files:
            raise ValueError(f"no library request JSON files under {root}")
        entry_dir = library_root / f"{idx:02d}_{path_label(root, f'entry{idx}')}"
        norm_dir = entry_dir / "normalized"
        input_dir = entry_dir / "mapping_inputs"
        mapping_json = entry_dir / "ours_static.json"
        entry_count = write_normalized_requests(
            norm_dir,
            files,
            layers=layers,
            max_decode_steps=args.decode_len,
            role="static_mapping_library_entry",
            source_root=root,
        )
        if entry_count != len(files):
            raise RuntimeError(
                f"static library entry {root} normalized {entry_count} requests "
                f"from {len(files)} prevalidated files"
            )
        build_mapping_inputs(
            normalized_dir=norm_dir,
            input_dir=input_dir,
            num_dies=num_dies,
            num_experts=args.num_experts,
            decode_len=args.decode_len,
            window_size=args.window_size,
            source_mode=args.trace_source_mode,
        )
        solve_mapping(
            method="ours_static",
            topo_cfg=topo_cfg,
            input_dir=input_dir,
            output_json=mapping_json,
            args=args,
            max_replicas=args.static_max_replicas,
        )
        sim = popularity_aligned_cosine(
            telemetry_vec,
            load_h_vector(input_dir, args.num_experts),
        )
        row = {
            "entry_id": idx,
            "source_role": "static_mapping_library_entry",
            "trace_root": str(root),
            "num_requests": entry_count,
            "discovered_request_files": len(library_candidates),
            "usable_request_files": len(library_candidates) - len(library_skipped),
            "skipped_unusable_request_files": [str(path) for path in library_skipped],
            "mapping_json": str(mapping_json),
            "mapping_input_dir": str(input_dir),
            "similarity_to_epoch_t": round(sim, 8),
            "similarity_metric": "cosine_on_hot_to_cold_frequency_ranks",
        }
        candidates.append(row)
        library_rows.append(row)

    selected = max(candidates, key=lambda r: float(r["similarity_to_epoch_t"]))
    mappings_dir = topo_dir / "mappings"
    mappings_dir.mkdir(parents=True, exist_ok=True)
    selected_mapping_path = Path(str(selected["mapping_json"]))
    selected_mapping_obj = json.loads(selected_mapping_path.read_text(encoding="utf-8"))
    selected_h = load_h_vector(
        Path(str(selected["mapping_input_dir"])), args.num_experts
    )
    transferred_mapping, rank_permutation = transfer_mapping_by_popularity(
        selected_mapping_obj["expert_to_dies"],
        selected_h,
        telemetry_vec,
    )
    transfer_metadata = {
        "policy": "popularity_rank_mapping_transfer",
        "selected_library_entry": selected["entry_id"],
        "selected_library_mapping_json": str(selected_mapping_path),
        "similarity_metric": selected["similarity_metric"],
        "similarity_to_epoch_t": selected["similarity_to_epoch_t"],
        "tie_break": "expert_id_ascending",
        "full_primary_and_replica_lists_transferred": True,
        "rank_permutation": rank_permutation,
    }
    transferred_mapping_obj = {
        **selected_mapping_obj,
        "expert_to_dies": {
            str(expert): dies for expert, dies in sorted(transferred_mapping.items())
        },
        "metadata": {
            **selected_mapping_obj.get("metadata", {}),
            "popularity_rank_transfer": transfer_metadata,
        },
    }
    write_json(mappings_dir / "ours_static.json", transferred_mapping_obj)

    if args.mapping_only:
        write_json(
            topo_dir / "mapping_only_metadata.json",
            {
                "mapping_only": True,
                "topology": label,
                "num_dies": num_dies,
                "selected_entry": selected,
                "epoch_t_requests": epoch_t_count,
                "static_mapping_library_roots": [
                    str(Path(root)) for root in args.library_hf_trace_roots
                ],
                "output_mapping": str(mappings_dir / "ours_static.json"),
            },
        )
        return []

    if "vllm" in args.methods:
        solve_mapping(
            method="uniform",
            topo_cfg=topo_cfg,
            input_dir=runtime_input,
            output_json=mappings_dir / "uniform.json",
            args=args,
            max_replicas=1,
        )
    if "moentwine" in args.methods:
        assert moentwine_tp is not None
        assert moentwine_tp_degree is not None
        assert moentwine_tp_shape is not None
        write_moentwine_primary_mapping(
            output_json=mappings_dir / "moentwine_static.json",
            num_experts=args.num_experts,
            num_dies=num_dies,
            query_heads=args.moentwine_query_heads,
            attention_tp_degree=moentwine_tp_degree,
            attention_tp_shape=moentwine_tp_shape,
            attention_tp_selection_policy=str(
                moentwine_tp["attention_tp_selection_policy"]
            ),
            attention_tp_selection_reason=str(moentwine_tp["selection_reason"]),
        )

    route_plan_by_run: dict[tuple[int, int, str], Path] = {}
    collective_route_plan_by_run: dict[tuple[int, int, str], Path] = {}
    dynamic_runtime_windows: dict[str, str] = {}
    moentwine_runtime_windows: dict[str, str] = {}
    rows: list[dict[str, object]] = []
    for seed in args.seeds:
        for batch_per_die in args.batch_sizes:
            sampled_input_dir = (
                topo_dir / "sampled_inputs" / f"seed{seed}" / f"bpd{batch_per_die}"
            )
            sampled = build_sampled_runtime_inputs(
                normalized_dir=epoch_t1_norm,
                initial_normalized_dir=epoch_t_norm,
                output_dir=sampled_input_dir,
                num_dies=num_dies,
                batch_per_die=batch_per_die,
                num_experts=args.num_experts,
                decode_len=args.decode_len,
                window_size=args.window_size,
                seed=seed,
                workload_mode=args.runtime_workload_mode,
                runtime_serving_steps=args.runtime_serving_steps,
            )
            window_stats = json.loads(
                (sampled_input_dir / "window_stats.json").read_text(encoding="utf-8")
            )

            runtime_json = None
            if "ours_dynamic" in args.methods:
                runtime_json = runtime_windows_path(
                    topo_dir,
                    method="ours_dynamic",
                    seed=seed,
                    batch_per_die=batch_per_die,
                )
                run_runtime_scheduler(
                    topo_cfg=topo_cfg,
                    static_mapping_json=mappings_dir / "ours_static.json",
                    runtime_input_dir=sampled_input_dir,
                    output_json=runtime_json,
                    batch_per_die=batch_per_die,
                    args=args,
                )
                dynamic_runtime_windows[f"seed{seed}_bpd{batch_per_die}"] = str(
                    runtime_json
                )

            moentwine_windows_json = None
            moentwine_source_domains = None
            if "moentwine" in args.methods:
                moentwine_mapping_obj = json.loads(
                    (mappings_dir / "moentwine_static.json").read_text(encoding="utf-8")
                )
                moentwine_windows = build_causal_ni_windows(
                    base_mapping=moentwine_mapping_obj["expert_to_dies"],
                    window_stats=window_stats,
                    topo=topo,
                    expert_weight_bytes=int(
                        3
                        * args.hidden_size
                        * args.expert_intermediate_size
                        * args.weight_bytes_per_elem
                    ),
                    max_replicas_per_expert=args.moentwine_max_replicas,
                    shadow_slots_per_die=args.moentwine_shadow_slots_per_die,
                    max_adjustments_per_window=args.moentwine_max_adjustments_per_window,
                    ewma_decay=args.moentwine_ewma_decay,
                )
                moentwine_windows_json = runtime_windows_path(
                    topo_dir,
                    method="moentwine",
                    seed=seed,
                    batch_per_die=batch_per_die,
                )
                write_json(moentwine_windows_json, moentwine_windows)
                moentwine_runtime_windows[f"seed{seed}_bpd{batch_per_die}"] = str(
                    moentwine_windows_json
                )
                assert moentwine_tp_shape is not None
                moentwine_source_domains = build_er_source_domains(
                    topo,
                    moentwine_tp_shape[0],
                    moentwine_tp_shape[1],
                    hierarchical=topo.cfg.num_wafers > 1,
                )

            for method in args.methods:
                solver_method = METHOD_TO_SOLVER[method]
                mapping_json = mappings_dir / f"{solver_method}.json"
                mapping_obj = json.loads(mapping_json.read_text(encoding="utf-8"))
                window_mappings = None
                source_domains = None
                moentwine_component_plans: dict[str, dict] = {}
                moentwine_candidate_plans: dict[str, dict] = {}
                moentwine_deployment_selection: dict[str, object] | None = None
                if method == "ours_dynamic" and runtime_json is not None:
                    window_mappings = json.loads(
                        runtime_json.read_text(encoding="utf-8")
                    )
                elif method == "moentwine" and moentwine_windows_json is not None:
                    window_mappings = json.loads(
                        moentwine_windows_json.read_text(encoding="utf-8")
                    )

                if method == "moentwine":
                    assert moentwine_tp is not None
                    assert moentwine_tp_degree is not None
                    assert moentwine_tp_shape is not None
                    assert moentwine_conventional_component_name is not None
                    hierarchical = topo.cfg.num_wafers > 1

                    def build_moentwine_plan(variant: str) -> dict:
                        if variant == "full_her":
                            return build_routed_steps(
                                steps=sampled["steps"],
                                static_mapping=mapping_obj["expert_to_dies"],
                                window_mappings=window_mappings,
                                topo=topo,
                                dispatch_bytes_per_token=(
                                    args.hidden_size
                                    * args.dispatch_bytes_per_hidden_elem
                                ),
                                combine_bytes_per_token=(
                                    args.hidden_size
                                    * args.combine_bytes_per_hidden_elem
                                ),
                                routing_policy="moentwine_er_nearest",
                                source_domains=moentwine_source_domains,
                                communication_weight=args.route_communication_weight,
                                compute_load_weight=args.route_compute_load_weight,
                                moentwine_her_wafer_local=hierarchical,
                            )
                        if variant == "no_her":
                            return build_routed_steps(
                                steps=sampled["steps"],
                                static_mapping=mapping_obj["expert_to_dies"],
                                window_mappings=window_mappings,
                                topo=topo,
                                dispatch_bytes_per_token=(
                                    args.hidden_size
                                    * args.dispatch_bytes_per_hidden_elem
                                ),
                                combine_bytes_per_token=(
                                    args.hidden_size
                                    * args.combine_bytes_per_hidden_elem
                                ),
                                routing_policy="nearest",
                                source_domains=None,
                                communication_weight=args.route_communication_weight,
                                compute_load_weight=args.route_compute_load_weight,
                                moentwine_her_wafer_local=False,
                            )
                        raise ValueError(f"unknown MoEntwine deployment variant: {variant}")

                    moentwine_candidate_plans = {"full_her": build_moentwine_plan("full_her")}
                    if args.moentwine_deployment_policy in {"no_her", "cost_gated_her"}:
                        moentwine_candidate_plans["no_her"] = build_moentwine_plan(
                            "no_her"
                        )

                    full_her_segments_by_step: list[list[dict]] = []
                    conventional_segments_by_step: list[list[dict]] = []
                    for step in moentwine_candidate_plans["full_her"]["steps"]:
                        full_her_segments_by_step.append(
                            build_moentwine_collective_segments(
                                tokens_per_die=step["tokens_per_die"],
                                topo=topo,
                                hidden_size=args.hidden_size,
                                bytes_per_hidden_elem=args.bytes_per_hidden_elem,
                                tp_rows=moentwine_tp_shape[0],
                                tp_cols=moentwine_tp_shape[1],
                            )
                        )
                        conventional_segments_by_step.append(
                            build_conventional_tp_collective_segments(
                                tokens_per_die=step["tokens_per_die"],
                                topo=topo,
                                hidden_size=args.hidden_size,
                                bytes_per_hidden_elem=args.bytes_per_hidden_elem,
                                tp_rows=moentwine_tp_shape[0],
                                tp_cols=moentwine_tp_shape[1],
                            )
                        )
                    full_her_component_plans = {
                        moentwine_conventional_component_name: build_collective_only_route_plan(
                            moentwine_candidate_plans["full_her"],
                            segments_by_step=conventional_segments_by_step,
                            component=moentwine_conventional_component_name,
                        ),
                        "full_her": build_collective_only_route_plan(
                            moentwine_candidate_plans["full_her"],
                            segments_by_step=full_her_segments_by_step,
                            component="full_her",
                        ),
                    }
                    full_her_total_us = None
                    no_her_total_us = None
                    if args.moentwine_deployment_policy == "cost_gated_her":
                        full_no_collective_total_us = analytical_route_total_us(
                            moentwine_candidate_plans["full_her"],
                            topo=topo,
                            num_ranks=num_dies,
                            args=args,
                        )
                        conventional_collective_us = analytical_route_collective_us(
                            full_her_component_plans[
                                moentwine_conventional_component_name
                            ],
                            topo=topo,
                            num_ranks=num_dies,
                            args=args,
                        )
                        full_her_collective_us = analytical_route_collective_us(
                            full_her_component_plans["full_her"],
                            topo=topo,
                            num_ranks=num_dies,
                            args=args,
                        )
                        full_her_total_us = incremental_collective_accounting(
                            no_collective_total_us=full_no_collective_total_us,
                            conventional_tp_collective_us=conventional_collective_us,
                            full_her_collective_us=full_her_collective_us,
                        )["charged_total_us"]
                        no_her_total_us = analytical_route_total_us(
                            moentwine_candidate_plans["no_her"],
                            topo=topo,
                            num_ranks=num_dies,
                            args=args,
                        )

                    moentwine_deployment_selection = select_moentwine_deployment(
                        policy=args.moentwine_deployment_policy,
                        full_her_total_us=full_her_total_us,
                        no_her_total_us=no_her_total_us,
                    )
                    selected_moentwine_variant = str(
                        moentwine_deployment_selection["selected_variant"]
                    )
                    route_plan = moentwine_candidate_plans[selected_moentwine_variant]
                    if selected_moentwine_variant == "full_her":
                        moentwine_component_plans = full_her_component_plans
                    full_her_bytes = segment_total_bytes(full_her_segments_by_step)
                    conventional_bytes = segment_total_bytes(
                        conventional_segments_by_step
                    )
                    route_plan["metadata"]["moentwine_collective"] = {
                        "tp_degree": moentwine_tp_degree,
                        "tp_shape": moentwine_tp_shape,
                        "tp_selection_policy": moentwine_tp[
                            "attention_tp_selection_policy"
                        ],
                        "tp_selection_reason": moentwine_tp["selection_reason"],
                        "tp_selection_batch_agnostic": moentwine_tp["batch_agnostic"],
                        "tp_selection_uses_runtime_trace": moentwine_tp[
                            "uses_runtime_trace_for_selection"
                        ],
                        "mode": (
                            "her"
                            if selected_moentwine_variant == "full_her"
                            and hierarchical
                            else "er"
                            if selected_moentwine_variant == "full_her"
                            else "no_her"
                        ),
                        "attention_bytes_per_hidden_elem": args.bytes_per_hidden_elem,
                        "accounting_scope": (
                            "moe_only_incremental_over_same_tp_conventional_collective"
                            if selected_moentwine_variant == "full_her"
                            else "moe_ni_balancer_without_her_collective"
                        ),
                        "formula": (
                            (
                                "moe_no_collective + "
                                f"max(full_her - {moentwine_conventional_component_name}, 0)"
                            )
                            if selected_moentwine_variant == "full_her"
                            else "moe_ni_balancer_normal_a2a"
                        ),
                        "full_her_collective_phase_instances": sum(
                            len(segments) for segments in full_her_segments_by_step
                        ),
                        "full_her_collective_bytes": full_her_bytes,
                        "conventional_collective_component": (
                            moentwine_conventional_component_name
                        ),
                        "conventional_tp_collective_phase_instances": sum(
                            len(segments) for segments in conventional_segments_by_step
                        ),
                        "conventional_tp_collective_bytes": conventional_bytes,
                        "main_route_plan_contains_collective": False,
                        "deployment_selection": moentwine_deployment_selection,
                    }
                    route_plan["metadata"][
                        "source_domain_policy"
                    ] = (
                        "moentwine_her_wafer_local_dispatch_after_inter_wafer_all_gather"
                        if selected_moentwine_variant == "full_her" and hierarchical
                        else "moentwine_er_tp_group_after_all_gather"
                        if selected_moentwine_variant == "full_her"
                        else "moentwine_no_her_normal_source_die_to_expert"
                    )
                else:
                    routing_policy = (
                        "hacmo_cost_aware"
                        if method in {"ours_static", "ours_dynamic"}
                        else "nearest"
                    )
                    route_plan = build_routed_steps(
                        steps=sampled["steps"],
                        static_mapping=mapping_obj["expert_to_dies"],
                        window_mappings=window_mappings,
                        topo=topo,
                        dispatch_bytes_per_token=(
                            args.hidden_size * args.dispatch_bytes_per_hidden_elem
                        ),
                        combine_bytes_per_token=(
                            args.hidden_size * args.combine_bytes_per_hidden_elem
                        ),
                        routing_policy=routing_policy,
                        source_domains=source_domains,
                        communication_weight=args.route_communication_weight,
                        compute_load_weight=args.route_compute_load_weight,
                        moentwine_her_wafer_local=False,
                    )
                route_plan["metadata"].update(
                    {
                        "method": method,
                        "seed": seed,
                        "batch_per_die": batch_per_die,
                        "global_batch": batch_per_die * num_dies,
                        "sampled_step_input_json": str(
                            sampled_input_dir / "step_inputs.json"
                        ),
                        "mapping_json": str(mapping_json),
                        "window_mappings_json": (
                            str(runtime_json)
                            if method == "ours_dynamic" and runtime_json is not None
                            else (
                                str(moentwine_windows_json)
                                if method == "moentwine"
                                and moentwine_windows_json is not None
                                else None
                            )
                        ),
                        "method_isolation": {
                            "uses_hacmo_static_objective": method
                            in {"ours_static", "ours_dynamic"},
                            "uses_hacmo_runtime_scheduler": method == "ours_dynamic",
                            "uses_moentwine_er_her": (
                                method == "moentwine"
                                and moentwine_deployment_selection is not None
                                and moentwine_deployment_selection.get(
                                    "selected_variant"
                                )
                                == "full_her"
                            ),
                            "uses_moentwine_ni_balancer": method == "moentwine",
                            "moentwine_deployment_policy": (
                                moentwine_deployment_selection.get("policy")
                                if moentwine_deployment_selection is not None
                                else None
                            ),
                            "moentwine_deployment_selected_variant": (
                                moentwine_deployment_selection.get(
                                    "selected_variant"
                                )
                                if moentwine_deployment_selection is not None
                                else None
                            ),
                        },
                    }
                )
                route_plan_path = (
                    topo_dir
                    / "route_plans"
                    / f"seed{seed}"
                    / f"{method}_bpd{batch_per_die}.json"
                )
                if method == "moentwine" and moentwine_candidate_plans:
                    candidate_paths: dict[str, Path] = {}
                    selected_variant = (
                        str(moentwine_deployment_selection["selected_variant"])
                        if moentwine_deployment_selection is not None
                        else "full_her"
                    )
                    for variant, candidate_plan in moentwine_candidate_plans.items():
                        if variant == selected_variant:
                            continue
                        candidate_copy = copy.deepcopy(candidate_plan)
                        candidate_copy["metadata"].update(
                            {
                                "artifact": "tp2_step_route_plan_diagnostic_candidate",
                                "method": method,
                                "seed": seed,
                                "batch_per_die": batch_per_die,
                                "global_batch": batch_per_die * num_dies,
                                "sampled_step_input_json": str(
                                    sampled_input_dir / "step_inputs.json"
                                ),
                                "mapping_json": str(mapping_json),
                                "window_mappings_json": (
                                    str(moentwine_windows_json)
                                    if moentwine_windows_json is not None
                                    else None
                                ),
                                "diagnostic_candidate_variant": variant,
                                "selected_route_plan_json": str(route_plan_path),
                                "moentwine_deployment_selection": (
                                    moentwine_deployment_selection
                                ),
                            }
                        )
                        candidate_path = route_plan_path.with_name(
                            f"moentwine_{variant}_candidate_bpd{batch_per_die}.json"
                        )
                        write_json(candidate_path, candidate_copy)
                        candidate_paths[variant] = candidate_path
                    if candidate_paths:
                        route_plan["metadata"]["moentwine_collective"][
                            "candidate_route_plans"
                        ] = {
                            key: str(path) for key, path in sorted(candidate_paths.items())
                        }
                write_json(route_plan_path, route_plan)
                route_plan_by_run[(seed, batch_per_die, method)] = route_plan_path

                component_paths: dict[str, Path] = {}
                for component, component_plan in moentwine_component_plans.items():
                    component_plan["metadata"].update(copy.deepcopy(route_plan["metadata"]))
                    component_plan["metadata"].update(
                        {
                            "artifact": "tp2_collective_only_route_plan",
                            "collective_component": component,
                            "moe_dispatch_and_combine_disabled": True,
                        }
                    )
                    component_path = route_plan_path.with_name(
                        f"moentwine_{component}_collective_only_bpd{batch_per_die}.json"
                    )
                    write_json(component_path, component_plan)
                    component_paths[component] = component_path
                    collective_route_plan_by_run[(seed, batch_per_die, component)] = (
                        component_path
                    )
                if component_paths:
                    route_plan["metadata"]["moentwine_collective"][
                        "component_route_plans"
                    ] = {key: str(path) for key, path in component_paths.items()}
                    write_json(route_plan_path, route_plan)

                traffic_json = (
                    topo_dir
                    / "traffic_matrices"
                    / f"seed{seed}"
                    / f"{method}_bpd{batch_per_die}.json"
                )
                row = run_analytical_method(
                    method=method,
                    topo_cfg=topo_cfg,
                    route_plan_json=route_plan_path,
                    batch_per_die=batch_per_die,
                    num_dies=num_dies,
                    seed=seed,
                    output_json=traffic_json,
                    args=args,
                )
                row.update(
                    {
                        "moentwine_no_collective_total_us": None,
                        "moentwine_conventional_tp_collective_us": None,
                        "moentwine_full_her_collective_us": None,
                        "moentwine_incremental_collective_us": None,
                        "moentwine_full_her_total_us": None,
                        "moentwine_conventional_tp_total_us": None,
                        "moentwine_deployment_policy": None,
                        "moentwine_deployment_selected_variant": None,
                        "moentwine_deployment_selection_reason": None,
                        "moentwine_deployment_selection_metric": None,
                        "moentwine_deployment_full_her_total_us": None,
                        "moentwine_deployment_no_her_total_us": None,
                    }
                )
                selected_deployment_variant = None
                if method == "moentwine":
                    selection_obj = (
                        route_plan.get("metadata", {})
                        .get("moentwine_collective", {})
                        .get("deployment_selection", {})
                    )
                    selected_deployment_variant = str(
                        selection_obj.get("selected_variant", "full_her")
                    )
                    row.update(
                        {
                            "moentwine_deployment_policy": selection_obj.get("policy"),
                            "moentwine_deployment_selected_variant": (
                                selected_deployment_variant
                            ),
                            "moentwine_deployment_selection_reason": selection_obj.get(
                                "selection_reason"
                            ),
                            "moentwine_deployment_selection_metric": selection_obj.get(
                                "selection_metric"
                            ),
                            "moentwine_deployment_full_her_total_us": selection_obj.get(
                                "full_her_total_us"
                            ),
                            "moentwine_deployment_no_her_total_us": selection_obj.get(
                                "no_her_total_us"
                            ),
                        }
                    )
                if component_paths and selected_deployment_variant == "full_her":
                    assert moentwine_conventional_component_name is not None
                    component_rows = {}
                    for component, component_path in component_paths.items():
                        component_rows[component] = run_analytical_method(
                            method=f"moentwine_{component}_collective_only",
                            topo_cfg=topo_cfg,
                            route_plan_json=component_path,
                            batch_per_die=batch_per_die,
                            num_dies=num_dies,
                            seed=seed,
                            output_json=traffic_json.with_name(
                                f"moentwine_{component}_collective_only_bpd{batch_per_die}.json"
                            ),
                            args=args,
                        )
                    no_collective_total_us = float(row["total_us"])
                    conventional_us = float(
                        component_rows[moentwine_conventional_component_name][
                            "collective_us"
                        ]
                    )
                    full_her_us = float(component_rows["full_her"]["collective_us"])
                    accounting = incremental_collective_accounting(
                        no_collective_total_us=no_collective_total_us,
                        conventional_tp_collective_us=conventional_us,
                        full_her_collective_us=full_her_us,
                    )
                    row.update(
                        {
                            "total_us": accounting["charged_total_us"],
                            "collective_us": accounting["incremental_collective_us"],
                            "serial_upper_us": float(row["serial_upper_us"])
                            + accounting["incremental_collective_us"],
                            "metric_source": "step_level_die_dag_with_incremental_her",
                            "moentwine_no_collective_total_us": no_collective_total_us,
                            "moentwine_conventional_tp_collective_us": conventional_us,
                            "moentwine_full_her_collective_us": full_her_us,
                            "moentwine_incremental_collective_us": accounting[
                                "incremental_collective_us"
                            ],
                            "moentwine_full_her_total_us": no_collective_total_us
                            + full_her_us,
                            "moentwine_conventional_tp_total_us": no_collective_total_us
                            + conventional_us,
                        }
                    )
                elif method == "moentwine" and selected_deployment_variant == "no_her":
                    no_collective_total_us = float(row["total_us"])
                    row.update(
                        {
                            "metric_source": "step_level_die_dag_moentwine_no_her",
                            "moentwine_no_collective_total_us": no_collective_total_us,
                            "moentwine_conventional_tp_collective_us": 0.0,
                            "moentwine_full_her_collective_us": 0.0,
                            "moentwine_incremental_collective_us": 0.0,
                            "moentwine_full_her_total_us": (
                                row["moentwine_deployment_full_her_total_us"]
                            ),
                            "moentwine_conventional_tp_total_us": None,
                        }
                    )
                rows.append(row)

    selection = {
        "selection_policy": "cosine_similarity_on_hot_to_cold_expert_frequency_ranks",
        "base_placement_source_mode": base_placement_source_mode(
            args.include_epoch_t_base_entry
        ),
        "selected_entry": selected,
        "popularity_rank_transfer": transfer_metadata,
        "transferred_mapping_json": str(mappings_dir / "ours_static.json"),
        "candidates": candidates,
        "epoch_t_telemetry": {
            "runtime_trace_root": str(epoch_t_root),
            "request_count": epoch_t_count,
            "source_files": [str(p) for p in epoch_t_files],
        },
        "epoch_t1_evaluation": {
            "runtime_trace_root": str(epoch_t1_root),
            "request_count": epoch_t1_count,
            "source_files": [str(p) for p in epoch_t1_files],
            "distribution_shift_from_epoch_t": epoch_shift,
        },
        "no_future_leak": True,
        "epoch_t1_used_for_selection": False,
        "trace_source_mode": args.trace_source_mode,
        "source_mode_semantics": source_mode_semantics(args.trace_source_mode),
        "sampling_contract": runtime_sampling_contract(
            args.runtime_workload_mode, args.trace_source_mode
        ),
        "trace_source_sampling_contract": sampling_contract(args.trace_source_mode),
    }
    write_json(topo_dir / "serving_epoch" / "selection.json", selection)
    write_csv(
        topo_dir / "serving_epoch" / "library_candidates.csv",
        library_rows,
        [
            "entry_id",
            "source_role",
            "trace_root",
            "num_requests",
            "discovered_request_files",
            "usable_request_files",
            "skipped_unusable_request_files",
            "mapping_json",
            "mapping_input_dir",
            "similarity_to_epoch_t",
            "similarity_metric",
        ],
    )
    write_json(
        topo_dir / "serving_epoch_metadata.json",
        {
            "topology": label,
            "num_dies": num_dies,
            "source_mode": args.trace_source_mode,
            "source_mode_semantics": source_mode_semantics(args.trace_source_mode),
            "sampling_contract": runtime_sampling_contract(
                args.runtime_workload_mode, args.trace_source_mode
            ),
            "trace_source_sampling_contract": sampling_contract(
                args.trace_source_mode
            ),
            "runtime_trace_root": str(epoch_t_root),
            "epoch_t1_trace_root": str(epoch_t1_root),
            "epoch_t_requests": epoch_t_count,
            "epoch_t1_requests": epoch_t1_count,
            "trace_pool_validation": {
                "criterion": "contains_selected_layer_decode_data",
                "selected_layers": (
                    sorted(layers) if layers is not None else "all"
                ),
                "epoch_t_source": {
                    "discovered_files": len(runtime_candidates),
                    "usable_files": len(runtime_files),
                    "skipped_files": [str(path) for path in runtime_skipped],
                },
                "epoch_t1_source": (
                    {
                        "discovered_files": len(epoch_t1_candidates),
                        "usable_files": len(epoch_t1_pool),
                        "skipped_files": [str(path) for path in epoch_t1_skipped],
                    }
                    if epoch_t1_candidates is not None and epoch_t1_pool is not None
                    else "shared_with_epoch_t_source"
                ),
            },
            "epoch_t_mapping_inputs": epoch_t_meta,
            "epoch_t1_mapping_inputs": runtime_meta,
            "batch_scope": "per_die",
            "runtime_workload_mode": args.runtime_workload_mode,
            "runtime_serving_steps": args.runtime_serving_steps or args.decode_len,
            "request_decode_len": args.decode_len,
            "batch_sizes": args.batch_sizes,
            "global_batches": [b * num_dies for b in args.batch_sizes],
            "static_mapping_library_roots": [
                str(Path(x)) for x in args.library_hf_trace_roots
            ],
            "include_epoch_t_base_entry": args.include_epoch_t_base_entry,
            "base_placement_source_mode": base_placement_source_mode(
                args.include_epoch_t_base_entry
            ),
            "selected_entry": selected,
            "moentwine": (
                {
                    "query_heads": args.moentwine_query_heads,
                    "attention_tp_degree": moentwine_tp_degree,
                    "attention_tp_shape": moentwine_tp_shape,
                    "attention_tp_selection_policy": moentwine_tp[
                        "attention_tp_selection_policy"
                    ],
                    "attention_tp_selection_reason": moentwine_tp[
                        "selection_reason"
                    ],
                    "attention_tp_selection_batch_agnostic": moentwine_tp[
                        "batch_agnostic"
                    ],
                    "attention_tp_selection_uses_runtime_trace": moentwine_tp[
                        "uses_runtime_trace_for_selection"
                    ],
                    "conventional_collective_component": (
                        moentwine_conventional_component_name
                    ),
                    "deployment_policy": args.moentwine_deployment_policy,
                    "moe_only_accounting": (
                        "moe_no_collective + max(full_her - same_tp_conventional, 0)"
                    ),
                }
                if moentwine_tp is not None
                else None
            ),
            "baseline_reference_scope": {
                "vllm": "uniform placement; does not use epoch_t1 telemetry for placement decisions",
                "moentwine": (
                    "popularity-independent ER/HER primary layout plus causal NI-Balancer; "
                    "window w observations affect window w+1 only"
                ),
                "ours_static": ours_static_reference_scope(
                    args.include_epoch_t_base_entry
                ),
                "ours_dynamic": "starts from ours_static and applies causal in-epoch runtime adaptation",
            },
            "evaluation_seeds": args.seeds,
            "dynamic_runtime_windows": dynamic_runtime_windows,
            "moentwine_runtime_windows": moentwine_runtime_windows,
            "route_plans": {
                f"seed{seed}_bpd{batch}_{method}": str(path)
                for (seed, batch, method), path in sorted(route_plan_by_run.items())
            },
            "moentwine_collective_component_route_plans": {
                f"seed{seed}_bpd{batch}_{component}": str(path)
                for (seed, batch, component), path in sorted(
                    collective_route_plan_by_run.items()
                )
            },
            "semantics": {
                "epoch_t": "telemetry only; selects a precomputed Static Mapping Library entry",
                "epoch_t1": "evaluation only; not used for selecting the Epoch Base Placement",
                "global_refresh": "deploy selected Epoch Base Placement at epoch boundary",
                "dynamic": "causal in-epoch local adaptation from the selected Epoch Base Placement",
            },
        },
    )

    if args.run_astra:
        astra_csv = run_astra_for_topology(
            topo_cfg=topo_cfg,
            topo_dir=topo_dir,
            batch_sizes=args.batch_sizes,
            args=args,
        )
        print(f"[astra] wrote {astra_csv}")

    return rows


def normalize_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for topology in sorted({str(r["topology"]) for r in rows}):
        topo_rows = [r for r in rows if str(r["topology"]) == topology]
        for seed in sorted({int(r["seed"]) for r in topo_rows}):
            seed_rows = [r for r in topo_rows if int(r["seed"]) == seed]
            for batch in sorted({int(r["batch_per_die"]) for r in seed_rows}):
                batch_rows = [r for r in seed_rows if int(r["batch_per_die"]) == batch]
                vllm = next((r for r in batch_rows if r["method"] == "vllm"), None)
                base = float(vllm["total_us"]) if vllm is not None else None
                for r in batch_rows:
                    total = float(r["total_us"])
                    out.append(
                        {
                            **r,
                            "normalized_to_vllm": (
                                total / max(base, 1e-9) if base is not None else None
                            ),
                            "speedup_over_vllm": (
                                base / max(total, 1e-9) if base is not None else None
                            ),
                        }
                    )
    return out


def aggregate_seed_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    metrics = [
        "total_us",
        "collective_us",
        "dispatch_us",
        "compute_us",
        "combine_us",
        "serial_upper_us",
        "normalized_to_vllm",
        "speedup_over_vllm",
        "cross_wafer_ratio",
        "avg_hops",
        "moentwine_no_collective_total_us",
        "moentwine_conventional_tp_collective_us",
        "moentwine_full_her_collective_us",
        "moentwine_incremental_collective_us",
        "moentwine_full_her_total_us",
        "moentwine_conventional_tp_total_us",
        "moentwine_deployment_full_her_total_us",
        "moentwine_deployment_no_her_total_us",
    ]
    string_fields = [
        "moentwine_deployment_policy",
        "moentwine_deployment_selected_variant",
        "moentwine_deployment_selection_reason",
        "moentwine_deployment_selection_metric",
    ]
    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["topology"]), str(row["method"]), int(row["batch_per_die"]))
        grouped.setdefault(key, []).append(row)

    out = []
    for key, samples in sorted(grouped.items()):
        topology, method, batch = key
        row: dict[str, object] = {
            "topology": topology,
            "method": method,
            "batch_per_die": batch,
            "global_batch": int(samples[0]["global_batch"]),
            "seed_count": len(samples),
            "seeds": ";".join(
                str(sample["seed"])
                for sample in sorted(samples, key=lambda x: int(x["seed"]))
            ),
            "metric_source": samples[0]["metric_source"],
        }
        for metric in metrics:
            values = [
                float(sample[metric])
                for sample in samples
                if sample.get(metric) is not None
            ]
            row[metric] = statistics.fmean(values) if values else None
            row[f"{metric}_std"] = (
                statistics.stdev(values)
                if len(values) > 1
                else (0.0 if values else None)
            )
        for field in string_fields:
            values = {
                str(sample[field])
                for sample in samples
                if sample.get(field) not in (None, "")
            }
            row[field] = ";".join(sorted(values)) if values else None
        out.append(row)
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run TP2 serving-epoch placement-refresh experiment."
    )
    p.add_argument(
        "--topologies",
        nargs="+",
        default=[
            str(PROJECT_DIR / "configs/network/hierarchical_4w_8x8_mesh_rail_900.yaml"),
            str(
                PROJECT_DIR / "configs/network/hierarchical_16w_4x4_mesh_rail_500.yaml"
            ),
        ],
    )
    p.add_argument(
        "--methods",
        nargs="+",
        default=["vllm", "moentwine", "ours_static", "ours_dynamic"],
    )
    p.add_argument("--library-hf-trace-roots", nargs="+", required=True)
    p.add_argument("--runtime-hf-trace-root", required=True)
    p.add_argument(
        "--epoch-t1-hf-trace-root",
        default=None,
        help=(
            "Optional independent epoch-t+1 evaluation trace root. If omitted, "
            "epoch t and epoch t+1 remain adjacent slices of runtime-hf-trace-root."
        ),
    )
    p.add_argument(
        "--include-epoch-t-base-entry",
        action="store_true",
        help=(
            "Add an HACMO Epoch Base Placement candidate solved from epoch_t telemetry. "
            "The epoch_t1 segment remains evaluation-only and is never used to select this candidate."
        ),
    )
    p.add_argument("--epoch-t-start", type=int, default=0)
    p.add_argument("--epoch-t1-start", type=int, default=0)
    p.add_argument("--epoch-t-requests", type=int, default=160)
    p.add_argument("--epoch-t1-requests", type=int, default=None)
    p.add_argument("--library-max-requests", type=int, default=256)
    p.add_argument("--hf-trace-layers", nargs="+", default=["23"])
    p.add_argument("--hf-trace-all-layers", action="store_true")
    p.add_argument("--decode-len", type=int, default=128)
    p.add_argument(
        "--runtime-serving-steps",
        type=int,
        default=None,
        help=(
            "Optional serving-horizon length for rolling_continuous_request_slots. "
            "The request decode length remains --decode-len."
        ),
    )
    p.add_argument("--window-size", type=int, default=8)
    p.add_argument(
        "--runtime-workload-mode",
        choices=RUNTIME_WORKLOAD_MODES,
        default="fixed_request_batch",
        help=(
            "fixed_request_batch preserves the existing pack protocol; "
            "continuous_request_slots models a fixed-concurrency serving epoch "
            "whose staggered epoch-t requests are replaced by epoch-t+1 requests; "
            "rolling_continuous_request_slots extends that model over a longer "
            "serving horizon with repeated admissions."
        ),
    )
    p.add_argument(
        "--trace-source-mode",
        choices=TRACE_SOURCE_MODES,
        default="empirical_per_die_sampling",
        help=(
            "How normalized trace requests become target-system source traffic. "
            "empirical_per_die_sampling is the paper-facing per-die batch setting; "
            "replicate_all_dies is a legacy alias."
        ),
    )
    p.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[64], help="Per-die batch sizes."
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    p.add_argument("--num-experts", type=int, default=256)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--hidden-size", type=int, default=7168)
    p.add_argument("--dispatch-bytes-per-hidden-elem", type=int, default=1)
    p.add_argument("--combine-bytes-per-hidden-elem", type=int, default=2)
    p.add_argument("--bytes-per-hidden-elem", type=int, default=2)
    p.add_argument("--expert-intermediate-size", type=int, default=2048)
    p.add_argument("--h200-effective-tflops", type=float, default=1979.0)
    p.add_argument("--h200-hbm-bw-tbps", type=float, default=4.8)
    p.add_argument("--weight-bytes-per-elem", type=int, default=1)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--delta", type=float, default=2.0)
    p.add_argument("--lambda", dest="lambda_", type=float, default=1.0)
    p.add_argument(
        "--placement-source-model",
        choices=["source_agnostic", "projected"],
        default="source_agnostic",
    )
    p.add_argument("--topk-hot-experts-for-ilp", type=int, default=32)
    p.add_argument("--static-max-replicas", type=int, default=4)
    p.add_argument("--die-memory-capacity-gib", type=float, default=141.0)
    p.add_argument("--die-compute-capacity", type=float, default=1.0)
    p.add_argument("--refinement-max-passes", type=int, default=16)
    p.add_argument("--moentwine-max-replicas", type=int, default=2)
    p.add_argument("--moentwine-query-heads", type=int, default=128)
    p.add_argument("--moentwine-attention-tp-degree", type=int, default=4)
    p.add_argument("--moentwine-attention-tp-shape", type=int, nargs=2, default=[2, 2])
    p.add_argument(
        "--moentwine-attention-tp-selection",
        choices=MOENTWINE_TP_SELECTION_POLICIES,
        default="fixed",
        help=(
            "fixed uses --moentwine-attention-tp-* for every topology. "
            "topology_predeclared keeps the CLI/default TP unless a topology has "
            "a preregistered MoEntwine deployment, currently TP16 on 16w_4x4."
        ),
    )
    p.add_argument(
        "--moentwine-deployment-policy",
        choices=MOENTWINE_DEPLOYMENT_POLICIES,
        default="full_her",
        help=(
            "MoEntwine-only deployment policy. full_her preserves historical "
            "ER/HER behavior; no_her keeps NI-Balancer but routes MoE as normal "
            "source-die all-to-all; cost_gated_her chooses the lower analytical "
            "full-HER vs no-HER candidate per topology/batch/seed."
        ),
    )
    p.add_argument("--moentwine-shadow-slots-per-die", type=int, default=1)
    p.add_argument("--moentwine-max-adjustments-per-window", type=int, default=256)
    p.add_argument("--moentwine-ewma-decay", type=float, default=0.8)
    p.add_argument("--route-communication-weight", type=float, default=1.0)
    p.add_argument("--route-compute-load-weight", type=float, default=1.0)
    p.add_argument(
        "--route-estimation-decisions-per-source-expert", type=int, default=1
    )
    p.add_argument("--max-replicas-per-expert", type=int, default=4)
    p.add_argument("--ewma-decay", type=float, default=0.8)
    p.add_argument("--hot-percentile", type=float, default=90.0)
    p.add_argument("--critical-load-weight", type=float, default=0.0)
    p.add_argument("--max-hot-experts-per-window", type=int, default=8)
    p.add_argument("--max-candidate-dies-per-expert", type=int, default=3)
    p.add_argument("--max-candidate-operations-per-window", type=int, default=12)
    p.add_argument(
        "--attention-migration-slack-us-per-window", type=float, default=1000.0
    )
    p.add_argument("--run-astra", action="store_true")
    p.add_argument(
        "--mapping-only",
        action="store_true",
        help=(
            "Build the Static Mapping Library, select and transfer the Epoch Base "
            "Placement, then stop before runtime sampling and route-plan generation."
        ),
    )
    p.add_argument("--output-root", default=None)
    args = p.parse_args()

    if not set(args.methods).issubset(METHOD_TO_SOLVER):
        raise ValueError(f"unknown method in {args.methods}")
    if args.runtime_serving_steps is not None:
        if args.runtime_workload_mode != "rolling_continuous_request_slots":
            raise ValueError(
                "--runtime-serving-steps is only valid with "
                "rolling_continuous_request_slots"
            )
        if args.runtime_serving_steps <= 0:
            raise ValueError("--runtime-serving-steps must be positive")

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = (
        Path(args.output_root)
        if args.output_root
        else PROJECT_DIR / "outputs" / f"tp2_serving_epoch_{ts}"
    )
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, object]] = []
    for topo in args.topologies:
        all_rows.extend(run_topology(args, Path(topo).resolve(), output_root))

    sample_rows = normalize_rows(all_rows)
    rows = aggregate_seed_rows(sample_rows)
    sample_fields = [
        "topology",
        "method",
        "seed",
        "batch_per_die",
        "global_batch",
        "total_us",
        "collective_us",
        "dispatch_us",
        "compute_us",
        "combine_us",
        "serial_upper_us",
        "normalized_to_vllm",
        "speedup_over_vllm",
        "cross_wafer_ratio",
        "avg_hops",
        "moentwine_no_collective_total_us",
        "moentwine_conventional_tp_collective_us",
        "moentwine_full_her_collective_us",
        "moentwine_incremental_collective_us",
        "moentwine_full_her_total_us",
        "moentwine_conventional_tp_total_us",
        "moentwine_deployment_policy",
        "moentwine_deployment_selected_variant",
        "moentwine_deployment_selection_reason",
        "moentwine_deployment_selection_metric",
        "moentwine_deployment_full_her_total_us",
        "moentwine_deployment_no_her_total_us",
        "max_rank_recv_bytes",
        "max_edge_port_bytes",
        "metric_source",
        "route_plan_json",
    ]
    write_csv(
        output_root / "csv" / "tp2_serving_epoch_analytical_samples.csv",
        sample_rows,
        sample_fields,
    )
    fields = [
        "topology",
        "method",
        "seed_count",
        "seeds",
        "batch_per_die",
        "global_batch",
        "total_us",
        "total_us_std",
        "collective_us",
        "collective_us_std",
        "dispatch_us",
        "dispatch_us_std",
        "compute_us",
        "compute_us_std",
        "combine_us",
        "combine_us_std",
        "serial_upper_us",
        "serial_upper_us_std",
        "normalized_to_vllm",
        "normalized_to_vllm_std",
        "speedup_over_vllm",
        "speedup_over_vllm_std",
        "cross_wafer_ratio",
        "cross_wafer_ratio_std",
        "avg_hops",
        "avg_hops_std",
        "moentwine_no_collective_total_us",
        "moentwine_no_collective_total_us_std",
        "moentwine_conventional_tp_collective_us",
        "moentwine_conventional_tp_collective_us_std",
        "moentwine_full_her_collective_us",
        "moentwine_full_her_collective_us_std",
        "moentwine_incremental_collective_us",
        "moentwine_incremental_collective_us_std",
        "moentwine_full_her_total_us",
        "moentwine_full_her_total_us_std",
        "moentwine_conventional_tp_total_us",
        "moentwine_conventional_tp_total_us_std",
        "moentwine_deployment_policy",
        "moentwine_deployment_selected_variant",
        "moentwine_deployment_selection_reason",
        "moentwine_deployment_selection_metric",
        "moentwine_deployment_full_her_total_us",
        "moentwine_deployment_full_her_total_us_std",
        "moentwine_deployment_no_her_total_us",
        "moentwine_deployment_no_her_total_us_std",
        "metric_source",
    ]
    write_csv(
        output_root / "csv" / "tp2_serving_epoch_analytical_summary.csv", rows, fields
    )
    write_json(
        output_root / "serving_epoch_run_metadata.json",
        {
            "runner": "run_serving_epoch_tp2.py",
            "created_at_utc": ts,
            "topologies": [str(Path(t).resolve()) for t in args.topologies],
            "methods": args.methods,
            "library_hf_trace_roots": [
                str(Path(x)) for x in args.library_hf_trace_roots
            ],
            "runtime_hf_trace_root": str(Path(args.runtime_hf_trace_root)),
            "epoch_t1_hf_trace_root": (
                str(Path(args.epoch_t1_hf_trace_root))
                if args.epoch_t1_hf_trace_root is not None
                else None
            ),
            "include_epoch_t_base_entry": args.include_epoch_t_base_entry,
            "epoch_t_start": args.epoch_t_start,
            "epoch_t1_start": args.epoch_t1_start,
            "epoch_t_requests": args.epoch_t_requests,
            "epoch_t1_requests": args.epoch_t1_requests,
            "runtime_workload_mode": args.runtime_workload_mode,
            "runtime_serving_steps": args.runtime_serving_steps or args.decode_len,
            "request_decode_len": args.decode_len,
            "trace_source_mode": args.trace_source_mode,
            "source_mode_semantics": source_mode_semantics(args.trace_source_mode),
            "sampling_contract": runtime_sampling_contract(
                args.runtime_workload_mode, args.trace_source_mode
            ),
            "trace_source_sampling_contract": sampling_contract(
                args.trace_source_mode
            ),
            "batch_scope": "per_die",
            "batch_sizes": args.batch_sizes,
            "seeds": args.seeds,
            "run_astra": args.run_astra,
            "route_plan_is_authoritative": True,
            "batch_scale": 1.0,
            "moentwine": {
                "query_heads": args.moentwine_query_heads,
                "attention_tp_selection_policy": (
                    args.moentwine_attention_tp_selection
                ),
                "requested_default_attention_tp_degree": (
                    args.moentwine_attention_tp_degree
                ),
                "requested_default_attention_tp_shape": (
                    args.moentwine_attention_tp_shape
                ),
                "topology_predeclared_overrides": [
                    {
                        "num_wafers": key[0],
                        "mesh_rows": key[1],
                        "mesh_cols": key[2],
                        **value,
                    }
                    for key, value in sorted(
                        MOENTWINE_TOPOLOGY_PREDECLARED_TP.items()
                    )
                ],
                "actual_attention_tp_by_topology": (
                    "see each topology's serving_epoch_metadata.json"
                ),
                "ni_balancer_is_causal": True,
                "deployment_policy": args.moentwine_deployment_policy,
                "deployment_policy_candidates": list(MOENTWINE_DEPLOYMENT_POLICIES),
                "moe_only_accounting": (
                    "moe_no_collective + max(full_her - same_tp_conventional, 0)"
                ),
            },
        },
    )
    print(f"[DONE] Results in: {output_root}")


if __name__ == "__main__":
    main()
