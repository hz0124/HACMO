#!/usr/bin/env python3
"""Run the TP2 Dynamic-vs-Static headroom diagnostic protocol.

This runner is intentionally separate from the paper final pack.  It builds a
fixed-concurrency continuous-slot serving stream, lets the existing Dynamic
scheduler adapt causally, emits authoritative routed-step plans, and optionally
replays the tail through the shared SupernodeSim/ASTRA backend.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
CORE_DIR = PROJECT_DIR / "tp2_core"
SCRIPTS_DIR = PROJECT_DIR / "scripts"
for path in (REPO_ROOT, CORE_DIR, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_step_inputs import (  # noqa: E402
    aggregate_steps_to_windows,
    build_continuous_step_inputs,
    build_mixed_continuous_step_inputs,
)
from build_window_traffic_matrix import analyze_routed_steps  # noqa: E402
from compute_model import H200RooflineConfig  # noqa: E402
from moentwine_baseline import (  # noqa: E402
    build_causal_ni_windows,
    build_conventional_tp_collective_segments,
    build_er_source_domains,
    build_moentwine_collective_segments,
    incremental_collective_accounting,
)
from route_steps import build_routed_steps  # noqa: E402
from runtime_scheduler import run_runtime  # noqa: E402
from run_serving_epoch_tp2 import (  # noqa: E402
    MOENTWINE_DEPLOYMENT_POLICIES,
    MOENTWINE_TP_SELECTION_POLICIES,
    build_collective_only_route_plan,
    iter_json_files,
    load_normalized_requests,
    moentwine_conventional_component,
    resolve_moentwine_attention_tp,
    segment_total_bytes,
    select_moentwine_deployment,
    select_request_segments,
    write_moentwine_primary_mapping,
    write_normalized_requests,
)
from utils_topology import load_topology  # noqa: E402


@dataclass(frozen=True)
class CaseSpec:
    name: str
    initial_domain: str
    initial_start: int
    initial_count: int
    replacement_domain: str
    replacement_start: int
    replacement_count: int
    replacement_slices: tuple[tuple[str, int, int], ...] = ()

    @property
    def initial_label(self) -> str:
        return (
            f"{self.initial_domain}["
            f"{self.initial_start}:{self.initial_start + self.initial_count}]"
        )

    @property
    def replacement_label(self) -> str:
        if self.replacement_slices:
            return "+".join(
                f"{domain}[{start}:{start + count}]"
                for domain, start, count in self.replacement_slices
            )
        return (
            f"{self.replacement_domain}["
            f"{self.replacement_start}:{self.replacement_start + self.replacement_count}]"
        )


CASE_SPECS = {
    "primary_law_accounting": CaseSpec(
        name="primary_law_accounting",
        initial_domain="professional_law",
        initial_start=256,
        initial_count=160,
        replacement_domain="professional_accounting",
        replacement_start=0,
        replacement_count=280,
    ),
    "control_law_law": CaseSpec(
        name="control_law_law",
        initial_domain="professional_law",
        initial_start=256,
        initial_count=160,
        replacement_domain="professional_law",
        replacement_start=416,
        replacement_count=280,
    ),
    "high_drift_uniform_pool_law_math_elem_human_aging": CaseSpec(
        name="high_drift_uniform_pool_law_math_elem_human_aging",
        initial_domain="professional_law",
        initial_start=256,
        initial_count=160,
        replacement_domain="mixed_replacement_pool",
        replacement_start=0,
        replacement_count=270,
        replacement_slices=(
            ("high_school_mathematics", 0, 90),
            ("elementary_mathematics", 0, 90),
            ("human_aging", 0, 90),
        ),
    ),
}

STAGE_ORDER = ("prepare", "dynamic", "route", "astra", "summarize")
METHODS = ("vllm", "moentwine", "ours_static", "ours_dynamic")
METHOD_TO_MAPPING_FILE = {
    "vllm": "uniform.json",
    "moentwine": "moentwine_static.json",
    "ours_static": "ours_static.json",
    "ours_dynamic": "ours_static.json",
}


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_uniform_mapping(path: Path, *, num_experts: int, num_dies: int) -> None:
    if num_experts <= 0 or num_dies <= 0:
        raise ValueError("num_experts and num_dies must be positive")
    write_json(
        path,
        {
            "method": "uniform",
            "expert_to_dies": {
                str(expert): [expert % num_dies] for expert in range(num_experts)
            },
            "metadata": {
                "placement": "deterministic_round_robin_single_copy",
                "uses_hacmo_objective": False,
            },
        },
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def trace_root_default() -> Path:
    return (
        PROJECT_DIR
        / "data/raw_traces/hf_moe_expert_selection_trace/cognitivecomputations"
        / "DeepSeek-R1-AWQ/mmlu"
    )


def topology_label(config_path: Path) -> str:
    stem = config_path.stem
    return stem.replace("hierarchical_", "").replace("_mesh_rail_", "_rail_")


def vector_from_windows(windows: list[dict[str, Any]], num_experts: int) -> list[float]:
    counts = [0.0] * num_experts
    for window in windows:
        for expert, count in window.get("expert_counts", {}).items():
            counts[int(expert)] += int(count)
    return counts


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def top_overlap(a: list[float], b: list[float], k: int) -> int:
    top_a = {idx for idx, _ in sorted(enumerate(a), key=lambda x: (-x[1], x[0]))[:k]}
    top_b = {idx for idx, _ in sorted(enumerate(b), key=lambda x: (-x[1], x[0]))[:k]}
    return len(top_a & top_b)


def case_root(output_root: Path, case: CaseSpec) -> Path:
    return output_root / case.name


def topo_dir(output_root: Path, case: CaseSpec, network_config: Path) -> Path:
    return case_root(output_root, case) / topology_label(network_config)


def sampled_dir(output_root: Path, case: CaseSpec, network_config: Path, seed: int, bpd: int) -> Path:
    return topo_dir(output_root, case, network_config) / "sampled_inputs" / f"seed{seed}" / f"bpd{bpd}"


def runtime_windows_json(output_root: Path, case: CaseSpec, network_config: Path, bpd: int) -> Path:
    return topo_dir(output_root, case, network_config) / "runtime_windows" / f"ours_dynamic_windows_bpd{bpd}.json"


def route_plan_json(
    output_root: Path,
    case: CaseSpec,
    network_config: Path,
    seed: int,
    bpd: int,
    method: str,
) -> Path:
    return (
        topo_dir(output_root, case, network_config)
        / "route_plans"
        / f"seed{seed}"
        / f"{method}_bpd{bpd}.json"
    )


def input_summary_json(output_root: Path, case: CaseSpec) -> Path:
    return case_root(output_root, case) / "input_summary.json"


def case_summary_json(output_root: Path, case: CaseSpec) -> Path:
    return case_root(output_root, case) / "summary.json"


def expand_stages(stages: list[str]) -> list[str]:
    if "all" in stages:
        return list(STAGE_ORDER)
    expanded: list[str] = []
    for stage in STAGE_ORDER:
        if stage in stages:
            expanded.append(stage)
    return expanded


def select_file_slice(
    files: list[Path], *, start: int, count: int, label: str
) -> list[Path]:
    if start < 0 or count <= 0:
        raise ValueError(f"{label}: start must be non-negative and count positive")
    selected = files[start : start + count]
    if len(selected) != count:
        raise ValueError(f"{label}: found {len(selected)} files, expected {count}")
    return selected


def ensure_prepared(case: CaseSpec, args: argparse.Namespace) -> None:
    root = case_root(args.output_root, case)
    topology_dir = topo_dir(args.output_root, case, args.network_config)
    step_inputs = sampled_dir(args.output_root, case, args.network_config, args.seed, args.batch_per_die) / "step_inputs.json"
    static_mapping_out = topology_dir / "mappings/ours_static.json"
    summary_out = input_summary_json(args.output_root, case)
    required = [
        step_inputs,
        topology_dir / "mapping_inputs/runtime/window_stats.json",
        topology_dir / "mapping_inputs/runtime/window_src_expert_counts.json",
        static_mapping_out,
        summary_out,
    ]
    if not args.force and all(path.exists() for path in required):
        log(case, "prepare", f"skip existing {summary_out}")
        return
    if args.force and root.exists():
        shutil.rmtree(root)

    start = time.perf_counter()
    topology = load_topology(args.network_config)
    initial_trace_root = args.trace_root / case.initial_domain
    initial_files = select_file_slice(
        iter_json_files(initial_trace_root),
        start=case.initial_start,
        count=case.initial_count,
        label=case.initial_label,
    )

    norm_initial = topology_dir / "serving_epoch/epoch_t_telemetry/normalized"
    layers = {str(args.layer)}
    initial_count = write_normalized_requests(
        norm_initial,
        initial_files,
        layers=layers,
        max_decode_steps=args.decode_len,
        role="epoch_t_initial",
        source_root=initial_trace_root,
    )
    replacement_counts_by_domain: dict[str, int] = {}
    if case.replacement_slices:
        replacement_payloads_by_domain: list[tuple[str, list[dict]]] = []
        replacement_parts_root = (
            topology_dir / "serving_epoch/epoch_t1_eval/normalized_parts"
        )
        for index, (domain, slice_start, slice_count) in enumerate(
            case.replacement_slices
        ):
            source_root = args.trace_root / domain
            files = select_file_slice(
                iter_json_files(source_root),
                start=slice_start,
                count=slice_count,
                label=f"{domain}[{slice_start}:{slice_start + slice_count}]",
            )
            normalized_dir = replacement_parts_root / f"{index:02d}_{domain}"
            normalized_count = write_normalized_requests(
                normalized_dir,
                files,
                layers=layers,
                max_decode_steps=args.decode_len,
                role=f"epoch_t1_replacement_{domain}",
                source_root=source_root,
            )
            replacement_counts_by_domain[domain] = normalized_count
            replacement_payloads_by_domain.append(
                (domain, load_normalized_requests(normalized_dir))
            )
        replacement_count = sum(replacement_counts_by_domain.values())
        sampled = build_mixed_continuous_step_inputs(
            initial_req_payloads=load_normalized_requests(norm_initial),
            replacement_req_payloads_by_domain=replacement_payloads_by_domain,
            num_dies=topology.total_dies,
            batch_per_die=args.batch_per_die,
            decode_len=args.decode_len,
            window_size=args.window_size,
            seed=args.seed,
            initial_domain=case.initial_domain,
        )
        domain_by_index = sampled["metadata"].pop(
            "replacement_request_domain_by_index"
        )
        sampled["metadata"].update(
            {
                "diagnostic_case": case.name,
                "admission_policy": "uniform_pool",
                "initial_slice": case.initial_label,
                "replacement_slices": [
                    f"{domain}[{start}:{start + count}]"
                    for domain, start, count in case.replacement_slices
                ],
                "replacement_request_domain_by_index": domain_by_index,
            }
        )
    else:
        replacement_trace_root = args.trace_root / case.replacement_domain
        replacement_files = select_file_slice(
            iter_json_files(replacement_trace_root),
            start=case.replacement_start,
            count=case.replacement_count,
            label=case.replacement_label,
        )
        norm_replacement = topology_dir / "serving_epoch/epoch_t1_eval/normalized"
        replacement_count = write_normalized_requests(
            norm_replacement,
            replacement_files,
            layers=layers,
            max_decode_steps=args.decode_len,
            role="epoch_t1_replacement",
            source_root=replacement_trace_root,
        )
        replacement_counts_by_domain[case.replacement_domain] = replacement_count
        sampled = build_continuous_step_inputs(
            initial_req_payloads=load_normalized_requests(norm_initial),
            replacement_req_payloads=load_normalized_requests(norm_replacement),
            num_dies=topology.total_dies,
            batch_per_die=args.batch_per_die,
            decode_len=args.decode_len,
            window_size=args.window_size,
            seed=args.seed,
            initial_domain=case.initial_domain,
            replacement_domain=case.replacement_domain,
        )
    window_stats, window_src = aggregate_steps_to_windows(
        sampled["steps"], window_size=args.window_size
    )

    sample_root = sampled_dir(args.output_root, case, args.network_config, args.seed, args.batch_per_die)
    write_json(sample_root / "step_inputs.json", sampled)
    write_json(sample_root / "window_stats.json", window_stats)
    write_json(sample_root / "window_src_expert_counts.json", window_src)

    mapping_inputs = topology_dir / "mapping_inputs/runtime"
    write_json(mapping_inputs / "window_stats.json", window_stats)
    write_json(mapping_inputs / "window_src_expert_counts.json", window_src)
    write_json(
        mapping_inputs / "metadata.json",
        {
            "sampling": "continuous_staggered_request_slots",
            "num_requests": replacement_count,
            "trace_num_requests": replacement_count,
            "initial_normalized_count": initial_count,
            "replacement_normalized_count": replacement_count,
            "batch_per_die": args.batch_per_die,
            "global_batch": topology.total_dies * args.batch_per_die,
            "decode_len": args.decode_len,
            "window_size": args.window_size,
            "seed": args.seed,
            "diagnostic_case": case.name,
        },
    )
    static_mapping_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.static_mapping_json, static_mapping_out)
    write_uniform_mapping(
        static_mapping_out.parent / "uniform.json",
        num_experts=args.num_experts,
        num_dies=topology.total_dies,
    )
    write_moentwine_primary_mapping(
        output_json=static_mapping_out.parent / "moentwine_static.json",
        num_experts=args.num_experts,
        num_dies=topology.total_dies,
        query_heads=args.moentwine_query_heads,
        attention_tp_degree=args.moentwine_attention_tp_degree,
        attention_tp_shape=args.moentwine_attention_tp_shape,
        attention_tp_selection_policy=args.moentwine_attention_tp_selection,
        attention_tp_selection_reason=args.moentwine_attention_tp_selection_reason,
    )

    first_half = vector_from_windows(window_stats[: len(window_stats) // 2], args.num_experts)
    second_half = vector_from_windows(window_stats[len(window_stats) // 2 :], args.num_experts)
    write_json(
        summary_out,
        {
            "diagnostic_case": case.name,
            "admission_policy": (
                "uniform_pool" if case.replacement_slices else "single_domain_pool"
            ),
            "initial_domain": case.initial_label,
            "replacement_domain": case.replacement_label,
            "replacement_slices": [
                f"{domain}[{start}:{start + count}]"
                for domain, start, count in case.replacement_slices
            ],
            "normalized_counts": {
                "initial": initial_count,
                "replacement_by_domain": replacement_counts_by_domain,
                "replacement_total": replacement_count,
            },
            "replacement_request_domain_counts": replacement_counts_by_domain,
            "replacement_event_domain_counts": dict(
                sorted(
                    Counter(
                        event["replacement_domain"]
                        for event in sampled["replacement_events"]
                    ).items()
                )
            ),
            "first_to_second_half_cosine": cosine(first_half, second_half),
            "first_to_second_half_top32_overlap": top_overlap(
                first_half, second_half, args.topk_hot_experts
            ),
            "params": params_summary(args, topology.total_dies),
            "input_hashes": {
                "step_inputs": sha256_file(sample_root / "step_inputs.json"),
                "window_stats": sha256_file(sample_root / "window_stats.json"),
                "window_src_expert_counts": sha256_file(
                    sample_root / "window_src_expert_counts.json"
                ),
            },
            "static_mapping_json": str(static_mapping_out),
            "static_mapping_source_json": str(args.static_mapping_json),
            "baseline_mapping_jsons": {
                "vllm": str(static_mapping_out.parent / "uniform.json"),
                "moentwine": str(static_mapping_out.parent / "moentwine_static.json"),
            },
            "elapsed_seconds": round(time.perf_counter() - start, 3),
        },
    )
    log(case, "prepare", f"wrote {summary_out}")


def run_dynamic(case: CaseSpec, args: argparse.Namespace) -> None:
    out = runtime_windows_json(args.output_root, case, args.network_config, args.batch_per_die)
    if out.exists() and not args.force:
        log(case, "dynamic", f"skip existing {out}")
        update_dynamic_operation_count(case, args)
        return
    start = time.perf_counter()
    topology = load_topology(args.network_config)
    topology_dir = topo_dir(args.output_root, case, args.network_config)
    static_raw = read_json(topology_dir / "mappings/ours_static.json")["expert_to_dies"]
    static_mapping = {int(e): [int(die) for die in dies] for e, dies in static_raw.items()}
    sample_root = sampled_dir(args.output_root, case, args.network_config, args.seed, args.batch_per_die)
    step_inputs_raw = read_json(sample_root / "step_inputs.json")
    runtime_windows = run_runtime(
        static_mapping=static_mapping,
        window_stats=read_json(sample_root / "window_stats.json"),
        window_src_expert_counts=read_json(sample_root / "window_src_expert_counts.json"),
        topo=topology,
        decay=args.ewma_decay,
        hot_percentile=args.hot_percentile,
        replica_persistence_windows=args.replica_persistence_windows,
        migration_budget_bytes_per_window=args.migration_budget_bytes_per_window,
        bytes_per_token=args.hidden_size * args.combine_bytes_per_hidden_elem,
        max_replicas_per_expert=args.max_replicas_per_expert,
        step_inputs=step_inputs_raw["steps"],
        alpha=args.alpha,
        gamma=args.gamma,
        delta=args.delta,
        lambda_=args.lambda_,
        critical_load_weight=args.critical_load_weight,
        runtime_compute_batch_scale=1.0,
        dispatch_bytes_per_token=(
            args.hidden_size * args.dispatch_bytes_per_hidden_elem
        ),
        route_communication_weight=args.route_communication_weight,
        route_compute_load_weight=args.route_compute_load_weight,
        route_estimation_decisions_per_source_expert=(
            args.route_estimation_decisions_per_source_expert
        ),
        roofline_cfg=roofline_config(args),
        max_adjustments_per_window=args.max_adjustments_per_window,
        max_hot_experts_per_window=args.max_hot_experts_per_window,
        max_candidate_dies_per_expert=args.max_candidate_dies_per_expert,
        max_candidate_operations_per_window=args.max_candidate_operations_per_window,
        attention_migration_slack_us_per_window=args.attention_migration_slack_us_per_window,
    )
    write_json(out, runtime_windows)
    update_dynamic_operation_count(case, args, elapsed=time.perf_counter() - start)
    log(case, "dynamic", f"wrote {out}")


def update_dynamic_operation_count(
    case: CaseSpec,
    args: argparse.Namespace,
    *,
    elapsed: float | None = None,
) -> None:
    summary_path = input_summary_json(args.output_root, case)
    runtime_path = runtime_windows_json(args.output_root, case, args.network_config, args.batch_per_die)
    if not summary_path.exists() or not runtime_path.exists():
        return
    summary = read_json(summary_path)
    summary["dynamic_operations"] = operation_summary(read_json(runtime_path))
    if elapsed is not None:
        summary["dynamic_elapsed_seconds"] = round(elapsed, 3)
    write_json(summary_path, summary)


def build_route_plans(case: CaseSpec, args: argparse.Namespace) -> None:
    outputs = [
        route_plan_json(args.output_root, case, args.network_config, args.seed, args.batch_per_die, method)
        for method in args.methods
    ]
    if not args.force and all(path.exists() for path in outputs):
        log(case, "route", f"skip existing {outputs[0].parent}")
        return
    start = time.perf_counter()
    topology = load_topology(args.network_config)
    topology_dir = topo_dir(args.output_root, case, args.network_config)
    sample_root = sampled_dir(args.output_root, case, args.network_config, args.seed, args.batch_per_die)
    sampled = read_json(sample_root / "step_inputs.json")
    dynamic_windows = (
        read_json(runtime_windows_json(args.output_root, case, args.network_config, args.batch_per_die))
        if "ours_dynamic" in args.methods
        else None
    )
    moentwine_windows = None
    moentwine_windows_path = topology_dir / "runtime_windows" / f"moentwine_windows_bpd{args.batch_per_die}.json"
    moentwine_source_domains = None
    if "moentwine" in args.methods:
        moentwine_mapping_obj = read_json(topology_dir / "mappings/moentwine_static.json")
        moentwine_windows = build_causal_ni_windows(
            base_mapping=moentwine_mapping_obj["expert_to_dies"],
            window_stats=read_json(sample_root / "window_stats.json"),
            topo=topology,
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
        write_json(moentwine_windows_path, moentwine_windows)
        moentwine_source_domains = build_er_source_domains(
            topology,
            args.moentwine_attention_tp_shape[0],
            args.moentwine_attention_tp_shape[1],
            hierarchical=topology.cfg.num_wafers > 1,
        )

    for method in args.methods:
        mapping_path = topology_dir / "mappings" / METHOD_TO_MAPPING_FILE[method]
        mapping_obj = read_json(mapping_path)
        window_mappings = None
        source_domains = None
        routing_policy = "nearest"
        moentwine_deployment_selection = None
        moentwine_candidate_plans: dict[str, dict[str, Any]] = {}
        if method == "ours_dynamic":
            window_mappings = dynamic_windows
            routing_policy = "hacmo_cost_aware"
        elif method == "ours_static":
            routing_policy = "hacmo_cost_aware"
        elif method == "moentwine":
            window_mappings = moentwine_windows
            source_domains = moentwine_source_domains
            routing_policy = "moentwine_er_nearest"

        def build_plan(*, policy: str, domains: Any, her_wafer_local: bool) -> dict[str, Any]:
            return build_routed_steps(
                steps=sampled["steps"],
                static_mapping=mapping_obj["expert_to_dies"],
                window_mappings=window_mappings,
                topo=topology,
                dispatch_bytes_per_token=(
                    args.hidden_size * args.dispatch_bytes_per_hidden_elem
                ),
                combine_bytes_per_token=(
                    args.hidden_size * args.combine_bytes_per_hidden_elem
                ),
                routing_policy=policy,
                source_domains=domains,
                communication_weight=args.route_communication_weight,
                compute_load_weight=args.route_compute_load_weight,
                moentwine_her_wafer_local=her_wafer_local,
            )

        if method == "moentwine":
            hierarchical = topology.cfg.num_wafers > 1
            moentwine_candidate_plans = {
                "full_her": build_plan(
                    policy="moentwine_er_nearest",
                    domains=source_domains,
                    her_wafer_local=hierarchical,
                )
            }
            if args.moentwine_deployment_policy in {"no_her", "cost_gated_her"}:
                moentwine_candidate_plans["no_her"] = build_plan(
                    policy="nearest",
                    domains=None,
                    her_wafer_local=False,
                )
            full_her_total_us = None
            no_her_total_us = None
            if args.moentwine_deployment_policy == "cost_gated_her":
                full_her_segments_by_step = [
                    build_moentwine_collective_segments(
                        tokens_per_die=step["tokens_per_die"],
                        topo=topology,
                        hidden_size=args.hidden_size,
                        bytes_per_hidden_elem=args.activation_bytes_per_elem,
                        tp_rows=args.moentwine_attention_tp_shape[0],
                        tp_cols=args.moentwine_attention_tp_shape[1],
                    )
                    for step in moentwine_candidate_plans["full_her"]["steps"]
                ]
                conventional_segments_by_step = [
                    build_conventional_tp_collective_segments(
                        tokens_per_die=step["tokens_per_die"],
                        topo=topology,
                        hidden_size=args.hidden_size,
                        bytes_per_hidden_elem=args.activation_bytes_per_elem,
                        tp_rows=args.moentwine_attention_tp_shape[0],
                        tp_cols=args.moentwine_attention_tp_shape[1],
                    )
                    for step in moentwine_candidate_plans["full_her"]["steps"]
                ]
                conventional_component = moentwine_conventional_component(
                    args.moentwine_attention_tp_degree
                )
                full_no_collective_total_us = analytical_route_total_us(
                    moentwine_candidate_plans["full_her"],
                    topology=topology,
                    args=args,
                )
                conventional_collective_us = analytical_route_collective_us(
                    build_collective_only_route_plan(
                        moentwine_candidate_plans["full_her"],
                        segments_by_step=conventional_segments_by_step,
                        component=conventional_component,
                    ),
                    topology=topology,
                    args=args,
                )
                full_her_collective_us = analytical_route_collective_us(
                    build_collective_only_route_plan(
                        moentwine_candidate_plans["full_her"],
                        segments_by_step=full_her_segments_by_step,
                        component="full_her",
                    ),
                    topology=topology,
                    args=args,
                )
                full_her_total_us = incremental_collective_accounting(
                    no_collective_total_us=full_no_collective_total_us,
                    conventional_tp_collective_us=conventional_collective_us,
                    full_her_collective_us=full_her_collective_us,
                )["charged_total_us"]
                no_her_total_us = analytical_route_total_us(
                    moentwine_candidate_plans["no_her"],
                    topology=topology,
                    args=args,
                )
            moentwine_deployment_selection = select_moentwine_deployment(
                policy=args.moentwine_deployment_policy,
                full_her_total_us=full_her_total_us,
                no_her_total_us=no_her_total_us,
            )
            plan = moentwine_candidate_plans[
                str(moentwine_deployment_selection["selected_variant"])
            ]
        else:
            plan = build_plan(
                policy=routing_policy,
                domains=source_domains,
                her_wafer_local=False,
            )
        plan["metadata"].update(
            {
                "diagnostic_case": case.name,
                "method": method,
                "seed": args.seed,
                "batch_per_die": args.batch_per_die,
                "global_batch": topology.total_dies * args.batch_per_die,
                "sampled_step_input_json": str(sample_root / "step_inputs.json"),
                "mapping_json": str(mapping_path),
                "window_mappings_json": (
                    str(runtime_windows_json(args.output_root, case, args.network_config, args.batch_per_die))
                    if method == "ours_dynamic"
                    else str(moentwine_windows_path)
                    if method == "moentwine"
                    else None
                ),
                "method_isolation": {
                    "uses_hacmo_static_objective": method
                    in {"ours_static", "ours_dynamic"},
                    "uses_hacmo_runtime_scheduler": method == "ours_dynamic",
                    "uses_moentwine_er_her": (
                        method == "moentwine"
                        and moentwine_deployment_selection is not None
                        and moentwine_deployment_selection.get("selected_variant")
                        == "full_her"
                    ),
                    "uses_moentwine_ni_balancer": method == "moentwine",
                    "moentwine_deployment_policy": (
                        moentwine_deployment_selection.get("policy")
                        if moentwine_deployment_selection is not None
                        else None
                    ),
                    "moentwine_deployment_selected_variant": (
                        moentwine_deployment_selection.get("selected_variant")
                        if moentwine_deployment_selection is not None
                        else None
                    ),
                },
            }
        )
        component_paths: dict[str, Path] = {}
        if method == "moentwine":
            selected_variant = str(
                moentwine_deployment_selection["selected_variant"]
                if moentwine_deployment_selection is not None
                else "full_her"
            )
            if selected_variant == "full_her":
                component_paths = write_moentwine_component_route_plans(
                    route_plan=plan,
                    route_plan_path=route_plan_json(
                        args.output_root,
                        case,
                        args.network_config,
                        args.seed,
                        args.batch_per_die,
                        method,
                    ),
                    topo=topology,
                    args=args,
                    deployment_selection=moentwine_deployment_selection,
                )
                plan["metadata"]["moentwine_collective"]["component_route_plans"] = {
                    key: str(path) for key, path in component_paths.items()
                }
            else:
                plan["metadata"]["source_domain_policy"] = (
                    "moentwine_no_her_normal_source_die_to_expert"
                )
                plan["metadata"]["moentwine_collective"] = {
                    "tp_degree": args.moentwine_attention_tp_degree,
                    "tp_shape": list(args.moentwine_attention_tp_shape),
                    "tp_selection_policy": args.moentwine_attention_tp_selection,
                    "tp_selection_reason": args.moentwine_attention_tp_selection_reason,
                    "tp_selection_batch_agnostic": (
                        args.moentwine_attention_tp_selection_batch_agnostic
                    ),
                    "tp_selection_uses_runtime_trace": (
                        args.moentwine_attention_tp_uses_runtime_trace
                    ),
                    "mode": "no_her",
                    "attention_bytes_per_hidden_elem": args.activation_bytes_per_elem,
                    "accounting_scope": "moe_ni_balancer_without_her_collective",
                    "formula": "moe_ni_balancer_normal_a2a",
                    "full_her_collective_phase_instances": 0,
                    "full_her_collective_bytes": 0,
                    "conventional_collective_component": moentwine_conventional_component(
                        args.moentwine_attention_tp_degree
                    ),
                    "conventional_tp_collective_phase_instances": 0,
                    "conventional_tp_collective_bytes": 0,
                    "main_route_plan_contains_collective": False,
                    "component_route_plans": {},
                    "deployment_selection": moentwine_deployment_selection,
                }
            candidate_paths: dict[str, str] = {}
            for variant, candidate_plan in moentwine_candidate_plans.items():
                if variant == selected_variant:
                    continue
                candidate_copy = dict(candidate_plan)
                candidate_copy["metadata"] = {
                    **candidate_plan.get("metadata", {}),
                    "artifact": "tp2_step_route_plan_diagnostic_candidate",
                    "diagnostic_case": case.name,
                    "method": method,
                    "seed": args.seed,
                    "batch_per_die": args.batch_per_die,
                    "global_batch": topology.total_dies * args.batch_per_die,
                    "diagnostic_candidate_variant": variant,
                    "moentwine_deployment_selection": moentwine_deployment_selection,
                }
                candidate_path = route_plan_json(
                    args.output_root,
                    case,
                    args.network_config,
                    args.seed,
                    args.batch_per_die,
                    f"moentwine_{variant}_candidate",
                )
                write_json(candidate_path, candidate_copy)
                candidate_paths[variant] = str(candidate_path)
            if candidate_paths:
                plan["metadata"]["moentwine_collective"][
                    "candidate_route_plans"
                ] = candidate_paths
        write_json(
            route_plan_json(
                args.output_root, case, args.network_config, args.seed, args.batch_per_die, method
            ),
            plan,
        )
    log(case, "route", f"wrote route plans in {round(time.perf_counter() - start, 3)}s")


def write_moentwine_component_route_plans(
    *,
    route_plan: dict[str, Any],
    route_plan_path: Path,
    topo: Any,
    args: argparse.Namespace,
    deployment_selection: dict[str, Any] | None,
) -> dict[str, Path]:
    hierarchical = topo.cfg.num_wafers > 1
    conventional_component = moentwine_conventional_component(
        args.moentwine_attention_tp_degree
    )
    full_her_segments_by_step: list[list[dict]] = []
    conventional_segments_by_step: list[list[dict]] = []
    for step in route_plan["steps"]:
        full_her_segments_by_step.append(
            build_moentwine_collective_segments(
                tokens_per_die=step["tokens_per_die"],
                topo=topo,
                hidden_size=args.hidden_size,
                bytes_per_hidden_elem=args.activation_bytes_per_elem,
                tp_rows=args.moentwine_attention_tp_shape[0],
                tp_cols=args.moentwine_attention_tp_shape[1],
            )
        )
        conventional_segments_by_step.append(
            build_conventional_tp_collective_segments(
                tokens_per_die=step["tokens_per_die"],
                topo=topo,
                hidden_size=args.hidden_size,
                bytes_per_hidden_elem=args.activation_bytes_per_elem,
                tp_rows=args.moentwine_attention_tp_shape[0],
                tp_cols=args.moentwine_attention_tp_shape[1],
            )
        )

    full_her_bytes = segment_total_bytes(full_her_segments_by_step)
    conventional_bytes = segment_total_bytes(conventional_segments_by_step)
    route_plan["metadata"]["moentwine_collective"] = {
        "tp_degree": args.moentwine_attention_tp_degree,
        "tp_shape": list(args.moentwine_attention_tp_shape),
        "tp_selection_policy": args.moentwine_attention_tp_selection,
        "tp_selection_reason": args.moentwine_attention_tp_selection_reason,
        "tp_selection_batch_agnostic": args.moentwine_attention_tp_selection_batch_agnostic,
        "tp_selection_uses_runtime_trace": args.moentwine_attention_tp_uses_runtime_trace,
        "mode": "her" if hierarchical else "er",
        "attention_bytes_per_hidden_elem": args.activation_bytes_per_elem,
        "accounting_scope": (
            "moe_only_incremental_over_same_tp_conventional_collective"
        ),
        "formula": (
            "moe_no_collective + "
            f"max(full_her - {conventional_component}, 0)"
        ),
        "full_her_collective_phase_instances": sum(
            len(segments) for segments in full_her_segments_by_step
        ),
        "full_her_collective_bytes": full_her_bytes,
        "conventional_collective_component": conventional_component,
        "conventional_tp_collective_phase_instances": sum(
            len(segments) for segments in conventional_segments_by_step
        ),
        "conventional_tp_collective_bytes": conventional_bytes,
        "main_route_plan_contains_collective": False,
        "deployment_selection": deployment_selection,
    }
    route_plan["metadata"]["source_domain_policy"] = (
        "moentwine_her_wafer_local_dispatch_after_inter_wafer_all_gather"
        if hierarchical
        else "moentwine_er_tp_group_after_all_gather"
    )

    component_specs = {
        conventional_component: conventional_segments_by_step,
        "full_her": full_her_segments_by_step,
    }
    component_paths: dict[str, Path] = {}
    for component, segments_by_step in component_specs.items():
        component_plan = build_collective_only_route_plan(
            route_plan,
            segments_by_step=segments_by_step,
            component=component,
        )
        component_path = route_plan_path.with_name(
            f"moentwine_{component}_collective_only_bpd{args.batch_per_die}.json"
        )
        write_json(component_path, component_plan)
        component_paths[component] = component_path
    return component_paths


def run_astra(case: CaseSpec, args: argparse.Namespace) -> None:
    for label in args.replay_labels:
        csv_path = case_root(args.output_root, case) / f"astra_{label}.csv"
        if csv_path.exists() and not args.force:
            log(case, "astra", f"skip existing {csv_path}")
            continue
        start = time.perf_counter()
        exclude = replay_exclude_leading_steps(label, args)
        cmd = [
            sys.executable,
            "-m",
            "supernode_sim.experiments.run_ablation2_from_artifacts",
            "--artifact-topology-dir",
            topo_dir(args.output_root, case, args.network_config),
            "--network-config",
            args.network_config,
            "--methods",
            *args.methods,
            "--seeds",
            args.seed,
            "--batch-scope",
            "per_die",
            "--batch-sizes",
            args.batch_per_die,
            "--decode-len",
            args.decode_len,
            "--window-size",
            args.window_size,
            "--num-experts",
            args.num_experts,
            "--topk",
            args.topk,
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
            args.activation_bytes_per_elem,
            "--astra-trace-mode",
            "communication_only",
            "--astra-network-backend",
            "congestion_aware",
            "--exclude-leading-windows",
            exclude,
            "--run-astra",
            "--output-root",
            case_root(args.output_root, case) / f"supernode_astra_{label}",
            "--output-csv",
            csv_path,
        ]
        try:
            run_cmd(cmd)
        except subprocess.CalledProcessError as exc:
            error_path = case_root(args.output_root, case) / f"astra_{label}.error.json"
            write_json(
                error_path,
                {
                    "diagnostic_case": case.name,
                    "stage": "astra",
                    "label": label,
                    "returncode": exc.returncode,
                    "cmd": [str(part) for part in exc.cmd],
                    "elapsed_seconds": round(time.perf_counter() - start, 3),
                },
            )
            log(case, "astra", f"failed; wrote {error_path}")
            if args.allow_astra_failure:
                continue
            raise
        log(case, "astra", f"wrote {csv_path} in {round(time.perf_counter() - start, 3)}s")


def summarize_case(case: CaseSpec, args: argparse.Namespace) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "diagnostic_case": case.name,
        "input": read_json(input_summary_json(args.output_root, case))
        if input_summary_json(args.output_root, case).exists()
        else None,
        "route_plan_hashes": {},
        "replays": {},
    }
    for method in args.methods:
        plan = route_plan_json(
            args.output_root, case, args.network_config, args.seed, args.batch_per_die, method
        )
        if plan.exists():
            summary["route_plan_hashes"][method] = sha256_file(plan)
    for label in args.replay_labels:
        csv_path = case_root(args.output_root, case) / f"astra_{label}.csv"
        if not csv_path.exists():
            error_path = case_root(args.output_root, case) / f"astra_{label}.error.json"
            summary["replays"][label] = {
                "status": "failed" if error_path.exists() else "missing",
                "csv": str(csv_path),
                "error_json": str(error_path) if error_path.exists() else None,
            }
            continue
        rows = read_csv(csv_path)
        by_method = {row["method"]: row for row in rows}
        replay = {
            "status": "complete",
            "csv": str(csv_path),
            "rows": rows,
        }
        present_methods = [method for method in args.methods if method in by_method]
        method_totals = {
            method: float(by_method[method]["total_us"]) for method in present_methods
        }
        replay.update(
            {
                "method_totals_us": method_totals,
                "route_plan_consumed": {
                    method: by_method[method].get("route_plan_consumed")
                    for method in present_methods
                },
                "compute_us": {
                    method: float(by_method[method]["compute_us"])
                    for method in present_methods
                },
                "communication_us": {
                    method: float(by_method[method]["communication_us"])
                    for method in present_methods
                },
            }
        )
        if "vllm" in method_totals:
            replay["speedup_over_vllm"] = {
                method: method_totals["vllm"] / total
                for method, total in method_totals.items()
            }
        if "ours_static" in method_totals and "ours_dynamic" in method_totals:
            replay.update(
                {
                    "ours_static_total_us": method_totals["ours_static"],
                    "ours_dynamic_total_us": method_totals["ours_dynamic"],
                    "dynamic_speedup_over_static": (
                        method_totals["ours_static"] / method_totals["ours_dynamic"]
                    ),
                }
            )
        summary["replays"][label] = replay
    write_json(case_summary_json(args.output_root, case), summary)
    log(case, "summarize", f"wrote {case_summary_json(args.output_root, case)}")
    return summary


def write_combined_summary(cases: list[CaseSpec], args: argparse.Namespace) -> None:
    combined = {
        "status": "diagnostic_summary",
        "output_root": str(args.output_root),
        "cases": {},
    }
    for case in cases:
        path = case_summary_json(args.output_root, case)
        if path.exists():
            combined["cases"][case.name] = read_json(path)
    write_json(args.output_root / "summary.json", combined)


def operation_summary(windows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    active_windows = []
    for window in windows:
        adds = len(window.get("replicas_added") or [])
        removes = len(window.get("replicas_removed") or [])
        swaps = len(window.get("swaps") or [])
        rollback = 1 if window.get("base_rollback") else 0
        if adds or removes or swaps or rollback:
            active_windows.append(
                {
                    "window_id": int(window["window_id"]),
                    "replicas_added": adds,
                    "replicas_removed": removes,
                    "swaps": swaps,
                    "rollback": rollback,
                    "planned_activation_window": window.get("planned_activation_window"),
                    "migration_hidden_by_attention": window.get(
                        "migration_hidden_by_attention"
                    ),
                }
            )
        counts["replicas_added"] += adds
        counts["replicas_removed"] += removes
        counts["swaps"] += swaps
        counts["rollback_windows"] += rollback
    return {
        "num_windows": len(windows),
        "num_operation_windows": len(active_windows),
        "operation_counts": dict(counts),
        "active_windows": active_windows,
    }


def replay_exclude_leading_steps(label: str, args: argparse.Namespace) -> int:
    if label == "full":
        return 0
    if label == "tail16":
        return max(args.decode_len - args.tail_steps, 0)
    raise ValueError(f"unsupported replay label: {label}")


def roofline_config(args: argparse.Namespace) -> H200RooflineConfig:
    return H200RooflineConfig(
        hidden_size=args.hidden_size,
        expert_intermediate_size=args.expert_intermediate_size,
        effective_tflops=args.h200_effective_tflops,
        hbm_bw_tbps=args.h200_hbm_bw_tbps,
        weight_bytes_per_elem=args.weight_bytes_per_elem,
        activation_bytes_per_elem=args.activation_bytes_per_elem,
    )


def analytical_route_total_us(
    route_plan: dict[str, Any],
    *,
    topology: Any,
    args: argparse.Namespace,
) -> float:
    summary = analyze_routed_steps(
        route_plan,
        topo=topology,
        roofline_cfg=roofline_config(args),
        num_ranks=topology.total_dies,
    )["summary"]
    return float(summary["avg_step_total_us"])


def analytical_route_collective_us(
    route_plan: dict[str, Any],
    *,
    topology: Any,
    args: argparse.Namespace,
) -> float:
    summary = analyze_routed_steps(
        route_plan,
        topo=topology,
        roofline_cfg=roofline_config(args),
        num_ranks=topology.total_dies,
    )["summary"]
    return float(summary.get("avg_step_collective_us", 0.0))


def params_summary(args: argparse.Namespace, num_dies: int) -> dict[str, Any]:
    return {
        "seed": args.seed,
        "batch_per_die": args.batch_per_die,
        "global_batch": num_dies * args.batch_per_die,
        "decode_len": args.decode_len,
        "window_size": args.window_size,
        "tail_steps": args.tail_steps,
        "num_experts": args.num_experts,
        "topk": args.topk,
        "layer": args.layer,
        "hidden_size": args.hidden_size,
        "expert_intermediate_size": args.expert_intermediate_size,
        "h200_effective_tflops": args.h200_effective_tflops,
        "h200_hbm_bw_tbps": args.h200_hbm_bw_tbps,
        "weight_bytes_per_elem": args.weight_bytes_per_elem,
        "activation_bytes_per_elem": args.activation_bytes_per_elem,
        "network_config": str(args.network_config),
        "moentwine_attention_tp_selection": args.moentwine_attention_tp_selection,
        "moentwine_attention_tp_degree": args.moentwine_attention_tp_degree,
        "moentwine_attention_tp_shape": args.moentwine_attention_tp_shape,
        "moentwine_attention_tp_selection_reason": getattr(
            args, "moentwine_attention_tp_selection_reason", "fixed_cli_or_default"
        ),
        "moentwine_deployment_policy": args.moentwine_deployment_policy,
    }


def apply_moentwine_attention_tp_selection(args: argparse.Namespace) -> None:
    selected = resolve_moentwine_attention_tp(
        args=args,
        topo=load_topology(args.network_config),
    )
    args.moentwine_attention_tp_degree = int(selected["attention_tp_degree"])
    args.moentwine_attention_tp_shape = list(selected["attention_tp_shape"])
    args.moentwine_attention_tp_selection_reason = str(selected["selection_reason"])
    args.moentwine_attention_tp_selection_batch_agnostic = bool(
        selected["batch_agnostic"]
    )
    args.moentwine_attention_tp_uses_runtime_trace = bool(
        selected["uses_runtime_trace_for_selection"]
    )


def run_cmd(cmd: list[Any]) -> None:
    printable = " ".join(str(part) for part in cmd[:16])
    if len(cmd) > 16:
        printable += " ..."
    print(f"[cmd] {printable}", flush=True)
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(REPO_ROOT)
    subprocess.run([str(part) for part in cmd], cwd=REPO_ROOT, env=env, check=True)


def log(case: CaseSpec, stage: str, message: str) -> None:
    print(f"[{case.name}][{stage}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run non-paper TP2 Dynamic headroom diagnostics."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/tmp/hacmo_tp2_dynamic_headroom_diagnostic"),
        help="Diagnostic output root. Defaults to /tmp and does not touch final packs.",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=sorted(CASE_SPECS),
        default=["primary_law_accounting", "control_law_law"],
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(METHODS),
        default=list(METHODS),
        help="Methods to include in route-plan generation and ASTRA replay.",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["all", *STAGE_ORDER],
        default=["all"],
        help="Stages to run. Use prepare dynamic route summarize for a no-ASTRA smoke.",
    )
    parser.add_argument(
        "--replay-labels",
        nargs="+",
        choices=["full", "tail16"],
        default=["tail16"],
    )
    parser.add_argument(
        "--allow-astra-failure",
        action="store_true",
        help="Record ASTRA failures and continue to later stages. Intended for smoke tests.",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild selected outputs.")
    parser.add_argument("--trace-root", type=Path, default=trace_root_default())
    parser.add_argument(
        "--network-config",
        type=Path,
        default=PROJECT_DIR / "configs/network/hierarchical_4w_8x8_mesh_rail_900.yaml",
    )
    parser.add_argument(
        "--static-mapping-json",
        type=Path,
        default=(
            PROJECT_DIR
            / "outputs/tp2_moderate_drift_gate_4w_bpd16_seed17_20260724"
            / "raw/mappings/ours_static.json"
        ),
        help="Epoch-t static mapping to reuse for this diagnostic.",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--batch-per-die", type=int, default=64)
    parser.add_argument("--decode-len", type=int, default=128)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--tail-steps", type=int, default=16)
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--topk-hot-experts", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument("--expert-intermediate-size", type=int, default=2048)
    parser.add_argument("--dispatch-bytes-per-hidden-elem", type=int, default=1)
    parser.add_argument("--combine-bytes-per-hidden-elem", type=int, default=2)
    parser.add_argument("--activation-bytes-per-elem", type=int, default=2)
    parser.add_argument("--h200-effective-tflops", type=float, default=1979.0)
    parser.add_argument("--h200-hbm-bw-tbps", type=float, default=4.8)
    parser.add_argument("--weight-bytes-per-elem", type=int, default=1)
    parser.add_argument("--ewma-decay", type=float, default=0.8)
    parser.add_argument("--hot-percentile", type=float, default=90.0)
    parser.add_argument("--replica-persistence-windows", type=int, default=2)
    parser.add_argument("--migration-budget-bytes-per-window", type=int, default=500_000_000)
    parser.add_argument("--max-replicas-per-expert", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--delta", type=float, default=2.0)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=1.0)
    parser.add_argument("--critical-load-weight", type=float, default=0.0)
    parser.add_argument("--route-communication-weight", type=float, default=1.0)
    parser.add_argument("--route-compute-load-weight", type=float, default=1.0)
    parser.add_argument("--route-estimation-decisions-per-source-expert", type=int, default=1)
    parser.add_argument("--max-adjustments-per-window", type=int, default=16)
    parser.add_argument("--max-hot-experts-per-window", type=int, default=8)
    parser.add_argument("--max-candidate-dies-per-expert", type=int, default=3)
    parser.add_argument("--max-candidate-operations-per-window", type=int, default=12)
    parser.add_argument("--attention-migration-slack-us-per-window", type=float, default=1000.0)
    parser.add_argument("--moentwine-max-replicas", type=int, default=2)
    parser.add_argument("--moentwine-query-heads", type=int, default=128)
    parser.add_argument("--moentwine-attention-tp-degree", type=int, default=4)
    parser.add_argument("--moentwine-attention-tp-shape", type=int, nargs=2, default=[2, 2])
    parser.add_argument(
        "--moentwine-attention-tp-selection",
        choices=MOENTWINE_TP_SELECTION_POLICIES,
        default="fixed",
    )
    parser.add_argument(
        "--moentwine-deployment-policy",
        choices=MOENTWINE_DEPLOYMENT_POLICIES,
        default="full_her",
        help=(
            "MoEntwine-only deployment policy. full_her preserves historical "
            "ER/HER; no_her keeps NI-Balancer but routes MoE from the normal "
            "source die; cost_gated_her chooses the lower analytical candidate."
        ),
    )
    parser.add_argument("--moentwine-shadow-slots-per-die", type=int, default=1)
    parser.add_argument("--moentwine-max-adjustments-per-window", type=int, default=256)
    parser.add_argument("--moentwine-ewma-decay", type=float, default=0.8)
    args = parser.parse_args()
    args.output_root = args.output_root.resolve()
    args.trace_root = args.trace_root.resolve()
    args.network_config = args.network_config.resolve()
    args.static_mapping_json = args.static_mapping_json.resolve()
    return args


def main() -> None:
    args = parse_args()
    if "moentwine" in args.methods:
        apply_moentwine_attention_tp_selection(args)
    stages = expand_stages(args.stages)
    selected_cases = [CASE_SPECS[name] for name in args.cases]
    print(
        f"[diagnostic] output_root={args.output_root} stages={','.join(stages)}",
        flush=True,
    )
    for case in selected_cases:
        if "prepare" in stages:
            ensure_prepared(case, args)
        if "dynamic" in stages and "ours_dynamic" in args.methods:
            run_dynamic(case, args)
        if "route" in stages:
            build_route_plans(case, args)
        if "astra" in stages:
            run_astra(case, args)
        if "summarize" in stages:
            summarize_case(case, args)
    if "summarize" in stages:
        write_combined_summary(selected_cases, args)
        print(f"[diagnostic] wrote {args.output_root / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
