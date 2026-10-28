#!/usr/bin/env python3
"""Rebuild Overall Performance from the rebuttal TP1/TP2/TP3 evidence packs.

This runner intentionally avoids the old paper-specific scale factors.  MoE
rows are loaded from model-specific TP2/TP3 artifacts.  GPU vLLM rows replay
the same TP2 runtime artifacts with DGX/NVL72 network configs.  WaferLLM is
represented only by an explicit in-pipeline matrix model; MoE placement and
dispatch/combine communication stay on the vLLM path.
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import json
import math
import os
import shutil
import sys
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[3]
OVERALL_DIR = REPO / "hacmo" / "overall"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from supernode_sim.api import simulate_layer
from supernode_sim.backends.utils import compute_attention_flops
from supernode_sim.config import load_config
from supernode_sim.types import CachePolicy, HardwareConfig, ParallelConfig, SimulationConfig, WorkloadConfig
from hacmo.attention.tp1_core.generate_tp1_final_pack import (
    estimate_latency as estimate_tp1_attention_latency,
    get_hit_rate as get_tp1_hit_rate,
    load_hbm_service_curve as load_tp1_hbm_service_curve,
    load_mapping_profile as load_tp1_mapping_profile,
    mapping_for_shape as tp1_mapping_for_shape,
    parse_profile_file as parse_tp1_profile_file,
)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    attention: str
    layers: int
    moe_layers: int
    hidden: int
    n_heads: int
    kv_heads: int
    head_dim: int
    dense_kv_bytes_per_token: int
    sparse_topk: int
    selected_vector_bytes: int
    num_experts: int
    moe_topk: int
    expert_intermediate: int
    representative_layer: str = "23"


@dataclass(frozen=True)
class BaselineSpec:
    key: str
    label: str
    platform: str
    network: str
    note: str


@dataclass(frozen=True)
class PackSpec:
    tp2_dir: Path
    tp3_dir: Path
    topology_labels: dict[str, str]
    scope_note: str


@dataclass(frozen=True)
class ReplayScale:
    """Explicit traffic-scale contract for a TP2 runtime-artifact GPU replay.

    TP2 packs stage their runtime inputs under two different conventions:

    ``raw_trace_scale``
        ``window_stats`` counts are at raw trace scale, and the replay must
        scale them up/down to the target batch.  Used by the
        ``fixed_request_batch`` / ``empirical_per_die_sampling`` packs.
    ``pre_expanded``
        ``window_stats`` counts are already expanded to the target batch, so the
        replay must not rescale them at all.  Used by the
        ``continuous_request_slots`` packs.

    The pack metadata only records the raw request count, which is not by itself
    enough to tell the two conventions apart.  ``resolve_replay_scale`` therefore
    derives the divisor from the staged ``window_stats`` themselves so that a
    single rule covers both conventions with no tunable parameter.
    """

    num_requests: int
    batch_scale: float
    convention: str
    observed_tokens_per_die_per_step: float
    metadata_num_requests: int
    derivation: str

    def as_metadata(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StepWindowScope:
    """Which replayed decode steps a GPU row must cover.

    The GPU replay has to cover exactly the same decode-step range as the LS
    ``vllm`` row it is compared against.  TP2 records that range as
    ``excluded_leading_windows`` over route-plan *steps*; the Overall GPU replay
    consumes window aggregates instead, so the count is converted to whole
    windows here.  Any pack whose exclusion does not land on a window boundary
    is rejected rather than silently rounded.
    """

    exclude_leading_windows: int
    excluded_leading_steps: int
    reported_steps: int
    source: str

    def as_metadata(self) -> dict[str, Any]:
        return asdict(self)


MODELS: OrderedDict[str, ModelSpec] = OrderedDict(
    [
        (
            "deepseek_v32",
            ModelSpec(
                key="deepseek_v32",
                label="DeepSeek-V3.2",
                attention="dsa",
                layers=61,
                moe_layers=58,
                hidden=7168,
                n_heads=128,
                kv_heads=128,
                head_dim=128,
                dense_kv_bytes_per_token=2 * 128 * 128 * 2,
                sparse_topk=2048,
                selected_vector_bytes=656,
                num_experts=256,
                moe_topk=8,
                expert_intermediate=2048,
            ),
        ),
        (
            "qwen3_235b_a22b",
            ModelSpec(
                key="qwen3_235b_a22b",
                label="Qwen3-235B-A22B-FP8",
                attention="dense",
                layers=94,
                moe_layers=94,
                hidden=4096,
                n_heads=64,
                kv_heads=4,
                head_dim=128,
                dense_kv_bytes_per_token=2 * 4 * 128 * 2,
                sparse_topk=0,
                selected_vector_bytes=2 * 4 * 128 * 2,
                num_experts=256,
                moe_topk=8,
                expert_intermediate=1536,
            ),
        ),
        (
            "llama4_maverick",
            ModelSpec(
                key="llama4_maverick",
                label="Llama-4-Maverick-17B-128E-Instruct",
                attention="dense",
                layers=48,
                moe_layers=24,
                hidden=5120,
                n_heads=40,
                kv_heads=8,
                head_dim=128,
                dense_kv_bytes_per_token=2 * 8 * 128 * 2,
                sparse_topk=0,
                selected_vector_bytes=2 * 8 * 128 * 2,
                num_experts=128,
                moe_topk=1,
                expert_intermediate=8192,
            ),
        ),
    ]
)


BASELINES: OrderedDict[str, BaselineSpec] = OrderedDict(
    [
        ("dgx", BaselineSpec("dgx", "vLLM+DGX", "gpu", "dgx_h200_32x8", "TP2 runtime trace replay on DGX/H200 effective scale-out fabric.")),
        ("nvl72", BaselineSpec("nvl72", "vLLM+NVL72", "gpu", "nvl72_4x64", "TP2 runtime trace replay on NVL72-like effective fabric.")),
        ("ls_vllm", BaselineSpec("ls_vllm", "vLLM+LS", "ls", "ls_16w4x4", "TP2 vLLM row on Layered Supernode.")),
        ("moentwine", BaselineSpec("moentwine", "MoEntwine", "ls", "ls_16w4x4", "TP2 moentwine_static row; attention remains naive.")),
        ("waferllm", BaselineSpec("waferllm", "WaferLLM", "ls", "ls_16w4x4", "vLLM MoE placement/communication plus HACMO-side WaferLLM matrix model.")),
        ("hacmo", BaselineSpec("hacmo", "HACMO", "ls", "ls_16w4x4", "TP1 + TP2 dynamic mapping + TP3 lane-level crossbar allocation.")),
    ]
)


NETWORKS = {
    "dgx_h200_32x8": {
        "rail": OVERALL_DIR / "configs/network/dgx_h200_32x8.yaml",
        "crossbar": None,
        "attention_bw_tbps": 0.05,
        "attention_latency_ns": 3500.0,
    },
    "nvl72_4x64": {
        "rail": OVERALL_DIR / "configs/network/nvl72_4x64.yaml",
        "crossbar": None,
        "attention_bw_tbps": 1.80,
        "attention_latency_ns": 1200.0,
    },
    "ls_16w4x4": {
        "rail": REPO / "hacmo/moe/configs/network/hierarchical_16w_4x4_mesh_rail_500.yaml",
        "crossbar": REPO / "hacmo/interconnect/configs/network/hierarchical_16w_4x4_mesh_crossbar_500.yaml",
        "attention_bw_tbps": 8.0,
        "attention_latency_ns": 500.0,
    },
    "ls_4w8x8": {
        "rail": REPO / "hacmo/moe/configs/network/hierarchical_4w_8x8_mesh_rail_900.yaml",
        "crossbar": REPO / "hacmo/interconnect/configs/network/hierarchical_4w_8x8_mesh_crossbar_900.yaml",
        "attention_bw_tbps": 8.0,
        "attention_latency_ns": 500.0,
    },
}

PACKS: dict[str, PackSpec] = {
    "deepseek_v32": PackSpec(
        tp2_dir=REPO / "hacmo/moe/outputs/tp2_serving_epoch_paper_pack_20260831",
        tp3_dir=REPO / "hacmo/interconnect/outputs/tp3_lane_level_multibatch_final_candidate_tp2_20260831",
        topology_labels={"ls_16w4x4": "16w_4x4_rail_500", "ls_4w8x8": "4w_8x8_rail_900"},
        scope_note="DeepSeek-R1-AWQ layer-23 HF trace pack, used as DeepSeek-style MoE evidence.",
    ),
    "qwen3_235b_a22b": PackSpec(
        tp2_dir=REPO / "hacmo/moe/outputs/tp2_overall_qwen3_layer23_epoch_t_base_railfix_astra_16w_bpd32_20260901",
        tp3_dir=REPO / "hacmo/interconnect/outputs/tp3_overall_qwen3_layer23_epoch_t_base_railfix_final_16w_bpd32_20260901",
        topology_labels={"ls_16w4x4": "16w_4x4_rail_500"},
        scope_note="Qwen3 livecodebench representative layer-23 serving epoch: epoch_t telemetry base plus mmlu/mmlu_ZH_CN static library, held-out epoch_t1 eval; ASTRA replay uses model-specific 256 experts/topk/hidden_size and TP3 lane-level pack regenerated from this single-layer TP2 evidence.",
    ),
    "llama4_maverick": PackSpec(
        tp2_dir=REPO / "hacmo/moe/outputs/tp2_overall_llama4_fulltrace_analytical_16w_bpd32_20260901",
        tp3_dir=REPO / "hacmo/interconnect/outputs/tp3_overall_llama4_fulltrace_final_16w_bpd32_20260901",
        topology_labels={"ls_16w4x4": "16w_4x4_rail_500"},
        scope_note="Llama4 livecodebench full-trace serving epoch: mmlu/mmlu_ZH_CN static library, held-out epoch_t1 eval; ASTRA replay uses model-specific num_experts/topk/hidden_size and TP3 lane-level fulltrace pack.",
    ),
}

TP1_PACK_ROOT = REPO / "hacmo/attention/outputs/technical_point1_final_pack_astra_ramulator_trace_driven_20260724"
TP1_CSV = TP1_PACK_ROOT / "csv/tp1_audited_metrics.csv"
TP1_GPU_BASELINE = TP1_PACK_ROOT / "summary/tp1_gpu_measured_baseline.csv"
TP1_CONTEXT_HBM_CURVE = REPO / "hacmo/attention/outputs/technical_point1_rebuttal_pipeline/hbm_service_curve/hbm_service_curve.json"
TP1_CONTEXT_HBM_CURVE_16K = OVERALL_DIR / "outputs/context_tps_tp1_ctx16k_trace_20260617/hbm_service_curve/hbm_service_curve.json"
TP1_CONTEXT_MAPPING_PROFILE = REPO / "hacmo/attention/outputs/technical_point1_rebuttal_pipeline/core_static_mapping_profile.json"
TP1_CONTEXT_GAMMA_JSON = REPO / "hacmo/attention/support/gamma_calibration_neighbor.json"
SYSTEM_CONFIG = "supernode_sim/configs/astra/system/p2p_only.json"
REMOTE_MEMORY_CONFIG = "supernode_sim/configs/astra/remote_memory/no_memory_expansion.json"
BASE_MOE_CONFIG = "supernode_sim/configs/deepseek_v32_moe_astra_smoke.json"
NUM_WORKERS = 256
CONTEXT_TPS_CONTEXTS = [8192, 16384, 32768, 65536]
# The released artifact ships the compact processed profiles; the raw H200
# measurement directories only exist on a machine that did the profiling, so
# point at them with --tp1-profile-dir or HACMO_TP1_PROFILE_DIRS when
# regenerating from raw measurements.
TP1_PROFILE_DIRS = [
    Path(entry).expanduser()
    for entry in (
        os.environ.get("HACMO_TP1_PROFILE_DIRS", "").split(os.pathsep)
        if os.environ.get("HACMO_TP1_PROFILE_DIRS")
        else [str(REPO / "ae/data/tp1/profiles")]
    )
    if entry
]
DENSE_HBM_EFFECTIVE_FRACTION = 0.47
SPARSE_HBM_EFFECTIVE_FRACTION = 0.21
WEIGHT_BYTES_PER_ELEM = 1
DENSE_NONUNIFORM_BASE_METHOD = "B"
DENSE_NONUNIFORM_STATIC_METHOD = "S"

ATTENTION_TP1_METHOD = {
    "dgx": "G",
    "nvl72": "G",
    "ls_vllm": "B",
    "moentwine": "B",
    "waferllm": "B",
    "hacmo": "O",
}


def main() -> None:
    args = parse_args()
    apply_input_overrides(args)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.output_root) if args.output_root else OVERALL_DIR / "outputs" / f"overall_rebuttal_{ts}"
    out_root = out_root.resolve()
    for sub in ["data", "panels", "figures", "ppt_csv"]:
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    runner = OverallRebuttalRunner(out_root=out_root, run_gpu_astra=not args.no_gpu_astra)
    rows = runner.run_all()
    write_csv(out_root / "data/simulated_layers.csv", rows)
    panels = build_panels(rows)
    for name, panel_rows in panels.items():
        write_csv(out_root / "panels" / f"{name}.csv", panel_rows)
    write_ppt_csv_bundle(out_root, panels)
    write_metadata(out_root, args, rows, gpu_replay_contracts=runner.gpu_replay_contracts)
    plot_overall(panels, out_root / "figures")
    if not args.skip_root_copy:
        refresh_root_copies(out_root / "figures")
    print(f"[overall] wrote {out_root}")
    print(f"[overall] figure: {out_root / 'figures/overall_performance_dense.png'}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run rebuttal Overall Performance from current TP evidence packs.")
    p.add_argument("--output-root", default="")
    p.add_argument("--no-gpu-astra", action="store_true", help="Use analytical GPU replay fallback instead of ASTRA.")
    p.add_argument("--skip-root-copy", action="store_true")
    p.add_argument("--tp1-final-pack", default=str(TP1_CSV.parents[1]))
    p.add_argument("--tp1-support-pack", default=str(TP1_CONTEXT_HBM_CURVE.parents[1]))
    p.add_argument("--tp1-context-16k-pack", default=str(TP1_CONTEXT_HBM_CURVE_16K.parents[1]))
    p.add_argument(
        "--tp1-profile-dir",
        action="append",
        default=None,
        help="Sparse-attention profile directory (repeatable).",
    )
    p.add_argument(
        "--tp1-gpu-baseline-csv",
        default=None,
        help=(
            "Measured H200 baseline table. The CPU-only workflow cannot "
            "regenerate it and the TP1 stage does not stage it, so point this "
            "at the archived copy shipped with the artifact."
        ),
    )
    for model_key, option_prefix in [
        ("deepseek_v32", "deepseek"),
        ("qwen3_235b_a22b", "qwen"),
        ("llama4_maverick", "llama"),
    ]:
        p.add_argument(f"--{option_prefix}-tp2-pack", default=str(PACKS[model_key].tp2_dir))
        p.add_argument(f"--{option_prefix}-tp3-pack", default=str(PACKS[model_key].tp3_dir))
    return p.parse_args()


def pack_relative(path: Any) -> str:
    """Name an input by pack and leaf, never by its absolute location.

    Provenance strings end up in the published result table. An absolute path
    there is both an author directory in a released file and a value that can
    never match between two machines.
    """
    if not path:
        return ""
    parts = Path(str(path)).parts
    return "/".join(parts[-2:]) if len(parts) >= 2 else str(path)


def resolve_tp1_gpu_baseline(tp1_final_pack: Path, *, override: str | None) -> Path:
    """Locate the measured H200 baseline the Overall stage needs.

    The TP1 stage only writes this table when it is handed GPU profiles, which
    the CPU-only workflow never has, so the Overall stage used to die on a
    missing file it could not possibly produce. An explicit override lets the
    workflow point at the archived copy instead.
    """
    if override:
        path = Path(override).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing TP1 GPU baseline table: {path}")
        return path
    return tp1_final_pack / "summary/tp1_gpu_measured_baseline.csv"


def apply_input_overrides(args: argparse.Namespace) -> None:
    global TP1_PACK_ROOT
    global TP1_CSV
    global TP1_GPU_BASELINE
    global TP1_CONTEXT_HBM_CURVE
    global TP1_CONTEXT_HBM_CURVE_16K
    global TP1_CONTEXT_MAPPING_PROFILE
    global TP1_PROFILE_DIRS

    tp1_final_pack = Path(args.tp1_final_pack).resolve()
    tp1_support_pack = Path(args.tp1_support_pack).resolve()
    tp1_context_16k_pack = Path(args.tp1_context_16k_pack).resolve()
    TP1_PACK_ROOT = tp1_final_pack
    tp1_legacy_csv = tp1_final_pack / "csv/ablation1_tp1.csv"
    tp1_audited_csv = tp1_final_pack / "csv/tp1_audited_metrics.csv"
    TP1_CSV = tp1_audited_csv if tp1_audited_csv.exists() else tp1_legacy_csv
    TP1_GPU_BASELINE = resolve_tp1_gpu_baseline(
        tp1_final_pack, override=getattr(args, "tp1_gpu_baseline_csv", None)
    )
    TP1_CONTEXT_HBM_CURVE = tp1_support_pack / "hbm_service_curve/hbm_service_curve.json"
    TP1_CONTEXT_HBM_CURVE_16K = tp1_context_16k_pack / "hbm_service_curve/hbm_service_curve.json"
    TP1_CONTEXT_MAPPING_PROFILE = tp1_support_pack / "core_static_mapping_profile.json"
    TP1_PROFILE_DIRS = [
        Path(raw).resolve()
        for raw in (args.tp1_profile_dir or [str(path) for path in TP1_PROFILE_DIRS])
    ]

    for model_key, option_prefix in [
        ("deepseek_v32", "deepseek"),
        ("qwen3_235b_a22b", "qwen"),
        ("llama4_maverick", "llama"),
    ]:
        current = PACKS[model_key]
        PACKS[model_key] = PackSpec(
            tp2_dir=Path(getattr(args, f"{option_prefix}_tp2_pack")).resolve(),
            tp3_dir=Path(getattr(args, f"{option_prefix}_tp3_pack")).resolve(),
            topology_labels=current.topology_labels,
            scope_note=current.scope_note,
        )


class OverallRebuttalRunner:
    def __init__(self, out_root: Path, run_gpu_astra: bool = True) -> None:
        self.out_root = out_root
        self.run_gpu_astra = run_gpu_astra
        self.tp1_use_audited = TP1_CSV.name == "tp1_audited_metrics.csv"
        if self.tp1_use_audited:
            self.tp1_rows = {}
            self.tp1_audited_rows = load_csv_by_key(
                TP1_CSV,
                key_fields=("context_group", "array_rows", "array_cols", "batch_size", "method"),
            )
            self.tp1_gpu_rows = load_csv_by_key(
                TP1_GPU_BASELINE,
                key_fields=("context", "batch"),
            )
            self.tp1_direct_contexts = {8192, 16384}
        else:
            self.tp1_rows = load_csv_by_key(
                TP1_CSV,
                key_fields=("context_len", "array_rows", "array_cols", "batch_label", "method"),
            )
            self.tp1_audited_rows = {}
            self.tp1_gpu_rows = {}
            self.tp1_direct_contexts = {8192, 32768}
        self.tp2_cache: dict[tuple[str, str, str, int], dict[str, Any]] = {}
        self.tp3_cache: dict[tuple[str, str, int], dict[str, Any]] = {}
        self.gpu_moe_cache: dict[tuple[str, str, int], dict[str, Any]] = {}
        self.gpu_replay_contracts: dict[tuple[str, str, int], dict[str, Any]] = {}
        self.attn_cache: dict[tuple[str, str, int, int, str], dict[str, Any]] = {}
        self.context_attn_cache: dict[tuple[str, str, int, int], dict[str, Any]] = {}
        self._tp1_context_state: dict[str, Any] | None = None

    def run_all(self) -> list[dict[str, Any]]:
        scenarios: list[tuple[str, ModelSpec, BaselineSpec, str, int, int, str]] = []
        main_topo = "ls_16w4x4"
        for model in MODELS.values():
            for baseline in BASELINES.values():
                scenarios.append(("model_compare", model, baseline, main_topo, 32, 8192, "decode"))
                scenarios.append(("tpot", model, baseline, main_topo, 32, 8192, "decode"))

        deepseek = MODELS["deepseek_v32"]
        for ctx in CONTEXT_TPS_CONTEXTS:
            for baseline in BASELINES.values():
                scenarios.append(("context_tps", deepseek, baseline, main_topo, 32, ctx, "decode"))

        for batch in [16, 32, 64]:
            for baseline in BASELINES.values():
                scenarios.append(("throughput_batch", deepseek, baseline, main_topo, batch, 8192, "decode"))

        for topo in ["ls_16w4x4", "ls_4w8x8"]:
            for batch in [16, 32, 64]:
                for key in ["ls_vllm", "moentwine", "waferllm", "hacmo"]:
                    scenarios.append(("topology", deepseek, BASELINES[key], topo, batch, 8192, "decode"))

        for baseline in BASELINES.values():
            scenarios.append(("breakdown", deepseek, baseline, main_topo, 32, 32768, "decode"))

        rows: list[dict[str, Any]] = []
        seen = set()
        unique = []
        for item in scenarios:
            key = (item[0], item[1].key, item[2].key, item[3], item[4], item[5], item[6])
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)
        for idx, (scenario, model, baseline, topo, batch, ctx, phase) in enumerate(unique, start=1):
            print(
                f"[overall] {idx:03d}/{len(unique)} {scenario}: {model.label} "
                f"{baseline.label} topo={topo} b/worker={batch} ctx={ctx} phase={phase}",
                flush=True,
            )
            rows.append(self.simulate(scenario, model, baseline, topo, batch, ctx, phase))
        return rows

    def simulate(
        self,
        scenario: str,
        model: ModelSpec,
        baseline: BaselineSpec,
        topology_key: str,
        batch_per_worker: int,
        context_len: int,
        phase: str,
    ) -> dict[str, Any]:
        global_batch = batch_per_worker * NUM_WORKERS
        if scenario == "context_tps" and model.key == "deepseek_v32" and phase == "decode":
            attn = self.deepseek_context_tps_attention(model, baseline, batch_per_worker, context_len)
        else:
            attn = self.attention_components(model, baseline, batch_per_worker, context_len, phase)
        moe = self.moe_components(model, baseline, topology_key, batch_per_worker)
        if phase == "decode":
            attention_matrix_us = model.layers * attn["matrix_us"]
            kv_attention_us = model.layers * (attn["kv_mem_us"] + attn["collective_us"])
            moe_compute_us = model.moe_layers * moe["compute_us"]
            moe_comm_us = model.moe_layers * moe["communication_us"]
            total_us = attention_matrix_us + kv_attention_us + moe_compute_us + moe_comm_us
            tpot_ms = total_us / 1000.0
            throughput = global_batch * 1e6 / max(total_us, 1e-9)
            prefill_ms = 0.0
            if scenario == "breakdown" and model.key == "deepseek_v32":
                attn["source"] += "|retained_from_overall_v11|tp1_final_pack_static_traffic_shared_20260612"
        else:
            prefill = self.prefill_proxy_components(model, baseline, batch_per_worker, context_len, moe)
            attention_matrix_us = model.layers * prefill["attention_matrix_us"]
            kv_attention_us = model.layers * prefill["kv_attention_us"]
            moe_compute_us = model.moe_layers * prefill["moe_compute_us"]
            moe_comm_us = model.moe_layers * prefill["moe_comm_us"]
            total_us = attention_matrix_us + kv_attention_us + moe_compute_us + moe_comm_us
            tpot_ms = 0.0
            throughput = 0.0
            prefill_ms = total_us / 1000.0
            attn["source"] = f"{attn['source']}|{prefill['source']}"

        return {
            "scenario": scenario,
            "model": model.label,
            "model_key": model.key,
            "baseline": baseline.label,
            "baseline_key": baseline.key,
            "topology": topology_key,
            "batch_scope": "per_worker",
            "batch_per_worker": batch_per_worker,
            "batch_per_die": batch_per_worker if baseline.platform == "ls" else "",
            "batch_per_gpu": batch_per_worker if baseline.platform == "gpu" else "",
            "num_workers": NUM_WORKERS,
            "global_batch": global_batch,
            "context_len": context_len,
            "phase": phase,
            "representative_moe_layer": model.representative_layer,
            "attention_matrix_stack_ms": round(attention_matrix_us / 1000.0, 6),
            "kv_attention_stack_ms": round(kv_attention_us / 1000.0, 6),
            "moe_compute_stack_ms": round(moe_compute_us / 1000.0, 6),
            "moe_comm_stack_ms": round(moe_comm_us / 1000.0, 6),
            "tpot_ms": round(tpot_ms, 6),
            "prefill_proxy_ms": round(prefill_ms, 6),
            "throughput_tps": round(throughput, 6),
            "attn_source": attn["source"],
            "moe_source": moe["source"],
            "moe_method": moe["method"],
            "moe_compute_us": round(moe["compute_us"], 6),
            "moe_communication_us": round(moe["communication_us"], 6),
            "moe_cross_wafer_ratio": round(float(moe.get("cross_wafer_ratio", 0.0)), 6),
            "baseline_note": baseline.note,
        }

    def attention_components(
        self,
        model: ModelSpec,
        baseline: BaselineSpec,
        batch_per_worker: int,
        context_len: int,
        phase: str,
    ) -> dict[str, Any]:
        key = (model.key, baseline.key, batch_per_worker, context_len, phase)
        if key in self.attn_cache:
            return self.attn_cache[key]
        if model.attention == "dsa" and phase == "decode":
            out = self.deepseek_tp1_attention(model, baseline, batch_per_worker, context_len)
        elif phase == "decode":
            out = self.dense_decode_attention(model, baseline, batch_per_worker, context_len)
        else:
            out = self.analytic_attention(model, baseline, batch_per_worker, context_len)
        if baseline.key == "waferllm" and phase == "decode":
            out = self.apply_waferllm_attention_model(out, model, batch_per_worker)
        self.attn_cache[key] = out
        return out

    def deepseek_tp1_attention(
        self,
        model: ModelSpec,
        baseline: BaselineSpec,
        batch_per_worker: int,
        context_len: int,
    ) -> dict[str, Any]:
        method = ATTENTION_TP1_METHOD[baseline.key]
        row = self.tp1_row(context_len, batch_per_worker, method)
        matrix_us = float(row.get("T_comp_us", 0.0))
        total_us = float(row["T_attn_total_us"])
        kv_us = max(total_us - matrix_us, 0.0)
        collective_us = self.attention_allreduce_us(model, baseline, batch_per_worker)
        metric_file = "summary/tp1_gpu_measured_baseline.csv" if method == "G" and self.tp1_use_audited else (
            "csv/tp1_audited_metrics.csv" if self.tp1_use_audited else "csv/ablation1_tp1.csv"
        )
        source_prefix = (
            "tp1_measured_h200_flashmla"
            if method == "G" and self.tp1_use_audited
            else ("tp1_audited_astra_ramulator_trace_driven" if self.tp1_use_audited else "tp1_final_pack_static_traffic_shared_20260612")
        )
        metric_name = "G_us" if method == "G" and self.tp1_use_audited else ("mean_exposed_us" if self.tp1_use_audited else "T_attn_total_us")
        return {
            "matrix_us": matrix_us,
            "kv_mem_us": kv_us,
            "collective_us": collective_us,
            "source": (
                f"{source_prefix}:"
                f"{TP1_PACK_ROOT.name}/{metric_file};"
                f"context={self.tp1_context_group(context_len)};array=68x68;"
                f"batch={batch_per_worker};method={method};metric={metric_name}"
                "|deployment_attention_allreduce"
            ),
        }

    def deepseek_context_tps_attention(
        self,
        model: ModelSpec,
        baseline: BaselineSpec,
        batch_per_worker: int,
        context_len: int,
    ) -> dict[str, Any]:
        key = (model.key, baseline.key, batch_per_worker, context_len)
        if key in self.context_attn_cache:
            return self.context_attn_cache[key]
        method = ATTENTION_TP1_METHOD[baseline.key]
        if context_len in self.tp1_direct_contexts:
            out = self.deepseek_tp1_attention(model, baseline, batch_per_worker, context_len)
            out = dict(out)
            out["source"] = f"{out['source']}|context_tps_direct_tp1_audited_pack"
            self.context_attn_cache[key] = out
            return out

        state = self.tp1_context_state()
        rows, cols = 68, 68
        mapping = tp1_mapping_for_shape(state["mapping_profile"], rows, cols)
        hbm_config = dict(state["hbm_config"])
        hbm_config["service_curve"] = state["hbm_service_curve"]
        lat_method = method
        hit_rate = 0.0
        if method in {"C", "S"}:
            hit_rate = self.tp1_context_hit_rate(method, batch_per_worker, context_len, state)
        elif method == "O":
            hit_rate = self.tp1_context_hit_rate("O", batch_per_worker, context_len, state)
        elif method == "G":
            lat_method = "B"

        lat = estimate_tp1_attention_latency(
            batch_size=batch_per_worker,
            context_len=context_len,
            baseline=lat_method,
            hit_rate=hit_rate,
            rows=rows,
            cols=cols,
            gamma=state["gamma"],
            mapping_derived=mapping,
            hbm_config=hbm_config,
        )
        if method == "G":
            gpu_mem_penalty_vs_b = 1.08
            matrix_us = float(lat["T_comp_us"])
            indexer_mem_us = float(lat["T_indexer_mem_us"]) * gpu_mem_penalty_vs_b
            indexer_compute_us = float(lat["T_indexer_compute_us"])
            qkv_mem_us = float(lat["T_qkv_mem_us"]) * gpu_mem_penalty_vs_b
            total_us = matrix_us + indexer_mem_us + indexer_compute_us + qkv_mem_us
        else:
            matrix_us = float(lat["T_comp_us"])
            total_us = float(lat["T_attn_total_us"])
        collective_us = self.attention_allreduce_us(model, baseline, batch_per_worker)
        out = {
            "matrix_us": matrix_us,
            "kv_mem_us": max(total_us - matrix_us, 0.0),
            "collective_us": collective_us,
            "source": (
                "tp1_context_tps_service_curve:"
                f"method_{method};ctx={context_len};"
                f"main_curve={TP1_CONTEXT_HBM_CURVE.name};"
                f"supplemental_curve={TP1_CONTEXT_HBM_CURVE_16K.name}"
                "|deployment_attention_allreduce"
            ),
        }
        self.context_attn_cache[key] = out
        return out

    def tp1_context_state(self) -> dict[str, Any]:
        if self._tp1_context_state is not None:
            return self._tp1_context_state
        main_curve = load_tp1_hbm_service_curve(str(TP1_CONTEXT_HBM_CURVE))
        supplemental_curve = load_tp1_hbm_service_curve(str(TP1_CONTEXT_HBM_CURVE_16K))
        merged_curve = dict(main_curve)
        merged_rows = list(main_curve.get("rows", [])) + list(supplemental_curve.get("rows", []))
        merged_curve["rows"] = merged_rows
        gamma_data = json.loads(TP1_CONTEXT_GAMMA_JSON.read_text(encoding="utf-8"))
        layer_ids = list(range(0, MODELS["deepseek_v32"].layers, 6))
        profile_files: list[Path] = []
        for profile_dir in TP1_PROFILE_DIRS:
            profile_files.extend(sorted(profile_dir.glob("*_profile.pkl")))
        profile_files = [p for p in profile_files if "gutenberg_0_profile.pkl" not in p.name][:3]
        layer_streams = [
            parse_tp1_profile_file(path, decode_steps_limit=50, layer_ids=layer_ids)
            for path in profile_files
        ]
        self._tp1_context_state = {
            "gamma": float(gamma_data.get("gamma_fit", gamma_data.get("gamma", 1.0))),
            "mapping_profile": load_tp1_mapping_profile(str(TP1_CONTEXT_MAPPING_PROFILE)),
            "hbm_service_curve": merged_curve,
            "hbm_config": {
                "model": "service_curve",
                "hbm_peak_tbps": 4.8,
                "dense_hbm_util": DENSE_HBM_EFFECTIVE_FRACTION,
                "sparse_pattern": "profile",
                "sparse_compute_tokens": 2048.0,
                "selected_kv_bytes_per_token": float(model_selected_kv_bytes(MODELS["deepseek_v32"])),
                "indexer_model": "deepseek_v32",
                "indexer_k_bytes_per_token": 128.0,
                "indexer_scale_bytes_per_token": 4.0,
                "indexer_index_bytes": 4.0,
                "indexer_n_heads": 64.0,
                "indexer_head_dim": 128.0,
                "indexer_compute_tflops": 2250.0,
                "noc_model": "shared",
                "noc_bw_scale_mode": "cols",
            },
            "layer_streams": layer_streams,
            "hit_rate_memo": {},
            "hit_rate_overrides": {},
        }
        return self._tp1_context_state

    @staticmethod
    def tp1_context_hit_rate(method: str, batch_per_worker: int, context_len: int, state: dict[str, Any]) -> float:
        return get_tp1_hit_rate(
            overrides=state["hit_rate_overrides"],
            memo=state["hit_rate_memo"],
            layer_streams_all=state["layer_streams"],
            batch_size=batch_per_worker,
            cache_ratio=0.40,
            context_len=context_len,
            policy=method,
            kv_entry_bytes=model_selected_kv_bytes(MODELS["deepseek_v32"]),
            total_sram_bytes=int(440.0 * 1024 * 1024),
            n_layers=MODELS["deepseek_v32"].layers,
        )

    def analytic_attention(
        self,
        model: ModelSpec,
        baseline: BaselineSpec,
        batch_per_worker: int,
        context_len: int,
    ) -> dict[str, Any]:
        dense = model.attention != "dsa"
        topk = context_len if dense else min(model.sparse_topk, context_len)
        selected = model.dense_kv_bytes_per_token if dense else model.selected_vector_bytes
        cfg = SimulationConfig(
            backend="analytic",
            hardware=HardwareConfig(
                array_rows=68,
                array_cols=68,
                hbm_peak_tbps=4.8,
                noc_bw_tbps=5.0,
                compute_peak_tflops=2250.0,
                total_sram_mb=440.0,
            ),
            workload=WorkloadConfig(
                model_name=model.label,
                layer_type="attention",
                batch_size=batch_per_worker,
                context_len=context_len,
                topk=topk,
                selected_vector_bytes=selected,
                hidden_size=model.hidden,
                n_heads=model.n_heads,
                head_dim=model.head_dim,
                n_layers=model.layers,
            ),
            parallel=ParallelConfig(cp_degree=1, horizontal_strategy="dp", tp_degree_horizontal=1),
            mapping=CachePolicy(name=f"{baseline.key}_dense", cache_policy="none"),
        )
        raw = simulate_layer(cfg)
        collective_us = self.attention_allreduce_us(model, baseline, batch_per_worker)
        return {
            "matrix_us": raw.compute_us,
            "kv_mem_us": raw.hbm_us + raw.sram_noc_us + raw.prefetch_exposed_us,
            "collective_us": collective_us,
            "source": "supernode_sim_analytic_attention_dense_or_prefill|deployment_attention_allreduce",
        }

    def dense_decode_attention(
        self,
        model: ModelSpec,
        baseline: BaselineSpec,
        batch_per_worker: int,
        context_len: int,
    ) -> dict[str, Any]:
        # Decode reads one token's dense KV cache per worker.  The generic
        # analytic backend models wafer-core sparse-KV movement and multiplies
        # HBM traffic by average core-to-edge hops; that is appropriate for TP1
        # sparse-attention studies but overcharges dense decode in this overall
        # composition.
        flops = dense_attention_decode_flops(model, batch_per_worker, context_len)
        matrix_us = flops / (2250.0e12) * 1e6
        kv_bytes = float(batch_per_worker) * context_len * model.dense_kv_bytes_per_token
        kv_hbm_us = self.hbm_stream_us(kv_bytes, DENSE_HBM_EFFECTIVE_FRACTION)
        # KV injection/NoC transport is charged on every row.  All rows share the
        # same per-die substrate (68x68 array, 4.8 TB/s HBM, 5 TB/s NoC) and
        # differ only in fabric, so removing this term from GPU rows alone would
        # make the Layered Supernode pay a substrate cost its GPU comparison does
        # not, which inverts the LS-vs-NVL72 ordering for reasons unrelated to
        # either fabric.  HACMO's credit on this term comes from TP1 static
        # mapping and is applied below.
        kv_injection_us = kv_bytes / (5.0e12) * 1e6
        source_extra = ""
        if baseline.key == "hacmo":
            nonuniform = self.dense_nonuniform_mapping_credit(batch_per_worker, context_len)
            kv_injection_us *= nonuniform["injection_scale"]
            source_extra = (
                "|dense_nonuniform_workload_mapping:"
                f"static_only_{DENSE_NONUNIFORM_STATIC_METHOD}_vs_{DENSE_NONUNIFORM_BASE_METHOD};"
                f"hbm_hops={nonuniform['static_hbm_hops']:.6g}/{nonuniform['base_hbm_hops']:.6g};"
                f"injection_scale={nonuniform['injection_scale']:.6g};"
                "applied_to=kv_injection_only;no_prefetch_no_hbm_service_credit"
            )
        collective_us = self.attention_allreduce_us(model, baseline, batch_per_worker)
        return {
            "matrix_us": matrix_us,
            "kv_mem_us": kv_hbm_us + kv_injection_us,
            "collective_us": collective_us,
            "source": (
                "overall_dense_decode_attention_hbm_stream:"
                f"dense_hbm_util={DENSE_HBM_EFFECTIVE_FRACTION}"
                f"{source_extra}|deployment_attention_allreduce"
            ),
        }

    def dense_nonuniform_mapping_credit(self, batch_per_worker: int, context_len: int) -> dict[str, float]:
        base = self.tp1_row(context_len, batch_per_worker, DENSE_NONUNIFORM_BASE_METHOD)
        static = self.tp1_row(context_len, batch_per_worker, DENSE_NONUNIFORM_STATIC_METHOD)
        base_hops = max(float(base["avg_hbm_hops"]), 1e-9)
        static_hops = max(float(static["avg_hbm_hops"]), 0.0)
        return {
            "base_hbm_hops": base_hops,
            "static_hbm_hops": static_hops,
            "injection_scale": min(1.0, max(0.0, static_hops / base_hops)),
        }

    def apply_waferllm_attention_model(
        self,
        base: dict[str, Any],
        model: ModelSpec,
        batch_per_worker: int,
    ) -> dict[str, Any]:
        out = dict(base)
        bytes_per_rank = batch_per_worker * model.hidden * 2
        out["collective_us"] = self.waferllm_two_tree_collective_us(bytes_per_rank, ranks=NUM_WORKERS)
        out["source"] = f"{base['source']}|waferllm_matrix_model_hacmo_reproduction:MeshGEMV_two_tree"
        return out

    def prefill_proxy_components(
        self,
        model: ModelSpec,
        baseline: BaselineSpec,
        batch_per_worker: int,
        context_len: int,
        moe: dict[str, Any],
    ) -> dict[str, float]:
        # Proxy semantics: prompt tokens are processed in explicit chunks, using
        # dense attention GEMM work for matrix cost and the representative TP2
        # MoE layer for routed FFN/communication pressure.
        tokens = batch_per_worker * context_len
        if model.attention == "dense":
            flops = dense_attention_decode_flops(model, tokens, context_len)
        else:
            flops = compute_attention_flops(tokens, context_len, model.hidden, model.n_heads, model.head_dim)
        matrix_us = flops / (2250.0e12) * 1e6
        weight_us = self.hbm_stream_us(self.attention_weight_bytes(model), DENSE_HBM_EFFECTIVE_FRACTION)
        source = "prefill_proxy_dense_gemm_roofline"
        if baseline.key == "waferllm":
            matrix_us = self.waferllm_meshgemm_us(matrix_us, tokens, model.hidden)
            source = "waferllm_matrix_model_hacmo_reproduction:MeshGEMM_prefill"
        kv_bytes = float(tokens) * model.dense_kv_bytes_per_token
        kv_us = self.hbm_stream_us(kv_bytes, DENSE_HBM_EFFECTIVE_FRACTION)
        moe_microbatches = max(math.ceil(context_len / 128), 1)
        return {
            "attention_matrix_us": matrix_us + weight_us,
            "kv_attention_us": kv_us,
            "moe_compute_us": moe["compute_us"] * moe_microbatches,
            "moe_comm_us": moe["communication_us"] * moe_microbatches,
            "source": f"{source}|attention_weight_hbm_stream:dense_hbm_util={DENSE_HBM_EFFECTIVE_FRACTION}",
        }

    def moe_components(
        self,
        model: ModelSpec,
        baseline: BaselineSpec,
        topology_key: str,
        batch_per_worker: int,
    ) -> dict[str, Any]:
        if baseline.key in {"dgx", "nvl72"}:
            return self.gpu_vllm_moe(model, baseline, batch_per_worker)
        topology = PACKS[model.key].topology_labels.get(topology_key)
        if topology is None:
            raise ValueError(f"No {model.label} TP2/TP3 pack for topology {topology_key}")
        if baseline.key == "hacmo":
            row = self.tp3_row(model, topology, batch_per_worker, "full_tp2")
            return self.moe_from_row(row, "tp3_lane_level_full_tp2", "full_tp2")
        if baseline.key == "moentwine":
            row = self.tp2_row(model, topology, batch_per_worker, "moentwine")
            return self.moe_from_row(row, "tp2_serving_epoch_moentwine_static", "moentwine_static")
        row = self.tp2_row(model, topology, batch_per_worker, "vllm")
        out = self.moe_from_row(row, "tp2_serving_epoch_vllm", "vllm")
        if baseline.key == "waferllm":
            out["compute_us"] = self.waferllm_moe_compute_us(out["compute_us"])
            out["total_us"] = out["compute_us"] + out["communication_us"]
            out["source"] += "|waferllm_matrix_model_hacmo_reproduction:MoE_FFN_matrix_only"
            out["method"] = "vllm_placement_comm|waferllm_matrix_model"
        return out

    def tp2_row(self, model: ModelSpec, topology: str, batch_per_worker: int, method: str) -> dict[str, Any]:
        key = (model.key, topology, method, batch_per_worker)
        if key in self.tp2_cache:
            return self.tp2_cache[key]
        pack = PACKS[model.key].tp2_dir
        rows: list[dict[str, Any]] = []
        for path in [
            pack / "paper_figures/paper_figure_summary.csv",
            pack / "csv/tp2_serving_epoch_astra_comm_only_summary.csv",
            pack / f"{topology}/supernode_astra_congestion_aware/metrics_{topology}.csv",
            pack / "csv/tp2_serving_epoch_analytical_summary.csv",
        ]:
            if path.exists():
                rows = read_csv(path)
                break
        if not rows:
            raise FileNotFoundError(f"No TP2 summary CSV found under {pack}")
        for row in rows:
            row_topo = row.get("topology", "")
            row_method = row.get("method", "")
            batch_raw = row.get("batch_size") or row.get("batch_per_die") or ""
            if row_topo == topology and row_method == method and int(float(batch_raw)) == batch_per_worker:
                row = dict(row)
                row["artifact_csv"] = str(path)
                self.tp2_cache[key] = row
                return row
        raise KeyError(f"Missing TP2 row model={model.key} topology={topology} method={method} batch={batch_per_worker}")

    def tp3_row(self, model: ModelSpec, topology: str, batch_per_worker: int, bar_key: str) -> dict[str, Any]:
        key = (model.key, topology, bar_key, batch_per_worker)
        if key in self.tp3_cache:
            return self.tp3_cache[key]
        path = PACKS[model.key].tp3_dir / "tp3_metrics.csv"
        for row in read_csv(path):
            if (
                row.get("topology") == topology
                and row.get("bar_key") == bar_key
                and int(float(row.get("batch_size") or row.get("batch_per_die"))) == batch_per_worker
            ):
                row = dict(row)
                row["artifact_csv"] = str(path)
                self.tp3_cache[key] = row
                return row
        raise KeyError(f"Missing TP3 row model={model.key} topology={topology} bar={bar_key} batch={batch_per_worker}")

    @staticmethod
    def moe_from_row(row: dict[str, Any], source_tag: str, method: str) -> dict[str, Any]:
        compute = float(row["compute_us"])
        comm = float(row.get("communication_us", row.get("comm_us", 0.0)))
        return {
            "compute_us": compute,
            "communication_us": comm,
            "total_us": compute + comm,
            "cross_wafer_ratio": float(row.get("cross_wafer_ratio", 0.0) or 0.0),
            "source": f"{source_tag}|{pack_relative(row.get('artifact_csv', ''))}",
            "method": method,
        }

    def gpu_vllm_moe(self, model: ModelSpec, baseline: BaselineSpec, batch_per_worker: int) -> dict[str, Any]:
        key = (model.key, baseline.key, batch_per_worker)
        if key in self.gpu_moe_cache:
            return self.gpu_moe_cache[key]
        pack = PACKS[model.key]
        topology = pack.topology_labels["ls_16w4x4"]
        artifact_dir = pack.tp2_dir / topology
        runtime_window_stats = artifact_dir / "mapping_inputs/runtime/window_stats.json"
        runtime_window_src = artifact_dir / "mapping_inputs/runtime/window_src_expert_counts.json"
        uniform_mapping_json = artifact_dir / "mappings/uniform.json"

        # A paper-facing GPU baseline must be a real replay.  Reusing the
        # published LS vLLM row here would silently produce a GPU row with no
        # GPU-fabric cost at all, so a missing runtime artifact is an error.
        if not (runtime_window_stats.exists() and runtime_window_src.exists()):
            raise FileNotFoundError(
                f"{model.key}: GPU replay requires TP2 runtime artifacts under "
                f"{artifact_dir / 'mapping_inputs/runtime'}. Stage window_stats.json and "
                "window_src_expert_counts.json into the pack, or pass a pack that retains them."
            )

        ls_row = find_ls_step_scope_row(pack.tp2_dir, topology, batch_per_worker)
        scale = resolve_replay_scale(
            artifact_dir,
            global_batch=batch_per_worker * NUM_WORKERS,
            num_dies=NUM_WORKERS,
            batch_per_die=batch_per_worker,
        )
        scope = resolve_step_window_scope(artifact_dir, ls_row)
        self.gpu_replay_contracts[(model.key, baseline.key, batch_per_worker)] = {
            "replay_scale": scale.as_metadata(),
            "step_window_scope": scope.as_metadata(),
        }

        cfg = copy.deepcopy(load_config(REPO / BASE_MOE_CONFIG))
        cfg.backend = "astra"
        cfg.workload.model_name = model.label
        cfg.workload.layer_type = "moe"
        cfg.workload.batch_size = batch_per_worker * NUM_WORKERS
        cfg.workload.decode_len = 128
        cfg.workload.hidden_size = model.hidden
        cfg.moe.num_experts = model.num_experts
        cfg.moe.topk = model.moe_topk
        cfg.moe.num_requests = scale.num_requests
        cfg.moe.exclude_leading_windows = scope.exclude_leading_windows
        cfg.moe.window_size = 8
        cfg.moe.hidden_size = model.hidden
        cfg.moe.bytes_per_hidden_elem = 2
        cfg.moe.dispatch_bytes_per_hidden_elem = 1
        cfg.moe.combine_bytes_per_hidden_elem = 2
        cfg.moe.compute_backend = "roofline_h200"
        cfg.moe.expert_intermediate_size = model.expert_intermediate
        cfg.moe.h200_effective_tflops = 1979.0
        cfg.moe.h200_hbm_bw_tbps = 4.8
        cfg.moe.weight_bytes_per_elem = 1
        cfg.moe.activation_bytes_per_elem = 2
        cfg.moe.mapping_method = "external_vllm"
        cfg.moe.window_stats_json = str(artifact_dir / "mapping_inputs/runtime/window_stats.json")
        cfg.moe.window_src_expert_json = str(artifact_dir / "mapping_inputs/runtime/window_src_expert_counts.json")
        if uniform_mapping_json.exists():
            cfg.moe.expert_mapping_json = str(uniform_mapping_json)
        else:
            cfg.moe.expert_mapping = self.uniform_expert_mapping(model.num_experts)
        cfg.moe.window_mappings_json = ""
        cfg.moe.astra_trace_mode = "communication_only"
        cfg.supernode.num_wafers = 16
        cfg.supernode.dies_per_wafer = 16
        cfg.supernode.network_config = str(NETWORKS[baseline.network]["rail"].relative_to(REPO))
        cfg.supernode.system_config = SYSTEM_CONFIG
        cfg.supernode.remote_memory_config = REMOTE_MEMORY_CONFIG
        cfg.supernode.astra_sim_dir = "astra-sim"
        cfg.supernode.astra_network_backend = "congestion_aware"
        cfg.supernode.run_astra = self.run_gpu_astra
        cfg.supernode.keep_traces = True
        cfg.supernode.output_root = str(self.out_root / "runs/gpu_moe" / model.key / baseline.key / f"bpd{batch_per_worker}")
        result = simulate_layer(cfg)
        profile = result.metadata.get("profile", {}) if result.metadata else {}
        steps = trace_steps(
            artifact_dir / "mapping_inputs/runtime/window_stats.json",
            128,
            exclude_leading_windows=scope.exclude_leading_windows,
        )
        compute = float(profile.get("analysis_compute_us_per_step", 0.0))
        comm = result.communication_us / steps if self.run_gpu_astra else 0.0
        out = {
            "compute_us": compute,
            "communication_us": comm,
            "total_us": compute + comm,
            "cross_wafer_ratio": float(profile.get("cross_wafer_ratio", 0.0)),
            "source": (
                "tp2_runtime_artifact_replay_gpu_vllm|"
                f"{pack_relative(artifact_dir)}|network={baseline.network}"
                f"|astra={self.run_gpu_astra}"
                f"|replay_convention={scale.convention}|batch_scale={scale.batch_scale:g}"
                f"|num_requests={scale.num_requests}"
                f"|exclude_leading_windows={scope.exclude_leading_windows}"
                f"|reported_steps={steps}"
            ),
            "method": f"vllm_uniform_replayed_on_{baseline.network}",
            "replay_dispatch_bytes_per_step": float(profile.get("total_dispatch_bytes", 0.0)) / max(steps, 1),
        }
        self.gpu_moe_cache[key] = out
        return out

    @staticmethod
    def uniform_expert_mapping(num_experts: int) -> dict[str, list[int]]:
        return {str(expert_id): [expert_id] for expert_id in range(num_experts)}

    def tp1_row(self, context_len: int, batch_per_worker: int, method: str) -> dict[str, Any]:
        if self.tp1_use_audited:
            context_group = self.tp1_context_group(context_len)
            if method == "G":
                gpu_key = (context_group, str(batch_per_worker))
                if gpu_key in self.tp1_gpu_rows:
                    gpu_row = self.tp1_gpu_rows[gpu_key]
                    return {
                        "T_comp_us": 0.0,
                        "T_attn_total_us": float(gpu_row["G_us"]),
                        "avg_hbm_hops": 0.0,
                    }
            key = (context_group, "68", "68", str(batch_per_worker), method)
            if key in self.tp1_audited_rows:
                row = self.tp1_audited_rows[key]
                compute_us = float(row["compute_busy_us_sum"]) / (
                    float(row["physical_cores"]) * float(row["sampled_windows"])
                )
                return {
                    "T_comp_us": compute_us,
                    "T_attn_total_us": float(row["mean_exposed_us"]),
                    "avg_hbm_hops": float(row["hbm_byte_hops"]) / max(float(row["hbm_requested_bytes"]), 1.0),
                }
        else:
            ctx = 32768 if context_len > 8192 else 8192
            candidates = [str(batch_per_worker), "64+CP2", "128+CP4"]
            for batch_label in candidates:
                key = (str(ctx), "68", "68", batch_label, method)
                if key in self.tp1_rows:
                    return self.tp1_rows[key]
        raise KeyError(f"Missing TP1 row ctx={context_len} batch={batch_per_worker} method={method}")

    @staticmethod
    def tp1_context_group(context_len: int) -> str:
        return "8k" if context_len <= 8192 else "16k"

    @staticmethod
    def attention_allreduce_us(model: ModelSpec, baseline: BaselineSpec, batch_per_worker: int, tp: int = 16) -> float:
        net = NETWORKS[baseline.network]
        bytes_per_rank = batch_per_worker * model.hidden * 2
        transfer = 2.0 * (tp - 1) / tp * bytes_per_rank / max(float(net["attention_bw_tbps"]) * 1e12, 1.0) * 1e6
        latency = math.ceil(math.log2(tp)) * float(net["attention_latency_ns"]) / 1000.0
        return transfer + latency

    @staticmethod
    def waferllm_two_tree_collective_us(bytes_per_rank: float, ranks: int) -> float:
        bw_tbps = 5.0
        latency_us = 0.5
        levels = max(math.ceil(math.log2(max(ranks, 2))), 1)
        transfer = 2.0 * (ranks - 1) / ranks * bytes_per_rank / (bw_tbps * 1e12) * 1e6
        return transfer + 2.0 * levels * latency_us

    @staticmethod
    def waferllm_moe_compute_us(base_compute_us: float) -> float:
        # Recompute the H200 roofline compute component on the HACMO wafer-core
        # peak.  This is not a paper multiplier; it is the explicit local matrix
        # roofline used for the WaferLLM reproduction on the same substrate.
        return base_compute_us * (1979.0 / 2250.0)

    @staticmethod
    def hbm_stream_us(num_bytes: float, effective_fraction: float) -> float:
        return float(num_bytes) / (4.8e12 * effective_fraction) * 1e6

    @staticmethod
    def attention_weight_bytes(model: ModelSpec) -> float:
        q_proj = model.hidden * model.hidden
        o_proj = model.hidden * model.hidden
        k_proj = model.hidden * model.kv_heads * model.head_dim
        v_proj = model.hidden * model.kv_heads * model.head_dim
        return float(q_proj + o_proj + k_proj + v_proj) * WEIGHT_BYTES_PER_ELEM

    @staticmethod
    def waferllm_meshgemm_us(base_compute_us: float, tokens: int, hidden: int) -> float:
        compute_us = base_compute_us
        shift_bytes = float(tokens) * hidden * 2
        shift_us = shift_bytes / (5.0e12) * 1e6 + 2.0
        return compute_us + shift_us


def build_panels(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    def select(name: str) -> list[dict[str, Any]]:
        return [r for r in rows if r["scenario"] == name]

    panels: dict[str, list[dict[str, Any]]] = {}
    model_rows = select("model_compare")
    ref = {r["model"]: float(r["throughput_tps"]) for r in model_rows if r["baseline_key"] == "dgx"}
    panels["model_compare"] = [
        {**r, "normalized_tps_vs_dgx": round(float(r["throughput_tps"]) / max(ref[r["model"]], 1e-9), 6)}
        for r in model_rows
    ]
    panels["tpot"] = select("tpot")

    context_rows = select("context_tps")
    context_ref = {r["context_len"]: float(r["throughput_tps"]) for r in context_rows if r["baseline_key"] == "dgx"}
    panels["context_tps"] = [
        {**r, "normalized_tps_vs_dgx": round(float(r["throughput_tps"]) / max(context_ref[r["context_len"]], 1e-9), 6)}
        for r in context_rows
    ]

    batch_rows = select("throughput_batch")
    batch_ref = {r["batch_per_worker"]: float(r["throughput_tps"]) for r in batch_rows if r["baseline_key"] == "dgx"}
    panels["throughput_batch"] = [
        {**r, "normalized_tps_vs_dgx": round(float(r["throughput_tps"]) / max(batch_ref[r["batch_per_worker"]], 1e-9), 6)}
        for r in batch_rows
    ]

    topo_rows = select("topology")
    topo_ref = {
        (r["topology"], r["batch_per_worker"]): float(r["throughput_tps"])
        for r in topo_rows
        if r["baseline_key"] == "ls_vllm"
    }
    panels["topology"] = [
        {
            **r,
            "topology_batch": f"{r['topology']}|B{r['batch_per_worker']}",
            "normalized_tps_vs_ls_vllm": round(
                float(r["throughput_tps"]) / max(topo_ref[(r["topology"], r["batch_per_worker"])], 1e-9), 6
            ),
        }
        for r in topo_rows
    ]
    panels["breakdown"] = select("breakdown")
    return panels


def plot_overall(panels: dict[str, list[dict[str, Any]]], fig_dir: Path) -> None:
    colors = {
        "dgx": "#737373",
        "nvl72": "#a6a6a6",
        "ls_vllm": "#4c78a8",
        "moentwine": "#f28e2b",
        "waferllm": "#b07aa1",
        "hacmo": "#d62728",
    }
    labels = {k: v.label for k, v in BASELINES.items()}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.4, "figure.dpi": 220})
    fig, axs = plt.subplots(2, 3, figsize=(12.8, 6.5))
    plot_grouped_bars(axs[0, 0], panels["model_compare"], "model", "baseline_key", "normalized_tps_vs_dgx", list(BASELINES), colors, "Norm. decode TPS", "(a) Overall decode TPS", labels, group_label_fn=model_short_label)
    plot_grouped_bars(axs[0, 1], panels["tpot"], "model", "baseline_key", "tpot_ms", list(BASELINES), colors, "TPOT (ms)", "(b) Decode TPOT, B/worker=32, ctx=8K", labels, group_label_fn=model_short_label, lower_is_better=True)
    plot_grouped_bars(axs[0, 2], panels["context_tps"], "context_len", "baseline_key", "normalized_tps_vs_dgx", list(BASELINES), colors, "Norm. decode TPS", "(c) Decode TPS vs context", labels, group_label_fn=lambda x: f"{int(x)//1024}K")
    plot_lines(axs[1, 0], panels["throughput_batch"], "batch_per_worker", "baseline_key", "throughput_tps", list(BASELINES), colors, "Decode TPS", "B/worker", "(d) Decode TPS scaling, 16W x 4x4", labels)
    topo_rows = [r for r in panels["topology"] if r["baseline_key"] in {"ls_vllm", "moentwine", "waferllm", "hacmo"}]
    plot_grouped_bars(axs[1, 1], topo_rows, "topology_batch", "baseline_key", "normalized_tps_vs_ls_vllm", ["ls_vllm", "moentwine", "waferllm", "hacmo"], colors, "Norm. TPS vs vLLM+LS", "(e) LS topology and B/worker scaling", labels, group_label_fn=topology_batch_label)
    rotate_xticklabels(axs[1, 1], 12, 7.0)
    plot_breakdown(axs[1, 2], panels["breakdown"], colors, labels)
    handles, leg_labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "overall_performance_dense.png", bbox_inches="tight")
    fig.savefig(fig_dir / "overall_performance_dense.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_breakdown(ax, rows: list[dict[str, Any]], colors: dict[str, str], labels: dict[str, str]) -> None:
    x = np.arange(len(rows))
    bkeys = [r["baseline_key"] for r in rows]
    matrix = [float(r["attention_matrix_stack_ms"]) for r in rows]
    kv = [float(r["kv_attention_stack_ms"]) for r in rows]
    moe_comp = [float(r["moe_compute_stack_ms"]) for r in rows]
    moe_comm = [float(r["moe_comm_stack_ms"]) for r in rows]
    ax.bar(x, matrix, color="#4c78a8", edgecolor="white", linewidth=0.3, label="Attn matrix")
    bottom = np.array(matrix)
    ax.bar(x, kv, bottom=bottom, color="#72b7b2", edgecolor="white", linewidth=0.3, label="KV/attn mem")
    bottom += np.array(kv)
    ax.bar(x, moe_comp, bottom=bottom, color="#f28e2b", edgecolor="white", linewidth=0.3, label="MoE compute")
    bottom += np.array(moe_comp)
    ax.bar(x, moe_comm, bottom=bottom, color="#d62728", edgecolor="white", linewidth=0.3, label="MoE comm")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[k].replace("+", "\n+") for k in bkeys], rotation=18, ha="right")
    ax.tick_params(axis="x", labelsize=7.0)
    ax.set_ylabel("Stack latency (ms)")
    ax.set_title("(f) DeepSeek component breakdown, ctx=32K")
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.legend(loc="upper right", frameon=False, fontsize=6.8)


def plot_grouped_bars(ax, rows, group_key, bar_key, value_key, bar_order, colors, ylabel, title, legend_labels, group_label_fn=None, lower_is_better=False):
    groups = list(OrderedDict((r[group_key], None) for r in rows).keys())
    idx = np.arange(len(groups))
    width = min(0.78 / max(len(bar_order), 1), 0.12)
    by = {(r[group_key], r[bar_key]): r for r in rows}
    ymax = 0.0
    for i, b in enumerate(bar_order):
        vals = [float(by[(g, b)][value_key]) if (g, b) in by else 0.0 for g in groups]
        ymax = max(ymax, max(vals) if vals else 0.0)
        ax.bar(idx + (i - (len(bar_order) - 1) / 2) * width, vals, width, label=legend_labels.get(b, b), color=colors[b], edgecolor="white", linewidth=0.35)
    ax.set_xticks(idx)
    ax.set_xticklabels([group_label_fn(g) if group_label_fn else str(g) for g in groups])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    if not lower_is_better:
        ax.axhline(1.0, color="#555555", linestyle="--", linewidth=0.7, alpha=0.6)
    ax.set_ylim(0, ymax * 1.18 if ymax > 0 else 1.0)


def plot_lines(ax, rows, x_key, series_key, y_key, series_order, colors, ylabel, xlabel, title, legend_labels):
    for s in series_order:
        pts = sorted([r for r in rows if r[series_key] == s], key=lambda r: int(r[x_key]))
        if not pts:
            continue
        ax.plot([int(r[x_key]) for r in pts], [float(r[y_key]) for r in pts], marker="o", linewidth=1.7, markersize=3.5, color=colors[s], label=legend_labels.get(s, s))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="both", alpha=0.25, linewidth=0.5)
    ax.set_xticks(sorted({int(r[x_key]) for r in rows}))


def topology_batch_label(value: str) -> str:
    topo, batch = value.split("|")
    topo_label = "16W x 4x4" if topo == "ls_16w4x4" else "4W x 8x8"
    return f"{topo_label}\n{batch}"


def model_short_label(value: str) -> str:
    labels = {
        "DeepSeek-V3.2": "DeepSeek\nV3.2",
        "Qwen3-235B-A22B-FP8": "Qwen3\n235B-A22B",
        "Llama-4-Maverick-17B-128E-Instruct": "Llama-4\nMaverick",
    }
    return labels.get(value, str(value))


def rotate_xticklabels(ax, rotation: float, fontsize: float) -> None:
    ax.tick_params(axis="x", labelsize=fontsize)
    for label in ax.get_xticklabels():
        label.set_rotation(rotation)
        label.set_ha("right")


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def model_selected_kv_bytes(model: ModelSpec) -> int:
    return int(model.selected_vector_bytes)


def dense_attention_decode_flops(model: ModelSpec, batch: int, context_len: int) -> float:
    """Dense attention matrix FLOPs with GQA-aware Q/K/V/O projections."""
    q_dim = model.n_heads * model.head_dim
    kv_dim = model.kv_heads * model.head_dim
    q_proj = 2.0 * batch * model.hidden * q_dim
    k_proj = 2.0 * batch * model.hidden * kv_dim
    v_proj = 2.0 * batch * model.hidden * kv_dim
    qk = 2.0 * 2.0 * batch * model.n_heads * model.head_dim * context_len
    out_proj = 2.0 * batch * q_dim * model.hidden
    return q_proj + k_proj + v_proj + qk + out_proj


def load_csv_by_key(path: Path, key_fields: tuple[str, ...]) -> dict[tuple[str, ...], dict[str, Any]]:
    return {tuple(str(r[field]) for field in key_fields): r for r in read_csv(path)}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_ppt_csv_bundle(out_root: Path, panels: dict[str, list[dict[str, Any]]]) -> None:
    ppt_dir = out_root / "ppt_csv"
    ppt_dir.mkdir(parents=True, exist_ok=True)
    panel_names = [
        "model_compare",
        "tpot",
        "context_tps",
        "throughput_batch",
        "topology",
        "breakdown",
    ]
    for name in panel_names:
        write_csv(ppt_dir / f"panel_{name}.csv", panels[name])
    context_rows = []
    for row in panels["context_tps"]:
        context_rows.append(
            {
                "context_len": row["context_len"],
                "context_label": f"{int(row['context_len']) // 1024}K",
                "baseline": row["baseline"],
                "baseline_key": row["baseline_key"],
                "tpot_ms": row["tpot_ms"],
                "throughput_tps": row["throughput_tps"],
                "normalized_tps_vs_dgx": row["normalized_tps_vs_dgx"],
                "attn_source": row["attn_source"],
                "moe_source": row["moe_source"],
            }
        )
    write_csv(ppt_dir / "panel_c_context_tps_for_ppt.csv", context_rows)


def load_window_stats(window_stats_json: Path) -> list[dict[str, Any]]:
    return yaml.safe_load(window_stats_json.read_text(encoding="utf-8")) or []


def active_windows(window_stats: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for ws in window_stats:
        tokens = sum(int(v) for v in (ws.get("tokens_per_die") or {}).values())
        activations = sum(int(v) for v in (ws.get("expert_counts") or {}).values())
        if tokens <= 0 and activations <= 0:
            continue
        out.append(ws)
    return out


def resolve_replay_scale(
    artifact_dir: Path,
    *,
    global_batch: int,
    num_dies: int,
    batch_per_die: int,
) -> ReplayScale:
    """Derive the GPU-replay traffic scale from the staged runtime inputs.

    ``supernode_sim`` scales replayed traffic by
    ``workload.batch_size / moe.num_requests``.  Rather than trusting the pack
    metadata -- whose ``num_requests`` means different things under the two
    staging conventions -- the divisor is derived so that post-scale traffic is
    exactly ``batch_per_die`` tokens per die per decode step.  This reproduces
    the metadata value exactly for ``raw_trace_scale`` packs and yields
    ``batch_scale == 1.0`` for ``pre_expanded`` packs.
    """
    window_stats_json = artifact_dir / "mapping_inputs/runtime/window_stats.json"
    meta_path = artifact_dir / "mapping_inputs/runtime/metadata.json"
    if not window_stats_json.exists():
        raise FileNotFoundError(
            f"GPU replay requires staged runtime window_stats: {window_stats_json}"
        )
    windows = active_windows(load_window_stats(window_stats_json))
    if not windows:
        raise ValueError(f"No active windows in {window_stats_json}")

    first = windows[0]
    tokens_per_die_total = sum(int(v) for v in (first.get("tokens_per_die") or {}).values())
    die_count = len(first.get("tokens_per_die") or {}) or num_dies
    start = int(first.get("start_token_idx", 1))
    end = int(first.get("end_token_idx", start))
    steps_in_window = max(end - start + 1, 1)
    observed = tokens_per_die_total / die_count / steps_in_window
    if observed <= 0:
        raise ValueError(f"Non-positive tokens_per_die in {window_stats_json}")

    # batch_scale must map observed tokens/die/step onto the target batch_per_die.
    batch_scale = batch_per_die / observed
    num_requests = int(round(global_batch / batch_scale))
    if num_requests <= 0:
        raise ValueError(f"Derived non-positive num_requests for {artifact_dir}")

    meta_num_requests = 0
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta_num_requests = int(meta.get("num_requests") or meta.get("trace_num_requests") or 0)

    convention = "raw_trace_scale" if abs(batch_scale - 1.0) > 1e-9 else "pre_expanded"
    agrees = meta_num_requests > 0 and abs(num_requests - meta_num_requests) <= 1
    derivation = (
        f"observed_tokens_per_die_per_step={observed:g}; target_batch_per_die={batch_per_die}; "
        f"batch_scale=target/observed={batch_scale:g}; num_requests=global_batch/batch_scale={num_requests}; "
        f"metadata_num_requests={meta_num_requests}; metadata_agrees={agrees}"
    )

    # Post-scale invariant: the replay must reproduce the target batch exactly.
    effective = observed * batch_scale
    if abs(effective - batch_per_die) > 1e-6 * max(batch_per_die, 1):
        raise ValueError(
            f"Replay scale invariant failed for {artifact_dir}: "
            f"post-scale tokens/die/step={effective:g} != batch_per_die={batch_per_die}"
        )
    return ReplayScale(
        num_requests=num_requests,
        batch_scale=batch_scale,
        convention=convention,
        observed_tokens_per_die_per_step=observed,
        metadata_num_requests=meta_num_requests,
        derivation=derivation,
    )


def find_ls_step_scope_row(
    pack_dir: Path, topology: str, batch_per_die: int
) -> dict[str, Any]:
    """Locate the LS ``vllm`` row that records the replayed decode-step range.

    The field lives in different CSVs across packs: the per-topology
    ``metrics_*.csv`` carries ``excluded_leading_windows``/``reported_steps``,
    while the condensed high-drift exports only carry a ``replay`` tag.  The
    published ``paper_figure_summary.csv`` carries neither, so the pack is
    searched for whichever CSV records the range.
    """
    for path in sorted(pack_dir.rglob("*.csv")):
        rows = read_csv(path)
        if not rows:
            continue
        cols = set(rows[0].keys())
        if not (cols & {"excluded_leading_windows", "replay"}):
            continue
        for row in rows:
            if row.get("method") != "vllm" or row.get("topology") != topology:
                continue
            raw_batch = row.get("batch_per_die") or row.get("batch_size") or ""
            if not raw_batch or int(float(raw_batch)) != batch_per_die:
                continue
            out = dict(row)
            out["step_scope_csv"] = str(path)
            return out
    raise FileNotFoundError(
        f"{pack_dir}: no TP2 CSV records the replayed decode-step range "
        f"(excluded_leading_windows or replay) for topology={topology} bpd={batch_per_die}"
    )


def resolve_step_window_scope(
    artifact_dir: Path, tp2_row: dict[str, Any]
) -> StepWindowScope:
    """Mirror the LS row's replayed decode-step range onto the GPU replay.

    TP2 reports the range as ``excluded_leading_windows`` counted in route-plan
    *steps*.  The GPU replay consumes window aggregates, so the count is
    converted to whole windows.  Packs whose published CSVs omit the field fall
    back to the ``replay`` provenance tag (e.g. ``tail16``).
    """
    window_stats_json = artifact_dir / "mapping_inputs/runtime/window_stats.json"
    windows = active_windows(load_window_stats(window_stats_json))
    first = windows[0]
    start = int(first.get("start_token_idx", 1))
    end = int(first.get("end_token_idx", start))
    steps_per_window = max(end - start + 1, 1)
    total_steps = sum(
        max(int(ws.get("end_token_idx", 1)) - int(ws.get("start_token_idx", 1)) + 1, 1)
        for ws in windows
    )

    excluded_steps = 0
    source = ""
    raw_excluded = tp2_row.get("excluded_leading_windows")
    if raw_excluded not in (None, ""):
        excluded_steps = int(float(raw_excluded))
        source = f"tp2_row.excluded_leading_windows={excluded_steps}"
    else:
        replay_tag = str(tp2_row.get("replay") or "").strip()
        if replay_tag.startswith("tail"):
            tail = int(replay_tag[4:])
            excluded_steps = max(total_steps - tail, 0)
            source = f"tp2_row.replay={replay_tag} -> excluded_leading_steps={excluded_steps}"
        else:
            source = "no_exclusion_recorded_in_tp2_row"

    if excluded_steps % steps_per_window != 0:
        raise ValueError(
            f"{artifact_dir}: LS excluded_leading_steps={excluded_steps} does not land on a "
            f"window boundary (steps_per_window={steps_per_window}); GPU replay cannot mirror "
            "the LS step range from window aggregates"
        )
    exclude_windows = excluded_steps // steps_per_window
    if exclude_windows >= len(windows):
        raise ValueError(
            f"{artifact_dir}: excluding {exclude_windows} of {len(windows)} windows leaves nothing"
        )
    return StepWindowScope(
        exclude_leading_windows=exclude_windows,
        excluded_leading_steps=excluded_steps,
        reported_steps=total_steps - excluded_steps,
        source=f"{source}; steps_per_window={steps_per_window}; total_steps={total_steps}",
    )


def trace_steps(
    window_stats_json: Path, fallback_decode_len: int, exclude_leading_windows: int = 0
) -> int:
    if not window_stats_json.exists():
        return fallback_decode_len
    total = 0
    for ws in active_windows(load_window_stats(window_stats_json))[exclude_leading_windows:]:
        start = int(ws.get("start_token_idx", 1))
        end = int(ws.get("end_token_idx", start))
        total += max(end - start + 1, 1)
    return max(total, 1)


def write_metadata(
    out_root: Path,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    gpu_replay_contracts: dict[tuple[str, str, int], dict[str, Any]] | None = None,
) -> None:
    meta = {
        "command_args": vars(args),
        "generated_at_utc": dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z"),
        "repo": str(REPO),
        "num_rows": len(rows),
        "models": {k: asdict(v) for k, v in MODELS.items()},
        "baselines": {k: asdict(v) for k, v in BASELINES.items()},
        "packs": {
            k: {
                "tp2_dir": str(v.tp2_dir),
                "tp3_dir": str(v.tp3_dir),
                "topology_labels": v.topology_labels,
                "scope_note": v.scope_note,
            }
            for k, v in PACKS.items()
        },
        "tp1_csv": str(TP1_CSV),
        "semantics": {
            "primary_metrics": "decode-only TPOT/TPS",
            "tpot": "full decode-stage service time per output token at stated B/worker",
            "throughput_tps": "global_batch / TPOT_seconds",
            "context_tps_panel": "DeepSeek-V3.2 decode TPS normalized vs DGX over context length; not TTFT and not prefill proxy",
            "representative_moe_layer": "layer 23 TP2/TP3 latency multiplied by model MoE-layer count",
            "dense_hbm_stream": "dense decode KV, prefill KV, and prefill attention-weight streaming use TP1 dense_hbm_util=0.47; sparse-gather analytic utility remains 0.21 only for legacy sparse paths",
            "dense_nonuniform_mapping": "HACMO dense decode applies only TP1 static non-uniform mapping hop credit from method S vs B to the KV injection/NoC term; dense HBM service, matrix compute, cache hit-rate, and dynamic prefetch remain unchanged",
            "equal_per_die_substrate": "All rows share the same per-die substrate (68x68 array, 2250 TFLOPS, 4.8 TB/s HBM, 5 TB/s NoC) and differ only in interconnect fabric. Dense decode attention therefore charges the KV injection/NoC term on every row, GPU included; only HACMO reduces it, via the TP1 static-mapping hop credit",
            "gpu_moe_replay_scale": "GPU MoE replay derives its traffic divisor from the staged window_stats so post-scale traffic is exactly batch_per_die tokens/die/step. See gpu_replay_contracts",
            "gpu_moe_step_scope": "GPU MoE replay mirrors the LS vLLM row's replayed decode-step range via exclude_leading_windows so both sides cover the same steps",
            "waferllm": "HACMO-side waferllm_matrix_model_hacmo_reproduction for MeshGEMV/MeshGEMM matrix stages; vLLM MoE placement and communication retained",
        },
        "hbm_streaming": {
            "dense_hbm_effective_fraction": DENSE_HBM_EFFECTIVE_FRACTION,
            "sparse_hbm_effective_fraction": SPARSE_HBM_EFFECTIVE_FRACTION,
            "dense_hbm_source": "TP1 current pack raw/run_meta.json hbm_model.dense_hbm_util",
            "sparse_hbm_source": "supernode_sim sparse analytic/service-curve fallback for sparse/random selected-KV access",
        },
        "dense_nonuniform_mapping": {
            "enabled_for": "baseline_key=hacmo and model.attention=dense and phase=decode",
            "base_method": DENSE_NONUNIFORM_BASE_METHOD,
            "static_mapping_method": DENSE_NONUNIFORM_STATIC_METHOD,
            "scale": "kv_injection_us *= min(1, avg_hbm_hops(static_mapping_method)/avg_hbm_hops(base_method))",
            "not_applied_to": ["kv_hbm_us", "matrix_us", "collective_us", "MoE", "prefill_proxy"],
        },
        "context_tps_panel": {
            "model": MODELS["deepseek_v32"].label,
            "phase": "decode",
            "contexts": CONTEXT_TPS_CONTEXTS,
            "batch_per_worker": 32,
            "normalization": "throughput_tps / vLLM+DGX throughput_tps at the same context",
            "tp1_main_curve": str(TP1_CONTEXT_HBM_CURVE),
            "tp1_supplemental_16k_curve": str(TP1_CONTEXT_HBM_CURVE_16K),
            "mapping_profile": str(TP1_CONTEXT_MAPPING_PROFILE),
            "gamma_json": str(TP1_CONTEXT_GAMMA_JSON),
        },
        "gpu_replay_contracts": {
            f"{model_key}|{baseline_key}|bpd{batch}": contract
            for (model_key, baseline_key, batch), contract in sorted(
                (gpu_replay_contracts or {}).items()
            )
        },
    }
    (out_root / "run_metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def refresh_root_copies(fig_dir: Path) -> None:
    for suffix in ["png", "pdf"]:
        shutil.copy2(fig_dir / f"overall_performance_dense.{suffix}", OVERALL_DIR / f"overall_performance_dense.{suffix}")


if __name__ == "__main__":
    main()
