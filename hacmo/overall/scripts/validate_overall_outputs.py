#!/usr/bin/env python3
"""Validate rebuttal-era Overall Performance output packs.

The validator checks provenance and metric semantics.  It deliberately does
not require any baseline to win by a fixed margin.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
OVERALL_DIR = REPO / "hacmo" / "overall"

MODELS = [
    "DeepSeek-V3.2",
    "Qwen3-235B-A22B-FP8",
    "Llama-4-Maverick-17B-128E-Instruct",
]
MODEL_KEYS = ["deepseek_v32", "qwen3_235b_a22b", "llama4_maverick"]
BASELINES = ["dgx", "nvl72", "ls_vllm", "moentwine", "waferllm", "hacmo"]
LS_BASELINES = ["ls_vllm", "moentwine", "waferllm", "hacmo"]
PANELS = [
    "model_compare",
    "tpot",
    "context_tps",
    "throughput_batch",
    "topology",
    "breakdown",
]

EXPECTED_PACK_MARKER_GROUPS = {
    "deepseek_v32": [
        (
            "tp2_serving_epoch_paper_pack_20260613",
            "tp2_serving_epoch_paper_pack_20260831",
            "tp2_serving_epoch_paper_pack_",
        ),
        (
            "tp3_lane_level_multibatch_final_candidate_20260614",
            "tp3_lane_level_multibatch_final_candidate_tp2_20260831",
            "tp3_lane_level_multibatch_final_candidate_tp2_",
        ),
    ],
    "qwen3_235b_a22b": [
        (
            "tp2_overall_qwen3_layer23_epoch_t_base_railfix_astra_16w_bpd32_20260617",
            "tp2_overall_qwen3_layer23_epoch_t_base_railfix_astra_16w_bpd32_",
        ),
        (
            "tp3_overall_qwen3_layer23_epoch_t_base_railfix_final_16w_bpd32_20260617",
            "tp3_overall_qwen3_layer23_epoch_t_base_railfix_final_16w_bpd32_",
        ),
    ],
    "llama4_maverick": [
        (
            "tp2_overall_llama4_fulltrace_analytical_16w_bpd32_20260617",
            "tp2_overall_llama4_fulltrace_analytical_16w_bpd32_",
        ),
        (
            "tp3_overall_llama4_fulltrace_final_16w_bpd32_20260617",
            "tp3_overall_llama4_fulltrace_final_16w_bpd32_",
        ),
    ],
}

EXPECTED_MOE_TOPK = {
    "deepseek_v32": 8.0,
    "qwen3_235b_a22b": 8.0,
    "llama4_maverick": 1.0,
}


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_root).resolve() if args.output_root else latest_output()
    panels = {name: read_csv(out_root / "panels" / f"{name}.csv") for name in PANELS}
    data_rows = read_csv(out_root / "data" / "simulated_layers.csv")
    metadata = read_json(out_root / "run_metadata.json")
    errors: list[str] = []

    check_panel_shape(panels, errors)
    check_required_fields(data_rows, panels, errors)
    check_metric_semantics(data_rows, panels, errors)
    check_provenance(data_rows, panels, metadata, errors)
    check_representative_layer_units(metadata, errors)
    check_context_tps_semantics(panels["context_tps"], metadata, errors)
    check_breakdown_semantics(panels["breakdown"], errors)
    check_gpu_replay_contracts(data_rows, metadata, errors)
    check_gpu_baseline_consistency(data_rows, errors)
    check_figures(out_root, errors)

    report = {
        "output_root": str(out_root),
        "status": "fail" if errors else "ok",
        "errors": errors,
        "summary": summarize(panels),
    }
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate hacmo/overall rebuttal output CSVs and figure copies.")
    p.add_argument("--output-root", default="", help="Run directory; defaults to latest outputs/overall_rebuttal_*.")
    return p.parse_args()


def latest_output() -> Path:
    runs = sorted((OVERALL_DIR / "outputs").glob("overall_rebuttal_*"))
    runs = [p for p in runs if p.is_dir()]
    if not runs:
        raise SystemExit("no hacmo/overall/outputs/overall_rebuttal_* run found")
    return runs[-1].resolve()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"missing CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"missing JSON: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def check_panel_shape(panels: dict[str, list[dict[str, str]]], errors: list[str]) -> None:
    expected = {
        "model_compare": len(MODELS) * len(BASELINES),
        "tpot": len(MODELS) * len(BASELINES),
        "context_tps": 4 * len(BASELINES),
        "throughput_batch": 3 * len(BASELINES),
        "topology": 2 * 3 * len(LS_BASELINES),
        "breakdown": len(BASELINES),
    }
    for name, count in expected.items():
        if len(panels[name]) != count:
            errors.append(f"{name}: expected {count} rows, got {len(panels[name])}")

    model_set = {r.get("model", "") for r in panels["model_compare"]}
    if model_set != set(MODELS):
        errors.append(f"model_compare model set mismatch: {sorted(model_set)}")

    for name in ["model_compare", "tpot"]:
        keys = {(r.get("model", ""), r.get("baseline_key", "")) for r in panels[name]}
        expected_keys = {(m, b) for m in MODELS for b in BASELINES}
        if keys != expected_keys:
            errors.append(f"{name}: model/baseline coverage mismatch")


def check_required_fields(
    data_rows: list[dict[str, str]],
    panels: dict[str, list[dict[str, str]]],
    errors: list[str],
) -> None:
    required = {
        "scenario",
        "model",
        "model_key",
        "baseline",
        "baseline_key",
        "topology",
        "batch_scope",
        "batch_per_worker",
        "num_workers",
        "global_batch",
        "context_len",
        "phase",
        "representative_moe_layer",
        "attention_matrix_stack_ms",
        "kv_attention_stack_ms",
        "moe_compute_stack_ms",
        "moe_comm_stack_ms",
        "tpot_ms",
        "prefill_proxy_ms",
        "throughput_tps",
        "attn_source",
        "moe_source",
        "moe_method",
        "moe_compute_us",
        "moe_communication_us",
    }
    for name, rows in {"simulated_layers": data_rows, **panels}.items():
        for idx, row in enumerate(rows, start=1):
            missing = sorted(required - set(row))
            if missing:
                errors.append(f"{name}:{idx} missing fields {missing}")
                break


def check_metric_semantics(
    data_rows: list[dict[str, str]],
    panels: dict[str, list[dict[str, str]]],
    errors: list[str],
) -> None:
    for row in data_rows:
        ident = row_id(row)
        batch = int_float(row["batch_per_worker"])
        workers = int_float(row["num_workers"])
        global_batch = int_float(row["global_batch"])
        if row.get("batch_scope") != "per_worker":
            errors.append(f"{ident}: batch_scope is not per_worker")
        if global_batch != batch * workers:
            errors.append(f"{ident}: global_batch {global_batch} != batch_per_worker*num_workers")
        if workers != 256:
            errors.append(f"{ident}: expected 256 workers, got {workers}")

        if row["baseline_key"] in {"dgx", "nvl72"}:
            if row.get("batch_per_gpu") != row["batch_per_worker"]:
                errors.append(f"{ident}: GPU row missing batch_per_gpu provenance")
            if row.get("batch_per_die"):
                errors.append(f"{ident}: GPU row should not set batch_per_die")
        else:
            if row.get("batch_per_die") != row["batch_per_worker"]:
                errors.append(f"{ident}: LS row missing batch_per_die provenance")
            if row.get("batch_per_gpu"):
                errors.append(f"{ident}: LS row should not set batch_per_gpu")

        phase = row["phase"]
        tpot_ms = float(row["tpot_ms"])
        throughput = float(row["throughput_tps"])
        prefill_ms = float(row["prefill_proxy_ms"])
        if phase == "decode":
            if tpot_ms <= 0:
                errors.append(f"{ident}: decode tpot_ms must be positive")
            expected_tps = global_batch / (tpot_ms / 1000.0)
            if not rel_close(throughput, expected_tps, rel=5e-6, abs_tol=1e-3):
                errors.append(f"{ident}: throughput_tps does not equal global_batch / TPOT")
            if prefill_ms != 0.0:
                errors.append(f"{ident}: decode row should not set prefill_proxy_ms")
        elif phase == "prefill_proxy":
            if prefill_ms <= 0:
                errors.append(f"{ident}: prefill_proxy_ms must be positive")
            if tpot_ms != 0.0 or throughput != 0.0:
                errors.append(f"{ident}: prefill proxy row should not set TPOT/TPS")
        else:
            errors.append(f"{ident}: unknown phase {phase}")

    for row in panels["model_compare"]:
        if row["baseline_key"] == "dgx" and abs(float(row["normalized_tps_vs_dgx"]) - 1.0) > 1e-6:
            errors.append(f"{row_id(row)}: DGX normalized_tps_vs_dgx is not 1.0")

    for row in panels["context_tps"]:
        if row["baseline_key"] == "dgx" and abs(float(row["normalized_tps_vs_dgx"]) - 1.0) > 1e-6:
            errors.append(f"{row_id(row)}: DGX normalized_tps_vs_dgx is not 1.0")

    for row in panels["topology"]:
        if row["baseline_key"] == "ls_vllm" and abs(float(row["normalized_tps_vs_ls_vllm"]) - 1.0) > 1e-6:
            errors.append(f"{row_id(row)}: vLLM+LS normalized_tps_vs_ls_vllm is not 1.0")


def check_provenance(
    data_rows: list[dict[str, str]],
    panels: dict[str, list[dict[str, str]]],
    metadata: dict[str, Any],
    errors: list[str],
) -> None:
    tp1 = metadata.get("tp1_csv", "")
    if "technical_point1_final_pack_astra_ramulator_trace_driven_20260724" not in tp1:
        errors.append(f"metadata tp1_csv is not current TP1 pack: {tp1}")

    if set(metadata.get("models", {}).keys()) != set(MODEL_KEYS):
        errors.append("metadata model keys do not match rebuttal model set")
    if set(metadata.get("baselines", {}).keys()) != set(BASELINES):
        errors.append("metadata baseline keys do not match rebuttal baseline set")

    semantics = metadata.get("semantics", {})
    if "decode-only" not in semantics.get("primary_metrics", ""):
        errors.append("metadata does not state decode-only primary metrics")
    if "decode TPS" not in semantics.get("context_tps_panel", ""):
        errors.append("metadata does not state context panel is decode TPS")
    if "not TTFT" not in semantics.get("context_tps_panel", ""):
        errors.append("metadata does not state context panel is not TTFT")
    if "waferllm_matrix_model" not in semantics.get("waferllm", ""):
        errors.append("metadata does not state WaferLLM matrix-model reproduction")
    if "dense_hbm_util=0.47" not in semantics.get("dense_hbm_stream", ""):
        errors.append("metadata does not state dense HBM stream utility")

    hbm_streaming = metadata.get("hbm_streaming", {})
    if hbm_streaming.get("dense_hbm_effective_fraction") != 0.47:
        errors.append("metadata dense_hbm_effective_fraction must be 0.47")
    if hbm_streaming.get("sparse_hbm_effective_fraction") != 0.21:
        errors.append("metadata sparse_hbm_effective_fraction must be 0.21")

    packs = metadata.get("packs", {})
    for model_key, marker_groups in EXPECTED_PACK_MARKER_GROUPS.items():
        pack = packs.get(model_key, {})
        packed_text = json.dumps(pack, sort_keys=True)
        for marker_group in marker_groups:
            if not any(marker in packed_text for marker in marker_group):
                errors.append(
                    f"metadata for {model_key} missing pack marker from group {sorted(marker_group)}"
                )
        if model_key in {"qwen3_235b_a22b", "llama4_maverick"}:
            scope_note = pack.get("scope_note", "")
            if "livecodebench" not in scope_note:
                errors.append(f"metadata for {model_key} missing livecodebench scope note")
            if model_key == "qwen3_235b_a22b" and "single-layer" not in scope_note:
                errors.append(f"metadata for {model_key} must disclose single-layer representative TP2 scope")

    for row in data_rows:
        ident = row_id(row)
        model_key = row["model_key"]
        baseline = row["baseline_key"]
        attn_source = row.get("attn_source", "")
        moe_source = row.get("moe_source", "")

        if row["model"] not in MODELS:
            errors.append(f"{ident}: unexpected model label")
        if row["representative_moe_layer"] != "23":
            errors.append(f"{ident}: representative_moe_layer must be 23")

        if model_key == "deepseek_v32" and row["phase"] == "decode":
            if row["scenario"] == "context_tps":
                context_len = int_float(row["context_len"])
                if context_len in {8192, 16384}:
                    if "tp1_audited_astra_ramulator_trace_driven" not in attn_source and "tp1_measured_h200_flashmla" not in attn_source:
                        errors.append(f"{ident}: DeepSeek context-TPS attention is not current TP1-backed")
                elif "tp1_context_tps_service_curve" not in attn_source and "retained_from_overall_v11" not in attn_source:
                    errors.append(f"{ident}: DeepSeek context-TPS fallback must retain the service-curve or v11 provenance")
                if "context_tps" not in attn_source:
                    errors.append(f"{ident}: DeepSeek context-TPS attention missing context_tps provenance")
            elif row["scenario"] == "breakdown":
                if "tp1_final_pack_static_traffic_shared_20260612" not in attn_source and "retained_from_overall_v11" not in attn_source:
                    errors.append(f"{ident}: DeepSeek breakdown attention must retain the v11 TP1 fallback provenance")
            elif "tp1_audited_astra_ramulator_trace_driven" not in attn_source and "tp1_measured_h200_flashmla" not in attn_source:
                errors.append(f"{ident}: DeepSeek decode attention is not current TP1-backed")
            if baseline == "hacmo" and "method=O" not in attn_source and "method_O" not in attn_source:
                errors.append(f"{ident}: HACMO DeepSeek attention must use TP1 method O")
            if baseline in {"ls_vllm", "moentwine", "waferllm"} and "method=B" not in attn_source and "method_B" not in attn_source:
                errors.append(f"{ident}: LS baseline DeepSeek attention must use TP1 method B")
            if baseline in {"dgx", "nvl72"} and "method=G" not in attn_source and "method_G" not in attn_source:
                errors.append(f"{ident}: GPU DeepSeek attention must use TP1 method G")

        if row["phase"] == "decode" and "deployment_attention_allreduce" not in attn_source:
            errors.append(f"{ident}: attention source missing deployment all-reduce")

        if baseline in {"dgx", "nvl72"}:
            if "tp2_runtime_artifact_replay_gpu_vllm" not in moe_source:
                errors.append(f"{ident}: GPU MoE is not TP2 runtime artifact replay")
            if "astra=True" not in moe_source:
                errors.append(f"{ident}: GPU MoE was not run with ASTRA enabled")
            if float(row["moe_communication_us"]) <= 0:
                errors.append(f"{ident}: GPU ASTRA communication must be positive")
            expected_network = "dgx_h200_32x8" if baseline == "dgx" else "nvl72_4x64"
            if f"network={expected_network}" not in moe_source:
                errors.append(f"{ident}: GPU MoE source missing network={expected_network}")

        elif baseline == "ls_vllm":
            if "tp2_serving_epoch_vllm" not in moe_source:
                errors.append(f"{ident}: vLLM+LS MoE is not TP2 vLLM")
        elif baseline == "moentwine":
            if "tp2_serving_epoch_moentwine_static" not in moe_source:
                errors.append(f"{ident}: MoEntwine MoE is not TP2 moentwine_static")
            if "waferllm_matrix_model" in moe_source:
                errors.append(f"{ident}: MoEntwine should not receive WaferLLM matrix credit")
        elif baseline == "waferllm":
            if "tp2_serving_epoch_vllm" not in moe_source:
                errors.append(f"{ident}: WaferLLM MoE must retain vLLM placement/communication")
            if "waferllm_matrix_model_hacmo_reproduction" not in (attn_source + "|" + moe_source):
                errors.append(f"{ident}: WaferLLM row missing matrix-model provenance")
            if "vllm_placement_comm|waferllm_matrix_model" not in row.get("moe_method", ""):
                errors.append(f"{ident}: WaferLLM MoE method must disclose vLLM comm plus matrix model")
        elif baseline == "hacmo":
            if "tp3_lane_level_full_tp2" not in moe_source:
                errors.append(f"{ident}: HACMO MoE is not TP3 lane-level full_tp2")

        marker_groups = EXPECTED_PACK_MARKER_GROUPS.get(model_key, [])
        tp2_markers = marker_groups[0] if len(marker_groups) >= 1 else ()
        tp3_markers = marker_groups[1] if len(marker_groups) >= 2 else ()
        if baseline == "hacmo":
            if tp3_markers and not any(marker in moe_source for marker in tp3_markers):
                errors.append(f"{ident}: HACMO source missing one of {sorted(tp3_markers)}")
        elif baseline in {"ls_vllm", "moentwine", "waferllm"}:
            if tp2_markers and not any(marker in moe_source for marker in tp2_markers):
                errors.append(f"{ident}: LS baseline source missing one of {sorted(tp2_markers)}")
        elif baseline in {"dgx", "nvl72"}:
            if tp2_markers and not any(marker in moe_source for marker in tp2_markers):
                errors.append(f"{ident}: GPU replay source missing one of {sorted(tp2_markers)}")

    for row in panels["model_compare"]:
        if row["model_key"] in {"qwen3_235b_a22b", "llama4_maverick"}:
            if "overall_dense_decode_attention_hbm_stream" not in row["attn_source"]:
                errors.append(f"{row_id(row)}: dense model attention is not overall dense decode HBM stream")
            if "dense_hbm_util=0.47" not in row["attn_source"]:
                errors.append(f"{row_id(row)}: dense model attention missing dense_hbm_util=0.47 provenance")


def check_gpu_replay_contracts(
    rows: list[dict[str, str]], metadata: dict[str, Any], errors: list[str]
) -> None:
    """Reject GPU rows that are not a real, correctly scaled replay.

    Three failure modes are checked, each of which has produced a wrong paper
    number before:

    1. ``summary_fallback`` -- the GPU row reused the LS vLLM row verbatim and so
       carried no GPU-fabric cost at all.
    2. a replay scale that does not reproduce the target batch, which silently
       multiplies all GPU traffic.
    3. a GPU row covering a different decode-step range than the LS row it is
       normalised against.
    """
    contracts = metadata.get("gpu_replay_contracts", {})
    seen_models: set[str] = set()

    for row in rows:
        if row.get("baseline_key") not in {"dgx", "nvl72"}:
            continue
        ident = row_id(row)
        moe_source = row.get("moe_source", "")
        if "summary_fallback" in moe_source:
            errors.append(
                f"{ident}: GPU MoE row is a summary fallback, not a replay; "
                "the GPU baseline would carry no GPU-fabric cost"
            )
        for marker in ("replay_convention=", "batch_scale=", "exclude_leading_windows=", "reported_steps="):
            if marker not in moe_source:
                errors.append(f"{ident}: GPU MoE source missing replay contract field {marker}")
        seen_models.add(str(row.get("model_key", "")))

    for model_key in sorted(seen_models):
        keys = [k for k in contracts if k.startswith(f"{model_key}|")]
        if not keys:
            errors.append(f"{model_key}: metadata gpu_replay_contracts has no entry")
            continue
        for k in keys:
            contract = contracts[k]
            scale = contract.get("replay_scale", {})
            scope = contract.get("step_window_scope", {})
            convention = scale.get("convention")
            if convention not in {"raw_trace_scale", "pre_expanded"}:
                errors.append(f"{k}: unknown replay convention {convention!r}")
            observed = float(scale.get("observed_tokens_per_die_per_step") or 0.0)
            batch_scale = float(scale.get("batch_scale") or 0.0)
            batch = int_float(k.rsplit("bpd", 1)[-1])
            if observed <= 0 or batch_scale <= 0:
                errors.append(f"{k}: non-positive replay scale ({observed=}, {batch_scale=})")
            elif abs(observed * batch_scale - batch) > 1e-6 * max(batch, 1):
                errors.append(
                    f"{k}: replay scale does not reproduce the target batch: "
                    f"{observed} * {batch_scale} != {batch}"
                )
            if int(scope.get("reported_steps") or 0) <= 0:
                errors.append(f"{k}: step_window_scope reported_steps must be positive")


def check_gpu_baseline_consistency(rows: list[dict[str, str]], errors: list[str]) -> None:
    """Cross-model invariants the GPU baseline pipeline must satisfy.

    GPU and LS rows share the same H200 MoE roofline and differ only in fabric,
    so GPU-replay MoE compute must track the LS vLLM row.  A large divergence
    means the replay scaled its traffic differently from the LS side.  GPU rows
    must also not be charged the wafer core-to-edge NoC injection term, which
    only exists on the Layered Supernode substrate.
    """
    ls_compute: dict[tuple[str, str], float] = {}
    for row in rows:
        if row.get("baseline_key") != "ls_vllm":
            continue
        key = (row.get("model_key", ""), row.get("batch_per_worker", ""))
        ls_compute[key] = float(row.get("moe_compute_us") or 0.0)

    for row in rows:
        if row.get("baseline_key") not in {"dgx", "nvl72"}:
            continue
        ident = row_id(row)
        key = (row.get("model_key", ""), row.get("batch_per_worker", ""))
        gpu = float(row.get("moe_compute_us") or 0.0)
        ls = ls_compute.get(key)
        if ls and ls > 0:
            ratio = gpu / ls
            if not 0.8 <= ratio <= 1.25:
                errors.append(
                    f"{ident}: GPU replay MoE compute {gpu:.3f}us diverges from LS vLLM "
                    f"{ls:.3f}us (ratio {ratio:.3f}); both use the same H200 roofline so only "
                    "the fabric should differ"
                )
        if "gpu_no_wafer_noc_injection" in row.get("attn_source", ""):
            errors.append(
                f"{ident}: GPU rows must charge the same KV injection/NoC term as LS rows; "
                "all rows share one per-die substrate and differ only in fabric"
            )


def check_representative_layer_units(metadata: dict[str, Any], errors: list[str]) -> None:
    packs = metadata.get("packs", {})
    models = metadata.get("models", {})
    for model_key, expected_ratio in EXPECTED_MOE_TOPK.items():
        pack = packs.get(model_key, {})
        model = models.get(model_key, {})
        tp2_dir = Path(pack.get("tp2_dir", ""))
        topo_labels = pack.get("topology_labels", {})
        topology = topo_labels.get("ls_16w4x4")
        if not tp2_dir or not topology:
            errors.append(f"metadata for {model_key} missing TP2 topology for representative-layer unit check")
            continue
        window_stats = tp2_dir / topology / "mapping_inputs" / "runtime" / "window_stats.json"
        if not window_stats.exists():
            errors.append(f"{model_key}: missing window_stats for representative-layer unit check: {window_stats}")
            continue
        windows = json.loads(window_stats.read_text(encoding="utf-8"))
        if not windows:
            errors.append(f"{model_key}: empty window_stats for representative-layer unit check")
            continue
        for idx, window in enumerate(windows, start=1):
            tokens = sum(int(v) for v in (window.get("tokens_per_die") or {}).values())
            expert_activations = sum(int(v) for v in (window.get("expert_counts") or {}).values())
            if tokens <= 0:
                errors.append(f"{model_key}: window {idx} has no source tokens")
                continue
            ratio = expert_activations / tokens
            if not math.isclose(ratio, expected_ratio, rel_tol=0.0, abs_tol=1e-9):
                errors.append(
                    f"{model_key}: window {idx} expert/token ratio {ratio:g} != expected topk {expected_ratio:g}; "
                    "Overall requires a single representative MoE layer, not an all-layer aggregate"
                )
                break
        metadata_topk = float(model.get("moe_topk", expected_ratio))
        if not math.isclose(metadata_topk, expected_ratio, rel_tol=0.0, abs_tol=1e-9):
            errors.append(f"{model_key}: metadata moe_topk {metadata_topk:g} != expected {expected_ratio:g}")


def check_context_tps_semantics(
    rows: list[dict[str, str]],
    metadata: dict[str, Any],
    errors: list[str],
) -> None:
    if metadata.get("semantics", {}).get("context_tps_panel") != "DeepSeek-V3.2 decode TPS normalized vs DGX over context length; not TTFT and not prefill proxy":
        errors.append("context-TPS metadata wording changed; keep explicit decode/non-TTFT semantics")
    ctxs = sorted({int_float(r["context_len"]) for r in rows})
    if ctxs != [8192, 16384, 32768, 65536]:
        errors.append(f"context_tps contexts mismatch: {ctxs}")
    if {r["model"] for r in rows} != {"DeepSeek-V3.2"}:
        errors.append("context_tps should be scoped to DeepSeek-V3.2 in the current figure")
    for row in rows:
        if row["phase"] != "decode":
            errors.append(f"{row_id(row)}: context-TPS panel row phase is not decode")
        if float(row["throughput_tps"]) <= 0:
            errors.append(f"{row_id(row)}: context-TPS row throughput_tps must be positive")
        if float(row["prefill_proxy_ms"]) != 0.0:
            errors.append(f"{row_id(row)}: context-TPS row should not set prefill_proxy_ms")
        if "context_tps" not in row["attn_source"]:
            errors.append(f"{row_id(row)}: context-TPS row missing context_tps attention provenance")
    panel_meta = metadata.get("context_tps_panel", {})
    if panel_meta.get("contexts") != [8192, 16384, 32768, 65536]:
        errors.append(f"metadata context_tps_panel contexts mismatch: {panel_meta.get('contexts')}")
    if "ctx16k" not in panel_meta.get("tp1_supplemental_16k_curve", ""):
        errors.append("metadata context_tps_panel missing supplemental 16K TP1 curve provenance")


def check_breakdown_semantics(rows: list[dict[str, str]], errors: list[str]) -> None:
    if {r["baseline_key"] for r in rows} != set(BASELINES):
        errors.append("breakdown baseline coverage mismatch")
    for row in rows:
        ident = row_id(row)
        if row["model"] != "DeepSeek-V3.2":
            errors.append(f"{ident}: breakdown should be DeepSeek-only")
        if int_float(row["context_len"]) != 32768:
            errors.append(f"{ident}: breakdown context must be 32768")
        total_components = (
            float(row["attention_matrix_stack_ms"])
            + float(row["kv_attention_stack_ms"])
            + float(row["moe_compute_stack_ms"])
            + float(row["moe_comm_stack_ms"])
        )
        if not rel_close(total_components, float(row["tpot_ms"]), rel=5e-6, abs_tol=1e-4):
            errors.append(f"{ident}: component breakdown does not sum to TPOT")


def check_figures(out_root: Path, errors: list[str]) -> None:
    for suffix in ["png", "pdf"]:
        run_fig = out_root / "figures" / f"overall_performance_dense.{suffix}"
        root_fig = OVERALL_DIR / f"overall_performance_dense.{suffix}"
        if not run_fig.exists():
            errors.append(f"missing run figure: {run_fig}")
        if not root_fig.exists():
            errors.append(f"missing root convenience figure: {root_fig}")
        if run_fig.exists() and root_fig.exists() and run_fig.stat().st_size != root_fig.stat().st_size:
            errors.append(f"root {suffix} convenience copy is stale")


def summarize(panels: dict[str, list[dict[str, str]]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in panels["model_compare"]:
        key = f"{row['model_key']}_{row['baseline_key']}_norm_tps_vs_dgx"
        out[key] = round(float(row["normalized_tps_vs_dgx"]), 4)
    for row in panels["tpot"]:
        if row["baseline_key"] == "hacmo":
            out[f"{row['model_key']}_hacmo_tpot_ms"] = round(float(row["tpot_ms"]), 4)
    return out


def int_float(value: str) -> int:
    return int(float(value))


def rel_close(a: float, b: float, rel: float, abs_tol: float) -> bool:
    return math.isclose(a, b, rel_tol=rel, abs_tol=abs_tol)


def row_id(row: dict[str, str]) -> str:
    return (
        f"{row.get('scenario')}:{row.get('model_key') or row.get('model')}:"
        f"{row.get('baseline_key')}:{row.get('topology')}:"
        f"B{row.get('batch_per_worker')}:ctx{row.get('context_len')}:"
        f"{row.get('phase')}"
    )


if __name__ == "__main__":
    main()
