#!/usr/bin/env python3
import argparse
import csv
import json
import math
import pickle
import subprocess
import sys
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from supernode_sim.backends.wafer_core_noc import (  # noqa: E402
    WaferCoreNoCConfig,
    WaferCoreNoCTraffic,
    estimate_wafer_core_noc,
)
from supernode_sim.backends.wafer_core_astra_noc import (  # noqa: E402
    WaferCoreAstraNoCConfig,
    WaferCoreAstraNoCTraffic,
    estimate_wafer_core_astra_noc,
)


def avg_hops_to_top_or_bottom(rows: int, cols: int) -> float:
    total = 0.0
    for r in range(rows):
        total += min(r, rows - 1 - r) * cols
    return total / float(rows * cols)


def avg_remote_manhattan_hops(rows: int, cols: int) -> float:
    n = rows * cols
    if n <= 1:
        return 0.0
    # Expected Manhattan distance between two distinct uniformly sampled grid
    # points. This is equivalent to the previous all-pairs enumeration but
    # avoids O((rows*cols)^2) runtime for 68x68 sweeps.
    exp_row = (rows * rows - 1) / (3.0 * rows)
    exp_col = (cols * cols - 1) / (3.0 * cols)
    return (exp_row + exp_col) * n / (n - 1)


def parse_profile_file(path: Path, decode_steps_limit: int, layer_ids: List[int]) -> List[Dict]:
    data = pickle.loads(path.read_bytes())
    prompt_len = int(data["prompt_len"])
    layer_streams: List[Dict] = []
    for layer_id in layer_ids:
        layer_data = data["profiling_data"][layer_id]
        if "topk_indices" in layer_data:
            arr = np.asarray(layer_data["topk_indices"])
            seq = arr[0]
            prefill = seq[:prompt_len, :]
            decode = seq[prompt_len:, :]
        else:
            prefill_list = layer_data.get("prefill_indices", [])
            decode_arr = np.asarray(layer_data.get("decode_indices", []))
            prefill = np.asarray(prefill_list[0]) if prefill_list else np.empty((0, 2048), dtype=np.int64)
            decode = decode_arr
        if decode_steps_limit > 0:
            decode = decode[:decode_steps_limit, :]
        prefill_counter = Counter(prefill.reshape(-1).tolist()) if prefill.size > 0 else Counter()
        layer_streams.append(
            {
                "prefill_indices": prefill.astype(np.int64, copy=False),
                "prefill_counter": prefill_counter,
                "decode_indices": decode.astype(np.int64, copy=False),
            }
        )
    return layer_streams


def lru_step_simulate(
    prefill_indices: np.ndarray,
    prefill_counter: Counter,
    decode_indices: np.ndarray,
    capacity_entries: int,
    context_len: int,
    warm_start: bool,
) -> Tuple[int, int]:
    if capacity_entries <= 0:
        valid_decode = decode_indices[(decode_indices >= 0) & (decode_indices < context_len)]
        return 0, int(valid_decode.size)
    cache: OrderedDict[int, None] = OrderedDict()
    if warm_start and prefill_indices.size > 0:
        hot = sorted(
            ((k, v) for k, v in prefill_counter.items() if 0 <= k < context_len),
            key=lambda x: x[1],
            reverse=True,
        )[:capacity_entries]
        for token_id, _ in hot:
            cache[token_id] = None
    hits = 0
    total = 0
    for row in decode_indices:
        step_tokens = [int(t) for t in np.asarray(row).tolist() if 0 <= int(t) < context_len]
        if not step_tokens:
            continue
        step_unique = list(dict.fromkeys(step_tokens))
        step_hits = [t for t in step_unique if t in cache]
        step_misses = [t for t in step_unique if t not in cache]
        hits += len(step_hits)
        total += len(step_unique)
        for t in step_hits:
            cache.move_to_end(t)
        for t in step_misses:
            cache[t] = None
            if len(cache) > capacity_entries:
                cache.popitem(last=False)
    return hits, total


def request_cache_entries(
    cache_ratio: float,
    batch_size: int,
    kv_entry_bytes: int,
    total_sram_bytes: int,
    n_layers: int,
    policy: str,
) -> int:
    total_cache_bytes = total_sram_bytes * cache_ratio
    layer_bytes = total_cache_bytes / n_layers if policy == "C" else total_cache_bytes
    entries = int(layer_bytes // max(batch_size * kv_entry_bytes, 1))
    return max(entries, 0)


def profile_hit_rate_cached(
    memo: Dict[Tuple, Dict],
    layer_streams_all: List[List[Dict]],
    batch_size: int,
    cache_ratio: float,
    context_len: int,
    policy: str,
    kv_entry_bytes: int,
    total_sram_bytes: int,
    n_layers: int,
) -> Dict:
    cap = request_cache_entries(cache_ratio, batch_size, kv_entry_bytes, total_sram_bytes, n_layers, policy)
    key = (policy, context_len, cap)
    if key not in memo:
        hits = 0
        total = 0
        warm = policy in {"C", "O"}
        for prompt_layers in layer_streams_all:
            for layer_stream in prompt_layers:
                h, t = lru_step_simulate(
                    prefill_indices=layer_stream["prefill_indices"],
                    prefill_counter=layer_stream["prefill_counter"],
                    decode_indices=layer_stream["decode_indices"],
                    capacity_entries=cap,
                    context_len=context_len,
                    warm_start=warm,
                )
                hits += h
                total += t
        memo[key] = {
            "hit_rate": float(hits) / float(total) if total > 0 else 0.0,
            "capacity_entries_per_request": cap,
        }
    return {
        "policy": policy,
        "cache_ratio": cache_ratio,
        "batch_size": batch_size,
        "context_len": context_len,
        "hit_rate": memo[key]["hit_rate"],
        "capacity_entries_per_request": memo[key]["capacity_entries_per_request"],
    }


def effective_hbm_model(baseline: str, batch_size: int, hit_rate: float, page_util_sparse: float, hbm_startup_ns: float):
    if baseline == "B":
        util_scale, startup_scale = 1.00, 1.00
    elif baseline == "C":
        batch_penalty = 2.3 / (1.0 + 0.35 * math.log2(max(batch_size, 2)))
        util_scale = min(1.20, max(0.60, batch_penalty))
        startup_scale = 0.75 + 0.20 * (1.0 - hit_rate)
    elif baseline == "S":
        util_scale, startup_scale = 1.00, 0.95
    elif baseline == "P":
        # dynamic-prefetch-only variant: keep C-like model, improve hit rate only
        batch_penalty = 2.3 / (1.0 + 0.35 * math.log2(max(batch_size, 2)))
        util_scale = min(1.20, max(0.60, batch_penalty))
        startup_scale = 0.75 + 0.20 * (1.0 - hit_rate)
    elif baseline == "O":
        util_scale, startup_scale = 1.00, 0.90
    else:
        util_scale, startup_scale = 1.00, 1.00
    return page_util_sparse * util_scale, hbm_startup_ns * startup_scale


def load_hbm_service_curve(path: str) -> Dict:
    if not path:
        return {}
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"HBM service curve JSON not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "rows" not in data:
        raise ValueError(f"Invalid HBM service curve JSON: {p}")
    return data


def load_mapping_profile(path: str) -> Dict:
    if not path:
        return {}
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"mapping_profile_json not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def load_hit_rate_overrides(path: str) -> Dict[Tuple[str, int, int], float]:
    if not path:
        return {}
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"hit_rate_override_csv not found: {p}")
    out: Dict[Tuple[str, int, int], float] = {}
    with p.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"policy", "batch_size", "context_len", "hit_rate"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{p} missing columns: {sorted(missing)}")
        for row in reader:
            out[(str(row["policy"]), int(row["batch_size"]), int(row["context_len"]))] = float(row["hit_rate"])
    return out


def get_hit_rate(
    overrides: Dict[Tuple[str, int, int], float],
    memo: Dict[Tuple, Dict],
    layer_streams_all: List[List[Dict]],
    batch_size: int,
    cache_ratio: float,
    context_len: int,
    policy: str,
    kv_entry_bytes: int,
    total_sram_bytes: int,
    n_layers: int,
) -> float:
    key = (policy, int(batch_size), int(context_len))
    if key in overrides:
        return overrides[key]
    return profile_hit_rate_cached(
        memo=memo,
        layer_streams_all=layer_streams_all,
        batch_size=batch_size,
        cache_ratio=cache_ratio,
        context_len=context_len,
        policy=policy,
        kv_entry_bytes=kv_entry_bytes,
        total_sram_bytes=total_sram_bytes,
        n_layers=n_layers,
    )["hit_rate"]


def mapping_for_shape(mapping_profile: Dict, rows: int, cols: int) -> Dict:
    if mapping_profile.get("profiles"):
        key = f"{rows}x{cols}"
        prof = mapping_profile["profiles"].get(key)
        if prof and prof.get("derived"):
            out = dict(prof["derived"])
            if prof.get("row_compute_weights"):
                out["row_compute_weights"] = prof["row_compute_weights"]
            if prof.get("row_cache_weights"):
                out["row_cache_weights"] = prof["row_cache_weights"]
            out["placement_axis"] = prof.get("placement_axis", "")
            return out
    if mapping_profile.get("derived"):
        return mapping_profile["derived"]
    return search_mapping_profile(rows, cols)


def service_curve_effective_gbps(curve: Dict, pattern: str, batch_size: int, context_len: int) -> float:
    rows = curve.get("rows", [])
    if not rows:
        raise ValueError("HBM service curve has no rows.")
    candidates = [r for r in rows if r.get("pattern") == pattern]
    if not candidates:
        candidates = rows

    def score(r: Dict) -> Tuple[int, int]:
        return (abs(int(r.get("batch_size", batch_size)) - batch_size), abs(int(r.get("context_len", context_len)) - context_len))

    best = min(candidates, key=score)
    return float(best["effective_gbps"])


def estimate_indexer_latency(
    batch_size: int,
    context_len: int,
    topk: float,
    hbm_bw_bps: float,
    dense_hbm_util: float,
    hbm_config: Dict,
) -> Dict:
    b = float(batch_size)
    s = float(context_len)
    mode = str(hbm_config.get("indexer_model", "deepseek_v32"))

    if mode == "metadata_scan":
        meta_bytes = float(hbm_config.get("selector_meta_bytes", 16.0))
        bytes_read = b * meta_bytes * s
        t_mem_s = bytes_read / max(hbm_bw_bps * dense_hbm_util, 1.0)
        return {
            "T_select_mem_us": t_mem_s * 1e6,
            "T_indexer_mem_us": t_mem_s * 1e6,
            "T_indexer_compute_us": 0.0,
            "indexer_bytes": bytes_read,
            "indexer_flops": 0.0,
        }

    if mode != "deepseek_v32":
        raise ValueError(f"Unsupported indexer_model: {mode}")

    k_bytes = float(hbm_config.get("indexer_k_bytes_per_token", 128.0))
    scale_bytes = float(hbm_config.get("indexer_scale_bytes_per_token", 4.0))
    extra_bytes = float(hbm_config.get("indexer_extra_bytes_per_token", 0.0))
    index_bytes = float(hbm_config.get("indexer_index_bytes", 4.0))
    bytes_read = b * s * (k_bytes + scale_bytes + extra_bytes)
    bytes_write = b * min(float(topk), s) * index_bytes

    indexer_hbm_util = float(hbm_config.get("indexer_hbm_util", dense_hbm_util))
    t_mem_s = (bytes_read + bytes_write) / max(hbm_bw_bps * indexer_hbm_util, 1.0)

    n_heads = float(hbm_config.get("indexer_n_heads", 64.0))
    head_dim = float(hbm_config.get("indexer_head_dim", 128.0))
    compute_tflops = float(hbm_config.get("indexer_compute_tflops", 2250.0))
    # DeepSeek-V3.2's Indexer scans the full prefix. The dominant arithmetic is
    # the FP8 dot product between each index query head and the cached index key.
    dot_flops = 2.0 * b * s * n_heads * head_dim
    t_compute_s = dot_flops / max(compute_tflops * 1e12, 1.0)

    return {
        "T_select_mem_us": (t_mem_s + t_compute_s) * 1e6,
        "T_indexer_mem_us": t_mem_s * 1e6,
        "T_indexer_compute_us": t_compute_s * 1e6,
        "indexer_bytes": bytes_read + bytes_write,
        "indexer_flops": dot_flops,
    }


def _normalize(xs):
    s = sum(xs)
    if s <= 0:
        return [1.0 / len(xs) for _ in xs]
    return [x / s for x in xs]


def _outerness(rows: int):
    c = (rows - 1) / 2.0
    m = max(c, 1e-9)
    return [abs(r - c) / m for r in range(rows)]


def _remote_hops_from_row_shares(src_share, dst_share, cols: int):
    total = 0.0
    for i, si in enumerate(src_share):
        for j, dj in enumerate(dst_share):
            hops = abs(i - j) + (cols / 2.0)
            total += si * dj * hops
    return total


def _uniform_row_weights(rows: int) -> List[float]:
    return [1.0 / rows for _ in range(rows)]


def _normalized_weights(weights: List[float], rows: int) -> List[float]:
    if len(weights) != rows:
        return _uniform_row_weights(rows)
    cleaned = [max(float(x), 0.0) for x in weights]
    total = sum(cleaned)
    if total <= 0.0:
        return _uniform_row_weights(rows)
    return [x / total for x in cleaned]


def _hbm_miss_row_weights(compute_w: List[float], cache_w: List[float], rows: int) -> List[float]:
    mismatch = [max(c - k, 0.0) for c, k in zip(compute_w, cache_w)]
    if sum(mismatch) <= 1e-12:
        return compute_w
    return _normalized_weights(mismatch, rows)


def _avg_hbm_hops_from_rows(row_weights: List[float], rows: int) -> float:
    return sum(w * min(r, rows - 1 - r) for r, w in enumerate(row_weights))


def _avg_remote_hops_from_rows(src_row_weights: List[float], dst_row_weights: List[float], rows: int, cols: int) -> float:
    row_hops = 0.0
    for src_row, src_share in enumerate(src_row_weights):
        for dst_row, dst_share in enumerate(dst_row_weights):
            row_hops += src_share * dst_share * abs(src_row - dst_row)
    expected_col_hops = (cols * cols - 1) / (3.0 * cols) if cols > 0 else 0.0
    return row_hops + expected_col_hops


def static_partition_traffic_profile(
    baseline: str,
    mapping_derived: Dict,
    rows: int,
    cols: int,
) -> Dict[str, object]:
    if baseline not in {"S", "O"}:
        return {}
    if len(mapping_derived.get("row_compute_weights", [])) != rows or len(mapping_derived.get("row_cache_weights", [])) != rows:
        return {}
    compute_w = _normalized_weights(list(mapping_derived.get("row_compute_weights", [])), rows)
    cache_w = _normalized_weights(list(mapping_derived.get("row_cache_weights", [])), rows)
    hbm_dst_w = _hbm_miss_row_weights(compute_w, cache_w, rows)
    remote_dst_w = compute_w
    remote_src_w = cache_w
    return {
        "hbm_dst_row_weights": tuple(hbm_dst_w),
        "remote_src_row_weights": tuple(remote_src_w),
        "remote_dst_row_weights": tuple(remote_dst_w),
        "avg_hbm_hops": _avg_hbm_hops_from_rows(hbm_dst_w, rows),
        "avg_remote_hops": _avg_remote_hops_from_rows(remote_src_w, remote_dst_w, rows, cols),
    }


def search_mapping_profile(rows: int, cols: int, compute_grad_max=1.2, cache_grad_max=1.6, grid_steps=13):
    def objective(g_compute, g_cache):
        out = _outerness(rows)
        compute_w = _normalize([1.0 + g_compute * o for o in out])
        cache_w = _normalize([1.0 + g_cache * (1.0 - o) for o in out])
        mismatch = [max(c - k, 0.0) for c, k in zip(compute_w, cache_w)]
        mismatch = _normalize(mismatch) if sum(mismatch) > 1e-12 else compute_w
        uni = [1.0 / rows for _ in range(rows)]
        edge_dist = [min(r, rows - 1 - r) for r in range(rows)]
        hbm_hops = sum(m * d for m, d in zip(mismatch, edge_dist))
        hbm_hops_ref = sum(u * d for u, d in zip(uni, edge_dist))
        hbm_scale = hbm_hops / max(hbm_hops_ref, 1e-9)
        remote_hops = _remote_hops_from_row_shares(mismatch, cache_w, cols)
        remote_hops_ref = _remote_hops_from_row_shares(uni, uni, cols)
        remote_scale = remote_hops / max(remote_hops_ref, 1e-9)
        mismatch_ref = sum(max(c - u, 0.0) for c, u in zip(compute_w, uni))
        mismatch_now = sum(max(c - k, 0.0) for c, k in zip(compute_w, cache_w))
        remote_fraction_scale = mismatch_now / max(mismatch_ref, 1e-9)
        pressure = [max(c - k, 0.0) for c, k in zip(compute_w, cache_w)]
        avg_p = sum(pressure) / rows
        tail_ratio = max(pressure) / max(avg_p, 1e-9)
        tail_ref_p = [max(c - u, 0.0) for c, u in zip(compute_w, uni)]
        tail_ref = max(tail_ref_p) / max(sum(tail_ref_p) / rows, 1e-9)
        tail_scale = tail_ratio / max(tail_ref, 1e-9)
        score = 0.45 * hbm_scale + 0.30 * remote_scale + 0.25 * tail_scale
        return {
            "score": score,
            "derived": {
                "avg_hbm_hops_scale": max(0.60, min(1.20, hbm_scale)),
                "avg_remote_hops_scale": max(0.60, min(1.20, remote_scale)),
                "remote_fraction_scale": max(0.50, min(1.20, remote_fraction_scale)),
                "tail_penalty_scale": max(0.70, min(1.20, tail_scale)),
            },
        }

    best = None
    for i in range(grid_steps):
        g_compute = compute_grad_max * i / max(grid_steps - 1, 1)
        for j in range(grid_steps):
            g_cache = cache_grad_max * j / max(grid_steps - 1, 1)
            cand = objective(g_compute, g_cache)
            if (best is None) or (cand["score"] < best["score"]):
                best = cand
    return best["derived"]


def estimate_latency(
    batch_size: int,
    context_len: int,
    baseline: str,
    hit_rate: float,
    rows: int,
    cols: int,
    gamma: float,
    mapping_derived: Dict,
    hbm_config: Dict,
):
    topk = 2048.0
    selected_kv_bytes_per_token = float(hbm_config.get("selected_kv_bytes_per_token", 1152.0))
    sparse_compute_tokens = float(hbm_config.get("sparse_compute_tokens", topk))
    b = float(batch_size)
    s = float(context_len)
    # breakdown base
    if baseline == "B":
        local_frac, near_frac, remote_frac = hit_rate * 0.40, 0.0, hit_rate * 0.60
        hbm_frac = 1.0 - hit_rate
        tail = 1.20
    elif baseline == "C":
        local_frac, near_frac, remote_frac = hit_rate * 0.30, 0.0, hit_rate * 0.70
        hbm_frac = 1.0 - hit_rate
        tail = 1.22
    elif baseline == "S":
        # static mapping only: keep C-style hit-rate behavior, optimize spatial terms
        local_frac, near_frac, remote_frac = hit_rate * 0.30, 0.0, hit_rate * 0.70
        hbm_frac = 1.0 - hit_rate
        tail = 1.22
    elif baseline == "P":
        # dynamic-prefetch only: use O-like hit-rate but C-style mapping
        local_frac, near_frac, remote_frac = hit_rate * 0.30, 0.0, hit_rate * 0.70
        hbm_frac = 1.0 - hit_rate
        tail = 1.22
    elif baseline == "O":
        # Dynamic prefetch stages current-layer hot KV into a dedicated buffer
        # close to the mapped compute tiles. Most hits should be local/near-row,
        # not long-distance remote SRAM traffic across the whole array.
        local_frac, near_frac, remote_frac = hit_rate * 0.75, hit_rate * 0.20, hit_rate * 0.05
        hbm_frac = 1.0 - hit_rate
        tail = 1.03
    else:
        raise ValueError(f"Unsupported baseline: {baseline}")

    avg_hbm = avg_hops_to_top_or_bottom(rows, cols)
    avg_remote = avg_remote_manhattan_hops(rows, cols)
    static_traffic = static_partition_traffic_profile(baseline, mapping_derived, rows, cols)
    if baseline in {"S", "O"}:
        remote_frac = max(0.0, min(hit_rate, remote_frac * mapping_derived["remote_fraction_scale"]))
        local_frac = max(0.0, hit_rate - near_frac - remote_frac)
        if static_traffic:
            avg_hbm = float(static_traffic["avg_hbm_hops"])
            avg_remote = float(static_traffic["avg_remote_hops"])
        else:
            avg_hbm *= mapping_derived["avg_hbm_hops_scale"]
            avg_remote *= mapping_derived["avg_remote_hops_scale"]
        tail *= mapping_derived["tail_penalty_scale"]

    # bytes
    v_qkv = b * selected_kv_bytes_per_token * topk
    bytes_hbm = v_qkv * hbm_frac
    bytes_near = v_qkv * near_frac
    bytes_remote = v_qkv * remote_frac
    bytes_local = v_qkv * local_frac

    # constants
    hbm_bw_bps = float(hbm_config.get("hbm_peak_tbps", 2.4)) * 1e12
    noc_bw_scale_mode = str(hbm_config.get("noc_bw_scale_mode", "cols"))
    noc_model = str(hbm_config.get("noc_model", "legacy"))
    if noc_bw_scale_mode == "cols":
        noc_bw_scale = max(cols / 20.0, 1.0)
    elif noc_bw_scale_mode == "sqrt_area":
        noc_bw_scale = max(math.sqrt((rows * cols) / 400.0), 1.0)
    else:
        noc_bw_scale = 1.0
    noc_bw_bps = 5.0 * 1e12 * noc_bw_scale
    page_bytes = 16384.0
    page_util_sparse = 0.35
    hbm_startup_ns = 200.0
    dense_hbm_util = float(hbm_config.get("dense_hbm_util", 0.36))
    mlp = 8.0
    near_sram_hops_cfg = hbm_config.get("near_sram_hops", None)
    if near_sram_hops_cfg is None:
        near_sram_hops = max(1.0, min(4.0, cols / 16.0))
    else:
        near_sram_hops = float(near_sram_hops_cfg)

    if hbm_config.get("model") == "service_curve":
        effective_gbps = service_curve_effective_gbps(
            curve=hbm_config["service_curve"],
            pattern=str(hbm_config.get("sparse_pattern", "profile")),
            batch_size=batch_size,
            context_len=context_len,
        )
        t_hbm_s = bytes_hbm / max(effective_gbps * 1e9, 1.0)
        util_eff_report = (effective_gbps * 1e9) / max(hbm_bw_bps, 1.0)
    else:
        util_eff, startup_eff_ns = effective_hbm_model(baseline, batch_size, hit_rate, page_util_sparse, hbm_startup_ns)
        n_req = bytes_hbm / max(util_eff * page_bytes, 1.0)
        t_hbm_start_s = (n_req / max(mlp, 1e-9)) * startup_eff_ns * 1e-9
        t_hbm_payload_s = (n_req * page_bytes) / max(hbm_bw_bps, 1.0)
        t_hbm_s = t_hbm_start_s + t_hbm_payload_s
        util_eff_report = (bytes_hbm / (t_hbm_s * hbm_bw_bps)) if t_hbm_s > 0 else 0.0
    if noc_model == "legacy":
        t_noc_hbm_s = ((bytes_hbm * avg_hbm) / max(noc_bw_bps, 1.0)) * gamma
        t_noc_near_s = ((bytes_near * near_sram_hops) / max(noc_bw_bps, 1.0)) * gamma
        t_noc_remote_s = ((bytes_remote * avg_remote) / max(noc_bw_bps, 1.0)) * gamma
        noc_byte_hops = {
            "hbm_byte_hops": bytes_hbm * avg_hbm,
            "near_sram_byte_hops": bytes_near * near_sram_hops,
            "remote_sram_byte_hops": bytes_remote * avg_remote,
            "effective_noc_bw_tbps": noc_bw_bps / 1e12,
        }
    elif noc_model == "shared":
        noc = estimate_wafer_core_noc(
            WaferCoreNoCConfig(
                rows=rows,
                cols=cols,
                base_noc_bw_tbps=5.0,
                gamma=gamma,
                noc_bw_scale_mode=noc_bw_scale_mode,
                hbm_edges=tuple(hbm_config.get("hbm_edges", ("top", "bottom"))),
                near_sram_hops=near_sram_hops,
            ),
            WaferCoreNoCTraffic(
                hbm_bytes=bytes_hbm,
                near_sram_bytes=bytes_near,
                remote_sram_bytes=bytes_remote,
                avg_hbm_hops=avg_hbm,
                near_sram_hops=near_sram_hops,
                avg_remote_hops=avg_remote,
            ),
        )
        t_noc_hbm_s = noc.hbm_us * 1e-6
        t_noc_near_s = noc.near_sram_us * 1e-6
        t_noc_remote_s = noc.remote_sram_us * 1e-6
        noc_bw_scale = noc.noc_bw_scale
        noc_byte_hops = {
            "hbm_byte_hops": noc.hbm_byte_hops,
            "near_sram_byte_hops": noc.near_sram_byte_hops,
            "remote_sram_byte_hops": noc.remote_sram_byte_hops,
            "effective_noc_bw_tbps": noc.effective_noc_bw_tbps,
        }
    elif noc_model == "astra":
        astra_root = str(hbm_config.get("astra_output_root", ""))
        astra_sim_dir = str(hbm_config.get("astra_sim_dir", ""))
        astra_system_config = str(hbm_config.get("astra_system_config", ""))
        astra_remote_memory_config = str(hbm_config.get("astra_remote_memory_config", ""))
        if not astra_root or not astra_sim_dir or not astra_system_config or not astra_remote_memory_config:
            raise ValueError(
                "noc_model='astra' requires astra_output_root, astra_sim_dir, "
                "astra_system_config, and astra_remote_memory_config"
            )
        noc = estimate_wafer_core_astra_noc(
            WaferCoreAstraNoCConfig(
                rows=rows,
                cols=cols,
                astra_sim_dir=Path(astra_sim_dir).resolve(),
                output_root=Path(astra_root).resolve(),
                system_config=Path(astra_system_config).resolve(),
                remote_memory_config=Path(astra_remote_memory_config).resolve(),
                astra_network_backend=str(hbm_config.get("astra_network_backend", "congestion_aware")),
                run_astra=bool(hbm_config.get("astra_run", True)),
                base_noc_bw_tbps=5.0,
                noc_bw_scale_mode=noc_bw_scale_mode,
                hop_latency_ns=float(hbm_config.get("astra_hop_latency_ns", 2.0)),
                gamma=gamma,
                hbm_edges=tuple(hbm_config.get("hbm_edges", ("top", "bottom"))),
                near_sram_hops=near_sram_hops,
                max_flows_per_component=int(hbm_config.get("astra_max_flows_per_component", 256)),
            ),
            WaferCoreAstraNoCTraffic(
                hbm_bytes=bytes_hbm,
                near_sram_bytes=bytes_near,
                remote_sram_bytes=bytes_remote,
                avg_hbm_hops=avg_hbm,
                near_sram_hops=near_sram_hops,
                avg_remote_hops=avg_remote,
                hbm_dst_row_weights=static_traffic.get("hbm_dst_row_weights") if static_traffic else None,
                remote_src_row_weights=static_traffic.get("remote_src_row_weights") if static_traffic else None,
                remote_dst_row_weights=static_traffic.get("remote_dst_row_weights") if static_traffic else None,
            ),
        )
        t_noc_hbm_s = noc.hbm_us * 1e-6
        t_noc_near_s = noc.near_sram_us * 1e-6
        t_noc_remote_s = noc.remote_sram_us * 1e-6
        noc_byte_hops = {
            "hbm_byte_hops": bytes_hbm * avg_hbm,
            "near_sram_byte_hops": bytes_near * near_sram_hops,
            "remote_sram_byte_hops": bytes_remote * avg_remote,
            "effective_noc_bw_tbps": noc.flow_summary["effective_noc_bw_tbps"],
        }
    else:
        raise ValueError(f"Unsupported noc_model: {noc_model}")
    t_qkv_pre_tail_s = t_hbm_s + t_noc_hbm_s + t_noc_near_s + t_noc_remote_s
    t_qkv_mem_s = t_qkv_pre_tail_s * tail
    indexer_lat = estimate_indexer_latency(
        batch_size=batch_size,
        context_len=context_len,
        topk=topk,
        hbm_bw_bps=hbm_bw_bps,
        dense_hbm_util=dense_hbm_util,
        hbm_config=hbm_config,
    )

    # compute model
    h = 7168.0
    nh_select = 64.0
    dh_select = 64.0
    nh = 128.0
    dh = 576.0
    flops_proj = 3.0 * 2.0 * b * h * h
    flops_select = 2.0 * b * nh_select * dh_select
    # Sparse attention computes QK/SV on selected KV tokens. DeepSeek-V3.2 DSA
    # Indexer full-prefix scan is accounted for by T_select_mem_us.
    flops_qk = 2.0 * 2.0 * b * nh * dh * min(s, sparse_compute_tokens)
    flops_mix = 2.0 * b * nh * dh * h
    t_comp_s = (flops_proj + flops_select + flops_qk + flops_mix) / max(2250.0 * 1e12, 1.0)

    return {
        "T_comp_us": t_comp_s * 1e6,
        "T_select_mem_us": indexer_lat["T_select_mem_us"],
        "T_indexer_mem_us": indexer_lat["T_indexer_mem_us"],
        "T_indexer_compute_us": indexer_lat["T_indexer_compute_us"],
        "T_hbm_service_us": t_hbm_s * 1e6,
        "T_noc_hbm_us": t_noc_hbm_s * 1e6,
        "T_noc_near_us": t_noc_near_s * 1e6,
        "T_noc_remote_us": t_noc_remote_s * 1e6,
        "T_qkv_pre_tail_us": t_qkv_pre_tail_s * 1e6,
        "T_qkv_mem_us": t_qkv_mem_s * 1e6,
        "T_attn_total_us": t_comp_s * 1e6 + indexer_lat["T_select_mem_us"] + t_qkv_mem_s * 1e6,
        "effective_hbm_bw_util": util_eff_report,
        "qkv_tail_factor": tail,
        "avg_hbm_hops": avg_hbm,
        "near_sram_hops": near_sram_hops,
        "avg_remote_hops": avg_remote,
        "hbm_fraction": hbm_frac,
        "local_sram_fraction": local_frac,
        "near_sram_fraction": near_frac,
        "remote_sram_fraction": remote_frac,
        "noc_model": noc_model,
        "noc_bw_scale": noc_bw_scale,
        "effective_noc_bw_tbps": noc_byte_hops["effective_noc_bw_tbps"],
        "hbm_byte_hops": noc_byte_hops["hbm_byte_hops"],
        "near_sram_byte_hops": noc_byte_hops["near_sram_byte_hops"],
        "remote_sram_byte_hops": noc_byte_hops["remote_sram_byte_hops"],
        "indexer_bytes": indexer_lat["indexer_bytes"],
        "indexer_flops": indexer_lat["indexer_flops"],
        "selected_kv_bytes_per_token": selected_kv_bytes_per_token,
        "bytes_from_hbm": bytes_hbm,
        "bytes_from_sram_local": bytes_local,
        "bytes_from_sram_near": bytes_near,
        "bytes_from_sram_far_remote": bytes_remote,
        "bytes_from_sram_remote": bytes_remote,
    }


def write_csv(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def git_revision(path: Path) -> Dict:
    try:
        head = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
        dirty = subprocess.run(
            ["git", "-C", str(path), "status", "--short"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"path": str(path), "head": "", "dirty": None}
    return {
        "path": str(path),
        "head": head.stdout.strip() if head.returncode == 0 else "",
        "dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
    }


def get_gpu_vllm_profile(profile: str) -> Dict[str, float]:
    """Preset knobs for GPU+vLLM baseline aligned to baseline-B modeling."""
    presets = {
        # optimistic GPU assumptions
        "conservative": {
            "gpu_mem_penalty_vs_B": 1.03,
            "gpu_cp_bw_tbps": 1.2,
        },
        # balanced assumptions
        "neutral": {
            "gpu_mem_penalty_vs_B": 1.08,
            "gpu_cp_bw_tbps": 0.9,
        },
        # pessimistic GPU assumptions
        "aggressive": {
            "gpu_mem_penalty_vs_B": 1.15,
            "gpu_cp_bw_tbps": 0.9,
        },
    }
    if profile not in presets:
        raise ValueError(f"Unsupported gpu_vllm_profile: {profile}")
    return dict(presets[profile])


def main():
    p = argparse.ArgumentParser(description="Generate final motivation + ablation pack for TP1.")
    p.add_argument("--profile_dirs", nargs="+", required=True)
    p.add_argument("--output_root", required=True)
    p.add_argument("--gamma_json", default="")
    p.add_argument("--decode_steps", type=int, default=50)
    p.add_argument("--layer_sample_stride", type=int, default=6)
    p.add_argument("--max_profile_files", type=int, default=3)
    p.add_argument("--total_sram_mb", type=float, default=440.0)
    p.add_argument("--kv_entry_bytes", type=int, default=1152)
    p.add_argument("--selected_kv_bytes_per_token", type=float, default=None)
    p.add_argument("--n_layers", type=int, default=61)
    p.add_argument("--ablation_cache_ratio", type=float, default=0.20)
    p.add_argument("--gpu_vllm_profile", choices=["conservative", "neutral", "aggressive"], default="neutral")
    p.add_argument("--gpu_mem_penalty_vs_B", type=float, default=None)
    p.add_argument("--gpu_cp_bw_tbps", type=float, default=None)
    p.add_argument("--hbm_model", choices=["legacy", "service_curve"], default="legacy")
    p.add_argument("--hbm_service_curve_json", default="")
    p.add_argument("--hbm_peak_tbps", type=float, default=2.4)
    p.add_argument("--dense_hbm_util", type=float, default=0.36)
    p.add_argument("--sparse_pattern", default="profile")
    p.add_argument("--near_sram_hops", type=float, default=0.0)
    p.add_argument("--sparse_compute_tokens", type=float, default=2048.0)
    p.add_argument("--indexer_model", choices=["metadata_scan", "deepseek_v32"], default="deepseek_v32")
    p.add_argument("--selector_meta_bytes", type=float, default=16.0, help="Legacy metadata_scan mode only.")
    p.add_argument("--indexer_k_bytes_per_token", type=float, default=128.0)
    p.add_argument("--indexer_scale_bytes_per_token", type=float, default=4.0)
    p.add_argument("--indexer_extra_bytes_per_token", type=float, default=0.0)
    p.add_argument("--indexer_index_bytes", type=float, default=4.0)
    p.add_argument("--indexer_hbm_util", type=float, default=None)
    p.add_argument("--indexer_n_heads", type=float, default=64.0)
    p.add_argument("--indexer_head_dim", type=float, default=128.0)
    p.add_argument("--indexer_compute_tflops", type=float, default=2250.0)
    p.add_argument("--noc_model", choices=["legacy", "shared", "astra"], default="legacy")
    p.add_argument("--noc_bw_scale_mode", choices=["fixed", "cols", "sqrt_area"], default="cols")
    p.add_argument("--astra_sim_dir", default="astra-sim")
    p.add_argument("--astra_system_config", default="hacmo/attention/astrasim_inner_die/configs/system/inner_die_attention_system.json")
    p.add_argument("--astra_remote_memory_config", default="hacmo/attention/astrasim_inner_die/configs/remote_memory/no_memory_expansion.json")
    p.add_argument("--astra_network_backend", choices=["congestion_aware", "congestion_unaware"], default="congestion_aware")
    p.add_argument("--astra_hop_latency_ns", type=float, default=2.0)
    p.add_argument("--astra_max_flows_per_component", type=int, default=256)
    p.add_argument("--astra_no_run", action="store_true", help="Generate ASTRA traces and use analytical fallback without running ASTRA.")
    p.add_argument("--astra_output_root", default="")
    p.add_argument("--mapping_profile_json", default="")
    p.add_argument("--hit_rate_override_csv", default="")
    args = p.parse_args()

    output_root = Path(args.output_root).resolve()
    (output_root / "csv").mkdir(parents=True, exist_ok=True)
    (output_root / "figs").mkdir(parents=True, exist_ok=True)
    (output_root / "raw").mkdir(parents=True, exist_ok=True)

    profile_files: List[Path] = []
    for d in args.profile_dirs:
        profile_files.extend(sorted(Path(d).glob("*_profile.pkl")))
    profile_files = [p for p in profile_files if "gutenberg_0_profile.pkl" not in p.name][: args.max_profile_files]
    if not profile_files:
        raise FileNotFoundError("No profile files found.")
    layer_ids = list(range(0, args.n_layers, max(args.layer_sample_stride, 1)))
    layer_streams_all = [parse_profile_file(p, decode_steps_limit=args.decode_steps, layer_ids=layer_ids) for p in profile_files]
    total_sram_bytes = int(args.total_sram_mb * 1024 * 1024)
    memo: Dict[Tuple, Dict] = {}

    gamma = 1.0
    if args.gamma_json:
        gcfg = json.loads(Path(args.gamma_json).resolve().read_text(encoding="utf-8"))
        gamma = float(gcfg.get("gamma_fit", gcfg.get("gamma", 1.0)))
    mapping_profile = load_mapping_profile(args.mapping_profile_json) if args.mapping_profile_json else {}
    hit_rate_overrides = load_hit_rate_overrides(args.hit_rate_override_csv)
    hbm_service_curve = load_hbm_service_curve(args.hbm_service_curve_json) if args.hbm_service_curve_json else {}
    hbm_config = {
        "model": args.hbm_model,
        "service_curve": hbm_service_curve,
        "hbm_peak_tbps": args.hbm_peak_tbps,
        "dense_hbm_util": args.dense_hbm_util,
        "sparse_pattern": args.sparse_pattern,
        "sparse_compute_tokens": args.sparse_compute_tokens,
        "selected_kv_bytes_per_token": args.selected_kv_bytes_per_token
        if args.selected_kv_bytes_per_token is not None
        else float(args.kv_entry_bytes),
        "indexer_model": args.indexer_model,
        "selector_meta_bytes": args.selector_meta_bytes,
        "indexer_k_bytes_per_token": args.indexer_k_bytes_per_token,
        "indexer_scale_bytes_per_token": args.indexer_scale_bytes_per_token,
        "indexer_extra_bytes_per_token": args.indexer_extra_bytes_per_token,
        "indexer_index_bytes": args.indexer_index_bytes,
        "indexer_n_heads": args.indexer_n_heads,
        "indexer_head_dim": args.indexer_head_dim,
        "indexer_compute_tflops": args.indexer_compute_tflops,
        "noc_model": args.noc_model,
        "noc_bw_scale_mode": args.noc_bw_scale_mode,
        "astra_sim_dir": str(Path(args.astra_sim_dir).resolve()),
        "astra_system_config": str(Path(args.astra_system_config).resolve()),
        "astra_remote_memory_config": str(Path(args.astra_remote_memory_config).resolve()),
        "astra_network_backend": args.astra_network_backend,
        "astra_hop_latency_ns": args.astra_hop_latency_ns,
        "astra_max_flows_per_component": args.astra_max_flows_per_component,
        "astra_run": not args.astra_no_run,
        "astra_output_root": str(
            Path(args.astra_output_root).resolve()
            if args.astra_output_root
            else (output_root / "raw" / "astra_wafer_core_noc")
        ),
    }
    if args.indexer_hbm_util is not None:
        hbm_config["indexer_hbm_util"] = args.indexer_hbm_util
    if args.near_sram_hops > 0:
        hbm_config["near_sram_hops"] = args.near_sram_hops
    if args.hbm_model == "service_curve" and not hbm_service_curve:
        raise ValueError("--hbm_model service_curve requires --hbm_service_curve_json")

    # ==== Motivation ====
    array_shapes = [(20, 20), (32, 32), (68, 68)]
    batch_sizes_mot = [8, 16, 32, 64, 128]
    cache_ratios = [0.05, 0.10, 0.20]
    context_len_mot = 32768
    motivation_rows = []
    for rows, cols in array_shapes:
        mapping_derived = mapping_for_shape(mapping_profile, rows, cols)  # for completeness; motivation uses B/C only
        for bsz in batch_sizes_mot:
            # B: no cache hit
            lat_b = estimate_latency(
                batch_size=bsz,
                context_len=context_len_mot,
                baseline="B",
                hit_rate=0.0,
                rows=rows,
                cols=cols,
                gamma=gamma,
                mapping_derived=mapping_derived,
                hbm_config=hbm_config,
            )
            motivation_rows.append(
                {
                    "array_rows": rows,
                    "array_cols": cols,
                    "baseline": "B",
                    "cache_ratio": 0.0,
                    "batch_size": bsz,
                    "hit_rate": 0.0,
                    "effective_hbm_bw_util": lat_b["effective_hbm_bw_util"],
                    "T_select_mem_us": lat_b["T_select_mem_us"],
                    "T_indexer_mem_us": lat_b["T_indexer_mem_us"],
                    "T_indexer_compute_us": lat_b["T_indexer_compute_us"],
                    "T_qkv_mem_us": lat_b["T_qkv_mem_us"],
                }
            )
            for ratio in cache_ratios:
                hr_c = get_hit_rate(
                    overrides=hit_rate_overrides,
                    memo=memo,
                    layer_streams_all=layer_streams_all,
                    batch_size=bsz,
                    cache_ratio=ratio,
                    context_len=context_len_mot,
                    policy="C",
                    kv_entry_bytes=args.kv_entry_bytes,
                    total_sram_bytes=total_sram_bytes,
                    n_layers=args.n_layers,
                )
                lat_c = estimate_latency(
                    batch_size=bsz,
                    context_len=context_len_mot,
                    baseline="C",
                    hit_rate=hr_c,
                    rows=rows,
                    cols=cols,
                    gamma=gamma,
                    mapping_derived=mapping_derived,
                    hbm_config=hbm_config,
                )
                motivation_rows.append(
                    {
                        "array_rows": rows,
                        "array_cols": cols,
                        "baseline": "C",
                        "cache_ratio": ratio,
                        "batch_size": bsz,
                        "hit_rate": hr_c,
                        "effective_hbm_bw_util": lat_c["effective_hbm_bw_util"],
                        "T_select_mem_us": lat_c["T_select_mem_us"],
                        "T_indexer_mem_us": lat_c["T_indexer_mem_us"],
                        "T_indexer_compute_us": lat_c["T_indexer_compute_us"],
                        "T_qkv_mem_us": lat_c["T_qkv_mem_us"],
                    }
                )

    write_csv(output_root / "csv" / "motivation_tp1.csv", motivation_rows)

    # figure-1: one figure with 3 subplots (one per array)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    for ax, (rows, cols) in zip(axes, array_shapes):
        sub = [r for r in motivation_rows if r["array_rows"] == rows and r["array_cols"] == cols]
        pts_b = sorted([r for r in sub if r["baseline"] == "B"], key=lambda x: x["batch_size"])
        ax.plot([x["batch_size"] for x in pts_b], [x["effective_hbm_bw_util"] for x in pts_b], marker="o", linewidth=2, label="vLLM-style (B)")
        for ratio in cache_ratios:
            pts_c = sorted([r for r in sub if r["baseline"] == "C" and abs(r["cache_ratio"] - ratio) < 1e-12], key=lambda x: x["batch_size"])
            ax.plot([x["batch_size"] for x in pts_c], [x["effective_hbm_bw_util"] for x in pts_c], marker="o", linewidth=2, label=f"Naive buffer C ({int(ratio*100)}%)")
        ax.axhline(args.dense_hbm_util, linestyle="--", color="gray", linewidth=1.2, label=f"Dense ref {args.dense_hbm_util:.0%}")
        ax.set_title(f"Array {rows}x{cols}")
        ax.set_xlabel("Batch size")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Effective HBM BW utilization")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("Motivation: B vs C under different SRAM ratios and array sizes")
    fig.tight_layout()
    fig.savefig(output_root / "figs" / "motivation_bw_util_tp1.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # figure-2: same plot with log-y for visual separation from dense ref.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    for ax, (rows, cols) in zip(axes, array_shapes):
        sub = [r for r in motivation_rows if r["array_rows"] == rows and r["array_cols"] == cols]
        pts_b = sorted([r for r in sub if r["baseline"] == "B"], key=lambda x: x["batch_size"])
        ax.plot([x["batch_size"] for x in pts_b], [x["effective_hbm_bw_util"] for x in pts_b], marker="o", linewidth=2, label="vLLM-style (B)")
        for ratio in cache_ratios:
            pts_c = sorted([r for r in sub if r["baseline"] == "C" and abs(r["cache_ratio"] - ratio) < 1e-12], key=lambda x: x["batch_size"])
            ax.plot([x["batch_size"] for x in pts_c], [x["effective_hbm_bw_util"] for x in pts_c], marker="o", linewidth=2, label=f"Naive buffer C ({int(ratio*100)}%)")
        ax.axhline(args.dense_hbm_util, linestyle="--", color="gray", linewidth=1.2, label=f"Dense ref {args.dense_hbm_util:.0%}")
        ax.set_yscale("log")
        ax.set_title(f"Array {rows}x{cols}")
        ax.set_xlabel("Batch size")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Effective HBM BW utilization (log scale)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("Motivation (log-y): B vs C")
    fig.tight_layout()
    fig.savefig(output_root / "figs" / "motivation_bw_util_tp1_logy.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # figure-3: context-length sweep (clearer separation)
    context_sweep = [4096, 8192, 16384, 32768, 65536]
    batch_for_sweep = [8, 32, 128]
    rows, cols = 20, 20
    mapping_derived_sweep = mapping_for_shape(mapping_profile, rows, cols)
    context_rows = []
    for bsz in batch_for_sweep:
        for ctx in context_sweep:
            lat_b = estimate_latency(
                batch_size=bsz,
                context_len=ctx,
                baseline="B",
                hit_rate=0.0,
                rows=rows,
                cols=cols,
                gamma=gamma,
                mapping_derived=mapping_derived_sweep,
                hbm_config=hbm_config,
            )
            hr_c20 = get_hit_rate(
                overrides=hit_rate_overrides,
                memo=memo,
                layer_streams_all=layer_streams_all,
                batch_size=bsz,
                cache_ratio=0.20,
                context_len=ctx,
                policy="C",
                kv_entry_bytes=args.kv_entry_bytes,
                total_sram_bytes=total_sram_bytes,
                n_layers=args.n_layers,
            )
            lat_c20 = estimate_latency(
                batch_size=bsz,
                context_len=ctx,
                baseline="C",
                hit_rate=hr_c20,
                rows=rows,
                cols=cols,
                gamma=gamma,
                mapping_derived=mapping_derived_sweep,
                hbm_config=hbm_config,
            )
            context_rows.append(
                {
                    "array_rows": rows,
                    "array_cols": cols,
                    "batch_size": bsz,
                    "context_len": ctx,
                    "B_bw_util": lat_b["effective_hbm_bw_util"],
                    "C20_bw_util": lat_c20["effective_hbm_bw_util"],
                    "C20_hit_rate": hr_c20,
                }
            )
    write_csv(output_root / "csv" / "motivation_context_sweep_tp1.csv", context_rows)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for bsz in batch_for_sweep:
        pts = sorted([r for r in context_rows if r["batch_size"] == bsz], key=lambda x: x["context_len"])
        ax.plot([x["context_len"] for x in pts], [x["B_bw_util"] for x in pts], marker="o", linewidth=1.8, linestyle="--", label=f"B, bs={bsz}")
        ax.plot([x["context_len"] for x in pts], [x["C20_bw_util"] for x in pts], marker="o", linewidth=2.2, label=f"C(20%), bs={bsz}")
    ax.axhline(args.dense_hbm_util, linestyle=":", color="gray", linewidth=1.2, label=f"Dense ref {args.dense_hbm_util:.0%}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Context length (log2)")
    ax.set_ylabel("Effective HBM BW utilization (log)")
    ax.set_title("Motivation context sweep (array 20x20)")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_root / "figs" / "motivation_context_sweep_tp1.png", dpi=220)
    plt.close(fig)

    # ==== Ablation ====
    # Grouped x-axis as requested:
    # 8,16,32,64,64+CP2,128+CP4
    batch_groups_ab = [
        {"label": "8", "global_batch": 8, "cp_degree": 1},
        {"label": "16", "global_batch": 16, "cp_degree": 1},
        {"label": "32", "global_batch": 32, "cp_degree": 1},
        {"label": "64", "global_batch": 64, "cp_degree": 1},
        {"label": "64+CP2", "global_batch": 64, "cp_degree": 2},
        {"label": "128+CP4", "global_batch": 128, "cp_degree": 4},
    ]
    contexts_ab = [8192, 32768]
    arrays_ab = [(20, 20), (68, 68)]
    ratio_ab = args.ablation_cache_ratio
    # GPU+vLLM proxy assumptions with 3-level preset + optional overrides
    gpu_cfg = get_gpu_vllm_profile(args.gpu_vllm_profile)
    for k in [
        "gpu_mem_penalty_vs_B",
        "gpu_cp_bw_tbps",
    ]:
        v = getattr(args, k)
        if v is not None:
            gpu_cfg[k] = float(v)

    gpu_mem_penalty_vs_b = float(gpu_cfg["gpu_mem_penalty_vs_B"])
    gpu_cp_bw_tbps = float(gpu_cfg["gpu_cp_bw_tbps"])
    ablation_rows = []
    for ctx in contexts_ab:
        for rows, cols in arrays_ab:
            mapping_derived = mapping_for_shape(mapping_profile, rows, cols)
            for g in batch_groups_ab:
                gbsz = int(g["global_batch"])
                cp = int(g["cp_degree"])
                per_bsz = max(1, gbsz // cp)
                hr_c = get_hit_rate(
                    overrides=hit_rate_overrides,
                    memo=memo,
                    layer_streams_all=layer_streams_all,
                    batch_size=per_bsz,
                    cache_ratio=ratio_ab,
                    context_len=ctx,
                    policy="C",
                    kv_entry_bytes=args.kv_entry_bytes,
                    total_sram_bytes=total_sram_bytes,
                    n_layers=args.n_layers,
                )
                hr_o = get_hit_rate(
                    overrides=hit_rate_overrides,
                    memo=memo,
                    layer_streams_all=layer_streams_all,
                    batch_size=per_bsz,
                    cache_ratio=ratio_ab,
                    context_len=ctx,
                    policy="O",
                    kv_entry_bytes=args.kv_entry_bytes,
                    total_sram_bytes=total_sram_bytes,
                    n_layers=args.n_layers,
                )
                cp_hops = 1.0 if cp == 2 else (1.5 if cp == 4 else 0.0)
                cp_bytes = 2.0 * per_bsz * 576.0 * 2048.0 * (cp - 1) / cp if cp > 1 else 0.0
                cp_comm_us_supernode = (cp_bytes * cp_hops / (5.0 * 1e12)) * 1e6 if cp > 1 else 0.0

                # Supernode methods
                for bl, hr in [("B", 0.0), ("C", hr_c), ("S", hr_c), ("O", hr_o)]:
                    lat = estimate_latency(
                        batch_size=per_bsz,
                        context_len=ctx,
                        baseline=bl,
                        hit_rate=hr,
                        rows=rows,
                        cols=cols,
                        gamma=gamma,
                        mapping_derived=mapping_derived,
                        hbm_config=hbm_config,
                    )
                    ablation_rows.append(
                        {
                            "context_len": ctx,
                            "array_rows": rows,
                            "array_cols": cols,
                            "batch_label": g["label"],
                            "global_batch": gbsz,
                            "cp_degree": cp,
                            "per_npu_batch": per_bsz,
                            "method": bl,
                            "hit_rate": hr,
                            "T_comp_us": lat["T_comp_us"],
                            "T_select_mem_us": lat["T_select_mem_us"],
                            "T_indexer_mem_us": lat["T_indexer_mem_us"],
                            "T_indexer_compute_us": lat["T_indexer_compute_us"],
                            "T_hbm_service_us": lat["T_hbm_service_us"],
                            "T_noc_hbm_us": lat["T_noc_hbm_us"],
                            "T_noc_near_us": lat["T_noc_near_us"],
                            "T_noc_remote_us": lat["T_noc_remote_us"],
                            "T_qkv_pre_tail_us": lat["T_qkv_pre_tail_us"],
                            "T_qkv_mem_us": lat["T_qkv_mem_us"],
                            "T_cp_comm_us": cp_comm_us_supernode,
                            "T_attn_total_us": lat["T_attn_total_us"] + cp_comm_us_supernode,
                            "attn_only_token_latency_ms_61L": (lat["T_attn_total_us"] + cp_comm_us_supernode) * args.n_layers / 1000.0,
                            "attn_only_throughput_tok_s_61L": float(gbsz)
                            / max((lat["T_attn_total_us"] + cp_comm_us_supernode) * args.n_layers * 1e-6, 1e-12),
                            "qkv_tail_factor": lat["qkv_tail_factor"],
                            "avg_hbm_hops": lat["avg_hbm_hops"],
                            "near_sram_hops": lat["near_sram_hops"],
                            "avg_remote_hops": lat["avg_remote_hops"],
                            "hbm_fraction": lat["hbm_fraction"],
                            "local_sram_fraction": lat["local_sram_fraction"],
                            "near_sram_fraction": lat["near_sram_fraction"],
                            "remote_sram_fraction": lat["remote_sram_fraction"],
                            "noc_model": lat["noc_model"],
                            "noc_bw_scale": lat["noc_bw_scale"],
                            "effective_noc_bw_tbps": lat["effective_noc_bw_tbps"],
                            "hbm_byte_hops": lat["hbm_byte_hops"],
                            "near_sram_byte_hops": lat["near_sram_byte_hops"],
                            "remote_sram_byte_hops": lat["remote_sram_byte_hops"],
                            "indexer_bytes": lat["indexer_bytes"],
                            "indexer_flops": lat["indexer_flops"],
                            "selected_kv_bytes_per_token": lat["selected_kv_bytes_per_token"],
                            "bytes_from_hbm": lat["bytes_from_hbm"],
                            "bytes_from_sram_local": lat["bytes_from_sram_local"],
                            "bytes_from_sram_near": lat["bytes_from_sram_near"],
                            "bytes_from_sram_far_remote": lat["bytes_from_sram_far_remote"],
                            "bytes_from_sram_remote": lat["bytes_from_sram_remote"],
                            "gamma": gamma,
                        }
                    )
                # GPU+vLLM baseline aligned with baseline-B memory model:
                # same 16KB-page/startup model, with a small device-side penalty.
                lat_gpu = estimate_latency(
                    batch_size=per_bsz,
                    context_len=ctx,
                    baseline="B",
                    hit_rate=0.0,
                    rows=rows,
                    cols=cols,
                    gamma=gamma,
                    mapping_derived=mapping_derived,
                    hbm_config=hbm_config,
                )
                lat_gpu_comp = lat_gpu["T_comp_us"]
                t_gpu_indexer_mem_us = lat_gpu["T_indexer_mem_us"] * gpu_mem_penalty_vs_b
                t_gpu_indexer_compute_us = lat_gpu["T_indexer_compute_us"]
                t_sel_us = t_gpu_indexer_mem_us + t_gpu_indexer_compute_us
                t_qkv_us = lat_gpu["T_qkv_mem_us"] * gpu_mem_penalty_vs_b

                # CP communication for GPU:
                # keep same communication volume/hops as supernode CP setting, but GPU link bandwidth is lower.
                cp_comm_us_gpu = (cp_bytes * cp_hops / (gpu_cp_bw_tbps * 1e12)) * 1e6 if cp > 1 else 0.0
                ablation_rows.append(
                    {
                        "context_len": ctx,
                        "array_rows": rows,
                        "array_cols": cols,
                        "batch_label": g["label"],
                        "global_batch": gbsz,
                        "cp_degree": cp,
                        "per_npu_batch": per_bsz,
                        "method": "G",
                        "hit_rate": 0.0,
                        "T_comp_us": lat_gpu_comp,
                        "T_select_mem_us": t_sel_us,
                        "T_indexer_mem_us": t_gpu_indexer_mem_us,
                        "T_indexer_compute_us": t_gpu_indexer_compute_us,
                        "T_hbm_service_us": lat_gpu["T_hbm_service_us"] * gpu_mem_penalty_vs_b,
                        "T_noc_hbm_us": lat_gpu["T_noc_hbm_us"] * gpu_mem_penalty_vs_b,
                        "T_noc_near_us": lat_gpu["T_noc_near_us"] * gpu_mem_penalty_vs_b,
                        "T_noc_remote_us": lat_gpu["T_noc_remote_us"] * gpu_mem_penalty_vs_b,
                        "T_qkv_pre_tail_us": lat_gpu["T_qkv_pre_tail_us"] * gpu_mem_penalty_vs_b,
                        "T_qkv_mem_us": t_qkv_us,
                        "T_cp_comm_us": cp_comm_us_gpu,
                        "T_attn_total_us": lat_gpu_comp + t_sel_us + t_qkv_us + cp_comm_us_gpu,
                        "attn_only_token_latency_ms_61L": (lat_gpu_comp + t_sel_us + t_qkv_us + cp_comm_us_gpu) * args.n_layers / 1000.0,
                        "attn_only_throughput_tok_s_61L": float(gbsz)
                        / max((lat_gpu_comp + t_sel_us + t_qkv_us + cp_comm_us_gpu) * args.n_layers * 1e-6, 1e-12),
                        "qkv_tail_factor": lat_gpu["qkv_tail_factor"],
                        "avg_hbm_hops": lat_gpu["avg_hbm_hops"],
                        "near_sram_hops": lat_gpu["near_sram_hops"],
                        "avg_remote_hops": lat_gpu["avg_remote_hops"],
                        "hbm_fraction": lat_gpu["hbm_fraction"],
                        "local_sram_fraction": 0.0,
                        "near_sram_fraction": 0.0,
                        "remote_sram_fraction": 0.0,
                        "noc_model": lat_gpu["noc_model"],
                        "noc_bw_scale": lat_gpu["noc_bw_scale"],
                        "effective_noc_bw_tbps": lat_gpu["effective_noc_bw_tbps"],
                        "hbm_byte_hops": lat_gpu["hbm_byte_hops"],
                        "near_sram_byte_hops": lat_gpu["near_sram_byte_hops"],
                        "remote_sram_byte_hops": lat_gpu["remote_sram_byte_hops"],
                        "indexer_bytes": lat_gpu["indexer_bytes"],
                        "indexer_flops": lat_gpu["indexer_flops"],
                        "selected_kv_bytes_per_token": lat_gpu["selected_kv_bytes_per_token"],
                        "bytes_from_hbm": lat_gpu["bytes_from_hbm"],
                        "bytes_from_sram_local": 0.0,
                        "bytes_from_sram_near": 0.0,
                        "bytes_from_sram_far_remote": 0.0,
                        "bytes_from_sram_remote": 0.0,
                        "gamma": gamma,
                    }
                )
    write_csv(output_root / "csv" / "ablation1_tp1.csv", ablation_rows)
    diagnostic_rows = []
    for r in ablation_rows:
        if r["method"] != "O":
            continue
        peers = [
            x
            for x in ablation_rows
            if x["context_len"] == r["context_len"]
            and x["array_rows"] == r["array_rows"]
            and x["array_cols"] == r["array_cols"]
            and x["batch_label"] == r["batch_label"]
            and x["method"] == "S"
        ]
        if not peers:
            continue
        s = peers[0]
        if r["T_attn_total_us"] > s["T_attn_total_us"]:
            diagnostic_rows.append(
                {
                    "context_len": r["context_len"],
                    "array_rows": r["array_rows"],
                    "array_cols": r["array_cols"],
                    "batch_label": r["batch_label"],
                    "S_total_us": s["T_attn_total_us"],
                    "O_total_us": r["T_attn_total_us"],
                    "O_over_S": r["T_attn_total_us"] / max(s["T_attn_total_us"], 1e-12),
                    "S_hit_rate": s["hit_rate"],
                    "O_hit_rate": r["hit_rate"],
                    "S_far_remote_bytes": s.get("bytes_from_sram_far_remote", s["bytes_from_sram_remote"]),
                    "O_far_remote_bytes": r.get("bytes_from_sram_far_remote", r["bytes_from_sram_remote"]),
                    "O_near_sram_bytes": r.get("bytes_from_sram_near", 0.0),
                }
            )
    if diagnostic_rows:
        write_csv(output_root / "csv" / "diagnostic_o_worse_than_s.csv", diagnostic_rows)

    label_map = {
        "G": "GPU+vLLM",
        "B": "vLLM",
        "C": "Naive Buffer",
        "S": "Static only",
        "O": "Static+Prefetch (O)",
    }
    color_map = {"G": "#595959", "B": "#7f7f7f", "C": "#a0a0a0", "S": "#4C78A8", "O": "#F58518"}
    for ctx in contexts_ab:
        for rows, cols in arrays_ab:
            sub = [r for r in ablation_rows if r["context_len"] == ctx and r["array_rows"] == rows and r["array_cols"] == cols]
            x = np.arange(len(batch_groups_ab))
            width = 0.15
            fig, ax = plt.subplots(figsize=(8.5, 4.8))
            methods = ["G", "B", "C", "S", "O"]
            for i, bl in enumerate(methods):
                ys = []
                for g in batch_groups_ab:
                    r = [t for t in sub if t["batch_label"] == g["label"] and t["method"] == bl][0]
                    ys.append(r["T_attn_total_us"])
                ax.bar(x + (i - 2.0) * width, ys, width=width, color=color_map[bl], label=label_map[bl])
            ax.set_xticks(x, [g["label"] for g in batch_groups_ab])
            ax.set_xlabel("Batch size")
            ax.set_ylabel("Attention single-run latency (us)")
            ax.set_title(f"Ablation-1 (ctx={ctx//1024}k, array={rows}x{cols})")
            ax.grid(alpha=0.3, axis="y")
            ax.legend()
            fig.tight_layout()
            fig.savefig(output_root / "figs" / f"ablation_ctx{ctx}_array{rows}x{cols}.png", dpi=200)
            plt.close(fig)

    # keep a compact CP-vs-GPU summary table from ablation rows (ctx=32k, array=20x20)
    cp_rows = []
    sub_cp = [r for r in ablation_rows if r["context_len"] == 32768 and r["array_rows"] == 20 and r["array_cols"] == 20]
    for label in ["64+CP2", "128+CP4"]:
        g = [r for r in sub_cp if r["batch_label"] == label and r["method"] == "G"][0]
        o = [r for r in sub_cp if r["batch_label"] == label and r["method"] == "O"][0]
        cp_rows.append(
            {
                "case": label,
                "global_batch": g["global_batch"],
                "cp_degree": g["cp_degree"],
                "per_npu_batch": g["per_npu_batch"],
                "ours_total_with_cp_us": o["T_attn_total_us"],
                "gpu_vllm_proxy_us": g["T_attn_total_us"],
                "ours_vs_gpu_speedup": g["T_attn_total_us"] / max(o["T_attn_total_us"], 1e-9),
            }
        )
    write_csv(output_root / "csv" / "cp_gpu_bridge_tp1.csv", cp_rows)
    # also keep single bridge figure for quick presentation
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = np.arange(len(cp_rows))
    w = 0.36
    ax.bar(x - w / 2, [r["gpu_vllm_proxy_us"] for r in cp_rows], width=w, label="GPU+vLLM")
    ax.bar(x + w / 2, [r["ours_total_with_cp_us"] for r in cp_rows], width=w, label="Ours")
    ax.set_xticks(x, [r["case"] for r in cp_rows])
    ax.set_ylabel("Attention latency (us)")
    ax.set_title("CP points: GPU+vLLM vs Ours")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_root / "figs" / "cp_gpu_bridge_tp1.png", dpi=220)
    plt.close(fig)

    meta = {
        "git": {
            "hacmo": git_revision(Path(__file__).resolve().parents[3]),
            "astra_sim": git_revision(
                Path(__file__).resolve().parents[3] / "astra-sim"
            ),
            "vllm": git_revision(Path(__file__).resolve().parents[3] / "vllm"),
        },
        "profile_files": [str(p) for p in profile_files],
        "layer_ids": layer_ids,
        "gamma": gamma,
        "motivation": {
            "batch_sizes": batch_sizes_mot,
            "cache_ratios_C": cache_ratios,
            "array_sizes": array_shapes,
            "dense_ref_bw_util": args.dense_hbm_util,
        },
        "ablation": {
            "batch_sizes": [8, 16, 32, 64],
            "context_lens": contexts_ab,
            "array_sizes": arrays_ab,
            "x_groups": [g["label"] for g in batch_groups_ab],
            "methods": ["G", "B", "C", "S", "O"],
            "P_definition": "Dynamic prefetch only; O-style hit-rate, C-style mapping terms.",
            "S_definition": "Static mapping only; C-style hit-rate with mapping optimization terms.",
            "O_definition": "Static mapping + dynamic prefetch.",
        },
        "cp_bridge_assumptions": {
            "gpu_vllm_profile": args.gpu_vllm_profile,
            "gpu_mem_penalty_vs_B": gpu_mem_penalty_vs_b,
            "supernode_cp_comm_bw_tbps": 5.0,
            "supernode_cp_comm_hops": {"cp2": 1.0, "cp4": 1.5},
            "gpu_cp_comm_bw_tbps": gpu_cp_bw_tbps,
            "gpu_cp_comm_hops": {"cp2": 1.0, "cp4": 1.5},
            "gpu_cp_collective_multiplier": 1.0,
        },
        "hbm_model": {
            "model": args.hbm_model,
            "hbm_service_curve_json": args.hbm_service_curve_json,
            "hbm_peak_tbps": args.hbm_peak_tbps,
            "dense_hbm_util": args.dense_hbm_util,
            "sparse_pattern": args.sparse_pattern,
            "near_sram_hops": args.near_sram_hops if args.near_sram_hops > 0 else None,
            "near_sram_hops_mode": "explicit" if args.near_sram_hops > 0 else "auto_cols_clamped",
            "near_sram_hops_auto_formula": "max(1.0, min(4.0, array_cols / 16.0))",
            "sparse_compute_tokens": args.sparse_compute_tokens,
            "kv_entry_bytes": args.kv_entry_bytes,
            "selected_kv_bytes_per_token": args.selected_kv_bytes_per_token
            if args.selected_kv_bytes_per_token is not None
            else float(args.kv_entry_bytes),
            "indexer_model": args.indexer_model,
            "selector_meta_bytes": args.selector_meta_bytes,
            "indexer_k_bytes_per_token": args.indexer_k_bytes_per_token,
            "indexer_scale_bytes_per_token": args.indexer_scale_bytes_per_token,
            "indexer_extra_bytes_per_token": args.indexer_extra_bytes_per_token,
            "indexer_index_bytes": args.indexer_index_bytes,
            "indexer_hbm_util": args.indexer_hbm_util if args.indexer_hbm_util is not None else args.dense_hbm_util,
            "indexer_n_heads": args.indexer_n_heads,
            "indexer_head_dim": args.indexer_head_dim,
            "indexer_compute_tflops": args.indexer_compute_tflops,
            "noc_model": args.noc_model,
            "noc_bw_scale_mode": args.noc_bw_scale_mode,
            "astra_sim_dir": str(Path(args.astra_sim_dir).resolve()),
            "astra_system_config": str(Path(args.astra_system_config).resolve()),
            "astra_remote_memory_config": str(Path(args.astra_remote_memory_config).resolve()),
            "astra_network_backend": args.astra_network_backend,
            "astra_hop_latency_ns": args.astra_hop_latency_ns,
            "astra_max_flows_per_component": args.astra_max_flows_per_component,
            "astra_run": not args.astra_no_run,
            "astra_output_root": str(
                Path(args.astra_output_root).resolve()
                if args.astra_output_root
                else (output_root / "raw" / "astra_wafer_core_noc")
            ),
        },
        "ablation_cache_ratio": args.ablation_cache_ratio,
        "hit_rate_override_csv": args.hit_rate_override_csv,
        "hit_rate_overrides": [
            {"policy": k[0], "batch_size": k[1], "context_len": k[2], "hit_rate": v}
            for k, v in sorted(hit_rate_overrides.items())
        ],
        "policy_semantics": {
            "C": "Naive all-layer on-chip cache. total_sram_mb * cache_ratio is divided across n_layers.",
            "S": "Static mapping only. Uses C hit-rate/capacity semantics plus spatial mapping scales.",
            "O": "Static mapping + layer-wise dynamic prefetch. Current layer can use the full dynamic buffer; hits are split into local/near/far SRAM traffic.",
        },
        "mapping_profile_json": args.mapping_profile_json,
    }
    (output_root / "raw" / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Generated final TP1 pack at: {output_root}")


if __name__ == "__main__":
    main()
