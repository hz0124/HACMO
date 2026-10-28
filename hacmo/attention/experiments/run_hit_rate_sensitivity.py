#!/usr/bin/env python3
"""Profile-derived hit-rate sensitivity for TP1.

This script is intentionally separate from the TP1 final-pack generator. It
audits how much the cache hit-rate depends on profile provenance and on the
assumption used when a shorter trace is extrapolated to a longer context.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np


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
    return max(int(layer_bytes // max(batch_size * kv_entry_bytes, 1)), 0)


def layer_data_at(profiling_data: dict[Any, Any], layer_id: int) -> dict[str, Any]:
    if layer_id in profiling_data:
        return profiling_data[layer_id]
    return profiling_data[str(layer_id)]


def observed_trace_len(prompt_len: int, layer_data: dict[str, Any]) -> int:
    max_token = prompt_len - 1
    if "topk_indices" in layer_data:
        arr = np.asarray(layer_data["topk_indices"])
        if arr.size:
            max_token = max(max_token, int(np.max(arr)))
    else:
        if layer_data.get("prefill_indices"):
            arr = np.asarray(layer_data["prefill_indices"][0])
            if arr.size:
                max_token = max(max_token, int(np.max(arr)))
        dec = np.asarray(layer_data.get("decode_indices", []))
        if dec.size:
            max_token = max(max_token, int(np.max(dec)))
    return max_token + 1


def split_prefill_decode(
    prompt_len: int,
    layer_data: dict[str, Any],
    decode_steps_limit: int,
) -> tuple[np.ndarray, np.ndarray]:
    if "topk_indices" in layer_data:
        arr = np.asarray(layer_data["topk_indices"])
        seq = arr[0]
        prefill = seq[:prompt_len, :]
        decode = seq[prompt_len:, :]
    else:
        prefill_list = layer_data.get("prefill_indices", [])
        prefill = np.asarray(prefill_list[0]) if prefill_list else np.empty((0, 2048), dtype=np.int64)
        decode = np.asarray(layer_data.get("decode_indices", []))
    if decode_steps_limit > 0:
        decode = decode[:decode_steps_limit, :]
    return prefill, decode


def prefill_counts(prefill: np.ndarray, source_len: int) -> np.ndarray:
    if prefill.size == 0 or source_len <= 0:
        return np.zeros(max(source_len, 0), dtype=np.int64)
    flat = np.asarray(prefill).reshape(-1)
    if flat.size == 0:
        return np.zeros(source_len, dtype=np.int64)
    if int(np.min(flat)) >= 0 and int(np.max(flat)) < source_len:
        return np.bincount(flat.astype(np.int64, copy=False), minlength=source_len)
    valid = flat[(flat >= 0) & (flat < source_len)]
    return np.bincount(valid.astype(np.int64, copy=False), minlength=source_len)


def projected_counts(counts: np.ndarray, target_context: int, projection: str) -> np.ndarray:
    source_len = int(counts.size)
    if target_context <= 0:
        return np.zeros(0, dtype=np.int64)
    if projection == "raw":
        if source_len >= target_context:
            return counts[:target_context].astype(np.int64, copy=False)
        out = np.zeros(target_context, dtype=np.int64)
        out[:source_len] = counts
        return out
    token_ids = np.arange(source_len, dtype=np.int64)
    dest = scaled_tokens(token_ids, source_len, target_context)
    return np.bincount(dest, weights=counts, minlength=target_context).astype(np.int64, copy=False)


def hot_cache_from_counts(counts: np.ndarray, cap: int) -> OrderedDict[int, None]:
    cache: OrderedDict[int, None] = OrderedDict()
    if cap <= 0 or counts.size == 0:
        return cache
    nonzero = np.flatnonzero(counts > 0)
    if nonzero.size == 0:
        return cache
    if cap >= nonzero.size:
        hot = nonzero[np.argsort(counts[nonzero], kind="stable")[::-1]]
    else:
        candidates = nonzero[np.argpartition(counts[nonzero], -cap)[-cap:]]
        hot = candidates[np.argsort(counts[candidates], kind="stable")[::-1]]
    for token_id in hot[:cap].tolist():
        cache[int(token_id)] = None
    return cache


def decode_unique_rows(decode: np.ndarray, source_len: int) -> list[list[int]]:
    rows: list[list[int]] = []
    for row in decode:
        vals = np.asarray(row)
        vals = vals[(vals >= 0) & (vals < source_len)]
        rows.append(list(dict.fromkeys(int(x) for x in vals.tolist())))
    return rows


def scaled_tokens(tokens: np.ndarray, source_len: int, target_context: int) -> np.ndarray:
    if source_len <= 0 or source_len == target_context:
        return tokens
    scaled = np.floor(tokens.astype(np.float64) * (float(target_context) / float(source_len))).astype(np.int64)
    return np.clip(scaled, 0, max(target_context - 1, 0))


def hot_cache_from_prefill(
    prefill: np.ndarray,
    source_len: int,
    target_context: int,
    cap: int,
    mode: str,
) -> OrderedDict[int, None]:
    cache: OrderedDict[int, None] = OrderedDict()
    if cap <= 0 or prefill.size == 0:
        return cache

    valid = prefill[(prefill >= 0) & (prefill < source_len)]
    if valid.size == 0:
        return cache
    if mode != "raw":
        valid = scaled_tokens(valid, source_len, target_context)
    counts = np.bincount(valid, minlength=target_context)
    if cap >= counts.size:
        hot = np.flatnonzero(counts > 0)
        hot = hot[np.argsort(counts[hot], kind="stable")[::-1]]
    else:
        candidates = np.argpartition(counts, -cap)[-cap:]
        candidates = candidates[counts[candidates] > 0]
        hot = candidates[np.argsort(counts[candidates], kind="stable")[::-1]]
    for token_id in hot[:cap].tolist():
        cache[int(token_id)] = None
    return cache


def build_layer_summary(path: Path, profile_idx: int, layer_ids: list[int], decode_steps: int) -> list[dict[str, Any]]:
    print(f"[load] {path}", flush=True)
    with path.open("rb") as f:
        data = pickle.load(f)
    prompt_len = int(data["prompt_len"])
    profiling_data = data["profiling_data"]
    summaries: list[dict[str, Any]] = []
    for layer_id in layer_ids:
        layer_data = layer_data_at(profiling_data, layer_id)
        source_len = observed_trace_len(prompt_len, layer_data)
        prefill, decode = split_prefill_decode(prompt_len, layer_data, decode_steps)
        summaries.append(
            {
                "profile_idx": profile_idx,
                "layer_id": layer_id,
                "prompt_len": prompt_len,
                "source_len": source_len,
                "prefill_counts": prefill_counts(prefill, source_len),
                "decode_rows": decode_unique_rows(decode, source_len),
            }
        )
    del data
    return summaries


def deterministic_tail_tokens(
    profile_idx: int,
    layer_id: int,
    step: int,
    needed: int,
    source_len: int,
    target_context: int,
) -> list[int]:
    if needed <= 0 or target_context <= source_len:
        return []
    tail_span = target_context - source_len
    base = (profile_idx * 1_000_003 + layer_id * 10_007 + step * 257) % tail_span
    return [source_len + ((base + i * 131) % tail_span) for i in range(needed)]


def projected_step_tokens(
    row: np.ndarray | list[int],
    source_len: int,
    target_context: int,
    mode: str,
    tail_beta: float,
    profile_idx: int,
    layer_id: int,
    step: int,
) -> list[int]:
    if isinstance(row, list):
        unique = row
    else:
        vals = np.asarray(row)
        vals = vals[(vals >= 0) & (vals < source_len)]
        if vals.size == 0:
            return []
        unique = list(dict.fromkeys(int(x) for x in vals.tolist()))
    if not unique:
        return []
    if mode == "raw":
        unique = [x for x in unique if x < target_context]
        return unique

    scaled = scaled_tokens(np.asarray(unique, dtype=np.int64), source_len, target_context)
    scaled_unique = list(dict.fromkeys(int(x) for x in scaled.tolist()))
    if mode == "scale":
        return scaled_unique

    if mode != "tail":
        raise ValueError(f"Unsupported mode: {mode}")

    if target_context <= source_len:
        return scaled_unique
    observed_mass = min(1.0, max(0.0, (float(source_len) / float(target_context)) ** tail_beta))
    keep = int(round(len(scaled_unique) * observed_mass))
    keep = max(0, min(len(scaled_unique), keep))
    kept = scaled_unique[:keep]
    tail = deterministic_tail_tokens(
        profile_idx=profile_idx,
        layer_id=layer_id,
        step=step,
        needed=len(scaled_unique) - keep,
        source_len=source_len,
        target_context=target_context,
    )
    return list(dict.fromkeys(kept + tail))


def simulate_rows(cache: OrderedDict[int, None], decode_rows: list[list[int]], token_projector) -> tuple[int, int]:
    hits = 0
    total = 0
    for step, row in enumerate(decode_rows):
        step_tokens = token_projector(step, row)
        if not step_tokens:
            continue
        step_hits = [t for t in step_tokens if t in cache]
        step_misses = [t for t in step_tokens if t not in cache]
        hits += len(step_hits)
        total += len(step_tokens)
        for token_id in step_hits:
            cache.move_to_end(token_id)
        for token_id in step_misses:
            cache[token_id] = None
            if len(cache) > token_projector.cap:
                cache.popitem(last=False)
    return hits, total


def profile_hit_rate(
    profile_files: list[Path],
    layer_ids: list[int],
    decode_steps: int,
    batch_size: int,
    policy: str,
    cache_ratio: float,
    target_context: int,
    projection: str,
    tail_beta: float,
    kv_entry_bytes: int,
    total_sram_bytes: int,
    n_layers: int,
) -> dict[str, Any]:
    cap = request_cache_entries(cache_ratio, batch_size, kv_entry_bytes, total_sram_bytes, n_layers, policy)
    warm = policy in {"C", "O"}
    hits = 0
    total = 0
    source_lens: list[int] = []
    prompt_lens: list[int] = []

    for profile_idx, path in enumerate(profile_files):
        with path.open("rb") as f:
            data = pickle.load(f)
        prompt_len = int(data["prompt_len"])
        prompt_lens.append(prompt_len)
        profiling_data = data["profiling_data"]

        for layer_id in layer_ids:
            layer_data = layer_data_at(profiling_data, layer_id)
            source_len = observed_trace_len(prompt_len, layer_data)
            source_lens.append(source_len)
            prefill, decode = split_prefill_decode(prompt_len, layer_data, decode_steps)
            cache = OrderedDict()
            if warm:
                cache = hot_cache_from_prefill(
                    prefill=prefill,
                    source_len=source_len,
                    target_context=target_context,
                    cap=cap,
                    mode="raw" if projection == "raw" else "projected",
                )

            def projector(step: int, row: np.ndarray) -> list[int]:
                return projected_step_tokens(
                    row=row,
                    source_len=source_len,
                    target_context=target_context,
                    mode=projection,
                    tail_beta=tail_beta,
                    profile_idx=profile_idx,
                    layer_id=layer_id,
                    step=step,
                )

            projector.cap = cap  # type: ignore[attr-defined]
            h, t = simulate_rows(cache, decode, projector)
            hits += h
            total += t

        del data

    return {
        "policy": policy,
        "batch_size": batch_size,
        "cache_ratio": cache_ratio,
        "target_context": target_context,
        "projection": projection,
        "tail_beta": tail_beta,
        "capacity_entries_per_request": cap,
        "hits": hits,
        "total_accesses": total,
        "hit_rate": float(hits) / float(total) if total else 0.0,
        "profile_count": len(profile_files),
        "layer_count": len(layer_ids),
        "prompt_len_min": min(prompt_lens) if prompt_lens else 0,
        "prompt_len_max": max(prompt_lens) if prompt_lens else 0,
        "source_len_min": min(source_lens) if source_lens else 0,
        "source_len_max": max(source_lens) if source_lens else 0,
    }


def profile_files_from_args(profile_dirs: list[str], profile_files: list[str], max_profile_files: int) -> list[Path]:
    files = [Path(p) for p in profile_files]
    for raw in profile_dirs:
        files.extend(sorted(Path(raw).glob("*_profile.pkl")))
    if max_profile_files > 0:
        files = files[:max_profile_files]
    if not files:
        raise FileNotFoundError("No profile files selected.")
    return files


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TP1 hit-rate locality extrapolation sensitivity.")
    parser.add_argument("--profile_dirs", nargs="*", default=[])
    parser.add_argument("--profile_files", nargs="*", default=[])
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--decode_steps", type=int, default=50)
    parser.add_argument("--layer_sample_stride", type=int, default=6)
    parser.add_argument("--n_layers", type=int, default=61)
    parser.add_argument("--max_profile_files", type=int, default=0)
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[8, 16, 32, 64])
    parser.add_argument("--policies", nargs="+", choices=["C", "O"], default=["C", "O"])
    parser.add_argument("--target_contexts", nargs="+", type=int, default=[8192, 32768])
    parser.add_argument("--projections", nargs="+", choices=["raw", "scale", "tail"], default=["raw", "scale", "tail"])
    parser.add_argument("--tail_betas", nargs="+", type=float, default=[0.5, 1.0])
    parser.add_argument("--total_sram_mb", type=float, default=440.0)
    parser.add_argument("--kv_entry_bytes", type=int, default=656)
    parser.add_argument("--cache_ratio", type=float, default=0.40)
    args = parser.parse_args()

    files = profile_files_from_args(args.profile_dirs, args.profile_files, args.max_profile_files)
    layer_ids = list(range(0, args.n_layers, max(args.layer_sample_stride, 1)))
    total_sram_bytes = int(args.total_sram_mb * 1024 * 1024)

    scenarios: list[dict[str, Any]] = []
    for target_context in args.target_contexts:
        for batch_size in args.batch_sizes:
            for policy in args.policies:
                for projection in args.projections:
                    betas = args.tail_betas if projection == "tail" else [0.0]
                    for beta in betas:
                        cap = request_cache_entries(
                            args.cache_ratio,
                            batch_size,
                            args.kv_entry_bytes,
                            total_sram_bytes,
                            args.n_layers,
                            policy,
                        )
                        scenarios.append(
                            {
                                "policy": policy,
                                "batch_size": batch_size,
                                "cache_ratio": args.cache_ratio,
                                "target_context": target_context,
                                "projection": projection,
                                "tail_beta": beta,
                                "capacity_entries_per_request": cap,
                                "hits": 0,
                                "total_accesses": 0,
                                "_prompt_lens": [],
                                "_source_lens": [],
                            }
                        )

    for profile_idx, path in enumerate(files):
        for summary in build_layer_summary(path, profile_idx, layer_ids, args.decode_steps):
            source_len = int(summary["source_len"])
            prompt_len = int(summary["prompt_len"])
            layer_id = int(summary["layer_id"])
            counts_cache: dict[tuple[str, int], np.ndarray] = {}
            for scenario in scenarios:
                projection = str(scenario["projection"])
                target_context = int(scenario["target_context"])
                cap = int(scenario["capacity_entries_per_request"])
                key = ("raw" if projection == "raw" else "projected", target_context)
                if key not in counts_cache:
                    counts_cache[key] = projected_counts(
                        summary["prefill_counts"],
                        target_context=target_context,
                        projection="raw" if projection == "raw" else "scale",
                    )
                cache = hot_cache_from_counts(counts_cache[key], cap)

                def projector(step: int, row: list[int]) -> list[int]:
                    return projected_step_tokens(
                        row=row,
                        source_len=source_len,
                        target_context=target_context,
                        mode=projection,
                        tail_beta=float(scenario["tail_beta"]),
                        profile_idx=profile_idx,
                        layer_id=layer_id,
                        step=step,
                    )

                projector.cap = cap  # type: ignore[attr-defined]
                h, t = simulate_rows(cache, summary["decode_rows"], projector)
                scenario["hits"] = int(scenario["hits"]) + h
                scenario["total_accesses"] = int(scenario["total_accesses"]) + t
                scenario["_prompt_lens"].append(prompt_len)
                scenario["_source_lens"].append(source_len)

    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        hits = int(scenario["hits"])
        total = int(scenario["total_accesses"])
        prompt_lens = scenario.pop("_prompt_lens")
        source_lens = scenario.pop("_source_lens")
        scenario["hit_rate"] = float(hits) / float(total) if total else 0.0
        scenario["profile_count"] = len(files)
        scenario["layer_count"] = len(layer_ids)
        scenario["prompt_len_min"] = min(prompt_lens) if prompt_lens else 0
        scenario["prompt_len_max"] = max(prompt_lens) if prompt_lens else 0
        scenario["source_len_min"] = min(source_lens) if source_lens else 0
        scenario["source_len_max"] = max(source_lens) if source_lens else 0
        rows.append(scenario)
        print(
            f"{scenario['policy']} B{scenario['batch_size']} ctx{scenario['target_context']} "
            f"{scenario['projection']} beta={scenario['tail_beta']}: "
            f"hit={scenario['hit_rate']:.6f}"
        )

    out = Path(args.output_root).resolve()
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "hit_rate_sensitivity.csv", rows)
    (out / "run_meta.json").write_text(
        json.dumps(
            {
                "profile_files": [str(p) for p in files],
                "layer_ids": layer_ids,
                "args": vars(args),
                "projection_semantics": {
                    "raw": "Current TP1 behavior: filter token ids by target_context only.",
                    "scale": "Preserve observed locality but stretch token positions to target_context.",
                    "tail": "Stretch observed tokens and replace a fraction with deterministic cold tail tokens. observed_mass=(source_len/target_context)^tail_beta.",
                },
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
