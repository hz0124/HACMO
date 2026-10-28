#!/usr/bin/env python3
"""Generate trace-driven HBM access streams from sparse-attention profiles.

The output format is simulator-neutral CSV so it can be consumed by local
analysis, Ramulator2 adapters, or DRAMsim3 adapters:

  case,op,addr,size,layer,step,batch,token,pattern

Addresses are synthetic but stable. They preserve the access distribution,
page grouping, batch scaling, and layer separation from the profile.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np


def _profile_files(paths: list[str], max_files: int) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        files.extend(sorted(Path(raw).glob("*_profile.pkl")))
    files = [p for p in files if "gutenberg_0_profile.pkl" not in p.name]
    return files[:max_files] if max_files > 0 else files


def _load_decode_rows(path: Path, layer_ids: list[int], decode_steps: int) -> dict[int, np.ndarray]:
    data = pickle.loads(path.read_bytes())
    prompt_len = int(data["prompt_len"])
    out: dict[int, np.ndarray] = {}
    for layer_id in layer_ids:
        layer_data = data["profiling_data"][layer_id]
        if "topk_indices" in layer_data:
            arr = np.asarray(layer_data["topk_indices"])
            decode = arr[0][prompt_len:, :]
        else:
            decode = np.asarray(layer_data.get("decode_indices", []))
        if decode_steps > 0:
            decode = decode[:decode_steps, :]
        out[layer_id] = decode.astype(np.int64, copy=False)
    return out


def _unique_valid(row: Iterable[int], context_len: int) -> list[int]:
    vals: list[int] = []
    seen: set[int] = set()
    for raw in row:
        token = int(raw)
        if token < 0 or token >= context_len or token in seen:
            continue
        vals.append(token)
        seen.add(token)
    return vals


def _tokens_for_pattern(
    pattern: str,
    profile_tokens: list[int],
    context_len: int,
    topk: int,
    rng: random.Random,
    page_tokens: int,
) -> list[int]:
    if pattern == "profile":
        return profile_tokens[:topk]
    if pattern == "page_grouped":
        return sorted(profile_tokens[:topk], key=lambda t: (t // page_tokens, t))
    if pattern == "random":
        k = min(topk, context_len)
        return rng.sample(range(context_len), k)
    if pattern == "dense":
        return list(range(min(topk, context_len)))
    raise ValueError(f"Unsupported pattern: {pattern}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate HBM traces from DeepSeek sparse-attention profiles.")
    parser.add_argument("--profile_dirs", nargs="+", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[8, 16, 32, 64])
    parser.add_argument("--context_lens", nargs="+", type=int, default=[8192, 32768, 65536])
    parser.add_argument("--patterns", nargs="+", choices=["profile", "page_grouped", "random", "dense"], default=["profile", "page_grouped", "random", "dense"])
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--selected_vector_bytes", type=int, default=576)
    parser.add_argument("--kv_entry_bytes", type=int, default=1152)
    parser.add_argument("--page_bytes", type=int, default=16384)
    parser.add_argument("--n_layers", type=int, default=61)
    parser.add_argument("--layer_sample_stride", type=int, default=6)
    parser.add_argument("--decode_steps", type=int, default=50)
    parser.add_argument("--max_profile_files", type=int, default=3)
    parser.add_argument("--max_events_per_case", type=int, default=250000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    root = Path(args.output_root).resolve()
    trace_dir = root / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    files = _profile_files(args.profile_dirs, args.max_profile_files)
    if not files:
        raise FileNotFoundError("No *_profile.pkl files found in profile_dirs.")

    layer_ids = list(range(0, args.n_layers, max(args.layer_sample_stride, 1)))
    decoded = [_load_decode_rows(path, layer_ids, args.decode_steps) for path in files]
    rng = random.Random(args.seed)
    page_tokens = max(args.page_bytes // max(args.kv_entry_bytes, 1), 1)
    summaries: list[dict] = []

    for context_len in args.context_lens:
        for batch_size in args.batch_sizes:
            for pattern in args.patterns:
                case = f"{pattern}_bs{batch_size}_ctx{context_len}_topk{args.topk}"
                trace_path = trace_dir / f"{case}.csv"
                events = 0
                token_counter: Counter[int] = Counter()
                with trace_path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=["case", "op", "addr", "size", "layer", "step", "batch", "token", "pattern"],
                    )
                    writer.writeheader()
                    stop = False
                    case_profile_stride = args.n_layers * batch_size * context_len * args.kv_entry_bytes
                    case_layer_stride = batch_size * context_len * args.kv_entry_bytes
                    case_batch_stride = context_len * args.kv_entry_bytes
                    for profile_idx, profile in enumerate(decoded):
                        if stop:
                            break
                        profile_base = profile_idx * case_profile_stride
                        for layer_id in layer_ids:
                            if stop:
                                break
                            layer_base = profile_base + layer_id * case_layer_stride
                            rows = profile[layer_id]
                            for step, row in enumerate(rows):
                                profile_tokens = _unique_valid(row, context_len)
                                tokens = _tokens_for_pattern(pattern, profile_tokens, context_len, args.topk, rng, page_tokens)
                                for b in range(batch_size):
                                    batch_base = b * case_batch_stride
                                    for token in tokens:
                                        addr = layer_base + batch_base + token * args.kv_entry_bytes
                                        writer.writerow(
                                            {
                                                "case": case,
                                                "op": "R",
                                                "addr": hex(addr),
                                                "size": args.selected_vector_bytes,
                                                "layer": layer_id,
                                                "step": step,
                                                "batch": b,
                                                "token": token,
                                                "pattern": pattern,
                                            }
                                        )
                                        token_counter[token] += 1
                                        events += 1
                                        if args.max_events_per_case > 0 and events >= args.max_events_per_case:
                                            stop = True
                                            break
                                    if stop:
                                        break
                                if stop:
                                    break
                requested_bytes = events * args.selected_vector_bytes
                summaries.append(
                    {
                        "case": case,
                        "trace": str(trace_path),
                        "pattern": pattern,
                        "batch_size": batch_size,
                        "context_len": context_len,
                        "topk": args.topk,
                        "events": events,
                        "requested_bytes": requested_bytes,
                        "unique_tokens": len(token_counter),
                        "max_token_reuse": max(token_counter.values()) if token_counter else 0,
                    }
                )

    (root / "trace_summary.json").write_text(json.dumps(summaries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (root / "trace_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    print(root)


if __name__ == "__main__":
    main()
