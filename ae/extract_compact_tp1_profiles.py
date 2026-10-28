#!/usr/bin/env python3
"""Extract the exact compact TP1 fields used by the AE workflow."""

from __future__ import annotations

import argparse
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def compact_profile(source: Path, decode_steps: int, layer_ids: list[int]) -> dict:
    with source.open("rb") as handle:
        data = pickle.load(handle)
    prompt_len = int(data["prompt_len"])
    compact: dict[int, dict[str, object]] = {}
    for layer_id in layer_ids:
        layer = data["profiling_data"][layer_id]
        if "topk_indices" in layer:
            sequence = np.asarray(layer["topk_indices"])[0]
            prefill = sequence[:prompt_len]
            decode = sequence[prompt_len : prompt_len + decode_steps]
        else:
            prefill_list = layer.get("prefill_indices", [])
            prefill = np.asarray(prefill_list[0]) if prefill_list else np.empty((0, 0), dtype=np.int64)
            decode = np.asarray(layer.get("decode_indices", []))[:decode_steps]
        counts = Counter(int(token) for token in prefill.reshape(-1).tolist())
        compact[layer_id] = {
            "prefill_counter": dict(counts),
            "decode_indices": decode.astype(np.int32, copy=False),
        }
    return {
        "format": "hacmo_compact_profile_v1",
        "source_name": source.name,
        "prompt_len": prompt_len,
        "profiling_data": compact,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Write one compact pickle to stdout (useful for read-only remote extraction).",
    )
    parser.add_argument("--decode-steps", type=int, default=50)
    parser.add_argument("--layers", type=int, nargs="+", default=list(range(0, 61, 6)))
    args = parser.parse_args()
    if args.stdout == (args.output_dir is not None):
        parser.error("choose exactly one of --output-dir or --stdout")
    if args.stdout and len(args.sources) != 1:
        parser.error("--stdout accepts exactly one source")
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    for source in args.sources:
        payload = compact_profile(source, args.decode_steps, args.layers)
        encoded = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        if args.stdout:
            sys.stdout.buffer.write(encoded)
            sys.stdout.buffer.flush()
        else:
            assert args.output_dir is not None
            output = args.output_dir / source.name
            output.write_bytes(encoded)
            print(f"{source} -> {output} ({len(encoded)} bytes)", flush=True)


if __name__ == "__main__":
    main()
