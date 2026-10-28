#!/usr/bin/env python3
"""Export 8-rank H200/DeepEP traffic shapes from TP2 runtime artifacts.

This tool intentionally rebuilds the H200 comparator traffic from the raw TP2
runtime source->expert counts instead of reusing HACMO's target-hardware
die-level matrices. The output is an EP=8 GPU-level matrix for validating the
sparse P2P communication model against vLLM/DeepEP on H200.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export 8-rank H200 DeepEP traffic JSON from TP2 artifacts."
    )
    parser.add_argument(
        "--artifact-topology-dir",
        required=True,
        help=(
            "TP2 topology artifact directory, e.g. "
            "hacmo/moe/outputs/.../16w_4x4_rail_500"
        ),
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument(
        "--num-requests",
        type=int,
        default=None,
        help="Base request count represented by the artifact. Defaults to metadata.",
    )
    parser.add_argument("--ep-size", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument("--dispatch-bytes-per-hidden-elem", type=int, default=1)
    parser.add_argument("--combine-bytes-per-hidden-elem", type=int, default=2)
    parser.add_argument(
        "--source-rank-map",
        choices=["modulo", "block"],
        default="modulo",
        help=(
            "How TP2 source IDs are folded onto 8 H200 GPU ranks. "
            "modulo is the default balanced DP-style fold."
        ),
    )
    parser.add_argument(
        "--window-limit",
        type=int,
        default=0,
        help="Optional debug limit on runtime windows. 0 means all windows.",
    )
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_topology_dir)
    runtime_dir = artifact_dir / "mapping_inputs" / "runtime"
    window_src_path = runtime_dir / "window_src_expert_counts.json"
    if not window_src_path.exists():
        raise FileNotFoundError(f"Missing runtime source/expert counts: {window_src_path}")

    window_src_expert = json.loads(window_src_path.read_text(encoding="utf-8"))
    if args.window_limit and args.window_limit > 0:
        window_src_expert = window_src_expert[: args.window_limit]

    num_requests = int(args.num_requests) if args.num_requests else _artifact_num_requests(artifact_dir)
    source_span = _source_span(window_src_expert)
    dispatch_bytes_per_token = args.hidden_size * args.dispatch_bytes_per_hidden_elem
    combine_bytes_per_token = args.hidden_size * args.combine_bytes_per_hidden_elem

    batches = []
    for batch_size in args.batch_sizes:
        batch_scale = float(batch_size) / max(num_requests, 1)
        windows = []
        for window_id, window in enumerate(window_src_expert):
            windows.append(
                _build_window(
                    window_id=window_id,
                    window=window,
                    batch_scale=batch_scale,
                    ep_size=args.ep_size,
                    source_span=source_span,
                    source_rank_map=args.source_rank_map,
                    num_experts=args.num_experts,
                    dispatch_bytes_per_token=dispatch_bytes_per_token,
                    combine_bytes_per_token=combine_bytes_per_token,
                )
            )
        batches.append(
            {
                "batch_size": batch_size,
                "batch_scale": batch_scale,
                "num_windows": len(windows),
                "summary": _summarize_windows(windows),
                "windows": windows,
            }
        )

    payload = {
        "schema_version": 1,
        "comparator": "h200_vllm_deepep_ep8",
        "source": {
            "artifact_topology_dir": str(artifact_dir),
            "window_src_expert_counts": str(window_src_path),
            "num_requests": num_requests,
            "source_rank_map": args.source_rank_map,
            "source_span": source_span,
        },
        "model": {
            "num_experts": args.num_experts,
            "topk": args.topk,
            "hidden_size": args.hidden_size,
            "ep_size": args.ep_size,
            "expert_placement": "vllm_linear",
            "dispatch_dtype": "fp8",
            "combine_dtype": "bf16_or_fp16",
            "dispatch_bytes_per_hidden_elem": args.dispatch_bytes_per_hidden_elem,
            "combine_bytes_per_hidden_elem": args.combine_bytes_per_hidden_elem,
            "dispatch_bytes_per_token": dispatch_bytes_per_token,
            "combine_bytes_per_token": combine_bytes_per_token,
        },
        "notes": [
            "Experts are remapped to 8 GPU ranks with vLLM default linear placement.",
            "Source IDs from TP2 runtime artifacts are folded to H200 ranks; HACMO 256-rank placement is not used.",
            "Matrices omit local src==dst transfers for network bytes but keep source_rank_expert_counts for DeepEP topk generation.",
        ],
        "batches": batches,
    }

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


def _artifact_num_requests(artifact_dir: Path) -> int:
    meta_path = artifact_dir / "mapping_inputs" / "runtime" / "metadata.json"
    if not meta_path.exists():
        return 64
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return int(meta.get("num_requests") or meta.get("trace_num_requests") or 64)


def _source_span(window_src_expert: list[dict[str, Any]]) -> int:
    max_src = 0
    for window in window_src_expert:
        for src in window.get("src_expert_counts", {}):
            max_src = max(max_src, int(src))
    return max_src + 1


def _source_rank(src: int, *, ep_size: int, source_span: int, mode: str) -> int:
    if mode == "modulo":
        return src % ep_size
    if mode == "block":
        return min((src * ep_size) // max(source_span, 1), ep_size - 1)
    raise ValueError(f"Unsupported source-rank map: {mode}")


def _expert_rank(expert: int, *, num_experts: int, ep_size: int) -> int:
    """Match vLLM's default linear expert placement."""
    base = num_experts // ep_size
    remainder = num_experts % ep_size
    first_regular = (base + 1) * remainder
    if expert < first_regular:
        return expert // (base + 1)
    return remainder + ((expert - first_regular) // max(base, 1))


def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def _scale_expert_counts(counts: dict[int, int], batch_scale: float) -> dict[int, int]:
    """Scale expert counts while preserving small nonzero batches.

    TP2 artifacts may spread requests over hundreds of source die IDs. For the
    H200 comparator, source IDs are first folded to 8 GPU ranks; then each
    rank's expert counts are scaled with largest-remainder rounding so
    batch=16 does not disappear due to per-die rounding.
    """
    if not counts:
        return {}
    scaled = {expert: float(count) * batch_scale for expert, count in counts.items()}
    floors = {expert: int(math.floor(value)) for expert, value in scaled.items()}
    target = _round_half_up(sum(counts.values()) * batch_scale)
    remainder = max(target - sum(floors.values()), 0)
    out = dict(floors)
    if remainder <= 0:
        return out
    order = sorted(
        scaled,
        key=lambda expert: (scaled[expert] - floors[expert], counts[expert], -expert),
        reverse=True,
    )
    for expert in order[:remainder]:
        out[expert] += 1
    return out


def _build_window(
    *,
    window_id: int,
    window: dict[str, Any],
    batch_scale: float,
    ep_size: int,
    source_span: int,
    source_rank_map: str,
    num_experts: int,
    dispatch_bytes_per_token: int,
    combine_bytes_per_token: int,
) -> dict[str, Any]:
    raw_source_rank_expert_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    for src_str, expert_counts in window.get("src_expert_counts", {}).items():
        src_rank = _source_rank(
            int(src_str),
            ep_size=ep_size,
            source_span=source_span,
            mode=source_rank_map,
        )
        for expert_str, raw_count in expert_counts.items():
            expert = int(expert_str)
            raw_source_rank_expert_counts[src_rank][expert] += int(raw_count)

    source_rank_expert_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for src_rank, expert_counts in raw_source_rank_expert_counts.items():
        for expert, count in _scale_expert_counts(expert_counts, batch_scale).items():
            if count > 0:
                source_rank_expert_counts[src_rank][expert] += count

    dispatch_tokens: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for src_rank, expert_counts in source_rank_expert_counts.items():
        for expert, count in expert_counts.items():
            dst_rank = _expert_rank(expert, num_experts=num_experts, ep_size=ep_size)
            if src_rank != dst_rank:
                dispatch_tokens[src_rank][dst_rank] += count

    combine_tokens: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for src_rank, dsts in dispatch_tokens.items():
        for dst_rank, token_count in dsts.items():
            combine_tokens[dst_rank][src_rank] += token_count

    dispatch_bytes = _scale_matrix(dispatch_tokens, dispatch_bytes_per_token)
    combine_bytes = _scale_matrix(combine_tokens, combine_bytes_per_token)

    return {
        "window_id": window_id,
        "source_rank_expert_counts": _stringify_nested(source_rank_expert_counts),
        "dispatch_token_matrix": _stringify_nested(dispatch_tokens),
        "combine_token_matrix": _stringify_nested(combine_tokens),
        "dispatch_matrix_bytes": _stringify_nested(dispatch_bytes),
        "combine_matrix_bytes": _stringify_nested(combine_bytes),
        "summary": {
            "total_expert_activations": _sum_nested(source_rank_expert_counts),
            "cross_rank_dispatch_tokens": _sum_nested(dispatch_tokens),
            "cross_rank_combine_tokens": _sum_nested(combine_tokens),
            "dispatch_bytes": _sum_nested(dispatch_bytes),
            "combine_bytes": _sum_nested(combine_bytes),
            "active_source_ranks": len(source_rank_expert_counts),
            "active_dispatch_edges": sum(len(dsts) for dsts in dispatch_tokens.values()),
        },
    }


def _scale_matrix(matrix: dict[int, dict[int, int]], bytes_per_token: int) -> dict[int, dict[int, int]]:
    return {
        src: {dst: int(tokens) * bytes_per_token for dst, tokens in dsts.items()}
        for src, dsts in matrix.items()
    }


def _sum_nested(matrix: dict[int, dict[int, int]]) -> int:
    return int(sum(v for dsts in matrix.values() for v in dsts.values()))


def _stringify_nested(matrix: dict[int, dict[int, int]]) -> dict[str, dict[str, int]]:
    return {
        str(src): {str(dst): int(value) for dst, value in sorted(dsts.items())}
        for src, dsts in sorted(matrix.items())
    }


def _summarize_windows(windows: list[dict[str, Any]]) -> dict[str, int]:
    keys = [
        "total_expert_activations",
        "cross_rank_dispatch_tokens",
        "cross_rank_combine_tokens",
        "dispatch_bytes",
        "combine_bytes",
    ]
    return {key: int(sum(w["summary"][key] for w in windows)) for key in keys}


if __name__ == "__main__":
    main()
