#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

PER_DIE_EMPIRICAL_SOURCE_MODES = {"replicate_all_dies", "empirical_per_die_sampling"}


def _list_request_files(normalized_dir: Path) -> List[Path]:
    return sorted(p for p in normalized_dir.glob("*.json") if p.name != "metadata.json")


def is_empirical_per_die_source_mode(source_mode: str) -> bool:
    return source_mode in PER_DIE_EMPIRICAL_SOURCE_MODES


def _token_experts(token_obj: dict) -> List[int]:
    experts: List[int] = []
    for _, eids in token_obj.get("layer_to_expert_ids", {}).items():
        experts.extend(int(e) for e in eids)
    return experts


def _iter_request_tokens(
    req_payloads: List[dict],
    *,
    start: int,
    end: int,
) -> Iterable[Tuple[int, List[int]]]:
    for req_idx, req in enumerate(req_payloads):
        for tk in req.get("decode", [])[start:end]:
            yield req_idx, _token_experts(tk)


def _source_die(req_idx: int, num_requests: int, num_dies: int) -> int:
    """Map normalized requests to source dies without low-rank wafer bias."""
    if num_dies <= 0:
        return 0
    if num_requests <= 0:
        return req_idx % num_dies
    if num_requests < num_dies:
        return min((req_idx * num_dies) // num_requests, num_dies - 1)
    return req_idx % num_dies


def build_mapping_inputs_from_payloads(
    *,
    req_payloads: List[dict],
    source_mode: str,
    num_dies: int,
    effective_decode_len: int,
    window_size: int,
) -> tuple[Dict[int, Counter], Counter, Counter, list[dict], list[dict]]:
    if is_empirical_per_die_source_mode(source_mode):
        return _build_empirical_per_die_inputs(
            req_payloads=req_payloads,
            num_dies=num_dies,
            effective_decode_len=effective_decode_len,
            window_size=window_size,
        )
    return _build_spread_request_inputs(
        req_payloads=req_payloads,
        num_dies=num_dies,
        effective_decode_len=effective_decode_len,
        window_size=window_size,
    )


def _build_empirical_per_die_inputs(
    *,
    req_payloads: List[dict],
    num_dies: int,
    effective_decode_len: int,
    window_size: int,
) -> tuple[Dict[int, Counter], Counter, Counter, list[dict], list[dict]]:
    """Build per-die empirical traffic without repeated per-die counting.

    In empirical-per-die mode every target die draws from the same finite trace
    distribution in expectation. The slow literal expansion counts the same
    request/token expert histogram once per die. This function computes that
    per-die histogram once, then replicates the resulting counters so the JSON
    contract remains identical to the literal expansion.
    """
    per_die_counts: Counter = Counter()
    tokens_per_die = 0
    for _, experts in _iter_request_tokens(req_payloads, start=0, end=effective_decode_len):
        tokens_per_die += 1
        per_die_counts.update(experts)

    p_ue_counts: Dict[int, Counter] = {u: Counter(per_die_counts) for u in range(num_dies)}
    s_u: Counter = Counter({u: tokens_per_die for u in range(num_dies)})
    h_counts: Counter = Counter({e: c * num_dies for e, c in per_die_counts.items()})

    window_stats = []
    window_src_expert_counts = []
    num_windows = (effective_decode_len + window_size - 1) // window_size
    for w in range(num_windows):
        st = w * window_size
        ed = min(effective_decode_len, (w + 1) * window_size)
        per_die_window_counts: Counter = Counter()
        per_die_window_tokens = 0
        for _, experts in _iter_request_tokens(req_payloads, start=st, end=ed):
            per_die_window_tokens += 1
            per_die_window_counts.update(experts)
        all_die_counts = {
            int(e): int(c) * num_dies for e, c in per_die_window_counts.items()
        }
        window_stats.append(
            {
                "window_id": w,
                "start_token_idx": st + 1,
                "end_token_idx": ed,
                "expert_counts": all_die_counts,
                "tokens_per_die": {u: int(per_die_window_tokens) for u in range(num_dies)},
                "active": per_die_window_tokens > 0,
            }
        )
        per_die_src_counts = {int(e): int(c) for e, c in per_die_window_counts.items()}
        window_src_expert_counts.append(
            {
                "window_id": w,
                "src_expert_counts": (
                    {str(u): dict(per_die_src_counts) for u in range(num_dies)}
                    if per_die_src_counts
                    else {}
                ),
            }
        )
    return p_ue_counts, h_counts, s_u, window_stats, window_src_expert_counts


def _build_spread_request_inputs(
    *,
    req_payloads: List[dict],
    num_dies: int,
    effective_decode_len: int,
    window_size: int,
) -> tuple[Dict[int, Counter], Counter, Counter, list[dict], list[dict]]:
    p_ue_counts: Dict[int, Counter] = {u: Counter() for u in range(num_dies)}
    h_counts: Counter = Counter()
    s_u: Counter = Counter()
    window_stats = []
    window_src_expert_counts = []
    trace_num_requests = len(req_payloads)

    for req_idx, experts in _iter_request_tokens(req_payloads, start=0, end=effective_decode_len):
        src_die = _source_die(req_idx, trace_num_requests, num_dies)
        s_u[src_die] += 1
        for e in experts:
            p_ue_counts[src_die][int(e)] += 1
            h_counts[int(e)] += 1

    num_windows = (effective_decode_len + window_size - 1) // window_size
    for w in range(num_windows):
        e_counter = Counter()
        u_counter = Counter()
        src_e_counter = {u: Counter() for u in range(num_dies)}
        st = w * window_size
        ed = min(effective_decode_len, (w + 1) * window_size)
        for req_idx, experts in _iter_request_tokens(req_payloads, start=st, end=ed):
            src_die = _source_die(req_idx, trace_num_requests, num_dies)
            for e in experts:
                e_counter[e] += 1
                src_e_counter[src_die][e] += 1
            u_counter[src_die] += 1
        window_stats.append(
            {
                "window_id": w,
                "start_token_idx": st + 1,
                "end_token_idx": ed,
                "expert_counts": dict(e_counter),
                "tokens_per_die": dict(u_counter),
                "active": bool(u_counter),
            }
        )
        window_src_expert_counts.append(
            {
                "window_id": w,
                "src_expert_counts": {str(u): dict(c) for u, c in src_e_counter.items() if c},
            }
        )
    return p_ue_counts, h_counts, s_u, window_stats, window_src_expert_counts


def main() -> None:
    p = argparse.ArgumentParser(description="Build mapping inputs p[u,e], h[e], s[u] and runtime windows.")
    p.add_argument("--normalized-trace-dir", required=True)
    p.add_argument("--num-dies", type=int, default=256)
    p.add_argument("--num-experts", type=int, required=True)
    p.add_argument("--decode-len", type=int, required=True)
    p.add_argument("--window-size", type=int, default=8)
    p.add_argument(
        "--source-mode",
        choices=["spread_requests", "replicate_all_dies", "empirical_per_die_sampling"],
        default="spread_requests",
        help=(
            "spread_requests maps each trace request to one target die. "
            "empirical_per_die_sampling treats the finite trace set as an empirical "
            "per-die request distribution and builds expected aggregate traffic for "
            "every target die. replicate_all_dies is a legacy alias."
        ),
    )
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    in_dir = Path(args.normalized_trace_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    req_files = _list_request_files(in_dir)
    req_payloads = [json.loads(f.read_text(encoding="utf-8")) for f in req_files]
    max_trace_decode_len = max((len(req.get("decode", [])) for req in req_payloads), default=0)
    effective_decode_len = min(args.decode_len, max_trace_decode_len) if max_trace_decode_len > 0 else args.decode_len
    if effective_decode_len <= 0:
        raise ValueError(f"No decode tokens found in normalized traces under {in_dir}")
    trace_num_requests = len(req_files)
    p_ue_counts, h_counts, s_u, window_stats, window_src_expert_counts = build_mapping_inputs_from_payloads(
        req_payloads=req_payloads,
        source_mode=args.source_mode,
        num_dies=args.num_dies,
        effective_decode_len=effective_decode_len,
        window_size=args.window_size,
    )

    total_per_u = {u: sum(c.values()) for u, c in p_ue_counts.items()}
    p_ue = defaultdict(dict)
    for u in range(args.num_dies):
        denom = float(total_per_u[u] or 1)
        for e in range(args.num_experts):
            p_ue[u][e] = p_ue_counts[u][e] / denom

    total_h = float(sum(h_counts.values()) or 1)
    h = {e: h_counts[e] / total_h for e in range(args.num_experts)}

    num_windows = (effective_decode_len + args.window_size - 1) // args.window_size

    # Static-to-dynamic compact signatures (sorted histogram top entries).
    trace_mapping_candidates = []
    for ws in window_stats:
        items = sorted(ws["expert_counts"].items(), key=lambda x: x[1], reverse=True)[:32]
        vals = [v for _, v in items]
        trace_mapping_candidates.append({"window_id": ws["window_id"], "signature": vals})

    (out_dir / "p_ue.json").write_text(json.dumps(p_ue, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "h.json").write_text(json.dumps(h, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "s_u.json").write_text(json.dumps(dict(s_u), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "window_stats.json").write_text(
        json.dumps(window_stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "trace_mapping_candidates.json").write_text(
        json.dumps(trace_mapping_candidates, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "window_src_expert_counts.json").write_text(
        json.dumps(window_src_expert_counts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    meta = {
        "requested_decode_len": args.decode_len,
        "effective_decode_len": effective_decode_len,
        "max_trace_decode_len": max_trace_decode_len,
        "window_size": args.window_size,
        "num_windows": num_windows,
        "source_mode": args.source_mode,
        "source_mode_semantics": (
            "empirical_per_die_sampling"
            if is_empirical_per_die_source_mode(args.source_mode)
            else "spread_requests_to_target_dies"
        ),
        "legacy_source_mode_alias": (
            "replicate_all_dies" if args.source_mode == "replicate_all_dies" else None
        ),
        "trace_sample_size": trace_num_requests,
        "sampling_contract": (
            "The finite trace set is treated as an empirical request distribution. "
            "For per-die batch evaluation, each target die sources traffic drawn "
            "from this distribution in expectation; downstream evaluators scale "
            "counts by batch_per_die / trace_sample_size. This does not claim one "
            "unique raw trace per global in-flight request."
            if is_empirical_per_die_source_mode(args.source_mode)
            else "Each normalized trace request is assigned to one source die."
        ),
        "trace_num_requests": trace_num_requests,
        "num_requests": trace_num_requests * args.num_dies
        if is_empirical_per_die_source_mode(args.source_mode)
        else trace_num_requests,
        "effective_source_dies": args.num_dies
        if is_empirical_per_die_source_mode(args.source_mode)
        else len({int(k) for k in s_u}),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote mapping inputs to: {out_dir} (effective_decode_len={effective_decode_len})")


if __name__ == "__main__":
    main()
