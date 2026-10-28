#!/usr/bin/env python3
"""Core-level static mapping search for TP1 rebuttal experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _normalize(xs: list[float]) -> list[float]:
    s = sum(xs)
    return [x / s for x in xs] if s > 0 else [1.0 / len(xs) for _ in xs]


def _avg_hbm_hops(cells: list[tuple[int, int]], rows: int, weights: list[float]) -> float:
    return sum(w * min(r, rows - 1 - r) for (r, _), w in zip(cells, weights))


def _expected_abs_distance(src: list[float], dst: list[float]) -> float:
    total = 0.0
    for i, si in enumerate(src):
        if si == 0:
            continue
        for j, dj in enumerate(dst):
            if dj == 0:
                continue
            total += si * dj * abs(i - j)
    return total


def _avg_remote_hops(cells: list[tuple[int, int]], src: list[float], dst: list[float], rows: int, cols: int) -> float:
    row_src = [0.0 for _ in range(rows)]
    row_dst = [0.0 for _ in range(rows)]
    col_src = [0.0 for _ in range(cols)]
    col_dst = [0.0 for _ in range(cols)]
    for (r, c), s, d in zip(cells, src, dst):
        row_src[r] += s
        row_dst[r] += d
        col_src[c] += s
        col_dst[c] += d
    return _expected_abs_distance(row_src, row_dst) + _expected_abs_distance(col_src, col_dst)


def _tail_scale(cells: list[tuple[int, int]], compute: list[float], cache: list[float], rows: int, cols: int) -> float:
    pressure = [max(c - k, 0.0) for c, k in zip(compute, cache)]
    if sum(pressure) <= 1e-12:
        return 1.0
    row_pressure = [0.0 for _ in range(rows)]
    for (r, _), p in zip(cells, pressure):
        row_pressure[r] += p
    avg = sum(row_pressure) / max(rows, 1)
    now = max(row_pressure) / max(avg, 1e-12)
    uniform_cache = [1.0 / (rows * cols) for _ in cells]
    ref_p = [max(c - k, 0.0) for c, k in zip(compute, uniform_cache)]
    if sum(ref_p) <= 1e-12:
        return 1.0
    ref_row = [0.0 for _ in range(rows)]
    for (r, _), p in zip(cells, ref_p):
        ref_row[r] += p
    ref_avg = sum(ref_row) / max(rows, 1)
    ref = max(ref_row) / max(ref_avg, 1e-12)
    return now / max(ref, 1e-12)


def _profile(rows: int, cols: int, compute_grad: float, cache_grad: float) -> dict:
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    row_center = (rows - 1) / 2.0
    max_row = max(row_center, 1.0)
    # HBM is modeled on the top/bottom die edges. Static compute and cache
    # placement therefore depends on row distance to HBM, not column position.
    # Column coordinates still matter later when explicit NoC paths are modeled.
    edge_score = [abs(r - row_center) / max_row for r, _ in cells]
    center_score = [1.0 - e for e in edge_score]
    compute = _normalize([1.0 + compute_grad * e for e in edge_score])
    cache = _normalize([1.0 + cache_grad * s for s in center_score])
    miss = _normalize([max(c - k, 0.0) for c, k in zip(compute, cache)])
    uniform = [1.0 / len(cells) for _ in cells]
    hbm_ref = _avg_hbm_hops(cells, rows, uniform)
    hbm_now = _avg_hbm_hops(cells, rows, miss)
    remote_ref = _avg_remote_hops(cells, uniform, uniform, rows, cols)
    remote_now = _avg_remote_hops(cells, miss, cache, rows, cols)
    mismatch_ref = sum(max(c - u, 0.0) for c, u in zip(compute, uniform))
    mismatch_now = sum(max(c - k, 0.0) for c, k in zip(compute, cache))
    if mismatch_ref <= 1e-12:
        remote_fraction_scale = 1.0
    else:
        remote_fraction_scale = mismatch_now / mismatch_ref
    derived = {
        "avg_hbm_hops_scale": max(0.50, min(1.10, hbm_now / max(hbm_ref, 1e-12))),
        "avg_remote_hops_scale": max(0.50, min(1.10, remote_now / max(remote_ref, 1e-12))),
        "remote_fraction_scale": max(0.35, min(1.00, remote_fraction_scale)),
        "tail_penalty_scale": max(0.60, min(1.05, _tail_scale(cells, compute, cache, rows, cols))),
    }
    score = (
        0.35 * derived["avg_hbm_hops_scale"]
        + 0.30 * derived["avg_remote_hops_scale"]
        + 0.20 * derived["remote_fraction_scale"]
        + 0.15 * derived["tail_penalty_scale"]
    )
    return {
        "score": score,
        "derived": derived,
        "row_compute_weights": [sum(compute[i] for i, (r, _) in enumerate(cells) if r == row) for row in range(rows)],
        "row_cache_weights": [sum(cache[i] for i, (r, _) in enumerate(cells) if r == row) for row in range(rows)],
        "compute_grad": compute_grad,
        "cache_grad": cache_grad,
        "placement_axis": "row_only_hbm_distance",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Search core-level static mapping profiles.")
    parser.add_argument("--array_shapes", nargs="+", default=["20x20", "68x68"])
    parser.add_argument("--compute_grad_max", type=float, default=2.5)
    parser.add_argument("--cache_grad_max", type=float, default=3.0)
    parser.add_argument("--grid_steps", type=int, default=21)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    profiles = {}
    for shape in args.array_shapes:
        rows, cols = [int(x) for x in shape.lower().split("x")]
        best = None
        for i in range(args.grid_steps):
            cg = args.compute_grad_max * i / max(args.grid_steps - 1, 1)
            for j in range(args.grid_steps):
                kg = args.cache_grad_max * j / max(args.grid_steps - 1, 1)
                cand = _profile(rows, cols, cg, kg)
                if best is None or cand["score"] < best["score"]:
                    best = cand
        profiles[shape] = {"rows": rows, "cols": cols, **best}

    payload = {
        "model": "core_static_mapping_v1",
        "profiles": profiles,
        "notes": "Core-level static mapping profile. remote_fraction_scale is capped at <=1.0 to avoid static mapping increasing remote traffic by construction.",
    }
    out = Path(args.output_json).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
