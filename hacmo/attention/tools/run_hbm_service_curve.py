#!/usr/bin/env python3
"""Build HBM service curves from generated traces.

This script provides a deterministic local fallback and an integration point
for external cycle-level simulators. The fallback is intentionally simple: it
derives request coalescing and bandwidth amplification from trace locality, then
maps that to calibrated dense/sparse effective bandwidths. If a Ramulator2 or
DRAMsim3 wrapper is available, pass --simulator_cmd_template and the wrapper can
consume each trace and emit per-case metrics JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from supernode_sim.backends.ramulator import run_hbm_trace


DEFAULT_SPARSE_UTIL_BY_BATCH = {
    1: 0.024,
    4: 0.090,
    8: 0.154,
    16: 0.181,
    32: 0.211,
    64: 0.186,
    128: 0.186,
}


def _interp_util(batch_size: int, table: dict[int, float]) -> float:
    if batch_size in table:
        return table[batch_size]
    xs = sorted(table)
    if batch_size <= xs[0]:
        return table[xs[0]]
    if batch_size >= xs[-1]:
        return table[xs[-1]]
    for lo, hi in zip(xs, xs[1:]):
        if lo <= batch_size <= hi:
            t = (math.log2(batch_size) - math.log2(lo)) / max(math.log2(hi) - math.log2(lo), 1e-9)
            return table[lo] * (1.0 - t) + table[hi] * t
    return table[xs[-1]]


def _read_trace_metrics(path: Path, cacheline_bytes: int) -> dict[str, Any]:
    requested = 0
    cachelines: set[int] = set()
    pages: set[int] = set()
    events = 0
    per_warp_lines: list[int] = []
    current_group: list[int] = []
    current_key: tuple[str, str, str] | None = None

    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            addr = int(row["addr"], 16)
            size = int(row["size"])
            requested += size
            first = addr // cacheline_bytes
            last = (addr + size - 1) // cacheline_bytes
            lines = list(range(first, last + 1))
            cachelines.update(lines)
            pages.add(addr // 16384)
            events += 1
            key = (row["layer"], row["step"], row["batch"])
            if current_key is None:
                current_key = key
            if key != current_key:
                per_warp_lines.append(len(set(current_group)))
                current_group = []
                current_key = key
            current_group.extend(lines)
    if current_group:
        per_warp_lines.append(len(set(current_group)))

    dram_bytes = len(cachelines) * cacheline_bytes
    amplification = dram_bytes / max(requested, 1)
    return {
        "events": events,
        "requested_bytes": requested,
        "dram_bytes_est": dram_bytes,
        "unique_cachelines": len(cachelines),
        "unique_pages": len(pages),
        "amplification_est": amplification,
        "avg_cachelines_per_step_batch": sum(per_warp_lines) / max(len(per_warp_lines), 1),
    }


def _run_external(template: str, trace: Path, out_dir: Path, case: str) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / f"{case}.metrics.json"
    cmd = template.format(trace=str(trace), output_dir=str(out_dir), metrics_json=str(metrics_path), case=case)
    proc = subprocess.run(cmd, shell=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"External simulator command failed for {case}: {cmd}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"External simulator did not write metrics JSON: {metrics_path}")
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate HBM service curves from trace CSVs.")
    parser.add_argument("--trace_summary", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--hbm_peak_tbps", type=float, default=4.8)
    parser.add_argument("--dense_util", type=float, default=0.47)
    parser.add_argument("--cacheline_bytes", type=int, default=128)
    parser.add_argument("--simulator_cmd_template", default="")
    parser.add_argument("--ramulator_exec", default="")
    parser.add_argument("--ramulator_channels", type=int, default=1)
    args = parser.parse_args()

    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    summaries = json.loads(Path(args.trace_summary).read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []

    for item in summaries:
        case = item["case"]
        trace = Path(item["trace"])
        local = _read_trace_metrics(trace, args.cacheline_bytes)
        external: dict[str, Any] = {}
        if args.ramulator_exec:
            external = run_hbm_trace(
                trace,
                root / "ramulator",
                ramulator_exec=args.ramulator_exec,
                cacheline_bytes=args.cacheline_bytes,
                channels=args.ramulator_channels,
            ).to_dict()
        elif args.simulator_cmd_template:
            external = _run_external(args.simulator_cmd_template, trace, root / "external", case)

        pattern = item["pattern"]
        batch_size = int(item["batch_size"])
        if external:
            effective_gbps = float(external["effective_gbps"])
            dram_gbps = float(external.get("dram_gbps", effective_gbps * local["amplification_est"]))
            source = str(external.get("source", "external"))
        else:
            if pattern == "dense":
                util = args.dense_util
            elif pattern == "page_grouped":
                util = min(0.42, _interp_util(batch_size, DEFAULT_SPARSE_UTIL_BY_BATCH) * 1.45)
            else:
                util = _interp_util(batch_size, DEFAULT_SPARSE_UTIL_BY_BATCH)
            effective_gbps = args.hbm_peak_tbps * 1000.0 * util
            dram_gbps = effective_gbps * local["amplification_est"]
            source = "local_calibrated"

        kernel_us = local["requested_bytes"] / max(effective_gbps * 1e9, 1.0) * 1e6
        rows.append(
            {
                "case": case,
                "source": source,
                "pattern": pattern,
                "batch_size": batch_size,
                "context_len": int(item["context_len"]),
                "topk": int(item["topk"]),
                "effective_gbps": effective_gbps,
                "dram_gbps": dram_gbps,
                "dram_util": dram_gbps / (args.hbm_peak_tbps * 1000.0),
                "kernel_us": kernel_us,
                **local,
            }
        )

    csv_path = root / "hbm_service_curve.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "hbm_peak_tbps": args.hbm_peak_tbps,
        "dense_util": args.dense_util,
        "rows": rows,
    }
    (root / "hbm_service_curve.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(csv_path)


if __name__ == "__main__":
    main()
