#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable


DEFAULT_METHODS = ("vllm", "moentwine")
DEFAULT_KEYS = ("method", "topology", "batch_size")
DEFAULT_METRICS = ("total_us", "compute_us", "communication_us")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _row_key(row: dict[str, str], key_columns: Iterable[str]) -> tuple[str, ...]:
    return tuple(str(row.get(k, "")) for k in key_columns)


def _as_float(value: str) -> float:
    return float(value.strip())


def _close(a: float, b: float, abs_tol: float, rel_tol: float) -> bool:
    return math.isclose(a, b, abs_tol=abs_tol, rel_tol=rel_tol)


def check_baseline_freeze(
    reference_rows: list[dict[str, str]],
    candidate_rows: list[dict[str, str]],
    *,
    methods: Iterable[str] = DEFAULT_METHODS,
    key_columns: Iterable[str] = DEFAULT_KEYS,
    metric_columns: Iterable[str] = DEFAULT_METRICS,
    abs_tol: float = 1e-6,
    rel_tol: float = 1e-6,
) -> list[str]:
    methods_set = set(methods)
    key_columns = tuple(key_columns)
    metric_columns = tuple(metric_columns)

    reference: dict[tuple[str, ...], dict[str, str]] = {}
    for row in reference_rows:
        if row.get("method") not in methods_set:
            continue
        key = _row_key(row, key_columns)
        if key in reference:
            raise ValueError(f"duplicate reference baseline key: {key}")
        reference[key] = row

    errors: list[str] = []
    checked = 0
    skipped = 0
    for row in candidate_rows:
        if row.get("method") not in methods_set:
            skipped += 1
            continue
        key = _row_key(row, key_columns)
        ref = reference.get(key)
        if ref is None:
            errors.append(f"candidate baseline row has no reference: key={key}")
            continue
        checked += 1
        for metric in metric_columns:
            if metric not in row or metric not in ref:
                continue
            if row[metric] == "" or ref[metric] == "":
                continue
            try:
                cur_v = _as_float(row[metric])
                ref_v = _as_float(ref[metric])
            except ValueError:
                if row[metric] != ref[metric]:
                    errors.append(
                        f"metric mismatch key={key} metric={metric}: "
                        f"candidate={row[metric]!r} reference={ref[metric]!r}"
                    )
                continue
            if not _close(cur_v, ref_v, abs_tol, rel_tol):
                errors.append(
                    f"metric drift key={key} metric={metric}: "
                    f"candidate={cur_v} reference={ref_v}"
                )

    if checked == 0:
        errors.append(
            "no candidate baseline rows were checked; this is allowed only for ours-only sweeps"
        )
    return errors


def main() -> None:
    p = argparse.ArgumentParser(description="Fail if TP2 baseline rows drift during HACMO-only experiments.")
    p.add_argument("--reference-csv", required=True)
    p.add_argument("--candidate-csv", nargs="+", required=True)
    p.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    p.add_argument("--key-columns", nargs="+", default=list(DEFAULT_KEYS))
    p.add_argument("--metric-columns", nargs="+", default=list(DEFAULT_METRICS))
    p.add_argument("--abs-tol", type=float, default=1e-6)
    p.add_argument("--rel-tol", type=float, default=1e-6)
    p.add_argument(
        "--allow-missing-baselines",
        action="store_true",
        help="Allow candidate CSVs that contain only HACMO rows and no baseline rows.",
    )
    args = p.parse_args()

    reference_rows = _read_rows(Path(args.reference_csv))
    all_errors: list[str] = []
    for candidate in args.candidate_csv:
        candidate_rows = _read_rows(Path(candidate))
        errors = check_baseline_freeze(
            reference_rows,
            candidate_rows,
            methods=args.methods,
            key_columns=args.key_columns,
            metric_columns=args.metric_columns,
            abs_tol=args.abs_tol,
            rel_tol=args.rel_tol,
        )
        if args.allow_missing_baselines:
            errors = [e for e in errors if not e.startswith("no candidate baseline rows")]
        if errors:
            all_errors.append(f"{candidate}:")
            all_errors.extend(f"  {e}" for e in errors)

    if all_errors:
        raise SystemExit("TP2 baseline freeze check failed:\n" + "\n".join(all_errors))
    print("TP2 baseline freeze check passed.")


if __name__ == "__main__":
    main()
