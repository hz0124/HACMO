#!/usr/bin/env python3
"""Compare keyed numeric fields in reproduced and expected CSV tables."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def read_rows(path: Path, keys: list[str]) -> dict[tuple[str, ...], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    indexed: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        key = tuple(row[name] for name in keys)
        if key in indexed:
            raise ValueError(f"duplicate key {key} in {path}")
        indexed[key] = row
    return indexed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--keys", nargs="+", required=True)
    parser.add_argument("--metrics", nargs="+", required=True)
    parser.add_argument("--rtol", type=float, default=0.05)
    parser.add_argument("--atol", type=float, default=1e-6)
    args = parser.parse_args()

    expected = read_rows(args.expected, args.keys)
    actual = read_rows(args.actual, args.keys)
    failures: list[str] = []
    extra = set(actual) - set(expected)
    if extra:
        print(f"note: actual table contains {len(extra)} additional rows")
    for key, expected_row in expected.items():
        actual_row = actual.get(key)
        if actual_row is None:
            failures.append(f"missing row {key}")
            continue
        for metric in args.metrics:
            reference = float(expected_row[metric])
            observed = float(actual_row[metric])
            if not math.isclose(observed, reference, rel_tol=args.rtol, abs_tol=args.atol):
                failures.append(
                    f"{key} {metric}: actual={observed:.9g} expected={reference:.9g}"
                )
    if failures:
        print(f"FAIL: {len(failures)} mismatches")
        for failure in failures[:40]:
            print(f"  {failure}")
        raise SystemExit(1)
    print(f"PASS: {args.actual} ({len(expected)} rows, metrics={','.join(args.metrics)})")


if __name__ == "__main__":
    main()
