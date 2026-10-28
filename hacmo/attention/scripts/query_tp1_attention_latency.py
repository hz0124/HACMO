#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Query one TP1 attention latency from ablation CSV.")
    p.add_argument("--csv_path", required=True)
    p.add_argument("--context_len", type=int, required=True)
    p.add_argument("--array_rows", type=int, required=True)
    p.add_argument("--array_cols", type=int, required=True)
    p.add_argument("--batch_label", required=True, help="e.g. 16, 32, 64+CP2, 128+CP4")
    p.add_argument("--method", required=True, choices=["G", "B", "C", "S", "O"])
    p.add_argument("--field", default="T_attn_total_us")
    args = p.parse_args()

    csv_path = Path(args.csv_path).resolve()
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        if (
            int(r["context_len"]) == args.context_len
            and int(r["array_rows"]) == args.array_rows
            and int(r["array_cols"]) == args.array_cols
            and r["batch_label"] == args.batch_label
            and r["method"] == args.method
        ):
            out = {
                "context_len": args.context_len,
                "array_rows": args.array_rows,
                "array_cols": args.array_cols,
                "batch_label": args.batch_label,
                "method": args.method,
                "field": args.field,
                "value": float(r[args.field]),
            }
            print(json.dumps(out, ensure_ascii=False))
            return

    raise SystemExit("No matching row found.")


if __name__ == "__main__":
    main()

