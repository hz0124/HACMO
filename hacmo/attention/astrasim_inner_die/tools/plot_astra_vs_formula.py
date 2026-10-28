#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", required=True)
    p.add_argument("--output_png", required=True)
    args = p.parse_args()

    rows = list(csv.DictReader(Path(args.input_csv).open(encoding="utf-8")))
    rows = [r for r in rows if r["kind"] == "ablation" and r["baseline"] in {"B", "D", "O"} and r["batch_size"] in {"32", "64"}]
    order = [("32", "B"), ("32", "D"), ("32", "O"), ("64", "B"), ("64", "D"), ("64", "O")]
    mp = {(r["batch_size"], r["baseline"]): r for r in rows}
    labels = [f"bs{bs}-{b}" for bs, b in order]
    astra = [float(mp[(bs, b)]["astra_wall_us"]) for bs, b in order]
    formula = [float(mp[(bs, b)]["formula_total_us"]) for bs, b in order]

    x = list(range(len(labels)))
    w = 0.36
    plt.figure(figsize=(8, 4.8))
    plt.bar([i - w / 2 for i in x], formula, width=w, label="Formula model", color="#7F7F7F")
    plt.bar([i + w / 2 for i in x], astra, width=w, label="ASTRA (qkv-only fragmented)", color="#4C78A8")
    plt.xticks(x, labels)
    plt.ylabel("Attention latency (us)")
    plt.title("Formula vs ASTRA (QKV-only Fragmentation)")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    out = Path(args.output_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=220)
    plt.close()
    print(out)


if __name__ == "__main__":
    main()
