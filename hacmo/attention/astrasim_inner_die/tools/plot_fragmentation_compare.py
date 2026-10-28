#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_rows(path: Path):
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description="Plot ASTRA fragmentation comparison.")
    parser.add_argument("--like_csv", required=True)
    parser.add_argument("--frag_csv", required=True)
    parser.add_argument("--output_png", required=True)
    args = parser.parse_args()

    like = read_rows(Path(args.like_csv))
    frag = read_rows(Path(args.frag_csv))

    def pick(rows):
        out = {}
        for r in rows:
            if r["kind"] == "ablation" and r["batch_size"] == "32" and r["baseline"] in {"B", "D", "O"}:
                out[r["baseline"]] = float(r["astra_wall_us"])
        return out

    like_v = pick(like)
    frag_v = pick(frag)
    order = ["B", "D", "O"]
    x = list(range(len(order)))
    w = 0.35

    plt.figure(figsize=(7.2, 4.8))
    plt.bar([i - w / 2 for i in x], [like_v[b] for b in order], width=w, label="Original model")
    plt.bar([i + w / 2 for i in x], [frag_v[b] for b in order], width=w, label="Fragmented model (4KB, 25% payload)")
    plt.xticks(x, order)
    plt.ylabel("Attention latency (us), bs=32")
    plt.title("ASTRA HBM Model Comparison")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()

    b_like = like_v["B"]
    d_like = like_v["D"]
    o_like = like_v["O"]
    b_frag = frag_v["B"]
    d_frag = frag_v["D"]
    o_frag = frag_v["O"]
    txt = (
        f"Original: D/B={((b_like-d_like)/b_like*100):.1f}%, O/B={((b_like-o_like)/b_like*100):.1f}%\n"
        f"Fragmented: D/B={((b_frag-d_frag)/b_frag*100):.1f}%, O/B={((b_frag-o_frag)/b_frag*100):.1f}%"
    )
    plt.text(0.02, 0.98, txt, transform=plt.gca().transAxes, va="top", ha="left", fontsize=9,
             bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "gray"})

    out = Path(args.output_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()
    print(out)


if __name__ == "__main__":
    main()
