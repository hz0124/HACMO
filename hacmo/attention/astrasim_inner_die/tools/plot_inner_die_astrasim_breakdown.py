#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def _read_rows(csv_path: Path):
    with csv_path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(x, default=0.0):
    if x is None or x == "":
        return default
    return float(x)


def plot_ablation_total(rows, out_path: Path):
    rows = [r for r in rows if r["kind"] == "ablation" and r["baseline"] in {"B", "D", "O"}]
    baselines = ["B", "D", "O"]
    by_b = {b: sorted([r for r in rows if r["baseline"] == b], key=lambda t: int(t["batch_size"])) for b in baselines}
    plt.figure(figsize=(7, 4.5))
    for b in baselines:
        xs = [int(r["batch_size"]) for r in by_b[b]]
        ys = [_to_float(r["astra_wall_us"]) for r in by_b[b]]
        plt.plot(xs, ys, marker="o", linewidth=2, label=b)
    plt.xlabel("Batch size")
    plt.ylabel("Attention latency (us)")
    plt.title("ASTRA-based Ablation: Total Attention Latency")
    plt.grid(alpha=0.25)
    plt.legend(title="Baseline")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_ablation_breakdown(rows, out_path: Path):
    rows = [r for r in rows if r["kind"] == "ablation" and r["baseline"] in {"B", "D", "O"} and r["batch_size"] in {"32", "64"}]
    order = [("32", "B"), ("32", "D"), ("32", "O"), ("64", "B"), ("64", "D"), ("64", "O")]
    row_map = {(r["batch_size"], r["baseline"]): r for r in rows}

    labels = []
    hbm = []
    remote = []
    for bs, b in order:
        r = row_map[(bs, b)]
        labels.append(f"bs{bs}-{b}")
        hbm.append(_to_float(r["astra_hbm_us"]))
        remote.append(_to_float(r["astra_remote_sram_us"]))

    x = range(len(labels))
    plt.figure(figsize=(8, 4.8))
    plt.bar(x, hbm, label="HBM-like (MEM_LOAD)", color="#4C78A8")
    plt.bar(x, remote, bottom=hbm, label="Remote-SRAM-like (mesh COMM)", color="#F58518")
    plt.xticks(list(x), labels)
    plt.ylabel("Latency (us)")
    plt.title("ASTRA-based Ablation Breakdown (bs=32/64)")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_motivation(rows, out_path: Path):
    rows = [r for r in rows if r["kind"] == "motivation"]
    ratios = sorted({float(r["cache_ratio"]) for r in rows})
    batches = sorted({int(r["batch_size"]) for r in rows})
    by_ratio = {c: sorted([r for r in rows if float(r["cache_ratio"]) == c], key=lambda t: int(t["batch_size"])) for c in ratios}

    plt.figure(figsize=(7, 4.5))
    for c in ratios:
        xs = [int(r["batch_size"]) for r in by_ratio[c]]
        ys = [_to_float(r["astra_wall_us"]) for r in by_ratio[c]]
        plt.plot(xs, ys, marker="o", linewidth=2, label=f"C-{int(c*100)}%")
    plt.xlabel("Batch size")
    plt.ylabel("Attention latency (us)")
    plt.title("ASTRA-based Motivation: Baseline C with SRAM Ratios")
    plt.grid(alpha=0.25)
    plt.legend(title="Cache ratio")
    plt.xticks(batches)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    csv_path = Path(args.input_csv).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_rows(csv_path)

    plot_ablation_total(rows, out_dir / "ablation_total_latency_astra.png")
    plot_ablation_breakdown(rows, out_dir / "ablation_breakdown_bs32_bs64_astra.png")
    plot_motivation(rows, out_dir / "motivation_cache_ratio_latency_astra.png")
    print(f"Saved figures to: {out_dir}")


if __name__ == "__main__":
    main()
