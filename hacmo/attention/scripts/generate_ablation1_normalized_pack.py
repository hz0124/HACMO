#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


METHOD_ORDER = ["G", "B", "C", "S", "O"]
METHOD_LABEL = {
    "G": "GPU+vLLM",
    "B": "vLLM",
    "C": "Naive Buffer",
    "S": "Static only",
    "O": "Static+Prefetch (O)",
}
METHOD_COLOR = {
    "G": "#595959",
    "B": "#7f7f7f",
    "C": "#a0a0a0",
    "S": "#4C78A8",
    "O": "#F58518",
}
BATCH_KEEP = ["16", "32", "64+CP2", "128+CP4"]
BATCH_CANONICAL_ORDER = ["8", "16", "32", "64", "64+CP2", "128+CP4"]


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def pct_speedup_vs_c(lat_c: float, lat_x: float) -> float:
    return (lat_c / max(lat_x, 1e-12) - 1.0) * 100.0


def pct_latency_reduction_vs_c(lat_c: float, lat_x: float) -> float:
    return (1.0 - lat_x / max(lat_c, 1e-12)) * 100.0


def choose_batch_labels(rows: List[Dict[str, str]], spec: str) -> List[str]:
    present = []
    for r in rows:
        label = r["batch_label"]
        if label not in present:
            present.append(label)
    if spec == "all":
        return [x for x in BATCH_CANONICAL_ORDER if x in present] + [x for x in present if x not in BATCH_CANONICAL_ORDER]
    labels = [x.strip() for x in spec.split(",") if x.strip()]
    missing = [x for x in labels if x not in present]
    if missing:
        raise SystemExit(f"Missing batch_label(s) in input CSV: {missing}")
    return labels


def main():
    p = argparse.ArgumentParser(description="Generate normalized ablation1 figures and PPT-friendly tables.")
    p.add_argument("--input_csv", required=True, help="Path to ablation1_tp1.csv")
    p.add_argument("--output_root", required=True, help="Output directory for normalized pack")
    p.add_argument(
        "--batch_labels",
        default=",".join(BATCH_KEEP),
        help="Comma-separated batch labels to include, or 'all' to mirror the input CSV.",
    )
    args = p.parse_args()

    input_csv = Path(args.input_csv).resolve()
    output_root = Path(args.output_root).resolve()
    fig_dir = output_root / "figs"
    csv_dir = output_root / "csv"
    fig_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv(input_csv)
    batch_keep = choose_batch_labels(rows, args.batch_labels)
    rows = [r for r in rows if r["batch_label"] in batch_keep]

    # Build index for quick lookup
    idx = {}
    for r in rows:
        key = (int(r["context_len"]), int(r["array_rows"]), int(r["array_cols"]), r["batch_label"], r["method"])
        idx[key] = r

    contexts = [8192, 32768]
    arrays = [(20, 20), (68, 68)]

    normalized_rows: List[Dict] = []
    ppt_rows: List[Dict] = []

    for ctx in contexts:
        for ar, ac in arrays:
            x = np.arange(len(batch_keep))
            w = 0.15
            fig, ax = plt.subplots(figsize=(8.8, 4.9))
            y_max = 1.0

            # bars: each method normalized by C under same batch/config
            y_map = {m: [] for m in METHOD_ORDER}
            anno_lines = []
            for bi, batch_label in enumerate(batch_keep):
                lat_c = float(idx[(ctx, ar, ac, batch_label, "C")]["T_attn_total_us"])
                lat_o = float(idx[(ctx, ar, ac, batch_label, "O")]["T_attn_total_us"])
                spd_o = pct_speedup_vs_c(lat_c, lat_o)
                red_o = pct_latency_reduction_vs_c(lat_c, lat_o)
                anno_lines.append((bi, spd_o, red_o))
                for m in METHOD_ORDER:
                    lat_m = float(idx[(ctx, ar, ac, batch_label, m)]["T_attn_total_us"])
                    norm = lat_m / max(lat_c, 1e-12)
                    y_map[m].append(norm)
                    y_max = max(y_max, norm)

                    normalized_rows.append(
                        {
                            "context_len": ctx,
                            "array_rows": ar,
                            "array_cols": ac,
                            "batch_label": batch_label,
                            "method": m,
                            "latency_us": lat_m,
                            "baseline_c_latency_us": lat_c,
                            "normalized_to_C": norm,
                            "speedup_vs_C_percent": pct_speedup_vs_c(lat_c, lat_m),
                            "latency_reduction_vs_C_percent": pct_latency_reduction_vs_c(lat_c, lat_m),
                        }
                    )

                ppt_rows.append(
                    {
                        "context_len": ctx,
                        "array_size": f"{ar}x{ac}",
                        "batch_label": batch_label,
                        "C_us": lat_c,
                        "O_us": lat_o,
                        "O_norm_to_C": lat_o / max(lat_c, 1e-12),
                        "O_speedup_vs_C_percent": spd_o,
                        "O_latency_reduction_vs_C_percent": red_o,
                    }
                )

            for i, m in enumerate(METHOD_ORDER):
                ax.bar(x + (i - 2.0) * w, y_map[m], width=w, color=METHOD_COLOR[m], label=METHOD_LABEL[m])

            for bi, spd_o, red_o in anno_lines:
                ax.text(
                    x[bi],
                    y_max * 1.03,
                    f"O vs C: +{spd_o:.1f}% spd\n(-{red_o:.1f}% lat)",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    color="#C75C00",
                )

            ax.axhline(1.0, linestyle="--", linewidth=1.1, color="gray")
            ax.set_ylim(0.0, y_max * 1.20)
            ax.set_xticks(x, batch_keep)
            ax.set_xlabel("Batch group")
            ax.set_ylabel("Normalized attention latency (C = 1.0)")
            ax.set_title(f"Ablation-1 Normalized (ctx={ctx//1024}k, array={ar}x{ac})")
            ax.grid(axis="y", alpha=0.3)
            ax.legend(ncol=3, fontsize=9)
            fig.tight_layout()
            fig.savefig(fig_dir / f"ablation_norm_ctx{ctx}_array{ar}x{ac}.png", dpi=220)
            plt.close(fig)

    write_csv(csv_dir / "ablation1_normalized_full.csv", normalized_rows)
    write_csv(csv_dir / "ablation1_ppt_values.csv", ppt_rows)

    md_path = output_root / "README_PPT_DATA.md"
    lines = []
    lines.append("# attention Normalized Figure Data (PPT)")
    lines.append("")
    lines.append("This pack is generated from `ablation1_tp1.csv` without re-running experiments.")
    lines.append("")
    lines.append("## Figure files")
    lines.append("")
    lines.append("- `figs/ablation_norm_ctx8192_array20x20.png`")
    lines.append("- `figs/ablation_norm_ctx8192_array68x68.png`")
    lines.append("- `figs/ablation_norm_ctx32768_array20x20.png`")
    lines.append("- `figs/ablation_norm_ctx32768_array68x68.png`")
    lines.append("")
    lines.append("## Normalization rule")
    lines.append("")
    lines.append("- For each `(context, array, batch_label)`, normalize by Naive Buffer (method `C`):")
    lines.append("- `normalized_to_C(method) = T_attn_total_us(method) / T_attn_total_us(C)`")
    lines.append("- Dashed line `y=1.0` means equal to `C`.")
    lines.append(f"- Included batch groups: `{', '.join(batch_keep)}`.")
    lines.append("")
    lines.append("## Annotation rule on figure")
    lines.append("")
    lines.append("- Text above each batch group is based on `O` vs `C`:")
    lines.append("- `speedup_percent = (C/O - 1) * 100`")
    lines.append("- `latency_reduction_percent = (1 - O/C) * 100`")
    lines.append("")
    lines.append("## CSV for PPT")
    lines.append("")
    lines.append("- `csv/ablation1_normalized_full.csv`: all methods (`G/B/C/S/O`) normalized data.")
    lines.append("- `csv/ablation1_ppt_values.csv`: compact table only for `O vs C` (recommended for labels).")
    lines.append("")
    lines.append("## Methods order in bars")
    lines.append("")
    lines.append("- `GPU+vLLM`, `vLLM`, `Naive Buffer`, `Static only`, `Static+Prefetch (O)`")
    lines.append("")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Generated normalized pack at: {output_root}")


if __name__ == "__main__":
    main()
