#!/usr/bin/env python3
"""TP2/TP3 motivation as three segments, derived from the confirmed idealLB pack.

The idealLB pack already decomposes the naive rail communication exactly:

    physical_lower_bound + hacmo_residual + challenge2 + challenge3 = naive

``hacmo_residual`` is not an unexplained remainder -- it is the distance between
HACMO's own result and the physical bound, i.e. the part of the placement problem
our method has not yet removed.  It therefore belongs with the challenge it comes
from, not with the ideal case.  Folding it into the TP2 segment gives three
segments that still close exactly:

    ideal        = physical_lower_bound   (capacity + hop-latency floor)
    tp2_challenge = challenge2 + hacmo_residual
    tp3_challenge = challenge3

No new simulation is required: every input is a column of the confirmed pack.

Reads the idealLB pack; writes only into the output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DEFAULT_SRC = (REPO / "hacmo/moe/motivation"
               / "tp2_serving_epoch_overhead_breakdown_idealLB_20260904"
               / "tp2_ideal_lower_bound.csv")
TOPO_ORDER = ["16w_4x4_rail_500", "4w_8x8_rail_900"]
TOPO_TITLES = {"16w_4x4_rail_500": "16W 4x4", "4w_8x8_rail_900": "4W 8x8"}


def build(src: Path) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for r in csv.DictReader(src.open(encoding="utf-8")):
        f = lambda k: float(r[k])  # noqa: E731
        ideal = f("physical_lower_bound_us")
        residual = f("hacmo_residual_us")
        c2 = f("challenge2_overhead_us")
        c3 = f("challenge3_overhead_us")
        naive = f("naive_rail_comm_us")

        tp2 = c2 + residual
        tp3 = c3
        total = ideal + tp2 + tp3
        if abs(total - naive) > 1e-6 * max(naive, 1.0):
            raise ValueError(
                f"{r['topology']} b{r['batch_size']}: segments {total:.6f} do not "
                f"close on naive {naive:.6f}"
            )
        if min(ideal, tp2, tp3) < 0:
            raise ValueError(
                f"{r['topology']} b{r['batch_size']}: negative segment"
            )
        out.append({
            "topology": r["topology"],
            "batch_size": int(r["batch_size"]),
            "ideal_us": f"{ideal:.6f}",
            "tp2_challenge_us": f"{tp2:.6f}",
            "tp3_challenge_us": f"{tp3:.6f}",
            "naive_us": f"{naive:.6f}",
            "naive_over_ideal_x": f"{naive / ideal:.6f}",
            "ideal_share_pct": f"{100.0 * ideal / naive:.6f}",
            "tp2_share_pct": f"{100.0 * tp2 / naive:.6f}",
            "tp3_share_pct": f"{100.0 * tp3 / naive:.6f}",
            "challenge_overhead_pct_vs_ideal": f"{100.0 * (tp2 + tp3) / ideal:.6f}",
            # provenance of the fold
            "src_challenge2_us": f"{c2:.6f}",
            "src_hacmo_residual_us": f"{residual:.6f}",
            "src_challenge3_us": f"{c3:.6f}",
            "src_full_tp2_comm_us": r["full_tp2_comm_us"],
            "bound_definition": r["bound_definition"],
        })
    if not out:
        raise ValueError(f"no rows read from {src}")
    return out


def plot(rows: list[dict[str, object]], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    topos = [t for t in TOPO_ORDER if any(r["topology"] == t for r in rows)]
    cols = {"ideal": "#b0bec5", "tp2": "#4e79a7", "tp3": "#f28e2b"}
    fig, axes = plt.subplots(1, len(topos), figsize=(5.1 * len(topos), 3.6),
                             squeeze=False)
    handles = labels = None
    for ax, topo in zip(axes[0], topos):
        sel = sorted([r for r in rows if r["topology"] == topo],
                     key=lambda r: int(r["batch_size"]))
        x = np.arange(len(sel))
        ideal = [float(r["ideal_us"]) for r in sel]
        tp2 = [float(r["tp2_challenge_us"]) for r in sel]
        tp3 = [float(r["tp3_challenge_us"]) for r in sel]
        b1 = ax.bar(x, ideal, 0.6, color=cols["ideal"], edgecolor="#37474f", lw=0.6,
                    label="Ideal (physical lower bound)")
        b2 = ax.bar(x, tp2, 0.6, bottom=ideal, color=cols["tp2"], edgecolor="#37474f",
                    lw=0.6, hatch="xx",
                    label="Challenge 2: expert placement (cross-wafer + multi-hop)")
        base = [a + b for a, b in zip(ideal, tp2)]
        b3 = ax.bar(x, tp3, 0.6, bottom=base, color=cols["tp3"], edgecolor="#37474f",
                    lw=0.6, hatch="//",
                    label="Challenge 3: wafer-edge I/O bandwidth imbalance")
        for xi, r in zip(x, sel):
            tot = float(r["naive_us"])
            ax.annotate(f"{float(r['naive_over_ideal_x']):.1f}x", (xi, tot),
                        xytext=(0, 4), textcoords="offset points", ha="center",
                        fontsize=8.5, weight="bold")
        ax.set_xticks(x, [str(int(r["batch_size"])) for r in sel])
        ax.set_xlabel("Batch per die")
        ax.set_title(TOPO_TITLES.get(topo, topo), loc="left", fontsize=10.5)
        ax.grid(axis="y", color="#d1d5db", lw=0.6, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        if handles is None:
            handles, labels = [b1, b2, b3], [h.get_label() for h in (b1, b2, b3)]
    axes[0][0].set_ylabel("MoE communication (us per decode step)")
    fig.legend(handles, labels, ncol=1, loc="upper center", frameon=False, fontsize=8)
    fig.subplots_adjust(top=0.70, bottom=0.15, wspace=0.22)
    fig.savefig(out_dir / "tp2_tp3_motivation.png", dpi=360, bbox_inches="tight")
    fig.savefig(out_dir / "tp2_tp3_motivation.pdf", bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def write_readme(out_dir: Path, rows: list[dict[str, object]], src: Path) -> None:
    mult = [float(r["naive_over_ideal_x"]) for r in rows]
    lines = [
        "# TP2/TP3 Motivation: Three Segments",
        "",
        "Heterogeneity-unaware MoE mapping against a physical lower bound, split",
        "into the two challenges HACMO addresses.",
        "",
        "## Segments",
        "",
        "```",
        "ideal         = physical lower bound (fabric capacity + hop-latency floor)",
        "tp2_challenge = expert placement: forced cross-wafer volume and extra hops",
        "tp3_challenge = wafer-edge rail I/O bandwidth imbalance",
        "ideal + tp2_challenge + tp3_challenge = naive        (exact)",
        "```",
        "",
        "Closure is checked to 1e-6 relative on every row.",
        "",
        "## Derivation",
        "",
        f"All values come from `{src.parent.name}/{src.name}`, which decomposes the",
        "same naive latency into four parts:",
        "",
        "```",
        "physical_lower_bound + hacmo_residual + challenge2 + challenge3 = naive",
        "```",
        "",
        "`hacmo_residual` is the gap between HACMO's own result and the bound -- the",
        "part of the placement problem our method has not yet removed. It is a",
        "property of the challenge, not of the ideal case, so it is folded into",
        "`tp2_challenge`. Keeping it separate would have implied the ideal case must",
        "pay it; moving it into `ideal` would have made the reference HACMO's own",
        "result rather than physics.",
        "",
        "Both source columns are retained (`src_challenge2_us`,",
        "`src_hacmo_residual_us`) so the fold can be audited or reversed.",
        "",
        "## Result",
        "",
        f"Naive is {min(mult):.2f}x to {max(mult):.2f}x the physical bound, rising",
        "monotonically with batch size on both topologies.",
        "",
        "| Topology | Batch | Ideal (us) | TP2 challenge | TP3 challenge | Naive | Naive/Ideal |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['topology']} | {r['batch_size']} | {float(r['ideal_us']):.2f} | "
            f"{float(r['tp2_challenge_us']):.2f} | {float(r['tp3_challenge_us']):.2f} | "
            f"{float(r['naive_us']):.2f} | {float(r['naive_over_ideal_x']):.2f}x |"
        )
    lines += [
        "",
        "## Scope",
        "",
        "Communication only; H200 expert compute is excluded, as in the source pack.",
        "Lower bounds are not comparable across topologies: the 8x8 mesh has twice",
        "the mean hop count of 4x4, so 4W has a higher latency floor. Compare each",
        "topology only against its own naive baseline.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "PYTHONPATH=$PWD python3 hacmo/moe/tools/build_tp2_motivation_from_idealLB.py \\",
        f"  --output-dir {out_dir}",
        "```",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src-csv", type=Path, default=DEFAULT_SRC)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()

    rows = build(args.src_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "tp2_tp3_motivation.csv", rows)
    plot(rows, args.output_dir)
    write_readme(args.output_dir, rows, args.src_csv)
    (args.output_dir / "metadata.json").write_text(json.dumps({
        "status": "motivation_candidate",
        "segments": {
            "ideal": "physical lower bound: fabric capacity plus all-to-all hop-latency floor",
            "tp2_challenge": "challenge2 + hacmo_residual",
            "tp3_challenge": "challenge3",
            "closure": "exact, checked to 1e-6 relative",
        },
        "residual_fold": "hacmo_residual is the gap from HACMO's result to the bound; "
                         "it is part of the placement challenge, not a cost the ideal "
                         "case pays, so it is folded into tp2_challenge",
        "source_csv": str(args.src_csv),
        "new_simulation_required": False,
    }, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {args.output_dir}")
    for r in rows:
        print(f"  {r['topology']:18s} b={r['batch_size']:>4} "
              f"ideal={float(r['ideal_us']):7.2f} tp2={float(r['tp2_challenge_us']):7.2f} "
              f"tp3={float(r['tp3_challenge_us']):6.2f} naive={float(r['naive_us']):7.2f} "
              f"({float(r['naive_over_ideal_x']):.2f}x)")


if __name__ == "__main__":
    main()
