#!/usr/bin/env python3
"""TP2/TP3 motivation, three segments against a physical lower bound.

Combines the two things the other two variants each got half right:

  * the reference is the analytic physical lower bound, not HACMO's own result,
    so the figure compares heterogeneity-unaware mapping against physics;
  * the part HACMO has not yet removed is folded into the two challenges using
    weights measured from the TP3 pack's four corners, so no weight is chosen by
    hand.

    ideal         = physical_lower_bound
    remainder     = full_tp2 - physical_lower_bound      (what HACMO has not removed)
    w_tp2         = shapley_tp2 / (shapley_tp2 + shapley_tp3)
    tp2_challenge = (naive - full_tp2) x w_tp2 + remainder x w_tp2
    tp3_challenge = (naive - full_tp2) x w_tp3 + remainder x w_tp3

Equivalently both the removed overhead and the remainder are apportioned by the
same measured ratio, so:

    ideal + tp2_challenge + tp3_challenge = naive        (exact)

The Shapley shares come from four measured corners of the TP3 pack
(naive_rail / rail_tp2 / crossbar_tp2 / full_tp2), averaging both orderings of
the two design choices. That fixes the interaction between them without an
assumption, which is what makes the weights defensible.

Known limitation
----------------
The analytic bound's ``min_cross_wafer_fraction`` assumes one expert per die.
HACMO uses ~96-100 extra replicas and its measured cross-wafer share (0.90 on
16w, 0.69 on 4w) is *below* that supposed minimum, so the bound is not tight for
a replicated placement: it sits higher than a true bound would. That makes the
ideal segment larger and the challenges smaller, i.e. the figure understates the
opportunity rather than overstating it.

Reads confirmed packs and the two sibling motivation CSVs; writes only to the
output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
MOTIV = REPO / "hacmo/moe/motivation"
DEFAULT_CORNERS = MOTIV / "tp2_tp3_motivation_corners_20260905" / "tp2_tp3_motivation.csv"
DEFAULT_BOUND = (MOTIV / "tp2_serving_epoch_overhead_breakdown_idealLB_20260904"
                 / "tp2_ideal_lower_bound.csv")
TOPO_ORDER = ["16w_4x4_rail_500", "4w_8x8_rail_900"]
TOPO_TITLES = {"16w_4x4_rail_500": "16W 4x4", "4w_8x8_rail_900": "4W 8x8"}


def key_of(row: dict) -> tuple[str, int]:
    return row["topology"], int(float(row["batch_size"]))


def build(args: argparse.Namespace) -> list[dict[str, object]]:
    corners = {key_of(r): r for r in csv.DictReader(args.corners_csv.open(encoding="utf-8"))}
    bounds = {key_of(r): r for r in csv.DictReader(args.bound_csv.open(encoding="utf-8"))}

    rows: list[dict[str, object]] = []
    for k in sorted(corners, key=lambda x: (TOPO_ORDER.index(x[0])
                                            if x[0] in TOPO_ORDER else 9, x[1])):
        if k not in bounds:
            raise KeyError(f"{k}: missing from the bound CSV")
        c, b = corners[k], bounds[k]

        naive = float(c["v00_naive_rail_us"])
        full = float(c["v11_full_tp2_us"])
        lb = float(b["physical_lower_bound_us"])

        # The two CSVs must describe the same operating point.
        naive_b = float(b["naive_rail_comm_us"])
        if abs(naive - naive_b) > 1e-6 * max(naive, 1.0):
            raise ValueError(
                f"{k}: naive differs between corner ({naive:.6f}) and bound "
                f"({naive_b:.6f}) inputs"
            )
        if lb > full:
            raise ValueError(
                f"{k}: bound {lb:.6f} exceeds HACMO's own result {full:.6f}; "
                "the bound is not valid at this point"
            )

        s2 = float(c["tp2_challenge_us"])
        s3 = float(c["tp3_challenge_us"])
        if s2 + s3 <= 0:
            raise ValueError(f"{k}: non-positive Shapley shares")
        w2 = s2 / (s2 + s3)
        w3 = 1.0 - w2

        gap = naive - lb          # everything above the bound
        tp2 = gap * w2
        tp3 = gap * w3
        remainder = full - lb

        total = lb + tp2 + tp3
        if abs(total - naive) > 1e-6 * max(naive, 1.0):
            raise ValueError(f"{k}: closure {total:.6f} != {naive:.6f}")
        if min(lb, tp2, tp3) < 0:
            raise ValueError(f"{k}: negative segment")

        rows.append({
            "topology": k[0],
            "batch_size": k[1],
            "ideal_us": f"{lb:.6f}",
            "tp2_challenge_us": f"{tp2:.6f}",
            "tp3_challenge_us": f"{tp3:.6f}",
            "naive_us": f"{naive:.6f}",
            "naive_over_ideal_x": f"{naive / lb:.6f}",
            "ideal_share_pct": f"{100.0 * lb / naive:.6f}",
            "tp2_share_pct": f"{100.0 * tp2 / naive:.6f}",
            "tp3_share_pct": f"{100.0 * tp3 / naive:.6f}",
            "challenge_overhead_pct_vs_ideal": f"{100.0 * (tp2 + tp3) / lb:.6f}",
            # how the weights were obtained
            "weight_tp2": f"{w2:.6f}",
            "weight_tp3": f"{w3:.6f}",
            "shapley_tp2_us": f"{s2:.6f}",
            "shapley_tp3_us": f"{s3:.6f}",
            # what HACMO has and has not removed, for auditing
            "hacmo_result_us": f"{full:.6f}",
            "removed_by_hacmo_us": f"{naive - full:.6f}",
            "not_removed_by_hacmo_us": f"{remainder:.6f}",
            "not_removed_share_of_gap_pct": f"{100.0 * remainder / gap:.6f}",
        })
    if not rows:
        raise ValueError("no rows built")
    return rows


def plot(rows: list[dict[str, object]], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    topos = [t for t in TOPO_ORDER if any(r["topology"] == t for r in rows)]
    cols = {"i": "#b0bec5", "t2": "#4e79a7", "t3": "#f28e2b"}
    fig, axes = plt.subplots(1, len(topos), figsize=(5.2 * len(topos), 3.7),
                             squeeze=False)
    h = l = None
    for ax, topo in zip(axes[0], topos):
        sel = sorted([r for r in rows if r["topology"] == topo],
                     key=lambda r: int(r["batch_size"]))
        x = np.arange(len(sel))
        i = [float(r["ideal_us"]) for r in sel]
        t2 = [float(r["tp2_challenge_us"]) for r in sel]
        t3 = [float(r["tp3_challenge_us"]) for r in sel]
        b1 = ax.bar(x, i, 0.6, color=cols["i"], edgecolor="#37474f", lw=0.6,
                    label="Ideal (physical lower bound)")
        b2 = ax.bar(x, t2, 0.6, bottom=i, color=cols["t2"], edgecolor="#37474f",
                    lw=0.6, hatch="xx",
                    label="Challenge 2: expert placement (cross-wafer + multi-hop)")
        base = [a + b for a, b in zip(i, t2)]
        b3 = ax.bar(x, t3, 0.6, bottom=base, color=cols["t3"], edgecolor="#37474f",
                    lw=0.6, hatch="//",
                    label="Challenge 3: static wafer-edge I/O ports")
        # mark where HACMO currently sits, so the figure shows both the
        # opportunity and how much of it is already captured
        for xi, r in zip(x, sel):
            ax.plot([xi - 0.32, xi + 0.32], [float(r["hacmo_result_us"])] * 2,
                    color="#2e7d32", lw=1.6, ls="--", zorder=5,
                    label="HACMO today" if xi == 0 and topo == topos[0] else None)
            ax.annotate(f"{float(r['naive_over_ideal_x']):.1f}x",
                        (xi, float(r["naive_us"])), xytext=(0, 4),
                        textcoords="offset points", ha="center",
                        fontsize=8.5, weight="bold")
        ax.set_xticks(x, [str(int(r["batch_size"])) for r in sel])
        ax.set_xlabel("Batch per die")
        ax.set_title(TOPO_TITLES.get(topo, topo), loc="left", fontsize=10.5)
        ax.grid(axis="y", color="#d1d5db", lw=0.6, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        if h is None:
            hh, ll = ax.get_legend_handles_labels()
            h, l = [b1, b2, b3] + [x for x in hh if x.get_label() == "HACMO today"], \
                   ["Ideal (physical lower bound)",
                    "Challenge 2: expert placement (cross-wafer + multi-hop)",
                    "Challenge 3: static wafer-edge I/O ports",
                    "HACMO today"]
    axes[0][0].set_ylabel("MoE communication (us per decode step)")
    fig.legend(h, l, ncol=2, loc="upper center", frameon=False, fontsize=8)
    fig.subplots_adjust(top=0.74, bottom=0.15, wspace=0.22)
    fig.savefig(out_dir / "tp2_tp3_motivation.png", dpi=360, bbox_inches="tight")
    fig.savefig(out_dir / "tp2_tp3_motivation.pdf", bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def write_readme(out_dir: Path, rows: list[dict[str, object]], args) -> None:
    mult = [float(r["naive_over_ideal_x"]) for r in rows]
    w2 = [float(r["weight_tp2"]) for r in rows]
    nr = [float(r["not_removed_share_of_gap_pct"]) for r in rows]
    lines = [
        "# TP2/TP3 Motivation (final): Three Segments Against A Physical Bound",
        "",
        "Heterogeneity-unaware MoE mapping measured against a physical lower bound,",
        "split into the two challenges HACMO addresses. Each challenge segment is",
        "how large that challenge *is* -- including the part HACMO has not yet",
        "removed -- so the figure states an opportunity, not a scoreboard.",
        "",
        "## Segments",
        "",
        "```",
        "ideal         = physical lower bound (fabric capacity + hop-latency floor)",
        "gap           = naive - ideal",
        "tp2_challenge = gap x w_tp2",
        "tp3_challenge = gap x w_tp3",
        "ideal + tp2_challenge + tp3_challenge = naive      (exact, 1e-6)",
        "```",
        "",
        "## Where The Weights Come From",
        "",
        "`w_tp2 : w_tp3` is the ratio of the two Shapley shares measured over the",
        "TP3 pack's four corners:",
        "",
        "```",
        "V00 = naive_rail      V10 = rail_tp2",
        "V01 = crossbar_tp2    V11 = full_tp2",
        "",
        "shapley_tp2 = 1/2 [ (V00-V10) + (V01-V11) ]",
        "shapley_tp3 = 1/2 [ (V00-V01) + (V10-V11) ]",
        "```",
        "",
        "Averaging both orderings resolves the interaction between the two design",
        "choices without an assumption, so the split is derived from measured",
        "configurations rather than chosen. All four corners are bars the confirmed",
        "TP3 pack already contains; no new simulation is needed.",
        "",
        f"Measured weights: w_tp2 ranges {min(w2):.2f}-{max(w2):.2f}. On 16W the edge",
        "I/O term dominates (only 12 of 16 dies own a rail port); on 4W the two are",
        "close to even.",
        "",
        "## Ideal Versus HACMO Today",
        "",
        "The dashed line marks HACMO's measured result (`full_tp2`). The distance",
        "from it down to the ideal is what remains unsolved:",
        f"{min(nr):.0f}%-{max(nr):.0f}% of the total gap above the bound. That portion",
        "stays inside the challenge segments, since it is difficulty that exists",
        "regardless of how much of it we have captured.",
        "",
        "## Known Limitation",
        "",
        "The bound's `min_cross_wafer_fraction` assumes one expert per die. HACMO",
        "uses ~96-100 extra replicas, and its measured cross-wafer share (0.90 on",
        "16W, 0.69 on 4W) is *below* that supposed minimum, so the bound is not",
        "tight for a replicated placement -- it sits higher than a true bound would.",
        "The effect is a larger ideal segment and smaller challenges, i.e. the",
        "figure understates the opportunity rather than overstating it.",
        "",
        "## Result",
        "",
        f"Naive is {min(mult):.2f}x to {max(mult):.2f}x the physical bound.",
        "",
        "| Topology | Batch | Ideal | TP2 challenge | TP3 challenge | Naive | Naive/Ideal | HACMO today | w_TP2 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['topology']} | {r['batch_size']} | {float(r['ideal_us']):.2f} | "
            f"{float(r['tp2_challenge_us']):.2f} | {float(r['tp3_challenge_us']):.2f} | "
            f"{float(r['naive_us']):.2f} | {float(r['naive_over_ideal_x']):.2f}x | "
            f"{float(r['hacmo_result_us']):.2f} | {float(r['weight_tp2']):.2f} |")
    lines += [
        "",
        "## Scope",
        "",
        "Communication only; H200 expert compute is excluded. Ideal values are not",
        "comparable across topologies, since the 8x8 mesh has twice the mean hop",
        "count of 4x4 and therefore a higher latency floor. Compare each topology",
        "against its own naive baseline.",
        "",
        "## Sibling Variants",
        "",
        "- `tp2_tp3_motivation_corners_20260905`: same four corners, but the ideal is",
        "  HACMO's own `full_tp2`. Mathematically the cleanest (pure Shapley, no",
        "  bound), but the challenges then measure only what we removed.",
        "- `tp2_serving_epoch_overhead_breakdown_idealLB_20260904`: the source of the",
        "  bound, as a four-segment breakdown with the remainder shown separately.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "PYTHONPATH=$PWD python3 \\",
        "  hacmo/moe/tools/build_tp2_motivation_final.py \\",
        f"  --output-dir {out_dir}",
        "```",
        "",
        "## Sources",
        "",
        f"- Corners / Shapley weights: `{args.corners_csv}`",
        f"- Physical lower bound: `{args.bound_csv}`",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corners-csv", type=Path, default=DEFAULT_CORNERS)
    p.add_argument("--bound-csv", type=Path, default=DEFAULT_BOUND)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()

    rows = build(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "tp2_tp3_motivation.csv", rows)
    plot(rows, args.output_dir)
    write_readme(args.output_dir, rows, args)
    (args.output_dir / "metadata.json").write_text(json.dumps({
        "status": "motivation_final_candidate",
        "segments": {
            "ideal": "analytic physical lower bound (capacity + all-to-all hop latency)",
            "tp2_challenge": "(naive - ideal) x w_tp2",
            "tp3_challenge": "(naive - ideal) x w_tp3",
            "closure": "exact, 1e-6 relative",
        },
        "weights": "ratio of Shapley shares over the TP3 pack's four measured "
                   "corners (naive_rail / rail_tp2 / crossbar_tp2 / full_tp2)",
        "challenges_include_unsolved_part": True,
        "new_simulation_required": False,
        "known_limitation": "the bound assumes one expert per die; HACMO's replicated "
                            "placement achieves a lower cross-wafer share, so the bound "
                            "is loose and the challenges are understated",
        "sources": {"corners_csv": str(args.corners_csv),
                    "bound_csv": str(args.bound_csv)},
    }, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {args.output_dir}")
    for r in rows:
        print(f"  {r['topology']:18s} b={r['batch_size']:>4} "
              f"ideal={float(r['ideal_us']):7.2f} tp2={float(r['tp2_challenge_us']):7.2f} "
              f"tp3={float(r['tp3_challenge_us']):7.2f} naive={float(r['naive_us']):7.2f} "
              f"({float(r['naive_over_ideal_x']):.2f}x)  w_tp2={float(r['weight_tp2']):.2f}")


if __name__ == "__main__":
    main()
