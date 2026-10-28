#!/usr/bin/env python3
"""TP2/TP3 motivation: two-factor (Shapley) split over the TP3 pack's four bars.

The TP3 pack already measures every combination of the two design choices, so no
new simulation and no synthetic configuration is needed:

    P = expert placement   (0 = uniform/naive, 1 = HACMO mapping)
    B = wafer edge I/O     (0 = static rail ports, 1 = edge-local crossbar)

    V00 = naive_rail        V10 = rail_tp2
    V01 = crossbar_tp2      V11 = full_tp2      == ideal

    tp2_challenge = 1/2 [ (V00-V10) + (V01-V11) ]
    tp3_challenge = 1/2 [ (V00-V01) + (V10-V11) ]
    ideal         = V11

    ideal + tp2_challenge + tp3_challenge = V00     (exact, algebraically)

Averaging both orderings is the Shapley value for two factors, so the joint
effect of the two causes is split without choosing a weight.  Because the
reference is V11, each challenge segment already includes the part HACMO has not
removed.

What TP3 actually changes
-------------------------
Its network config sets ``crossbar-mode: true`` and ``rail-optimized: false``:
each wafer edge owns a few external I/O ports that an edge-local crossbar may
reassign among the die groups on that same edge.  The naive fabric binds those
ports statically.  So the B axis is port *reassignability*, not port position.

An earlier attempt built the B axis by relocating rail ports to a uniform grid.
That was wrong twice over: it does not model the crossbar, and it invalidates
HACMO's placement, which routes to the real port positions.  It produced
V11 = 46.35 against the pack's true full_tp2 = 23.89, and a negative TP3 share.

Reads the confirmed TP3 pack; writes only to the output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DEFAULT_TP3 = (REPO / "hacmo/interconnect/outputs"
               / "tp3_lane_level_multibatch_final_candidate_tp2_20260831"
               / "tp3_metrics.csv")
DEFAULT_TP2_PACK = REPO / "hacmo/moe/outputs/tp2_serving_epoch_paper_pack_20260831"
CORNERS = {"v00": "naive_rail", "v10": "rail_tp2",
           "v01": "crossbar_tp2", "v11": "full_tp2"}
TOPO_ORDER = ["16w_4x4_rail_500", "4w_8x8_rail_900"]
TOPO_TITLES = {"16w_4x4_rail_500": "16W 4x4", "4w_8x8_rail_900": "4W 8x8"}


def read_corners(path: Path) -> dict[tuple[str, int], dict[str, float]]:
    out: dict[tuple[str, int], dict[str, float]] = {}
    for r in csv.DictReader(path.open(encoding="utf-8")):
        key = (r["topology"], int(float(r["batch_size"])))
        out.setdefault(key, {})[r["bar_key"]] = float(r["communication_us"])
    return out


def pack_naive(pack: Path) -> dict[tuple[str, int], float]:
    """Published vllm communication, to cross-check V00."""
    out: dict[tuple[str, int], float] = {}
    for name in ("tp2_high_drift_astra_16w_comm_only.csv",
                 "tp2_high_drift_astra_4w_comm_only.csv"):
        p = pack / "csv" / name
        if not p.is_file():
            continue
        for r in csv.DictReader(p.open(encoding="utf-8")):
            if r.get("method") == "vllm":
                out[(r["topology"], int(float(r["batch_per_die"])))] = \
                    float(r["communication_us"])
    return out


def build(args: argparse.Namespace) -> list[dict[str, object]]:
    corners = read_corners(args.tp3_metrics)
    naive_ref = pack_naive(args.tp2_pack)
    rows: list[dict[str, object]] = []

    for (topo, batch) in sorted(corners, key=lambda k: (TOPO_ORDER.index(k[0])
                                                        if k[0] in TOPO_ORDER else 9,
                                                        k[1])):
        bars = corners[(topo, batch)]
        missing = [b for b in CORNERS.values() if b not in bars]
        if missing:
            raise KeyError(f"{topo} b{batch}: TP3 pack lacks bars {missing}")
        v00, v10 = bars["naive_rail"], bars["rail_tp2"]
        v01, v11 = bars["crossbar_tp2"], bars["full_tp2"]

        # Cross-check V00 against the TP2 pack's published vllm row.
        ref = naive_ref.get((topo, batch))
        if ref is not None:
            rel = abs(v00 - ref) / ref
            if rel > 2e-3:
                raise ValueError(
                    f"{topo} b{batch}: TP3 naive_rail {v00:.6f} disagrees with TP2 "
                    f"pack vllm {ref:.6f} (rel {rel:.4%})"
                )
        else:
            rel = float("nan")

        tp2 = 0.5 * ((v00 - v10) + (v01 - v11))
        tp3 = 0.5 * ((v00 - v01) + (v10 - v11))
        total = v11 + tp2 + tp3
        if abs(total - v00) > 1e-6 * max(v00, 1.0):
            raise ValueError(f"{topo} b{batch}: closure {total:.6f} != {v00:.6f}")
        if min(tp2, tp3) < 0:
            raise ValueError(
                f"{topo} b{batch}: negative share (tp2={tp2:.3f}, tp3={tp3:.3f}); "
                "the two factors are not combining as expected"
            )
        rows.append({
            "topology": topo,
            "batch_size": batch,
            "ideal_us": f"{v11:.6f}",
            "tp2_challenge_us": f"{tp2:.6f}",
            "tp3_challenge_us": f"{tp3:.6f}",
            "naive_us": f"{v00:.6f}",
            "naive_over_ideal_x": f"{v00 / v11:.6f}",
            "ideal_share_pct": f"{100.0 * v11 / v00:.6f}",
            "tp2_share_pct": f"{100.0 * tp2 / v00:.6f}",
            "tp3_share_pct": f"{100.0 * tp3 / v00:.6f}",
            "challenge_overhead_pct_vs_ideal": f"{100.0 * (tp2 + tp3) / v11:.6f}",
            "v00_naive_rail_us": f"{v00:.6f}",
            "v10_rail_tp2_us": f"{v10:.6f}",
            "v01_crossbar_tp2_us": f"{v01:.6f}",
            "v11_full_tp2_us": f"{v11:.6f}",
            "tp2_first_gain_us": f"{v00 - v10:.6f}",
            "tp2_second_gain_us": f"{v01 - v11:.6f}",
            "tp3_first_gain_us": f"{v00 - v01:.6f}",
            "tp3_second_gain_us": f"{v10 - v11:.6f}",
            "tp2_pack_naive_rel_err": f"{rel:.3e}",
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
                    label="Ideal: HACMO placement + edge crossbar")
        b2 = ax.bar(x, t2, 0.6, bottom=i, color=cols["t2"], edgecolor="#37474f",
                    lw=0.6, hatch="xx",
                    label="Challenge 2: expert placement (cross-wafer + multi-hop)")
        base = [a + b for a, b in zip(i, t2)]
        b3 = ax.bar(x, t3, 0.6, bottom=base, color=cols["t3"], edgecolor="#37474f",
                    lw=0.6, hatch="//",
                    label="Challenge 3: static wafer-edge I/O ports")
        for xi, r in zip(x, sel):
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
            h, l = [b1, b2, b3], [z.get_label() for z in (b1, b2, b3)]
    axes[0][0].set_ylabel("MoE communication (us per decode step)")
    fig.legend(h, l, ncol=1, loc="upper center", frameon=False, fontsize=8)
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


def write_readme(out_dir: Path, rows: list[dict[str, object]], args) -> None:
    mult = [float(r["naive_over_ideal_x"]) for r in rows]
    lines = [
        "# TP2/TP3 Motivation: Two-Factor Split Over Four Measured Corners",
        "",
        "Heterogeneity-unaware MoE mapping against an ideal that applies both",
        "techniques, with the joint effect split by the Shapley value. No new",
        "simulation and no synthetic configuration: all four corners are bars the",
        "confirmed TP3 pack already measures.",
        "",
        "## Method",
        "",
        "```",
        "P = expert placement  (0 = uniform/naive, 1 = HACMO mapping)",
        "B = wafer edge I/O    (0 = static rail ports, 1 = edge-local crossbar)",
        "",
        "V00 = naive_rail       V10 = rail_tp2",
        "V01 = crossbar_tp2     V11 = full_tp2   == ideal",
        "",
        "tp2_challenge = 1/2 [ (V00-V10) + (V01-V11) ]",
        "tp3_challenge = 1/2 [ (V00-V01) + (V10-V11) ]",
        "ideal + tp2_challenge + tp3_challenge = V00     (exact)",
        "```",
        "",
        "Averaging both orderings splits the interaction without choosing a weight.",
        "Because the reference is V11 rather than either technique alone, each",
        "challenge segment already contains the part HACMO has not removed, and no",
        "unattributed remainder is left over.",
        "",
        "Closure is checked to 1e-6 relative, and `V00` is cross-checked against the",
        "TP2 pack's published vllm row on every point.",
        "",
        "## What The Edge-I/O Axis Is",
        "",
        "TP3's network config sets `crossbar-mode: true` and `rail-optimized: false`:",
        "each wafer edge owns external I/O ports that an edge-local crossbar can",
        "reassign among the die groups on that edge, whereas the naive fabric binds",
        "them statically. The axis is therefore port *reassignability*, not port",
        "position or aggregate bandwidth -- both fabrics have the same total edge",
        "bandwidth.",
        "",
        "A previous attempt built this axis by relocating rail ports onto a uniform",
        "grid. That modelled the wrong mechanism and also invalidated HACMO's",
        "placement, which routes to the real port positions: it gave V11 = 46.35",
        "against the pack's true `full_tp2` = 23.89, and a negative TP3 share. The",
        "lesson is to take all four corners from measured configurations.",
        "",
        "## Result",
        "",
        f"Naive is {min(mult):.2f}x to {max(mult):.2f}x the ideal.",
        "",
        "| Topology | Batch | Ideal | TP2 challenge | TP3 challenge | Naive | Naive/Ideal |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['topology']} | {r['batch_size']} | {float(r['ideal_us']):.2f} | "
            f"{float(r['tp2_challenge_us']):.2f} | {float(r['tp3_challenge_us']):.2f} | "
            f"{float(r['naive_us']):.2f} | {float(r['naive_over_ideal_x']):.2f}x |")
    lines += [
        "",
        "### Corner values (us/step)",
        "",
        "| Topology | Batch | V00 naive_rail | V10 rail_tp2 | V01 crossbar_tp2 | V11 full_tp2 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['topology']} | {r['batch_size']} | {float(r['v00_naive_rail_us']):.2f} | "
            f"{float(r['v10_rail_tp2_us']):.2f} | {float(r['v01_crossbar_tp2_us']):.2f} | "
            f"{float(r['v11_full_tp2_us']):.2f} |")
    lines += [
        "",
        "Both marginal gains per factor are reported in the CSV",
        "(`tp2_first_gain_us` / `tp2_second_gain_us` and the TP3 pair), so the size",
        "of the interaction can be read directly.",
        "",
        "## Scope",
        "",
        "Communication only; H200 expert compute is excluded. Ideal values are not",
        "comparable across topologies, since the 8x8 mesh has twice the mean hop",
        "count of 4x4. Compare each topology against its own naive baseline.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "PYTHONPATH=$PWD python3 \\",
        "  hacmo/moe/tools/build_tp2_motivation_corners.py \\",
        f"  --output-dir {out_dir}",
        "```",
        "",
        "## Sources",
        "",
        f"- TP3 metrics: `{args.tp3_metrics}`",
        f"- TP2 pack (V00 cross-check): `{args.tp2_pack}`",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tp3-metrics", type=Path, default=DEFAULT_TP3)
    p.add_argument("--tp2-pack", type=Path, default=DEFAULT_TP2_PACK)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()

    rows = build(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "tp2_tp3_motivation.csv", rows)
    plot(rows, args.output_dir)
    write_readme(args.output_dir, rows, args)
    (args.output_dir / "metadata.json").write_text(json.dumps({
        "status": "motivation_candidate",
        "variant": "two-factor Shapley over the TP3 pack's four measured bars",
        "factors": {
            "placement": "naive uniform vs HACMO mapping",
            "edge_io": "static rail ports vs edge-local crossbar "
                       "(crossbar-mode: true, rail-optimized: false)",
        },
        "corners": CORNERS,
        "closure": "exact, checked to 1e-6 relative",
        "no_weighting_assumption": True,
        "new_simulation_required": False,
        "v00_cross_checked_against": str(args.tp2_pack),
        "sources": {"tp3_metrics": str(args.tp3_metrics)},
    }, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {args.output_dir}")
    for r in rows:
        print(f"  {r['topology']:18s} b={r['batch_size']:>4} "
              f"ideal={float(r['ideal_us']):7.2f} tp2={float(r['tp2_challenge_us']):7.2f} "
              f"tp3={float(r['tp3_challenge_us']):7.2f} naive={float(r['naive_us']):7.2f} "
              f"({float(r['naive_over_ideal_x']):.2f}x)")


if __name__ == "__main__":
    main()
