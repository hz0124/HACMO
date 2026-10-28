#!/usr/bin/env python3
"""TP2/TP3 motivation by two-factor (Shapley) decomposition over four corners.

Two independent causes make heterogeneity-unaware MoE mapping slow, and they
interact: cross-wafer volume worsens edge congestion, and clustered edge ports
amplify multi-hop cost. Splitting their joint effect by assuming a weighting is
avoidable -- measure all four corners instead and average both orderings, which
is the Shapley value for two factors:

    P = placement      (0 = uniform/naive, 1 = HACMO's own route plan)
    B = edge I/O       (0 = real edge-clustered rail ports, 1 = ports relocated)

    V(0,0) = naive        V(1,0) = placement fixed only
    V(0,1) = ports fixed only     V(1,1) = both fixed  == ideal

    tp2_challenge = 1/2 [ (V00 - V10) + (V01 - V11) ]
    tp3_challenge = 1/2 [ (V00 - V01) + (V10 - V11) ]
    ideal         = V11

    ideal + tp2_challenge + tp3_challenge = V00        (exact, algebraically)

Each challenge therefore includes the part HACMO has not removed, because the
reference is V11 rather than HACMO's own result. No lower-bound assumption is
needed, which also sidesteps the replicated-placement problem: HACMO's measured
cross-wafer share is lower than the analytic bound's "forced minimum", so that
bound cannot be used as a reference for a replicated mapping.

The edge-I/O control keeps the port count and each port's bandwidth scale
identical and only relocates ports onto a uniform grid, so exactly one variable
moves. Spreading aggregate bandwidth over all dies instead would change per-port
capacity too, and was measured to make the control slower than the real fabric
at low load on 4w_8x8.

Reads confirmed packs and staged route plans; writes only to the output dir.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
RUNNER = "supernode_sim.experiments.run_ablation2_from_artifacts"
NETWORKS = {
    "16w_4x4_rail_500": "hacmo/moe/configs/network/hierarchical_16w_4x4_mesh_rail_500.yaml",
    "4w_8x8_rail_900": "hacmo/moe/configs/network/hierarchical_4w_8x8_mesh_rail_900.yaml",
}
DEFAULT_PACK = REPO / "hacmo/moe/outputs/tp2_serving_epoch_paper_pack_20260831"
TOPO_ORDER = ["16w_4x4_rail_500", "4w_8x8_rail_900"]
TOPO_TITLES = {"16w_4x4_rail_500": "16W 4x4", "4w_8x8_rail_900": "4W 8x8"}


@dataclass(frozen=True)
class Corner:
    key: str
    placement_fixed: bool
    ports_fixed: bool


CORNERS = (
    Corner("v00_naive", False, False),
    Corner("v10_placement_fixed", True, False),
    Corner("v01_ports_fixed", False, True),
    Corner("v11_both_fixed", True, True),
)


def relocated_port_network(src: Path, dst: Path) -> dict:
    """Same port count and per-port bandwidth, relocated onto a uniform grid."""
    cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
    dpw = int(cfg["npus_count"][1])
    side = int(round(dpw ** 0.5))
    if side * side != dpw:
        raise ValueError(f"{src}: mesh is not square")
    ports = [int(p) for p in cfg["rail-port-nodes"]]
    scales = [float(s) for s in (cfg.get("rail-port-bw-scales") or [1.0] * len(ports))]
    agg: dict[int, float] = {}
    for p, s in zip(ports, scales):
        agg[p] = agg.get(p, 0.0) + s
    n = len(agg)
    stride = dpw / n
    chosen = sorted({min(int(round(i * stride)), dpw - 1) for i in range(n)})
    for d in range(dpw):
        if len(chosen) >= n:
            break
        if d not in chosen:
            chosen.append(d)
    chosen = sorted(chosen)[:n]
    out = copy.deepcopy(cfg)
    out["rail-port-nodes"] = chosen
    out["rail-port-bw-scales"] = sorted(agg.values(), reverse=True)
    if abs(sum(out["rail-port-bw-scales"]) - sum(scales)) > 1e-9:
        raise ValueError("relocation changed aggregate rail bandwidth")

    def hops(a: int, b: int) -> int:
        return abs(a // side - b // side) + abs(a % side - b % side)

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(yaml.safe_dump(out, sort_keys=False), encoding="utf-8")
    return {
        "port_dies": n,
        "mean_hops_original": sum(min(hops(d, q) for q in agg) for d in range(dpw)) / dpw,
        "mean_hops_relocated": sum(min(hops(d, q) for q in chosen) for d in range(dpw)) / dpw,
        "aggregate_scale": sum(scales),
    }


def run_corner(corner: Corner, *, topology: str, batch: int, artifact_dir: Path,
               work: Path, args: argparse.Namespace) -> dict:
    """One ASTRA replay. Returns us/step plus provenance."""
    net_src = REPO / NETWORKS[topology]
    scratch = work / f"{topology}_b{batch}_{corner.key}"
    scratch.mkdir(parents=True, exist_ok=True)

    net = net_src
    port_info: dict = {}
    if corner.ports_fixed:
        net = scratch / "network_relocated.yaml"
        port_info = relocated_port_network(net_src, net)

    # Placement axis: which method's route plan to replay. Both plans ship in the
    # staged artifacts, so no plan is synthesised.
    method = "ours_dynamic" if corner.placement_fixed else "vllm"
    plan = artifact_dir / "route_plans" / f"seed{args.seed}" / f"{method}_bpd{batch}.json"
    if not plan.is_file():
        raise FileNotFoundError(f"{corner.key}: {plan} missing")

    csv_out = scratch / "row.csv"
    cmd = [
        sys.executable, "-m", RUNNER,
        "--artifact-topology-dir", str(artifact_dir),
        "--network-config", str(net),
        "--methods", method,
        "--seeds", str(args.seed),
        "--batch-size", str(batch),
        "--batch-scope", "per_die",
        "--num-requests", str(batch * _num_dies(net_src)),
        "--window-size", str(args.window_size),
        "--decode-len", str(args.decode_len),
        "--exclude-leading-windows", str(args.exclude_leading_steps),
        "--communication-only", "--run-astra",
        "--topology-label", topology,
        "--output-root", str(scratch / "run"),
        "--output-csv", str(csv_out),
    ]
    env = {"PYTHONPATH": str(REPO), "MPLCONFIGDIR": "/tmp/matplotlib-shapley",
           "PATH": "/usr/bin:/bin"}
    proc = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    if proc.returncode != 0 or not csv_out.exists():
        tail = (proc.stderr or "")[-600:]
        raise RuntimeError(f"{corner.key} rc={proc.returncode}: {tail}")
    rows = list(csv.DictReader(csv_out.open(encoding="utf-8")))
    if len(rows) != 1:
        raise RuntimeError(f"{corner.key}: expected 1 row, got {len(rows)}")
    row = rows[0]
    steps = int(float(row["reported_steps"]))
    if steps != args.expect_steps:
        raise RuntimeError(f"{corner.key}: {steps} steps, expected {args.expect_steps}")
    return {
        "us_per_step": float(row["communication_us"]),
        "method": method,
        "cross_wafer_ratio": float(row.get("cross_wafer_ratio") or 0.0),
        "port_info": port_info,
    }


def _num_dies(net_cfg: Path) -> int:
    cfg = yaml.safe_load(net_cfg.read_text(encoding="utf-8"))
    return int(cfg["npus_count"][0]) * int(cfg["npus_count"][1])


def pack_points(pack: Path) -> dict[tuple[str, int], dict[str, str]]:
    out: dict[tuple[str, int], dict[str, str]] = {}
    for name in ("tp2_high_drift_astra_16w_comm_only.csv",
                 "tp2_high_drift_astra_4w_comm_only.csv"):
        path = pack / "csv" / name
        if not path.is_file():
            continue
        for r in csv.DictReader(path.open(encoding="utf-8")):
            if r.get("method") != "vllm":
                continue
            out[(r["topology"], int(float(r["batch_per_die"])))] = {
                "naive_us": r["communication_us"],
                "source_root": Path(r.get("source_root", "")).name,
            }
    if not out:
        raise FileNotFoundError(f"no vllm rows under {pack}/csv")
    return out


def resolve_artifact_dir(info: dict, topology: str, args) -> Path:
    staged = args.staging_parent / info["source_root"]
    if not staged.is_dir():
        raise FileNotFoundError(f"staged root {staged} not found")
    cands = [c for c in staged.iterdir() if c.is_dir() and (c / topology).is_dir()]
    if len(cands) != 1:
        raise FileNotFoundError(
            f"expected one case dir under {staged} holding {topology}, got {len(cands)}")
    return cands[0] / topology


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def plot(rows: list[dict], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    topos = [t for t in TOPO_ORDER if any(r["topology"] == t for r in rows)]
    cols = {"i": "#b0bec5", "t2": "#4e79a7", "t3": "#f28e2b"}
    fig, axes = plt.subplots(1, len(topos), figsize=(5.1 * len(topos), 3.6),
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
                    label="Ideal (good placement + balanced edge I/O)")
        b2 = ax.bar(x, t2, 0.6, bottom=i, color=cols["t2"], edgecolor="#37474f",
                    lw=0.6, hatch="xx",
                    label="Challenge 2: expert placement (cross-wafer + multi-hop)")
        base = [a + b for a, b in zip(i, t2)]
        b3 = ax.bar(x, t3, 0.6, bottom=base, color=cols["t3"], edgecolor="#37474f",
                    lw=0.6, hatch="//",
                    label="Challenge 3: wafer-edge I/O bandwidth imbalance")
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


def write_readme(out_dir: Path, rows: list[dict], skipped: list[str]) -> None:
    mult = [float(r["naive_over_ideal_x"]) for r in rows]
    lines = [
        "# TP2/TP3 Motivation: Two-Factor (Shapley) Decomposition",
        "",
        "Heterogeneity-unaware MoE mapping against an ideal that fixes both causes,",
        "with the joint effect split by measuring all four corners.",
        "",
        "## Method",
        "",
        "```",
        "P = placement   (0 = uniform/naive, 1 = HACMO route plan)",
        "B = edge I/O    (0 = real edge-clustered ports, 1 = ports relocated)",
        "",
        "V00 = naive          V10 = placement fixed only",
        "V01 = ports fixed    V11 = both fixed  == ideal",
        "",
        "tp2_challenge = 1/2 [ (V00-V10) + (V01-V11) ]",
        "tp3_challenge = 1/2 [ (V00-V01) + (V10-V11) ]",
        "ideal + tp2_challenge + tp3_challenge = V00      (exact)",
        "```",
        "",
        "Averaging both orderings is the Shapley value for two factors. It needs no",
        "weighting assumption, so nothing is apportioned by judgement.",
        "",
        "Because the reference is V11 and not HACMO's own result, each challenge",
        "segment already contains the part HACMO has not removed.",
        "",
        "## Why Not An Analytic Lower Bound",
        "",
        "The analytic bound's `min_cross_wafer_fraction` assumes one expert per die.",
        "HACMO uses ~96-100 extra replicas, and its measured cross-wafer share (0.90",
        "on 16w, 0.69 on 4w) is *below* that supposed minimum, so the bound does not",
        "hold for a replicated placement and cannot serve as the reference. Measuring",
        "V11 avoids the assumption entirely.",
        "",
        "## Controls",
        "",
        "- Placement: both route plans ship in the staged artifacts; neither is",
        "  synthesised.",
        "- Edge I/O: port count and each port's bandwidth scale are held identical;",
        "  only their positions move to a uniform grid. Spreading aggregate bandwidth",
        "  over all dies instead would move two variables, and was measured to make",
        "  the control slower than the real fabric at low load on 4w_8x8.",
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
        "| Topology | Batch | V00 naive | V10 placement | V01 ports | V11 both |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['topology']} | {r['batch_size']} | {float(r['v00_naive_us']):.2f} | "
            f"{float(r['v10_placement_fixed_us']):.2f} | {float(r['v01_ports_fixed_us']):.2f} | "
            f"{float(r['v11_both_fixed_us']):.2f} |")
    if skipped:
        lines += ["", "## Points Not Included", ""]
        lines += [f"- {s}" for s in skipped]
    lines += [
        "",
        "## Scope",
        "",
        "Communication only; H200 expert compute excluded. Ideal values are not",
        "comparable across topologies (the 8x8 mesh has twice the mean hop count of",
        "4x4). Compare each topology against its own naive baseline.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tp2-pack", type=Path, default=DEFAULT_PACK)
    p.add_argument("--staging-parent", type=Path, default=Path("/tmp"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--work-dir", type=Path, default=Path("ae/work/tp2-motivation/shapley"))
    p.add_argument("--topologies", nargs="+", default=TOPO_ORDER)
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[16, 32, 64, 128])
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--window-size", type=int, default=8)
    p.add_argument("--decode-len", type=int, default=128)
    p.add_argument("--exclude-leading-steps", type=int, default=112)
    p.add_argument("--expect-steps", type=int, default=16)
    p.add_argument("--skip-failures", action="store_true",
                   help="record and continue past points whose corners cannot run")
    args = p.parse_args()

    points = pack_points(args.tp2_pack)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    skipped: list[str] = []

    for topo in args.topologies:
        for batch in args.batch_sizes:
            key = (topo, batch)
            if key not in points:
                skipped.append(f"{topo} b{batch}: no vllm row in pack")
                continue
            info = points[key]
            try:
                art = resolve_artifact_dir(info, topo, args)
                got: dict[str, dict] = {}
                for c in CORNERS:
                    print(f"[run] {topo} b{batch} {c.key} ...", flush=True)
                    got[c.key] = run_corner(c, topology=topo, batch=batch,
                                            artifact_dir=art, work=args.work_dir,
                                            args=args)
                    print(f"      {c.key} = {got[c.key]['us_per_step']:.6f}", flush=True)
            except Exception as exc:  # noqa: BLE001
                msg = f"{topo} b{batch}: {type(exc).__name__}: {exc}"
                if not args.skip_failures:
                    raise
                print(f"  SKIP {msg}", flush=True)
                skipped.append(msg)
                continue

            v00 = got["v00_naive"]["us_per_step"]
            v10 = got["v10_placement_fixed"]["us_per_step"]
            v01 = got["v01_ports_fixed"]["us_per_step"]
            v11 = got["v11_both_fixed"]["us_per_step"]

            expect = float(info["naive_us"])
            rel = abs(v00 - expect) / expect
            if rel > 2e-3:
                msg = (f"{topo} b{batch}: V00 {v00:.6f} does not reproduce pack "
                       f"{expect:.6f} (rel {rel:.4%})")
                if not args.skip_failures:
                    raise RuntimeError(msg)
                print(f"  SKIP {msg}", flush=True)
                skipped.append(msg)
                continue

            tp2 = 0.5 * ((v00 - v10) + (v01 - v11))
            tp3 = 0.5 * ((v00 - v01) + (v10 - v11))
            total = v11 + tp2 + tp3
            if abs(total - v00) > 1e-6 * max(v00, 1.0):
                raise RuntimeError(f"{topo} b{batch}: closure {total} vs {v00}")
            if min(tp2, tp3) < 0:
                msg = (f"{topo} b{batch}: negative Shapley share "
                       f"(tp2={tp2:.3f}, tp3={tp3:.3f})")
                if not args.skip_failures:
                    raise RuntimeError(msg)
                print(f"  SKIP {msg}", flush=True)
                skipped.append(msg)
                continue

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
                "v00_naive_us": f"{v00:.6f}",
                "v10_placement_fixed_us": f"{v10:.6f}",
                "v01_ports_fixed_us": f"{v01:.6f}",
                "v11_both_fixed_us": f"{v11:.6f}",
                "pack_published_naive_us": f"{expect:.6f}",
                "pack_reproduction_rel_err": f"{rel:.3e}",
                "source_root": info["source_root"],
            })
            print(f"  -> ideal={v11:.3f} tp2={tp2:.3f} tp3={tp3:.3f} "
                  f"naive={v00:.3f} ({v00/v11:.2f}x)", flush=True)

    if not rows:
        raise RuntimeError("no point completed; nothing to write")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "tp2_tp3_motivation.csv", rows)
    plot(rows, args.output_dir)
    write_readme(args.output_dir, rows, skipped)
    (args.output_dir / "metadata.json").write_text(json.dumps({
        "status": "motivation_candidate_shapley",
        "variant": "two-factor Shapley over four simulated corners",
        "closure": "exact by construction, checked to 1e-6 relative",
        "no_weighting_assumption": True,
        "points_completed": len(rows),
        "points_skipped": skipped,
        "corners": {c.key: {"placement_fixed": c.placement_fixed,
                            "ports_fixed": c.ports_fixed} for c in CORNERS},
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {args.output_dir}  ({len(rows)} points, {len(skipped)} skipped)")


if __name__ == "__main__":
    main()
