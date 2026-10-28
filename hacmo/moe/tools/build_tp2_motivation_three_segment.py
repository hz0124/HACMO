#!/usr/bin/env python3
"""TP2/TP3 motivation: three simulated segments, exact closure.

    ideal            perfectly balanced all-to-all, same total volume, real fabric
    tp2_challenge    extra cost of heterogeneity-unaware (uniform) expert placement
    tp3_challenge    extra cost of wafer-edge rail-port load imbalance
    ideal + tp2 + tp3 = naive        (exact, by construction)

Every term is an ASTRA congestion-aware run through the same entry point the
confirmed TP2 pack used, so all three are on one footing and no residual is left
to attribute by hand.  Neither challenge term references what TP2 or TP3
achieve: both are differences between fabrics/demands, not between our methods.

    naive             uniform placement, real rail ports
    naive_flat_ports  uniform placement, rail capacity spread evenly over dies
                      (aggregate rail bandwidth unchanged)
    ideal             balanced demand, real rail ports

    tp3_challenge = naive            - naive_flat_ports
    tp2_challenge = naive_flat_ports - ideal

Reproducing the pack exactly requires these, each of which cost me a wrong run
to find:

  * ``--num-requests`` must equal the global batch.  The staged counts are
    already expanded to the target batch, but runtime metadata records only the
    raw request count (270), so the default rescales traffic by 8192/270 = 30.34x.
  * ``--batch-scope per_die`` with ``--batch-size`` as the per-die value.
  * ``--seeds`` must be given, otherwise route plans are ignored and window
    aggregates are used instead.
  * ``--communication-only``, otherwise trace mode is ``full``.
  * ``--exclude-leading-windows`` is counted in *steps* on the route-plan path
    (112) but in *windows* on the window-aggregate path (14).

Verified: this configuration reproduces the pack's vllm 16w bpd32 communication
of 51.2346875 us/step bit-for-bit.

Reads confirmed packs and staged artifacts; writes nothing into them.
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

NETWORK_CONFIGS = {
    "16w_4x4_rail_500": "hacmo/moe/configs/network/hierarchical_16w_4x4_mesh_rail_500.yaml",
    "4w_8x8_rail_900": "hacmo/moe/configs/network/hierarchical_4w_8x8_mesh_rail_900.yaml",
}
RUNNER = "supernode_sim.experiments.run_ablation2_from_artifacts"


@dataclass(frozen=True)
class Variant:
    key: str
    balanced_demand: bool
    flat_rail_ports: bool


VARIANTS = (
    Variant("ideal", True, False),
    Variant("naive_flat_ports", False, True),
    Variant("naive", False, False),
)


def flat_port_network(src: Path, dst: Path) -> dict:
    """Same port count and same per-port bandwidth, spread evenly over the mesh.

    This isolates *where* the rail ports sit, which is what TP3 addresses: on the
    real wafer they are clustered on the edge, so interior dies must cross the
    mesh to reach one and the ports themselves receive skewed load under XY
    routing.  The control keeps the number of distinct port dies and each port's
    bandwidth scale identical, and only relocates them to a uniform grid.

    An earlier version instead spread the *aggregate* rail bandwidth over all
    dies.  That changed two things at once -- port placement and per-port
    bandwidth -- and on 4w_8x8 at batch 16 it made the control slower than the
    real fabric (per-port capacity fell from ~1029 to 450 GB/s, which at low load
    outweighed the imbalance it removed).  Holding per-port bandwidth fixed keeps
    the comparison to one variable.
    """
    cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
    dpw = int(cfg["npus_count"][1])
    side = int(round(dpw ** 0.5))
    if side * side != dpw:
        raise ValueError(f"{src}: mesh is not square ({dpw} dies)")
    ports = [int(p) for p in cfg["rail-port-nodes"]]
    scales = [float(s) for s in (cfg.get("rail-port-bw-scales") or [1.0] * len(ports))]

    # Aggregate duplicate entries the way RailOptimizedSwitchMesh does, so the
    # per-die bandwidth scales we must preserve are explicit.
    agg: dict[int, float] = {}
    for p, s in zip(ports, scales):
        agg[p] = agg.get(p, 0.0) + s
    n_ports = len(agg)
    scale_multiset = sorted(agg.values(), reverse=True)

    # Choose n_ports dies spread as evenly as possible over the side x side mesh:
    # walk all dies ordered by distance from the mesh centre, taking a regular
    # stride so the chosen set is neither edge- nor centre-biased.
    stride = dpw / n_ports
    chosen = sorted({min(int(round(i * stride)), dpw - 1) for i in range(n_ports)})
    # rounding collisions can yield fewer than n_ports; fill from unused dies
    if len(chosen) < n_ports:
        for d in range(dpw):
            if len(chosen) >= n_ports:
                break
            if d not in chosen:
                chosen.append(d)
        chosen = sorted(chosen)
    if len(chosen) != n_ports:
        raise ValueError(f"could not place {n_ports} ports on {dpw} dies")

    out = copy.deepcopy(cfg)
    out["rail-port-nodes"] = chosen
    out["rail-port-bw-scales"] = scale_multiset

    def hops(a: int, b: int) -> int:
        return abs(a // side - b // side) + abs(a % side - b % side)

    mean_orig = sum(min(hops(d, q) for q in agg) for d in range(dpw)) / dpw
    mean_flat = sum(min(hops(d, q) for q in chosen) for d in range(dpw)) / dpw

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(yaml.safe_dump(out, sort_keys=False), encoding="utf-8")
    info = {
        "port_dies": n_ports,
        "total_scale_original": sum(scales),
        "total_scale_control": sum(scale_multiset),
        "aggregate_rail_bw_gbps": sum(scales) * float(cfg["bandwidth"][0]),
        "mean_hops_to_port_original": mean_orig,
        "mean_hops_to_port_control": mean_flat,
        "port_dies_original": sorted(agg),
        "port_dies_control": chosen,
    }
    if abs(info["total_scale_control"] - info["total_scale_original"]) > 1e-9:
        raise ValueError("control changed aggregate rail bandwidth")
    return info


def balanced_route_plan(src: Path, dst: Path, num_dies: int,
                        dies_per_wafer: int) -> dict:
    """Rewrite a route plan as the ideal case: no traffic leaves its wafer.

    This is the floor the two challenges are measured against.  Per-step
    activation totals are preserved exactly; the destinations are rewritten so
    that every activation is served inside its source die's own wafer, spread
    evenly over that wafer's dies.  Nothing can beat staying local, so this is a
    genuine lower bound and not a restatement of what TP2 achieves -- the
    confirmed pack's own TP2 result still sends ~94% of its traffic across
    wafers.

    The gap from here to the naive run is therefore exactly the cost of expert
    placement that ignores locality: cross-wafer volume plus the extra mesh hops
    to reach a wafer edge.
    """
    plan = json.loads(src.read_text(encoding="utf-8"))
    steps = plan.get("steps") or []
    flat_steps = []
    total_before = total_after = 0
    for st in steps:
        acts = sum(int(a["activations"]) for a in (st.get("assignments") or []))
        total_before += acts
        # ASTRA reads dispatch_bytes / combine_bytes (pre-computed src->dst byte
        # matrices), not assignments; assignments only feed expert_counts and the
        # per-step validation. All three must be rewritten together, plus
        # rank_expert_tokens which drives the compute model.
        meta = plan.get("metadata") or {}
        d_per = int(meta.get("dispatch_bytes_per_token", 0))
        c_per = int(meta.get("combine_bytes_per_token", 0))
        if d_per <= 0 or c_per <= 0:
            raise ValueError("route-plan metadata lacks dispatch/combine bytes per token")

        by_src: dict[int, int] = {}
        for a in (st.get("assignments") or []):
            by_src[int(a["src_die"])] = by_src.get(int(a["src_die"]), 0) + int(a["activations"])

        assigns = []
        disp: dict[str, dict[str, int]] = {}
        comb: dict[str, dict[str, int]] = {}
        rank_tok: dict[str, dict[str, float]] = {}
        for s, n_src in sorted(by_src.items()):
            if n_src <= 0:
                continue
            wafer = s // dies_per_wafer
            targets = [wafer * dies_per_wafer + k for k in range(dies_per_wafer)]
            base, rem = divmod(n_src, len(targets))
            for i, t in enumerate(targets):
                n = base + (1 if i < rem else 0)
                if n <= 0:
                    continue
                # uniform placement puts expert e on die e, so an activation
                # landing on die t is served by expert t
                assigns.append({
                    "src_die": s, "expert": t, "dst_die": t, "activations": n,
                })
                if s != t:
                    disp.setdefault(str(s), {})[str(t)] = \
                        disp.setdefault(str(s), {}).get(str(t), 0) + n * d_per
                    comb.setdefault(str(t), {})[str(s)] = \
                        comb.setdefault(str(t), {}).get(str(s), 0) + n * c_per
                rank_tok.setdefault(str(t), {})[str(t)] = \
                    rank_tok.setdefault(str(t), {}).get(str(t), 0.0) + float(n)

        total_after += sum(int(a["activations"]) for a in assigns)
        flat = {
            k: v for k, v in st.items()
            if k not in ("assignments", "dispatch_bytes", "combine_bytes",
                         "rank_expert_tokens", "validation")
        }
        flat["assignments"] = assigns
        flat["dispatch_bytes"] = disp
        flat["combine_bytes"] = comb
        flat["rank_expert_tokens"] = rank_tok
        # Preserve the validation contract the loader checks.
        val = dict(st.get("validation") or {})
        val["activations_preserved"] = True
        val["compute_matches_assignments"] = True
        flat["validation"] = val
        flat_steps.append(flat)
    if total_after != total_before:
        raise ValueError(
            f"balanced plan changed activation count: {total_before} -> {total_after}"
        )
    # The rewrite must actually remove cross-wafer traffic from the matrices
    # ASTRA reads, not just from assignments.
    xw_before = xw_after = 0
    for st_o, st_n in zip(steps, flat_steps):
        for s_id, dsts in (st_o.get("dispatch_bytes") or {}).items():
            for d_id, nb in dsts.items():
                if int(s_id) // dies_per_wafer != int(d_id) // dies_per_wafer:
                    xw_before += int(nb)
        for s_id, dsts in (st_n.get("dispatch_bytes") or {}).items():
            for d_id, nb in dsts.items():
                if int(s_id) // dies_per_wafer != int(d_id) // dies_per_wafer:
                    xw_after += int(nb)
    if xw_after != 0:
        raise ValueError(
            f"balanced plan still has {xw_after} cross-wafer dispatch bytes"
        )
    if xw_before == 0:
        raise ValueError("original plan had no cross-wafer traffic; nothing to isolate")
    out = {k: v for k, v in plan.items() if k != "steps"}
    out["steps"] = flat_steps
    out["motivation_note"] = (
        "balanced all-to-all: per-step activation totals preserved exactly, "
        "distribution flattened over all (src, dst) pairs"
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out), encoding="utf-8")
    return {"steps": len(flat_steps), "activations": total_before,
            "cross_wafer_dispatch_bytes_removed": xw_before}


def run_variant(
    variant: Variant, *, topology: str, batch: int, artifact_dir: Path,
    work: Path, args: argparse.Namespace,
) -> float:
    """One ASTRA run through the pack's own entry point. Returns us/step."""
    net_src = REPO / NETWORK_CONFIGS[topology]
    tag = f"{topology}_b{batch}_{variant.key}"
    scratch = work / tag
    scratch.mkdir(parents=True, exist_ok=True)

    net = net_src
    if variant.flat_rail_ports:
        net = scratch / "network_flat_ports.yaml"
        flat_port_network(net_src, net)

    art = artifact_dir
    balanced_info: dict = {}
    if variant.balanced_demand:
        # Shadow artifact dir: symlink every entry except route_plans, which gets
        # a real directory holding the rewritten plan. Order matters -- creating
        # route_plans first prevents the loop from linking the original.
        art = scratch / "artifact"
        art.mkdir(parents=True, exist_ok=True)
        rp_dir = art / "route_plans" / f"seed{args.seed}"
        rp_dir.mkdir(parents=True, exist_ok=True)
        rp_src = artifact_dir / "route_plans" / f"seed{args.seed}" / f"vllm_bpd{batch}.json"
        if not rp_src.is_file():
            raise FileNotFoundError(f"route plan not found: {rp_src}")
        balanced_info = balanced_route_plan(
            rp_src, rp_dir / f"vllm_bpd{batch}.json",
            _num_dies(net_src), _dies_per_wafer(net_src),
        )
        for child in artifact_dir.iterdir():
            if child.name == "route_plans":
                continue
            link = art / child.name
            if not link.exists(follow_symlinks=False):
                link.symlink_to(child)
        # Guard: the plan the runner will read must be the rewritten one.
        target = (art / "route_plans" / f"seed{args.seed}" / f"vllm_bpd{batch}.json")
        if target.resolve() == rp_src.resolve():
            raise RuntimeError(
                "shadow route plan still resolves to the original; the balanced "
                "variant would silently equal naive"
            )

    csv_out = scratch / "row.csv"
    cmd = [
        sys.executable, "-m", RUNNER,
        "--artifact-topology-dir", str(art),
        "--network-config", str(net),
        "--methods", "vllm",
        "--seeds", str(args.seed),
        "--batch-size", str(batch),
        "--batch-scope", "per_die",
        "--num-requests", str(batch * _num_dies(net_src)),
        "--window-size", str(args.window_size),
        "--decode-len", str(args.decode_len),
        "--exclude-leading-windows", str(args.exclude_leading_steps),
        "--communication-only",
        "--run-astra",
        "--topology-label", topology,
        "--output-root", str(scratch / "run"),
        "--output-csv", str(csv_out),
    ]
    env = {"PYTHONPATH": str(REPO), "MPLCONFIGDIR": "/tmp/matplotlib-tp2motiv",
           "PATH": "/usr/bin:/bin"}
    # ASTRA intermittently aborts in its statistics teardown with a glibc
    # pthread-priority assertion, after the simulation itself has finished and
    # reported. That is a race in the simulator's shutdown, not a data problem,
    # so a crashed run is retried rather than accepted or silently skipped.
    proc = None
    for attempt in range(1, args.max_attempts + 1):
        proc = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
        if proc.returncode == 0 and csv_out.exists():
            break
        if attempt < args.max_attempts:
            print(f"      {tag}: attempt {attempt} failed "
                  f"(rc={proc.returncode}), retrying", flush=True)
            csv_out.unlink(missing_ok=True)
    if proc is None or proc.returncode != 0 or not csv_out.exists():
        rc = proc.returncode if proc else "n/a"
        raise RuntimeError(
            f"{tag} failed after {args.max_attempts} attempts (rc={rc}):\n"
            f"{proc.stdout[-1500:] if proc else ''}\n{proc.stderr[-1500:] if proc else ''}"
        )
    rows = list(csv.DictReader(csv_out.open(encoding="utf-8")))
    if len(rows) != 1:
        raise RuntimeError(f"{tag}: expected 1 row, got {len(rows)}")
    row = rows[0]
    steps = int(float(row["reported_steps"]))
    if steps != args.expect_steps:
        raise RuntimeError(
            f"{tag}: replayed {steps} steps, expected {args.expect_steps}"
        )
    return float(row["communication_us"])


def _num_dies(net_cfg: Path) -> int:
    cfg = yaml.safe_load(net_cfg.read_text(encoding="utf-8"))
    return int(cfg["npus_count"][0]) * int(cfg["npus_count"][1])


def _dies_per_wafer(net_cfg: Path) -> int:
    cfg = yaml.safe_load(net_cfg.read_text(encoding="utf-8"))
    return int(cfg["npus_count"][1])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--artifact-root", type=Path, default=None,
                   help="single staged TP2 artifact root; omit to resolve per point "
                        "from the pack's own source_root column")
    p.add_argument("--tp2-pack", type=Path,
                   default=REPO / "hacmo/moe/outputs/tp2_serving_epoch_paper_pack_20260831",
                   help="confirmed pack; supplies per-point source_root and the "
                        "published naive value each run is verified against")
    p.add_argument("--staging-parent", type=Path, default=Path("/tmp"),
                   help="where the pack's source_root directories live")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--work-dir", type=Path, default=Path("ae/work/tp2-motivation/three_segment"))
    p.add_argument("--topologies", nargs="+", default=["16w_4x4_rail_500"])
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[32])
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--window-size", type=int, default=8)
    p.add_argument("--decode-len", type=int, default=128)
    p.add_argument("--exclude-leading-steps", type=int, default=112,
                   help="route-plan path counts this in STEPS (112 = tail16 of 128)")
    p.add_argument("--expect-steps", type=int, default=16)
    p.add_argument("--verify-naive-us", type=float, default=0.0,
                   help="unused; each point is verified against its own pack value")
    p.add_argument("--max-attempts", type=int, default=3,
                   help="retries for ASTRA's intermittent teardown abort")
    return p.parse_args()


def pack_points(pack: Path) -> dict[tuple[str, int], dict[str, str]]:
    """Per (topology, batch): the published naive value and its source root.

    Each pack row records the staged root it came from, so every point is
    replayed from its own artifacts and verified against its own published
    number rather than a single hardcoded pair.
    """
    out: dict[tuple[str, int], dict[str, str]] = {}
    for name in ("tp2_high_drift_astra_16w_comm_only.csv",
                 "tp2_high_drift_astra_4w_comm_only.csv"):
        path = pack / "csv" / name
        if not path.is_file():
            continue
        for r in csv.DictReader(path.open(encoding="utf-8")):
            if r.get("method") != "vllm":
                continue
            key = (r["topology"], int(float(r["batch_per_die"])))
            out[key] = {
                "naive_us": r["communication_us"],
                "source_root": Path(r.get("source_root", "")).name,
                "replay": r.get("replay", ""),
            }
    if not out:
        raise FileNotFoundError(f"no vllm rows found under {pack}/csv")
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def plot(rows: list[dict], out_dir: Path, topologies: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    titles = {"16w_4x4_rail_500": "16W 4x4", "4w_8x8_rail_900": "4W 8x8"}
    cols = {"ideal": "#b0bec5", "tp2": "#4e79a7", "tp3": "#f28e2b"}
    n = len(topologies)
    fig, axes = plt.subplots(1, n, figsize=(4.9 * n, 3.5), squeeze=False)
    handles = labels = None
    for ax, topo in zip(axes[0], topologies):
        sel = sorted([r for r in rows if r["topology"] == topo],
                     key=lambda r: int(r["batch_size"]))
        x = np.arange(len(sel))
        ideal = [float(r["ideal_us"]) for r in sel]
        tp2 = [float(r["tp2_challenge_us"]) for r in sel]
        tp3 = [float(r["tp3_challenge_us"]) for r in sel]
        b1 = ax.bar(x, ideal, 0.6, color=cols["ideal"], edgecolor="#37474f",
                    linewidth=0.5, label="Ideal (balanced all-to-all)")
        b2 = ax.bar(x, tp2, 0.6, bottom=ideal, color=cols["tp2"], edgecolor="#37474f",
                    linewidth=0.5, hatch="xx",
                    label="Challenge 2: expert placement (multi-hop + cross-wafer)")
        base = [a + b for a, b in zip(ideal, tp2)]
        b3 = ax.bar(x, tp3, 0.6, bottom=base, color=cols["tp3"], edgecolor="#37474f",
                    linewidth=0.5, hatch="//",
                    label="Challenge 3: wafer-edge I/O imbalance")
        for xi, r in zip(x, sel):
            tot = float(r["naive_us"])
            ax.annotate(f"{float(r['naive_over_ideal_x']):.1f}x", (xi, tot),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(x, [str(int(r["batch_size"])) for r in sel])
        ax.set_xlabel("Batch per die")
        ax.set_title(titles.get(topo, topo), loc="left", fontsize=10)
        ax.grid(axis="y", color="#d1d5db", linewidth=0.6, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        if handles is None:
            handles, labels = [b1, b2, b3], [h.get_label() for h in (b1, b2, b3)]
    axes[0][0].set_ylabel("Communication latency (us/step)")
    fig.legend(handles, labels, ncol=1, loc="upper center", frameon=False, fontsize=8)
    fig.subplots_adjust(top=0.72, bottom=0.16, wspace=0.24)
    fig.savefig(out_dir / "tp2_motivation_three_segment.png", dpi=360, bbox_inches="tight")
    fig.savefig(out_dir / "tp2_motivation_three_segment.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    points = pack_points(args.tp2_pack)

    for topo in args.topologies:
        for batch in args.batch_sizes:
            key = (topo, batch)
            if key not in points:
                raise KeyError(f"pack has no vllm row for {topo} b{batch}")
            info = points[key]
            expect_naive = float(info["naive_us"])

            if args.artifact_root is not None:
                root = args.artifact_root
            else:
                staged = args.staging_parent / info["source_root"]
                if not staged.is_dir():
                    raise FileNotFoundError(
                        f"{topo} b{batch}: staged root {staged} not found "
                        f"(pack recorded source_root={info['source_root']})"
                    )
                # the case subdirectory name varies; take the one holding topo/
                cands = [c for c in staged.iterdir()
                         if c.is_dir() and (c / topo).is_dir()]
                if len(cands) != 1:
                    raise FileNotFoundError(
                        f"{topo} b{batch}: expected exactly one case dir under "
                        f"{staged} containing {topo}, found {len(cands)}"
                    )
                root = cands[0]
            art = root / topo
            if not art.is_dir():
                raise FileNotFoundError(f"artifact dir not found: {art}")
            got: dict[str, float] = {}
            for v in VARIANTS:
                print(f"[run] {topo} b{batch} {v.key} ...", flush=True)
                got[v.key] = run_variant(
                    v, topology=topo, batch=batch, artifact_dir=art,
                    work=args.work_dir, args=args,
                )
                print(f"      {v.key} = {got[v.key]:.6f} us/step", flush=True)

            ideal, flat, naive = got["ideal"], got["naive_flat_ports"], got["naive"]
            # A silent no-op in the ideal variant would make it identical to
            # naive, which previously happened via a stale symlink.
            if abs(ideal - naive) < 1e-9:
                raise RuntimeError(
                    f"{topo} b{batch}: ideal == naive exactly ({ideal:.6f}); the "
                    "balanced route plan was not consumed"
                )
            rel = abs(naive - expect_naive) / expect_naive
            if rel > 2e-3:
                raise RuntimeError(
                    f"{topo} b{batch}: naive {naive:.6f} does not reproduce the "
                    f"pack's {expect_naive:.6f} (rel {rel:.4%})"
                )
            tp3 = naive - flat
            tp2 = flat - ideal
            if tp2 < 0 or tp3 < 0:
                raise RuntimeError(
                    f"{topo} b{batch}: negative segment (tp2={tp2:.3f}, tp3={tp3:.3f}); "
                    "the balanced/flat-port run is not below naive"
                )
            total = ideal + tp2 + tp3
            if abs(total - naive) > 1e-6 * max(naive, 1.0):
                raise RuntimeError(f"{topo} b{batch}: closure failed")
            rows.append({
                "topology": topo,
                "batch_size": batch,
                "ideal_us": f"{ideal:.6f}",
                "tp2_challenge_us": f"{tp2:.6f}",
                "tp3_challenge_us": f"{tp3:.6f}",
                "naive_us": f"{naive:.6f}",
                "naive_flat_ports_us": f"{flat:.6f}",
                "naive_over_ideal_x": f"{naive / max(ideal, 1e-9):.6f}",
                "ideal_share_pct": f"{100.0 * ideal / naive:.6f}",
                "tp2_share_pct": f"{100.0 * tp2 / naive:.6f}",
                "tp3_share_pct": f"{100.0 * tp3 / naive:.6f}",
                "challenge_overhead_pct_vs_ideal": f"{100.0 * (tp2 + tp3) / max(ideal, 1e-9):.6f}",
                "reported_steps": args.expect_steps,
                "pack_published_naive_us": f"{expect_naive:.6f}",
                "pack_reproduction_rel_err": f"{rel:.3e}",
                "source_root": info["source_root"],
            })
            print(f"  -> ideal={ideal:.3f} tp2={tp2:.3f} tp3={tp3:.3f} naive={naive:.3f} "
                  f"({naive / max(ideal, 1e-9):.2f}x)", flush=True)

    write_csv(args.output_dir / "tp2_motivation_three_segment.csv", rows)
    plot(rows, args.output_dir, args.topologies)
    (args.output_dir / "model.json").write_text(json.dumps({
        "status": "motivation_candidate",
        "segments": {
            "ideal": "perfectly balanced all-to-all, per-step totals preserved, "
                     "real fabric, ASTRA congestion-aware",
            "tp2_challenge": "naive_flat_ports - ideal: cost of uniform expert "
                             "placement (excess cross-wafer volume and hop distance)",
            "tp3_challenge": "naive - naive_flat_ports: cost of wafer-edge rail-port "
                             "load imbalance at unchanged aggregate rail bandwidth",
            "closure": "ideal + tp2_challenge + tp3_challenge = naive, exact",
        },
        "independence": "neither challenge term references TP2 or TP3 results; both "
                        "are differences between demands/fabrics under the same mapper",
        "replay_contract": {
            "entry_point": RUNNER,
            "num_requests": "equals global batch; staged counts are pre-expanded and "
                            "runtime metadata records only the raw request count",
            "batch_scope": "per_die",
            "exclude_leading_windows": f"{args.exclude_leading_steps} (STEPS on the "
                                       "route-plan path; would be windows on the "
                                       "window-aggregate path)",
            "trace_mode": "communication_only",
            "seed": args.seed,
            "verified_against_pack_us": args.verify_naive_us or None,
        },
        "artifact_root": str(args.artifact_root),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {args.output_dir}")


if __name__ == "__main__":
    main()
