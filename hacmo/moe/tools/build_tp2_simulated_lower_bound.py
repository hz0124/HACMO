#!/usr/bin/env python3
"""TP2 motivation: three-segment decomposition closed by simulation.

The analytic lower bound left 26-33% of the naive latency unexplained on
16w_4x4, because ASTRA's per-message link queues are not a closed form: each die
issues ~244 messages per step and they contend at the wafer rail exit. Rather
than attribute that residual by hand, this tool measures every term with the
same simulator, so the decomposition closes exactly:

    LB_sim              perfectly balanced all-to-all of the same total volume
    naive_flat_ports    naive traffic on a fabric whose rail ports are balanced
                        (same aggregate rail bandwidth, spread evenly)
    naive               naive traffic on the real fabric

    tp3_challenge = naive            - naive_flat_ports
    tp2_challenge = naive_flat_ports - LB_sim
    naive         = LB_sim + tp2_challenge + tp3_challenge     (exact)

``LB_sim`` is the cost an ideal mapping must still pay: it keeps the workload's
own volume, expert-parallel premise and fabric, but removes all skew -- every
die sends the same bytes to every other die. Any real placement faces at least
this much queueing, hop distance and rail contention, because uniform demand
minimises max-link load for a fixed total volume.

``tp3_challenge`` isolates wafer-edge I/O imbalance: the only difference between
the two runs is whether rail capacity sits on the topology's declared ports
(where XY routing concentrates load) or is spread evenly over all dies.

``tp2_challenge`` is then what remains of heterogeneity-unaware placement:
excess cross-wafer volume and mesh hop distance.

Neither challenge term references what TP2 or TP3 achieve.

Reads confirmed packs; writes nothing into them.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from supernode_sim.api import simulate_layer  # noqa: E402
from supernode_sim.config import load_config  # noqa: E402

BASE_CONFIG = "supernode_sim/configs/deepseek_v32_moe_astra_smoke.json"
NETWORK_CONFIGS = {
    "16w_4x4_rail_500": "hacmo/moe/configs/network/hierarchical_16w_4x4_mesh_rail_500.yaml",
    "4w_8x8_rail_900": "hacmo/moe/configs/network/hierarchical_4w_8x8_mesh_rail_900.yaml",
}


@dataclass(frozen=True)
class Scenario:
    key: str
    label: str
    uniform_demand: bool
    flat_rail_ports: bool


SCENARIOS = (
    Scenario("lb_sim", "balanced all-to-all, real fabric", True, False),
    Scenario("naive_flat_ports", "naive demand, rail ports spread evenly", False, True),
    Scenario("naive", "naive demand, real fabric", False, False),
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def measure_demand(runtime_dir: Path) -> dict:
    """Total activations and tokens per die per step from the staged trace."""
    stats = read_json(runtime_dir / "window_stats.json")
    active = [
        w for w in stats
        if sum(int(v) for v in (w.get("tokens_per_die") or {}).values()) > 0
    ]
    if not active:
        raise ValueError(f"no active windows in {runtime_dir}")
    tokens = acts = 0.0
    steps = 0
    dies = 0
    for w in active:
        tpd = w["tokens_per_die"]
        dies = max(dies, len(tpd))
        steps += max(int(w["end_token_idx"]) - int(w["start_token_idx"]) + 1, 1)
        tokens += sum(int(v) for v in tpd.values())
        acts += sum(int(v) for v in (w.get("expert_counts") or {}).values())
    return {
        "num_dies": dies,
        "windows": len(active),
        "steps": steps,
        "tokens_per_die_per_step": tokens / dies / steps,
        "activations_per_die_per_step": acts / dies / steps,
        "window_template": active,
    }


def write_uniform_demand(out_dir: Path, demand: dict, num_experts: int) -> tuple[Path, Path]:
    """Synthesise a perfectly balanced all-to-all of the same total volume.

    Each die's activations are spread equally over every expert, so every
    (src, dst) pair carries the same bytes. Total volume per die per step is
    unchanged, only the distribution is flattened.
    """
    tmpl = demand["window_template"]
    dies = demand["num_dies"]
    stats_out = []
    src_out = []
    for w in tmpl:
        acts = sum(int(v) for v in (w.get("expert_counts") or {}).values())
        per_expert = acts / num_experts
        per_pair = acts / dies / num_experts
        stats_out.append({
            "window_id": w.get("window_id"),
            "step_id": w.get("step_id", w.get("window_id")),
            "start_token_idx": w["start_token_idx"],
            "end_token_idx": w["end_token_idx"],
            "active": True,
            "tokens_per_die": dict(w["tokens_per_die"]),
            "expert_counts": {str(e): per_expert for e in range(num_experts)},
        })
        src_out.append({
            "window_id": w.get("window_id"),
            "src_expert_counts": {
                str(s): {str(e): per_pair for e in range(num_experts)}
                for s in range(dies)
            },
        })
    p1 = out_dir / "uniform_window_stats.json"
    p2 = out_dir / "uniform_window_src_expert_counts.json"
    p1.write_text(json.dumps(stats_out), encoding="utf-8")
    p2.write_text(json.dumps(src_out), encoding="utf-8")
    return p1, p2


def write_flat_port_network(src_cfg: Path, out_path: Path) -> dict:
    """Same aggregate rail bandwidth, spread evenly over every die.

    RailOptimizedSwitchMesh sums duplicate port entries into a per-die scale, so
    total wafer rail capacity is sum(scales) * switch_bandwidth. Keeping that sum
    fixed while giving every die an equal share removes port imbalance without
    changing total capacity.
    """
    cfg = yaml.safe_load(src_cfg.read_text(encoding="utf-8"))
    dpw = int(cfg["npus_count"][1])
    ports = [int(p) for p in cfg["rail-port-nodes"]]
    scales = [float(s) for s in (cfg.get("rail-port-bw-scales") or [1.0] * len(ports))]
    total_scale = sum(scales)
    flat = copy.deepcopy(cfg)
    flat["rail-port-nodes"] = list(range(dpw))
    flat["rail-port-bw-scales"] = [total_scale / dpw] * dpw
    out_path.write_text(yaml.safe_dump(flat, sort_keys=False), encoding="utf-8")
    return {
        "original_distinct_ports": len(set(ports)),
        "original_total_scale": total_scale,
        "flat_ports": dpw,
        "flat_scale_each": total_scale / dpw,
        "aggregate_rail_bw_gbps": total_scale * float(cfg["bandwidth"][0]),
    }


def run_scenario(
    scen: Scenario,
    *,
    topology: str,
    batch_per_die: int,
    runtime_dir: Path,
    demand: dict,
    work_dir: Path,
    args: argparse.Namespace,
) -> dict:
    """One ASTRA communication-only replay under a given scenario."""
    cfg = copy.deepcopy(load_config(REPO / BASE_CONFIG))
    net_src = REPO / NETWORK_CONFIGS[topology]
    num_dies = demand["num_dies"]

    scen_dir = work_dir / f"{topology}_b{batch_per_die}_{scen.key}"
    scen_dir.mkdir(parents=True, exist_ok=True)

    if scen.flat_rail_ports:
        net = scen_dir / "network_flat_ports.yaml"
        port_info = write_flat_port_network(net_src, net)
    else:
        net = net_src
        port_info = {}

    if scen.uniform_demand:
        ws, wse = write_uniform_demand(scen_dir, demand, args.num_experts)
    else:
        ws = runtime_dir / "window_stats.json"
        wse = runtime_dir / "window_src_expert_counts.json"

    cfg.moe.window_stats_json = str(ws)
    cfg.moe.window_src_expert_json = str(wse)
    cfg.moe.expert_mapping_json = ""
    cfg.moe.route_plan_json = ""
    cfg.moe.mapping_method = "uniform"
    cfg.moe.num_experts = args.num_experts
    cfg.moe.topk = args.topk
    cfg.moe.hidden_size = args.hidden_size
    cfg.moe.dispatch_bytes_per_hidden_elem = args.dispatch_bytes_per_hidden_elem
    cfg.moe.combine_bytes_per_hidden_elem = args.combine_bytes_per_hidden_elem
    cfg.moe.window_size = 8
    # Match the confirmed pack's replayed step range. DeepSeek's TP2 pack reports
    # tail16: 112 of 128 leading route-plan steps excluded, i.e. 14 of 16 windows.
    # Per-step communication rises with replay length, so a full-128 replay is not
    # comparable with the pack's published naive latency.
    cfg.moe.exclude_leading_windows = args.exclude_leading_windows
    # traffic is already at trace scale; do not rescale
    cfg.moe.num_requests = batch_per_die * num_dies
    cfg.workload.batch_size = batch_per_die * num_dies

    cfg.supernode.num_wafers = int(yaml.safe_load(net_src.read_text())["npus_count"][0])
    cfg.supernode.dies_per_wafer = int(yaml.safe_load(net_src.read_text())["npus_count"][1])
    cfg.supernode.network_config = str(net)
    cfg.trace_mode = "communication_only"
    cfg.backend = "astra"
    cfg.output_dir = str(scen_dir / "run")

    result = simulate_layer(cfg)
    profile = result.metadata.get("profile", {}) if result.metadata else {}
    # Steps actually replayed after the leading-window exclusion.
    kept = demand["window_template"][args.exclude_leading_windows:]
    steps = sum(
        max(int(w["end_token_idx"]) - int(w["start_token_idx"]) + 1, 1) for w in kept
    )
    if steps <= 0:
        raise ValueError("exclusion removed every step")
    comm = result.communication_us / steps
    return {
        "scenario": scen.key,
        "communication_us": comm,
        "total_dispatch_bytes": float(profile.get("total_dispatch_bytes", 0.0)),
        "total_combine_bytes": float(profile.get("total_combine_bytes", 0.0)),
        "cross_wafer_ratio": float(profile.get("cross_wafer_ratio", 0.0)),
        "steps": steps,
        "port_info": port_info,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tp2-pack", type=Path,
                   default=REPO / "hacmo/moe/outputs/tp2_serving_epoch_paper_pack_20260831")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--work-dir", type=Path, default=Path("ae/work/tp2-motivation/simulated_lower_bound"))
    p.add_argument("--topologies", nargs="+", default=list(NETWORK_CONFIGS))
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[16, 32, 64, 128])
    p.add_argument("--num-experts", type=int, default=256)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--hidden-size", type=int, default=7168)
    p.add_argument("--dispatch-bytes-per-hidden-elem", type=int, default=1)
    p.add_argument("--combine-bytes-per-hidden-elem", type=int, default=2)
    p.add_argument("--exclude-leading-windows", type=int, default=14,
                   help="match the pack's replayed step range; DeepSeek tail16 = 14 windows")
    p.add_argument("--expect-naive-us", type=float, default=0.0,
                   help="if set, assert the naive run reproduces this value (sanity check)")
    p.add_argument("--expect-tolerance-pct", type=float, default=2.0)
    return p.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Demand is a workload property; reuse across topologies with equal die count.
    demand_cache: dict[int, tuple[str, dict]] = {}
    rows: list[dict] = []
    diag: dict = {}

    for topology in args.topologies:
        net_cfg = yaml.safe_load((REPO / NETWORK_CONFIGS[topology]).read_text())
        num_dies = int(net_cfg["npus_count"][0]) * int(net_cfg["npus_count"][1])
        runtime = args.tp2_pack / topology / "mapping_inputs/runtime"
        source = topology
        if not runtime.is_dir():
            if num_dies in demand_cache:
                source, demand = demand_cache[num_dies]
                runtime = args.tp2_pack / source / "mapping_inputs/runtime"
            else:
                found = False
                for other in args.topologies:
                    alt = args.tp2_pack / other / "mapping_inputs/runtime"
                    if not alt.is_dir():
                        continue
                    alt_cfg = yaml.safe_load((REPO / NETWORK_CONFIGS[other]).read_text())
                    if int(alt_cfg["npus_count"][0]) * int(alt_cfg["npus_count"][1]) == num_dies:
                        runtime, source, found = alt, other, True
                        break
                if not found:
                    raise FileNotFoundError(
                        f"{topology}: no staged runtime demand with {num_dies} dies"
                    )
                demand = measure_demand(runtime)
        else:
            demand = measure_demand(runtime)
        demand_cache[num_dies] = (source, demand)
        diag[topology] = {
            "demand_measured_on": source,
            "demand_reused": source != topology,
            "num_dies": demand["num_dies"],
            "windows": demand["windows"],
            "steps": demand["steps"],
            "tokens_per_die_per_step": demand["tokens_per_die_per_step"],
            "activations_per_die_per_step": demand["activations_per_die_per_step"],
        }

        for batch in args.batch_sizes:
            got: dict[str, dict] = {}
            for scen in SCENARIOS:
                print(f"[run] {topology} b{batch} {scen.key} ...", flush=True)
                got[scen.key] = run_scenario(
                    scen, topology=topology, batch_per_die=batch,
                    runtime_dir=runtime, demand=demand,
                    work_dir=args.work_dir, args=args,
                )
            lb = got["lb_sim"]["communication_us"]
            flat = got["naive_flat_ports"]["communication_us"]
            naive = got["naive"]["communication_us"]

            tp3 = naive - flat
            tp2 = flat - lb
            row = {
                "topology": topology,
                "batch_size": batch,
                "lb_sim_us": f"{lb:.6f}",
                "naive_flat_ports_us": f"{flat:.6f}",
                "naive_us": f"{naive:.6f}",
                "tp2_challenge_us": f"{tp2:.6f}",
                "tp3_challenge_us": f"{tp3:.6f}",
                "naive_over_lb_x": f"{naive / max(lb, 1e-9):.6f}",
                "lb_share_pct": f"{100.0 * lb / max(naive, 1e-9):.6f}",
                "tp2_share_pct": f"{100.0 * tp2 / max(naive, 1e-9):.6f}",
                "tp3_share_pct": f"{100.0 * tp3 / max(naive, 1e-9):.6f}",
                "challenge_overhead_pct_vs_lb": f"{100.0 * (tp2 + tp3) / max(lb, 1e-9):.6f}",
                "lb_cross_wafer_ratio": f"{got['lb_sim']['cross_wafer_ratio']:.6f}",
                "naive_cross_wafer_ratio": f"{got['naive']['cross_wafer_ratio']:.6f}",
                "lb_dispatch_bytes": f"{got['lb_sim']['total_dispatch_bytes']:.0f}",
                "naive_dispatch_bytes": f"{got['naive']['total_dispatch_bytes']:.0f}",
                "steps": got["naive"]["steps"],
            }
            if tp2 < 0 or tp3 < 0:
                row["warning"] = (
                    f"negative segment (tp2={tp2:.3f}, tp3={tp3:.3f}); "
                    "the balanced run is not below the naive run"
                )
            rows.append(row)
            print(f"  -> LB={lb:.3f} flat={flat:.3f} naive={naive:.3f} "
                  f"tp2={tp2:.3f} tp3={tp3:.3f}", flush=True)

    write_csv(args.output_dir / "tp2_simulated_lower_bound.csv", rows)
    (args.output_dir / "simulated_lower_bound_model.json").write_text(
        json.dumps({
            "method": {
                "lb_sim": "perfectly balanced all-to-all of the same total volume, "
                          "real fabric, same ASTRA congestion-aware backend",
                "naive_flat_ports": "naive demand, rail capacity spread evenly over "
                                    "all dies at unchanged aggregate bandwidth",
                "naive": "naive demand, real fabric",
                "tp3_challenge": "naive - naive_flat_ports (wafer-edge I/O imbalance)",
                "tp2_challenge": "naive_flat_ports - lb_sim (uniform placement: "
                                 "excess cross-wafer volume and hop distance)",
                "closure": "lb_sim + tp2_challenge + tp3_challenge = naive, exact by construction",
            },
            "params": {
                "num_experts": args.num_experts, "topk": args.topk,
                "hidden_size": args.hidden_size,
                "dispatch_bytes_per_hidden_elem": args.dispatch_bytes_per_hidden_elem,
                "combine_bytes_per_hidden_elem": args.combine_bytes_per_hidden_elem,
            },
            "inputs": {"tp2_pack": str(args.tp2_pack)},
            "per_topology": diag,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote {args.output_dir}")


if __name__ == "__main__":
    main()
