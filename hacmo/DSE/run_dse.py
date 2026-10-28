#!/usr/bin/env python3
"""Run HACMO architecture DSE.

The DSE keeps all experiment-specific modeling under hacmo/DSE.  Performance
uses the same decomposition as the overall figure: sparse attention on
supernode_sim's attention backend, MoE communication through ASTRA plus the
TP3 crossbar plugin, and end-to-end TPS from the decode TPOT.
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import json
import math
import sys
from pathlib import Path
from typing import Any

import yaml

DSE_DIR = Path(__file__).resolve().parent
REPO = DSE_DIR.parents[1]
if str(DSE_DIR) not in sys.path:
    sys.path.insert(0, str(DSE_DIR))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from area_model import ArchCase, area_breakdown, validate_case
from case_generator import generate_cases
from energy_model import estimate_energy
from plot_dse import plot_all
from supernode_sim.api import simulate_layer
from supernode_sim.backends.utils import avg_hbm_hops, avg_remote_hops, compute_attention_flops, near_sram_hops
from supernode_sim.config import load_config
from supernode_sim.plugins.crossbar import crossbar_comm_us_per_step, load_crossbar_config
from supernode_sim.tracegen.moe import (
    build_combine_matrices,
    build_dispatch_matrices,
    build_window_counts,
    build_window_mappings,
    solve_mapping,
)
from supernode_sim.types import CachePolicy, HardwareConfig, ParallelConfig, SimulationConfig, WorkloadConfig

BASE_MOE_CONFIG = "supernode_sim/configs/deepseek_v32_moe_astra_smoke.json"


def main() -> None:
    args = parse_args()
    const = yaml.safe_load(Path(args.constants).read_text(encoding="utf-8"))
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.output_root) if args.output_root else DSE_DIR / "outputs" / f"dse_{ts}"
    out_root = out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        rows = read_csv(out_root / "raw_results.csv")
        pareto_rows = read_csv(out_root / "summary_pareto.csv")
        plot_all(rows, pareto_rows, out_root)
        if {r.get("mode", "") for r in rows} != {"edge_sensitivity"}:
            invalid_rows = read_csv(out_root / "invalid_cases.csv")
            write_analysis(out_root / "dse_analysis.md", rows, pareto_rows, invalid_rows)
        print(f"[dse] redrew figures under {out_root / 'figures'}")
        return

    runner = DSERunner(const=const, out_root=out_root, run_astra=not args.no_astra)
    rows, invalid_rows = runner.run(args.mode)
    pareto_rows = mark_pareto(rows)
    write_csv(out_root / "raw_results.csv", rows)
    write_csv(out_root / "summary_pareto.csv", pareto_rows)
    write_csv(out_root / "invalid_cases.csv", invalid_rows)
    dump_config(out_root, args, const, rows, invalid_rows)
    plot_all(rows, pareto_rows, out_root)
    write_analysis(out_root / "dse_analysis.md", rows, pareto_rows, invalid_rows)
    print(f"[dse] wrote {out_root}")
    print(f"[dse] raw results: {out_root / 'raw_results.csv'}")
    print(f"[dse] figures: {out_root / 'figures'}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run HACMO DSE.")
    p.add_argument(
        "--mode",
        choices=["sram_compute", "core_hbm", "wafer_scale", "edge_bandwidth", "edge_sensitivity", "batch_sweep", "pareto", "all"],
        default="all",
    )
    p.add_argument("--constants", default=str(DSE_DIR / "dse_constants.yaml"))
    p.add_argument("--output-root", default="")
    p.add_argument("--no-astra", action="store_true", help="Use generated traces/crossbar analytical comm only.")
    p.add_argument("--plot-only", action="store_true", help="Reuse raw_results.csv and summary_pareto.csv to redraw figures.")
    return p.parse_args()


class DSERunner:
    def __init__(self, const: dict[str, Any], out_root: Path, run_astra: bool = True) -> None:
        self.const = const
        self.out_root = out_root
        self.run_astra = run_astra
        self.net_dir = out_root / "generated_configs" / "network"
        self.net_dir.mkdir(parents=True, exist_ok=True)
        self.moe_cache: dict[tuple, dict[str, Any]] = {}

    def run(self, mode: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        cases = generate_cases(mode, self.const)
        rows: list[dict[str, Any]] = []
        invalid: list[dict[str, Any]] = []
        for idx, case in enumerate(cases, start=1):
            ok, reason, area = validate_case(case, self.const)
            if not ok:
                invalid.append(self.invalid_row(case, reason, area))
                print(f"[dse] {idx:03d}/{len(cases)} invalid {case.case_id}: {reason}", flush=True)
                continue
            print(
                f"[dse] {idx:03d}/{len(cases)} run {case.case_id}: "
                f"{case.label}, {case.num_wafers}W {case.wafer_side}x{case.wafer_side}",
                flush=True,
            )
            row = self.simulate_case(case, area)
            rows.append(row)

        if not rows:
            return [], invalid
        base = choose_base(rows)
        base_tps = float(base["throughput_tps"])
        for row in rows:
            row["normalized_throughput"] = round(float(row["throughput_tps"]) / max(base_tps, 1e-12), 6)
        pareto_ids = {r["case_id"] for r in pareto_front(rows)}
        for row in rows:
            row["is_pareto"] = "1" if row["case_id"] in pareto_ids else "0"
        return rows, invalid

    def invalid_row(self, case: ArchCase, reason: str, area: dict[str, float]) -> dict[str, Any]:
        return {
            **case.to_dict(),
            "reason": reason,
            "A_required": round(area["wafer_required_area_mm2"], 6),
            "A_budget": round(area["wafer_budget_area_mm2"], 6),
            "hbm_capacity_gb_per_die": round(area["hbm_capacity_gb_per_die"], 6),
        }

    def simulate_case(self, case: ArchCase, area: dict[str, float]) -> dict[str, Any]:
        wl = self.const["workload"]
        original_batch = int(wl["batch_size"])
        if case.batch_size_override is not None:
            wl["batch_size"] = int(case.batch_size_override)
        try:
            return self._simulate_case_inner(case, area)
        finally:
            wl["batch_size"] = original_batch

    def _simulate_case_inner(self, case: ArchCase, area: dict[str, float]) -> dict[str, Any]:
        wl = self.const["workload"]
        batch_per_die = int(wl["batch_size"])
        global_batch = batch_per_die * case.active_die_count
        inter_bw_gbps, mesh_bw_gbps = resolved_bandwidths(case, self.const)
        switch_latency_ns = float(case.switch_latency_ns if case.switch_latency_ns is not None else self.const["topology"]["switch_latency_ns"])
        crossbar_latency_ns = float(case.crossbar_latency_ns if case.crossbar_latency_ns is not None else 20.0)
        attn = self.attention(case)
        moe = self.moe(case)
        layer_attn_us = attn["total_us"]
        layer_moe_us = moe["total_us"]
        tpot_us = int(wl["layers"]) * layer_attn_us + int(wl["moe_layers"]) * layer_moe_us
        throughput = global_batch * 1e6 / max(tpot_us, 1e-12)

        counters = self.energy_counters(case, attn, moe, tpot_us)
        energy = estimate_energy(case, counters, self.const)
        row = {
            **case.to_dict(),
            "hbm_capacity_gb_per_die": round(area["hbm_capacity_gb_per_die"], 6),
            "wafer_area_utilization": round(area["wafer_area_utilization"], 6),
            "wafer_area_headroom": round(area["wafer_area_headroom"], 6),
            "wafer_area_margin_mm2": round(area["wafer_area_margin_mm2"], 6),
            "hbm_peak_tbps": round(hbm_peak_tbps(case, self.const), 6),
            "noc_bw_tbps": round(noc_bw_tbps(case, self.const), 6),
            "inter_wafer_bw_gbps": round(inter_bw_gbps, 6),
            "mesh_bw_gbps": round(mesh_bw_gbps, 6),
            "switch_latency_ns": round(switch_latency_ns, 6),
            "crossbar_latency_ns": round(crossbar_latency_ns, 6),
            "edge_bandwidth_index": round(edge_bandwidth_index(case, self.const), 6),
            "wafer_required_area_mm2": round(area["wafer_required_area_mm2"], 6),
            "wafer_budget_area_mm2": round(area["wafer_budget_area_mm2"], 6),
            "die_area_mm2": round(area["die_area_mm2"], 6),
            "batch_size": batch_per_die,
            "batch_size_per_die": batch_per_die,
            "global_batch_size": global_batch,
            "batch_semantics": "per_die",
            "context_len": int(wl["context_len"]),
            "attention_layer_us": round(layer_attn_us, 6),
            "moe_layer_us": round(layer_moe_us, 6),
            "attention_stack_ms": round(int(wl["layers"]) * layer_attn_us / 1000.0, 6),
            "moe_stack_ms": round(int(wl["moe_layers"]) * layer_moe_us / 1000.0, 6),
            "tpot_ms": round(tpot_us / 1000.0, 6),
            "throughput_tps": round(throughput, 6),
            "sparse_hit_rate": round(attn["hit_rate"], 6),
            "attention_compute_us": round(attn["compute_us"], 6),
            "attention_hbm_us": round(attn["hbm_us"], 6),
            "attention_sram_noc_us": round(attn["sram_noc_us"], 6),
            "attention_allreduce_us": round(attn["allreduce_us"], 6),
            "moe_compute_us": round(moe["compute_us"], 6),
            "moe_comm_us": round(moe["communication_us"], 6),
            "communication_us": round(moe["communication_us"], 6),
            "crossbar_comm_us": round(moe["crossbar_comm_us"], 6),
            "rail_analytical_comm_us": round(moe["rail_analytical_comm_us"], 6),
            "per_port_bw_gbps": round(moe["per_port_bw_gbps"], 6),
            "moe_cross_wafer_ratio": round(moe["cross_wafer_ratio"], 6),
            "cross_wafer_locality": round(max(0.0, 1.0 - float(moe["cross_wafer_ratio"])), 6),
            "bottleneck": classify_bottleneck(case, attn, moe),
            **{k: round(v, 6) for k, v in counters.items()},
            **{k: round(v, 6) for k, v in energy.items()},
            "attn_source": "supernode_sim_analytic_attention|dse_capacity_hit_model",
            "moe_source": moe["source"],
        }
        return row

    def attention(self, case: ArchCase) -> dict[str, float]:
        wl = self.const["workload"]
        hw = self.const["hardware"]
        batch = int(wl["batch_size"])
        topk = int(wl["sparse_topk"])
        cfg = SimulationConfig(
            backend="analytic",
            hardware=HardwareConfig(
                array_rows=case.array_rows,
                array_cols=case.array_cols,
                hbm_peak_tbps=hbm_peak_tbps(case, self.const),
                noc_bw_tbps=noc_bw_tbps(case, self.const),
                compute_peak_tflops=case.compute_peak_tflops,
                total_sram_mb=case.total_sram_mb,
            ),
            workload=WorkloadConfig(
                model_name=str(wl["model_name"]),
                layer_type="attention",
                batch_size=batch,
                context_len=topk,
                topk=topk,
                selected_vector_bytes=int(wl["selected_vector_bytes"]),
                hidden_size=int(wl["hidden_size"]),
                n_heads=int(wl["attention_heads"]),
                head_dim=int(wl["head_dim"]),
                n_layers=int(wl["layers"]),
            ),
            parallel=ParallelConfig(cp_degree=1, horizontal_strategy="dp", tp_degree_horizontal=1),
            mapping=CachePolicy(
                name=f"dse_{case.case_id}",
                cache_policy="layer_prefetch",
                static_mapping=True,
                dynamic_prefetch=True,
                row_only_static_mapping=True,
                prefetch_overlap_us=100.0,
            ),
        )
        raw = simulate_layer(cfg)
        hit = sparse_hit_rate(case, self.const)
        selected_bytes = 2.0 * batch * int(wl["selected_vector_bytes"]) * topk
        hbm_bytes = selected_bytes * (1.0 - hit)
        sram_local = selected_bytes * hit * 0.75
        sram_near = selected_bytes * hit * 0.20
        sram_remote = selected_bytes * hit * 0.05
        hbm_us = hbm_bytes / max(hbm_peak_tbps(case, self.const) * 1e12 * 0.21, 1.0) * 1e6
        hbm_us += hbm_bytes * avg_hbm_hops(case.array_rows, case.array_cols) / max(noc_bw_tbps(case, self.const) * 1e12, 1.0) * 1e6
        near_us = sram_near * near_sram_hops(case.array_cols) / max(noc_bw_tbps(case, self.const) * 1e12, 1.0) * 1e6
        remote_us = sram_remote * avg_remote_hops(case.array_rows, case.array_cols) / max(noc_bw_tbps(case, self.const) * 1e12, 1.0) * 1e6
        prefetch_us = max((sram_local + sram_near + sram_remote) / max(hbm_peak_tbps(case, self.const) * 1e12 * 0.21, 1.0) * 1e6 - 100.0, 0.0)
        allreduce_us = attention_allreduce_us(batch, self.const)
        total = raw.compute_us + hbm_us + near_us + remote_us + prefetch_us + allreduce_us
        flops = compute_attention_flops(
            batch,
            topk,
            int(wl["hidden_size"]),
            int(wl["attention_heads"]),
            int(wl["head_dim"]),
        )
        return {
            "total_us": total,
            "compute_us": raw.compute_us,
            "hbm_us": hbm_us,
            "sram_noc_us": near_us + remote_us,
            "prefetch_exposed_us": prefetch_us,
            "allreduce_us": allreduce_us,
            "hit_rate": hit,
            "flops": flops,
            "hbm_bytes": hbm_bytes,
            "sram_read_bytes": sram_local + sram_near + sram_remote,
            "sram_write_bytes": sram_local + sram_near + sram_remote,
            "noc_byte_hops": (
                hbm_bytes * avg_hbm_hops(case.array_rows, case.array_cols)
                + sram_near * near_sram_hops(case.array_cols)
                + sram_remote * avg_remote_hops(case.array_rows, case.array_cols)
            ),
        }

    def moe(self, case: ArchCase) -> dict[str, Any]:
        comm = self.moe_comm(case)
        ref_tflops = 1979.0
        ref_hbm = 4.8
        arch_tflops = max(case.compute_peak_tflops, 1e-9)
        arch_hbm = max(hbm_peak_tbps(case, self.const), 1e-9)
        compute_scale = max(ref_tflops / arch_tflops, ref_hbm / arch_hbm)
        compute_us = comm["ref_compute_us"] * compute_scale
        return {
            **comm,
            "compute_us": compute_us,
            "total_us": compute_us + float(comm["communication_us"]),
        }

    def moe_comm(self, case: ArchCase) -> dict[str, Any]:
        inter_bw_gbps, mesh_bw_gbps = resolved_bandwidths(case, self.const)
        switch_latency_ns = float(case.switch_latency_ns if case.switch_latency_ns is not None else self.const["topology"]["switch_latency_ns"])
        crossbar_latency_ns = float(case.crossbar_latency_ns if case.crossbar_latency_ns is not None else 20.0)
        topo_key = (
            case.num_wafers,
            case.wafer_side,
            round(inter_bw_gbps, 6),
            round(mesh_bw_gbps, 6),
            round(switch_latency_ns, 6),
            round(crossbar_latency_ns, 6),
            int(self.const["workload"]["batch_size"]),
        )
        if topo_key in self.moe_cache:
            return self.moe_cache[topo_key]
        rail_cfg, crossbar_cfg = self.write_network_configs(case)
        cfg = self.base_moe_config(case, rail_cfg)
        result = simulate_layer(cfg)
        profile = result.metadata.get("profile", {}) if result.metadata else {}
        steps = max(int(self.const["workload"]["decode_steps"]), 1)
        rail_comm_us = result.communication_us / steps if self.run_astra else 0.0
        xb = self.crossbar_comm(cfg, crossbar_cfg, rail_comm_us, crossbar_latency_ns)
        out = {
            "source": "supernode_sim_astra_plus_crossbar_plugin",
            "communication_us": xb["communication_us"],
            "crossbar_comm_us": xb["crossbar_comm_us"],
            "rail_comm_us": rail_comm_us,
            "rail_analytical_comm_us": xb["rail_analytical_comm_us"],
            "crossbar_ratio": xb["ratio"],
            "switch_latency_ns": switch_latency_ns,
            "crossbar_latency_ns": xb["crossbar_latency_ns"],
            "per_port_bw_gbps": xb["per_port_bw_gbps"],
            "ref_compute_us": float(profile.get("analysis_compute_us_per_step", 0.0)),
            "cross_wafer_ratio": float(profile.get("cross_wafer_ratio", 0.0)),
            "avg_extra_replicas": float(profile.get("avg_extra_replicas", 0.0)),
            "max_extra_replicas": float(profile.get("max_extra_replicas", 0.0)),
            "total_dispatch_bytes": float(profile.get("total_dispatch_bytes", 0.0)) / steps,
            "total_combine_bytes": float(profile.get("total_combine_bytes", 0.0)) / steps,
            "cross_wafer_bytes": float(profile.get("cross_wafer_bytes", 0.0)) / steps,
            "same_wafer_bytes": float(profile.get("same_wafer_bytes", 0.0)) / steps,
            "output_root": result.metadata.get("output_root", "") if result.metadata else "",
        }
        self.moe_cache[topo_key] = out
        return out

    def base_moe_config(self, case: ArchCase, rail_cfg: Path) -> SimulationConfig:
        wl = self.const["workload"]
        topo = self.const["topology"]
        base = load_config(REPO / BASE_MOE_CONFIG)
        cfg = copy.deepcopy(base)
        cfg.backend = "astra"
        cfg.workload.model_name = str(wl["model_name"])
        cfg.workload.layer_type = "moe"
        cfg.workload.batch_size = int(wl["batch_size"])
        cfg.workload.decode_len = int(wl["decode_steps"])
        cfg.workload.hidden_size = int(wl["hidden_size"])
        cfg.moe.num_experts = int(wl["num_experts"])
        cfg.moe.topk = int(wl["moe_topk"])
        cfg.moe.num_requests = 64
        cfg.moe.window_size = 1
        cfg.moe.hidden_size = int(wl["hidden_size"])
        cfg.moe.bytes_per_hidden_elem = 2
        cfg.moe.dispatch_bytes_per_hidden_elem = 1
        cfg.moe.combine_bytes_per_hidden_elem = 2
        cfg.moe.compute_backend = "roofline_h200"
        cfg.moe.expert_intermediate_size = int(wl["expert_intermediate_size"])
        cfg.moe.h200_effective_tflops = 1979.0
        cfg.moe.h200_hbm_bw_tbps = 4.8
        cfg.moe.weight_bytes_per_elem = 1
        cfg.moe.activation_bytes_per_elem = 2
        cfg.moe.mapping_method = "ours_dynamic"
        cfg.moe.real_freq_json = str(wl["freq_json"])
        cfg.moe.locality_factor = 5.0
        cfg.moe.locality_wafer_shift = 4
        cfg.moe.seed = stable_seed("dse", "deepseek_v32")
        cfg.moe.window_stats_json = ""
        cfg.moe.window_src_expert_json = ""
        cfg.moe.expert_mapping_json = ""
        cfg.moe.window_mappings_json = ""
        cfg.moe.expert_mapping = {}
        cfg.moe.route_via_edge_ports = False
        cfg.supernode.num_wafers = case.num_wafers
        cfg.supernode.dies_per_wafer = case.dies_per_wafer
        try:
            cfg.supernode.network_config = str(rail_cfg.relative_to(REPO))
        except ValueError:
            cfg.supernode.network_config = str(rail_cfg)
        cfg.supernode.system_config = str(topo["system_config"])
        cfg.supernode.remote_memory_config = str(topo["remote_memory_config"])
        cfg.supernode.astra_sim_dir = str(topo["astra_sim_dir"])
        cfg.supernode.astra_network_backend = "congestion_aware"
        cfg.supernode.run_astra = self.run_astra
        cfg.supernode.keep_traces = True
        cfg.supernode.output_root = str(
            self.out_root / "runs" / "moe" / case.case_id
        )
        return cfg

    def crossbar_comm(self, cfg: SimulationConfig, crossbar_cfg: Path, rail_comm_us: float, crossbar_latency_ns: float) -> dict[str, float]:
        window_stats, window_src = build_window_counts(cfg, REPO)
        static_mapping = solve_mapping(cfg, window_src)
        window_mappings = build_window_mappings(cfg, window_stats, window_src, static_mapping)
        dispatch = build_dispatch_matrices(cfg, window_src, static_mapping, window_mappings=window_mappings)
        combine = build_combine_matrices(cfg, dispatch)
        xb_cfg = load_crossbar_config(str(crossbar_cfg))
        batch_scale = cfg.workload.batch_size / max(cfg.moe.num_requests, 1)
        xb = crossbar_comm_us_per_step(
            dispatch,
            combine,
            xb_cfg,
            method="dynamic_bw",
            batch_scale=batch_scale,
            window_size=cfg.moe.window_size,
            beta=0.5,
            rho=0.4,
            crossbar_latency_ns=crossbar_latency_ns,
        )
        rail_analytical = max(float(xb.get("rail_analytical_comm_us", 0.0)), 1e-9)
        if self.run_astra:
            ratio = min(float(xb["comm_us"]) / rail_analytical, 1.0)
            comm_us = rail_comm_us * ratio
        else:
            ratio = 1.0
            comm_us = float(xb["comm_us"])
        return {
            "communication_us": comm_us,
            "crossbar_comm_us": float(xb["comm_us"]),
            "rail_analytical_comm_us": float(xb["rail_analytical_comm_us"]),
            "crossbar_latency_ns": float(xb.get("crossbar_latency_ns", crossbar_latency_ns)),
            "per_port_bw_gbps": float(xb.get("per_port_bw_gbps", 0.0)),
            "ratio": ratio,
        }

    def write_network_configs(self, case: ArchCase) -> tuple[Path, Path]:
        topo = self.const["topology"]
        side = case.wafer_side
        inter_bw, mesh_bw = resolved_bandwidths(case, self.const)
        switch_latency_ns = float(case.switch_latency_ns if case.switch_latency_ns is not None else topo["switch_latency_ns"])
        common = {
            "topology": ["Switch", "Mesh"],
            "npus_count": [case.num_wafers, case.dies_per_wafer],
            "bandwidth": [inter_bw, mesh_bw],
            "latency": [switch_latency_ns, float(topo["mesh_latency_ns"])],
            "switch-dim-index": 0,
            "mesh-dim-index": 1,
        }
        prefix = self.net_dir / f"{case.case_id}_{case.num_wafers}w_{side}x{side}"
        rail = {
            **common,
            "rail-optimized": True,
            "rail-port-nodes": edge_port_nodes(side),
            "rail-port-bw-scales": [1.0] * (4 * side),
        }
        crossbar = {
            **common,
            "rail-optimized": False,
            "crossbar-mode": True,
            "crossbar-ports": 4 * side,
            "crossbar-latency-ns": float(case.crossbar_latency_ns if case.crossbar_latency_ns is not None else 20.0),
        }
        rail_path = prefix.with_name(prefix.name + "_rail.yaml")
        crossbar_path = prefix.with_name(prefix.name + "_crossbar.yaml")
        rail_path.write_text(yaml.safe_dump(rail, sort_keys=False), encoding="utf-8")
        crossbar_path.write_text(yaml.safe_dump(crossbar, sort_keys=False), encoding="utf-8")
        return rail_path, crossbar_path

    def energy_counters(self, case: ArchCase, attn: dict[str, float], moe: dict[str, Any], tpot_us: float) -> dict[str, float]:
        wl = self.const["workload"]
        batch_per_die = float(wl["batch_size"])
        global_batch = batch_per_die * float(case.active_die_count)
        layers = float(wl["layers"])
        moe_layers = float(wl["moe_layers"])
        hidden = float(wl["hidden_size"])
        topk = float(wl["moe_topk"])
        inter = float(wl["expert_intermediate_size"])
        moe_flops = global_batch * topk * 6.0 * hidden * inter
        moe_weight_bytes, moe_activation_bytes, moe_active_expert_copies = moe_hbm_read_bytes_per_layer(
            case,
            moe,
            global_batch,
            self.const,
        )
        moe_bytes = (
            float(moe["total_dispatch_bytes"]) + float(moe["total_combine_bytes"])
        ) * float(case.active_die_count)
        cross_ratio = float(moe["cross_wafer_ratio"])
        cross_bytes = moe_bytes * cross_ratio
        same_bytes = max(moe_bytes - cross_bytes, 0.0)
        mesh_hops = avg_remote_hops(case.wafer_side, case.wafer_side)
        switch_count = max(1, math.ceil(case.num_wafers / 16.0))
        return {
            "latency_s": tpot_us / 1e6,
            "num_output_tokens": global_batch,
            "flops_total": layers * attn["flops"] * float(case.active_die_count)
            + moe_layers * moe_flops,
            "sram_read_bytes": layers * attn["sram_read_bytes"] * float(case.active_die_count),
            "sram_write_bytes": layers * attn["sram_write_bytes"] * float(case.active_die_count),
            "noc_byte_hops": layers * attn["noc_byte_hops"] * float(case.active_die_count),
            "hbm_read_bytes_actual": layers * attn["hbm_bytes"] * float(case.active_die_count)
            + moe_layers * (moe_weight_bytes + moe_activation_bytes),
            "hbm_write_bytes_actual": 0.0,
            "moe_weight_read_bytes": moe_layers * moe_weight_bytes,
            "moe_activation_read_bytes": moe_layers * moe_activation_bytes,
            "moe_active_expert_copies": moe_active_expert_copies,
            "d2d_byte_hops": moe_layers * same_bytes * mesh_hops,
            "inter_wafer_bytes_electrical": moe_layers * cross_bytes,
            "inter_wafer_bytes_optical": 0.0,
            "switch_bytes": moe_layers * cross_bytes,
            "active_die_count": float(case.active_die_count),
            "active_hbm_stack_count": float(case.active_die_count * case.hbm_stacks_per_die),
            "active_switch_count": float(switch_count),
        }


def hbm_peak_tbps(case: ArchCase, const: dict[str, Any]) -> float:
    return case.hbm_stacks_per_die * float(const["hardware"]["hbm_bandwidth_per_stack_tbps"])


def resolved_bandwidths(case: ArchCase, const: dict[str, Any]) -> tuple[float, float]:
    topo = const["topology"]
    bw_by_side = topo["inter_wafer_bw_gbps_by_side"]
    inter_bw = case.inter_wafer_bw_gbps
    if inter_bw is None:
        inter_bw = float(bw_by_side.get(case.wafer_side, bw_by_side.get(str(case.wafer_side))))
    mesh_bw = case.mesh_bw_gbps
    if mesh_bw is None:
        mesh_bw = float(topo["mesh_bandwidth_gbps"])
    return float(inter_bw), float(mesh_bw)


def edge_bandwidth_index(case: ArchCase, const: dict[str, Any]) -> float:
    inter_bw, mesh_bw = resolved_bandwidths(case, const)
    topo = const["topology"]
    bw_by_side = topo["inter_wafer_bw_gbps_by_side"]
    base_inter = float(bw_by_side.get(case.wafer_side, bw_by_side.get(str(case.wafer_side))))
    base_mesh = float(topo["mesh_bandwidth_gbps"])
    return math.sqrt(max(inter_bw, 1e-9) / max(base_inter, 1e-9) * max(mesh_bw, 1e-9) / max(base_mesh, 1e-9))


def noc_bw_tbps(case: ArchCase, const: dict[str, Any]) -> float:
    hw = const["hardware"]
    return float(hw["base_noc_bw_tbps"]) * math.sqrt(case.num_cores / float(hw["base_noc_array_cores"]))


def sparse_hit_rate(case: ArchCase, const: dict[str, Any]) -> float:
    hw = const["hardware"]
    wl = const["workload"]
    cache_bytes = case.total_sram_mb * float(hw["cache_ratio_for_sparse_attention"]) * 1024.0 * 1024.0
    entry_bytes = int(wl["selected_vector_bytes"]) * 2.0
    entries_per_request = cache_bytes / max(int(wl["batch_size"]) * entry_bytes, 1.0)
    topk = float(wl["sparse_topk"])
    lo = float(hw["min_sparse_hit_rate"])
    hi = float(hw["max_sparse_hit_rate"])
    tau = float(hw["sparse_cache_tau"])
    return min(hi, lo + (hi - lo) * (1.0 - math.exp(-entries_per_request / max(tau * topk, 1e-9))))


def attention_allreduce_us(batch: int, const: dict[str, Any]) -> float:
    hw = const["hardware"]
    wl = const["workload"]
    tp = int(hw["attention_allreduce_tp"])
    if tp <= 1:
        return 0.0
    bytes_per_rank = batch * int(wl["hidden_size"]) * 2
    transfer = 2.0 * (tp - 1) / tp * bytes_per_rank / max(float(hw["attention_allreduce_bw_tbps"]) * 1e12, 1.0) * 1e6
    latency = math.ceil(math.log2(tp)) * float(hw["attention_allreduce_latency_ns"]) / 1000.0
    return transfer + latency


def moe_hbm_read_bytes_per_layer(
    case: ArchCase,
    moe: dict[str, Any],
    global_batch: float,
    const: dict[str, Any],
) -> tuple[float, float, float]:
    wl = const["workload"]
    hidden = float(wl["hidden_size"])
    topk = float(wl["moe_topk"])
    inter = float(wl["expert_intermediate_size"])
    num_experts = float(wl["num_experts"])
    expert_tokens = global_batch * topk
    active_experts = min(num_experts, max(1.0, expert_tokens))
    extra_replicas = max(float(moe.get("avg_extra_replicas", 0.0)), 0.0)
    active_expert_copies = min(active_experts + extra_replicas, float(case.active_die_count))

    # Batched expert GEMMs reuse weights across all tokens routed to the same
    # expert copy in one decode step.  Activations still scale with expert-token
    # count.
    weight_bytes_per_expert = 3.0 * hidden * inter
    activation_bytes_per_expert_token = 3.0 * hidden * 2.0
    weight_bytes = active_expert_copies * weight_bytes_per_expert
    activation_bytes = expert_tokens * activation_bytes_per_expert_token
    return weight_bytes, activation_bytes, active_expert_copies


def edge_port_nodes(side: int) -> list[int]:
    ports: list[int] = []
    for c in range(side):
        ports.append(c)
    for r in range(side):
        ports.append(r * side + side - 1)
    for c in range(side - 1, -1, -1):
        ports.append((side - 1) * side + c)
    for r in range(side - 1, -1, -1):
        ports.append(r * side)
    return ports


def classify_bottleneck(case: ArchCase, attn: dict[str, float], moe: dict[str, Any]) -> str:
    if case.sram_per_core_mb >= 8.0 and case.compute_peak_tflops < 100.0:
        return "SRAM-over-provisioned"
    if attn["hbm_us"] > max(attn["compute_us"], attn["sram_noc_us"]) * 1.15:
        return "HBM-bound"
    if attn["sram_noc_us"] > attn["compute_us"] * 1.15:
        return "NoC-bound"
    if moe["communication_us"] > moe["compute_us"] * 1.2:
        return "edge-I/O-bound"
    if attn["compute_us"] + moe["compute_us"] > attn["hbm_us"] + attn["sram_noc_us"] + moe["communication_us"]:
        return "compute-bound"
    return "balanced"


def choose_base(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in rows:
        if row["case_id"] == "core_hbm_a16_h2":
            return row
    return min(rows, key=lambda r: abs(float(r["sram_per_core_mb"]) - 2.0) + abs(float(r["compute_per_core_tflops"]) - 0.512))


def pareto_front(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        t = float(row["throughput_tps"])
        h = float(row.get("wafer_area_headroom", 0.0))
        dominated = False
        for other in rows:
            if other is row:
                continue
            ot = float(other["throughput_tps"])
            oh = float(other.get("wafer_area_headroom", 0.0))
            if ot >= t and oh >= h and (ot > t or oh > h):
                dominated = True
                break
        if not dominated:
            out.append(row)
    return sorted(out, key=lambda r: float(r["throughput_tps"]))


def mark_pareto(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    front = pareto_front(rows)
    ids = {r["case_id"] for r in front}
    return [{**r, "is_pareto": "1" if r["case_id"] in ids else "0"} for r in front]


def write_analysis(path: Path, rows: list[dict[str, Any]], pareto_rows: list[dict[str, Any]], invalid: list[dict[str, Any]]) -> None:
    best_t = max(rows, key=lambda r: float(r["throughput_tps"]))
    best_headroom = max(rows, key=lambda r: float(r.get("wafer_area_headroom", 0.0)))
    wafer_best = max([r for r in rows if r["mode"] == "wafer_scale"], key=lambda r: float(r["throughput_tps"]), default=best_t)
    edge_best = min([r for r in rows if r["mode"] == "edge_bandwidth"], key=lambda r: float(r["moe_comm_us"]), default=best_t)
    frontier_rows = [
        r
        for r in rows
        if r["case_id"]
        in {
            "core_hbm_a16_h2",
            "core_hbm_a24_h2",
            "core_hbm_a24_h4",
            "core_hbm_a32_h2",
            "core_hbm_a32_h4",
            "core_hbm_a32_h8",
        }
    ]
    frontier_best = max(frontier_rows, key=lambda r: float(r["throughput_tps"]), default=best_t)
    frontier_tightest = max(frontier_rows, key=lambda r: float(r["wafer_area_utilization"]), default=best_t)
    sram_mid = find_row(rows, "sram_balanced_mid")
    sram_max = find_row(rows, "sram_sram_max")
    compute_heavy = find_row(rows, "sram_sram_small_compute_heavy")
    hbm2 = find_row(rows, "core_hbm_a32_h2")
    hbm4 = find_row(rows, "core_hbm_a32_h4")
    hbm8 = find_row(rows, "core_hbm_a32_h8")

    lines = [
        "# HACMO DSE Analysis",
        "",
        f"- Valid cases: {len(rows)}",
        f"- Invalid cases: {len(invalid)}",
        "- Batch semantics: `batch_size` is per active die; `global_batch_size = batch_size_per_die * active_die_count`.",
        f"- Best throughput: `{best_t['case_id']}` ({best_t['label']}), {float(best_t['throughput_tps']):.1f} tokens/s.",
        f"- Best area headroom: `{best_headroom['case_id']}` ({best_headroom['label']}), headroom {float(best_headroom.get('wafer_area_headroom', 0.0)):.3f}.",
        f"- Best wafer-scale throughput: `{wafer_best['case_id']}` with {int(wafer_best['num_wafers'])} wafers and {int(wafer_best['wafer_side'])}x{int(wafer_best['wafer_side'])} dies/wafer.",
        f"- Area-constrained core/HBM frontier best: `{frontier_best['case_id']}` with TPOT {float(frontier_best['tpot_ms']):.3f} ms and area utilization {float(frontier_best['wafer_area_utilization']):.3f}.",
        f"- Tightest valid frontier point: `{frontier_tightest['case_id']}` uses area utilization {float(frontier_tightest['wafer_area_utilization']):.3f}.",
        "",
        "## Modeling Audit",
        "",
        "- Performance follows the same decode formula as `hacmo/overall`: `TPS = batch * 1e6 / TPOT_us`, where TPOT is the attention stack plus MoE stack.",
        "- The DSE prompt defines batch size as per die. The previous result used 32 tokens as the whole-system batch, which under-counted throughput and tokens/J by roughly the active-die count.",
        "- Energy counters are still emitted in `raw_results.csv` as a first-order diagnostic, but no main DSE figure or ranking uses tokens/J.",
        "- Area feasibility uses a conservative wafer-envelope filter with explicit HBM keepout and fixed edge-crossbar budget; the resulting `wafer_area_headroom` is reported per case so the rebuttal can discuss slack explicitly.",
        "- The fourth main chart uses an area-constrained core-array/HBM frontier because it exposes the binding compute, HBM, and wafer-envelope trade-off more clearly than the flat edge-bandwidth sweep.",
        "- Edge bandwidth rows remain in `raw_results.csv` and diagnostic figures, but the current workload is not primarily external-edge-bandwidth limited under this endpoint model.",
        "",
        "## Area And Frontier Breakdown",
        "",
        f"- Best area headroom case `{best_headroom['case_id']}` uses required area {float(best_headroom['wafer_required_area_mm2']):.1f} mm^2 against budget {float(best_headroom['wafer_budget_area_mm2']):.1f} mm^2.",
        f"- Frontier best `{frontier_best['case_id']}` uses required area {float(frontier_best['wafer_required_area_mm2']):.1f} mm^2 against budget {float(frontier_best['wafer_budget_area_mm2']):.1f} mm^2.",
        f"- Best edge-bandwidth case `{edge_best['case_id']}` uses inter-wafer {float(edge_best.get('inter_wafer_bw_gbps', 0.0)):.1f} Gb/s and mesh {float(edge_best.get('mesh_bw_gbps', 0.0)):.1f} Gb/s.",
        "",
        "## Pareto Cases",
        "",
    ]
    for r in pareto_rows:
        lines.append(
            f"- `{r['case_id']}`: TPS={float(r['throughput_tps']):.1f}, "
            f"area_headroom={float(r.get('wafer_area_headroom', 0.0)):.3f}, bottleneck={r['bottleneck']}"
        )
    lines.extend(["", "## Design Insights", ""])
    if sram_mid and sram_max:
        lines.append(
            "Insight-1: Moderate SRAM gives a better hit-rate / throughput balance than maximum SRAM. "
            f"Evidence: `{sram_mid['case_id']}` reaches hit-rate {float(sram_mid['sparse_hit_rate']):.3f} "
            f"and {float(sram_mid['throughput_tps']):.1f} TPS, while `{sram_max['case_id']}` only raises hit-rate to "
            f"{float(sram_max['sparse_hit_rate']):.3f} but drops to {float(sram_max['throughput_tps']):.1f} TPS "
            f"because per-core compute falls to {float(sram_max['compute_per_core_tflops']):.3f} TFLOP/s."
        )
    if compute_heavy and sram_mid:
        lines.append(
            "Insight-2: Compute-heavy designs help throughput, but the HBM/capacity trade-off still matters. "
            f"Evidence: `{compute_heavy['case_id']}` has {float(compute_heavy['compute_peak_tflops']):.1f} TFLOP/s per die "
            f"and {float(compute_heavy['throughput_tps']):.1f} TPS; `{sram_mid['case_id']}` has "
            f"{float(sram_mid['compute_peak_tflops']):.1f} TFLOP/s and hit-rate {float(sram_mid['sparse_hit_rate']):.3f}."
        )
    if hbm2 and hbm4 and hbm8:
        lines.append(
            "Insight-3: Adding HBM stacks improves throughput only until area/headroom and compute/NoC dominate. "
            f"Evidence: at 32x32 cores, H2 gives {float(hbm2['throughput_tps']):.1f} TPS and "
            f"HBM capacity {float(hbm2['hbm_capacity_gb_per_die']):.1f} GB/die, while H4 gives "
            f"{float(hbm4['throughput_tps']):.1f} TPS and H8 gives {float(hbm8['throughput_tps']):.1f} TPS "
            f"at area utilization {float(hbm8['wafer_area_utilization']):.3f}."
        )
    lines.append(
        "Insight-4: Larger wafer meshes are useful only while edge traffic stays controlled by the TP3 crossbar. "
        f"Evidence: the best wafer-scale throughput point is `{wafer_best['case_id']}` with "
        f"MoE comm {float(wafer_best['moe_comm_us']):.3f} us/layer and cross-wafer locality "
        f"{float(wafer_best['cross_wafer_locality']):.3f}."
    )
    lines.extend(["", "## Reproduce", "", "```bash", f"python hacmo/DSE/run_dse.py --output-root {path.parent}", "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def find_row(rows: list[dict[str, Any]], case_id: str) -> dict[str, Any] | None:
    return next((r for r in rows if r["case_id"] == case_id), None)


def dump_config(out_root: Path, args: argparse.Namespace, const: dict[str, Any], rows: list[dict[str, Any]], invalid: list[dict[str, Any]]) -> None:
    payload = {
        "args": vars(args),
        "constants": const,
        "num_valid_cases": len(rows),
        "num_invalid_cases": len(invalid),
        "valid_case_ids": [r["case_id"] for r in rows],
        "invalid_case_ids": [r["case_id"] for r in invalid],
    }
    (out_root / "dse_config_dump.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def stable_seed(*parts: str) -> int:
    s = "|".join(parts)
    out = 17
    for ch in s:
        out = (out * 131 + ord(ch)) % 1000003
    return out


if __name__ == "__main__":
    main()
