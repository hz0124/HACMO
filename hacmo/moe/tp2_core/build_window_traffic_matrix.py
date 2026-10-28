#!/usr/bin/env python3
"""Build per-window src->dst byte traffic matrices and compute analytical latencies.

This is the core analysis script for moe. It takes normalized traces,
a mapping, and topology config, then produces:
1. Per-window sparse traffic matrices (dispatch and combine)
2. Routing statistics (cross-wafer bytes, avg hops, edge load)
3. Analytical latency estimates using the topology's physical parameters
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from .compute_model import (
        H200RooflineConfig,
        compute_rank_roofline_from_rank_expert_tokens,
    )
    from .utils_topology import HierarchicalTopology, load_topology
except ImportError:
    from compute_model import (
        H200RooflineConfig,
        compute_rank_roofline_from_rank_expert_tokens,
    )
    from utils_topology import HierarchicalTopology, load_topology


def _message_latency_ns(
    topo: HierarchicalTopology, src: int, dst: int, nbytes: int
) -> float:
    """Compute latency for one message traversing the routed path from src to dst.

    Model: sum over edges on path of (edge_fixed_latency + nbytes / edge_bw).
    GB/s numerically equals bytes/ns, so we can divide directly.
    """
    if src == dst or nbytes <= 0:
        return 0.0
    path = topo.get_path(src, dst)
    if not path:
        return 0.0
    total = 0.0
    for edge in path:
        kind = edge[0]
        if kind == "mesh":
            bw = max(topo.get_link_bandwidth(edge), 1e-9)
            total += topo.cfg.mesh_latency_ns + nbytes / bw
        elif kind == "switch":
            bw = max(topo.get_link_bandwidth(edge), 1e-9)
            total += topo.cfg.switch_latency_ns + nbytes / bw
    return total


def _path_latency_ns(topo: HierarchicalTopology, src: int, dst: int) -> float:
    """Compute zero-load path latency (hop latency only, no serialization).

    Used for replica selection where we want the nearest replica by
    actual latency (accounting for switch-vs-mesh hop cost difference)
    rather than raw hop count.
    """
    if src == dst:
        return 0.0
    path = topo.get_path(src, dst)
    if not path:
        return 0.0
    total = 0.0
    for edge in path:
        kind = edge[0]
        if kind == "mesh":
            total += topo.cfg.mesh_latency_ns
        elif kind == "switch":
            total += topo.cfg.switch_latency_ns
    return total


def _path_type(topo: HierarchicalTopology, src: int, dst: int) -> str:
    src_w, _ = topo.die_to_wafer_local(src)
    dst_w, _ = topo.die_to_wafer_local(dst)
    return "same_wafer" if src_w == dst_w else "cross_wafer"


def _hop_count(topo: HierarchicalTopology, src: int, dst: int) -> int:
    return len(topo.get_path(src, dst))


def build_window_matrices(
    window_src_expert_counts: List[Dict],
    mapping: Dict[int, List[int]],
    topo: HierarchicalTopology,
    batch_scale: float,
    dispatch_bytes_per_token: int,
    combine_bytes_per_token: int,
    num_ranks: int,
    expert_compute_us_per_token: float,
    compute_backend: str,
    roofline_cfg: H200RooflineConfig,
    window_size: int,
    window_stats: List[Dict],
    edge_congestion_factor: float = 1.0,
    runtime_windows: List[Dict] | None = None,
    profile_topk: int = 16,
    exclude_leading_windows: int = 0,
) -> Dict:
    """Build traffic matrices and compute analytical latencies for all windows."""
    path_cache: Dict[Tuple[int, int], List[Tuple]] = {}
    path_latency_cache: Dict[Tuple[int, int], float] = {}
    path_type_cache: Dict[Tuple[int, int], str] = {}
    hop_count_cache: Dict[Tuple[int, int], int] = {}
    message_latency_base_cache: Dict[Tuple[int, int], Tuple[float, float]] = {}

    def cached_path(src: int, dst: int) -> List[Tuple]:
        key = (src, dst)
        if key not in path_cache:
            path_cache[key] = topo.get_path(src, dst)
        return path_cache[key]

    def cached_path_latency(src: int, dst: int) -> float:
        key = (src, dst)
        if key not in path_latency_cache:
            if src == dst:
                path_latency_cache[key] = 0.0
            else:
                total = 0.0
                for edge in cached_path(src, dst):
                    kind = edge[0]
                    if kind == "mesh":
                        total += topo.cfg.mesh_latency_ns
                    elif kind == "switch":
                        total += topo.cfg.switch_latency_ns
                path_latency_cache[key] = total
        return path_latency_cache[key]

    def cached_path_type(src: int, dst: int) -> str:
        key = (src, dst)
        if key not in path_type_cache:
            src_w, _ = topo.die_to_wafer_local(src)
            dst_w, _ = topo.die_to_wafer_local(dst)
            path_type_cache[key] = "same_wafer" if src_w == dst_w else "cross_wafer"
        return path_type_cache[key]

    def cached_hop_count(src: int, dst: int) -> int:
        key = (src, dst)
        if key not in hop_count_cache:
            hop_count_cache[key] = len(cached_path(src, dst))
        return hop_count_cache[key]

    def cached_message_latency_ns(src: int, dst: int, nbytes: int) -> float:
        if src == dst or nbytes <= 0:
            return 0.0
        key = (src, dst)
        if key not in message_latency_base_cache:
            fixed_ns = 0.0
            per_byte_ns = 0.0
            for edge in cached_path(src, dst):
                kind = edge[0]
                if kind == "mesh":
                    bw = max(topo.get_link_bandwidth(edge), 1e-9)
                    fixed_ns += topo.cfg.mesh_latency_ns
                    per_byte_ns += 1.0 / bw
                elif kind == "switch":
                    bw = max(topo.get_link_bandwidth(edge), 1e-9)
                    fixed_ns += topo.cfg.switch_latency_ns
                    per_byte_ns += 1.0 / bw
            message_latency_base_cache[key] = (fixed_ns, per_byte_ns)
        fixed_ns, per_byte_ns = message_latency_base_cache[key]
        return fixed_ns + nbytes * per_byte_ns

    results = []
    summary_cross_wafer_bytes = 0
    summary_same_wafer_bytes = 0
    summary_total_hops = 0
    summary_total_messages = 0
    summary_max_edge_exit_bytes = 0
    summary_dispatch_ns_list = []
    summary_combine_ns_list = []
    summary_compute_ns_list = []
    summary_link_loads: Dict[str, int] = defaultdict(int)
    summary_wafer_exit_load: Dict[int, int] = defaultdict(int)
    summary_expert_counts: Dict[int, int] = defaultdict(int)
    summary_replica_events = {"added": 0, "removed": 0, "migration_bytes": 0}
    summary_max_rank_recv_bytes = 0
    summary_max_edge_port_bytes = 0
    summary_dispatch_actual_bytes = 0
    summary_combine_actual_bytes = 0
    excluded_window_ids: List[int] = []
    evaluated_window_ids: List[int] = []
    active_seen = 0

    def _is_active_window(w_id: int, wdata: Dict) -> bool:
        ws = window_stats[w_id] if w_id < len(window_stats) else {}
        tokens = sum(int(v) for v in (ws.get("tokens_per_die") or {}).values())
        activations = sum(int(v) for v in (ws.get("expert_counts") or {}).values())
        if tokens > 0 or activations > 0:
            return True
        src_counts = wdata.get("src_expert_counts") or {}
        return any((e_counts or {}) for e_counts in src_counts.values())

    for wdata in window_src_expert_counts:
        w_id = int(wdata["window_id"])
        if _is_active_window(w_id, wdata):
            if active_seen < exclude_leading_windows:
                excluded_window_ids.append(w_id)
                active_seen += 1
                continue
            active_seen += 1
        evaluated_window_ids.append(w_id)
        se = wdata["src_expert_counts"]
        if runtime_windows is not None and w_id < len(runtime_windows):
            cur_mapping = {
                int(k): [int(x) for x in v]
                for k, v in runtime_windows[w_id].get("mapping", {}).items()
            }
            summary_replica_events["added"] += len(
                runtime_windows[w_id].get("replicas_added", [])
            )
            summary_replica_events["removed"] += len(
                runtime_windows[w_id].get("replicas_removed", [])
            )
            summary_replica_events["migration_bytes"] += int(
                runtime_windows[w_id].get("migration_bytes", 0)
            )
        else:
            cur_mapping = mapping

        dispatch_matrix: Dict[int, Dict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        for src_str, e_counts in se.items():
            src = int(src_str)
            for e_str, c in e_counts.items():
                e = int(e_str)
                ds = cur_mapping.get(e, [])
                if not ds:
                    continue
                best_d = min(ds, key=lambda d: cached_path_latency(src, d))
                b = int(round(float(c) * batch_scale * dispatch_bytes_per_token))
                if b <= 0:
                    continue
                if src != best_d:
                    dispatch_matrix[src][best_d] += b

        combine_matrix: Dict[int, Dict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        for src, dsts in dispatch_matrix.items():
            for dst, b in dsts.items():
                combine_b = int(
                    round(
                        b * combine_bytes_per_token / max(dispatch_bytes_per_token, 1)
                    )
                )
                combine_matrix[dst][src] += combine_b

        w_cross_wafer = 0
        w_same_wafer = 0
        w_hops = 0
        w_msgs = 0
        edge_exit_bytes: Dict[int, int] = defaultdict(int)
        link_loads: Dict[str, int] = defaultdict(int)

        rank_recv_bytes: Dict[int, int] = defaultdict(int)
        for src, dsts in dispatch_matrix.items():
            for dst, b in dsts.items():
                rank_recv_bytes[dst] += b
                pt = cached_path_type(src, dst)
                hops = cached_hop_count(src, dst)
                for edge in cached_path(src, dst):
                    link_loads[str(edge)] += b
                    summary_link_loads[str(edge)] += b
                if pt == "cross_wafer":
                    w_cross_wafer += b
                    src_w, _ = topo.die_to_wafer_local(src)
                    dst_w, _ = topo.die_to_wafer_local(dst)
                    edge_exit_bytes[src_w] += b
                    edge_exit_bytes[dst_w] += b
                    summary_wafer_exit_load[src_w] += b
                else:
                    w_same_wafer += b
                w_hops += hops * b
                w_msgs += 1

        w_total_bytes = w_cross_wafer + w_same_wafer
        w_combine_bytes = sum(sum(dsts.values()) for dsts in combine_matrix.values())
        summary_dispatch_actual_bytes += w_total_bytes
        summary_combine_actual_bytes += w_combine_bytes
        avg_hops = w_hops / w_total_bytes if w_total_bytes > 0 else 0.0
        max_edge_exit = max(edge_exit_bytes.values()) if edge_exit_bytes else 0
        summary_max_edge_port_bytes = max(summary_max_edge_port_bytes, max_edge_exit)
        summary_max_rank_recv_bytes = max(
            summary_max_rank_recv_bytes,
            max(rank_recv_bytes.values()) if rank_recv_bytes else 0,
        )

        # --- Analytical latency ---
        # Edge-port congestion: cross-wafer bytes through each wafer's edge ports
        # create queuing delay = wafer_exit_bytes / (num_edge_ports * switch_bw)
        wafer_exit_load: Dict[int, int] = defaultdict(int)
        for src, dsts in dispatch_matrix.items():
            for dst, b in dsts.items():
                if cached_path_type(src, dst) == "cross_wafer":
                    sw, _ = topo.die_to_wafer_local(src)
                    wafer_exit_load[sw] += b

        # The YAML lists corner rail ports twice because a corner die exposes
        # two physical directions. load_topology merges duplicate local dies
        # into rail_bw_scales, so use the scale sum for congestion capacity.
        num_edge_ports = max(1.0, sum(topo.cfg.rail_bw_scales))
        switch_bw_bpns = max(topo.cfg.switch_bw_gbps, 1e-9)

        # Dispatch: for each receiving rank, sum latencies of all incoming messages.
        # Phase time = max over all ranks.
        # Edge-port congestion is modelled as a shared queuing delay per source
        # wafer: all cross-wafer messages from the same wafer share the same
        # edge-port egress, so the queuing delay is applied once per source wafer
        # as a serialization cost, NOT per individual message.
        wafer_congestion_ns: Dict[int, float] = {}
        if edge_congestion_factor > 0:
            for sw, load_bytes in wafer_exit_load.items():
                wafer_congestion_ns[sw] = (
                    load_bytes
                    / (num_edge_ports * switch_bw_bpns)
                    * edge_congestion_factor
                )

        rank_dispatch_ns = [0.0] * num_ranks
        for src, dsts in dispatch_matrix.items():
            src_w, _ = topo.die_to_wafer_local(src)
            has_cross_wafer_dispatch = any(
                cached_path_type(src, dst) == "cross_wafer" for dst in dsts
            )
            for dst, b in dsts.items():
                lat = cached_message_latency_ns(src, dst, b)
                rank_dispatch_ns[dst] += lat
            # Apply edge-port queuing delay once for this source wafer's
            # cross-wafer dispatch phase, not per-message.
            if has_cross_wafer_dispatch and src_w in wafer_congestion_ns:
                # Distribute the queuing delay to the busiest receiving rank.
                busiest_dst = max(dsts, key=lambda d: dsts[d]) if dsts else None
                if busiest_dst is not None:
                    rank_dispatch_ns[busiest_dst] += wafer_congestion_ns[src_w]

        # Combine: reverse direction.
        wafer_exit_load_combine: Dict[int, int] = defaultdict(int)
        for src, dsts in combine_matrix.items():
            for dst, b in dsts.items():
                if cached_path_type(src, dst) == "cross_wafer":
                    sw, _ = topo.die_to_wafer_local(src)
                    wafer_exit_load_combine[sw] += b

        wafer_congestion_combine_ns: Dict[int, float] = {}
        if edge_congestion_factor > 0:
            for sw, load_bytes in wafer_exit_load_combine.items():
                wafer_congestion_combine_ns[sw] = (
                    load_bytes
                    / (num_edge_ports * switch_bw_bpns)
                    * edge_congestion_factor
                )

        rank_combine_ns = [0.0] * num_ranks
        for src, dsts in combine_matrix.items():
            src_w, _ = topo.die_to_wafer_local(src)
            has_cross_wafer_combine = any(
                cached_path_type(src, dst) == "cross_wafer" for dst in dsts
            )
            for dst, b in dsts.items():
                lat = cached_message_latency_ns(src, dst, b)
                rank_combine_ns[dst] += lat
            if has_cross_wafer_combine and src_w in wafer_congestion_combine_ns:
                busiest_dst = max(dsts, key=lambda d: dsts[d]) if dsts else None
                if busiest_dst is not None:
                    rank_combine_ns[busiest_dst] += wafer_congestion_combine_ns[src_w]

        # Expert compute: derive from actual dispatched token counts,
        # not from uniform expert_counts division.
        #
        # The dispatch_matrix already encodes which die actually processes
        # each expert's tokens (nearest-replica routing).  We invert the
        # byte->token conversion to get per-rank token counts that are
        # consistent with the communication model.
        ws_data = window_stats[w_id] if w_id < len(window_stats) else {}
        expert_counts = ws_data.get("expert_counts", {})
        for e_str, ec in expert_counts.items():
            e = int(e_str)
            summary_expert_counts[e] += int(ec)

        # Derive per-rank token counts from the same nearest-replica routing
        # used for communication.  This must include local src==dst tokens,
        # which are intentionally omitted from the sparse dispatch matrix.
        rank_expert_tokens: List[Dict[int, float]] = [
            defaultdict(float) for _ in range(num_ranks)
        ]
        for src_str, e_counts in se.items():
            src = int(src_str)
            for e_str, c in e_counts.items():
                e = int(e_str)
                ds = cur_mapping.get(e, [])
                if not ds:
                    continue
                best_d = min(ds, key=lambda d: cached_path_latency(src, d))
                if 0 <= best_d < num_ranks:
                    rank_expert_tokens[best_d][e] += float(c) * batch_scale

        if compute_backend == "roofline_h200":
            roof = compute_rank_roofline_from_rank_expert_tokens(
                [dict(expert_tokens) for expert_tokens in rank_expert_tokens],
                roofline_cfg,
            )
            rank_compute_ns = [x * 1000.0 for x in roof["rank_total_us"]]
            compute_profile = {
                "backend": "roofline_h200_route_aware",
                "max_rank": roof["max_rank"],
                "max_rank_expert_tokens": roof["max_rank_expert_tokens"],
                "max_rank_active_experts": roof["max_rank_active_experts"],
                "max_rank_flops": roof["max_rank_flops"],
                "max_rank_bytes": roof["max_rank_bytes"],
                "max_rank_compute_us": roof["max_rank_compute_us"],
                "max_rank_memory_us": roof["max_rank_memory_us"],
            }
        else:
            rank_compute_ns = [0.0] * num_ranks
            for d, expert_tokens in enumerate(rank_expert_tokens):
                rank_compute_ns[d] = (
                    sum(expert_tokens.values()) * expert_compute_us_per_token * 1000.0
                )
            compute_profile = {"backend": "route_aware_per_token"}

        dispatch_ns = max(rank_dispatch_ns) if rank_dispatch_ns else 0.0
        combine_ns = max(rank_combine_ns) if rank_combine_ns else 0.0
        compute_ns = max(rank_compute_ns) if rank_compute_ns else 0.0
        total_ns = dispatch_ns + compute_ns + combine_ns

        summary_cross_wafer_bytes += w_cross_wafer
        summary_same_wafer_bytes += w_same_wafer
        summary_total_hops += w_hops
        summary_total_messages += w_msgs
        summary_max_edge_exit_bytes = max(summary_max_edge_exit_bytes, max_edge_exit)
        summary_dispatch_ns_list.append(dispatch_ns)
        summary_combine_ns_list.append(combine_ns)
        summary_compute_ns_list.append(compute_ns)

        results.append(
            {
                "window_id": w_id,
                "stats": {
                    "cross_wafer_bytes": w_cross_wafer,
                    "same_wafer_bytes": w_same_wafer,
                    "total_bytes": w_total_bytes,
                    "avg_hops": round(avg_hops, 4),
                    "max_edge_exit_bytes": max_edge_exit,
                    "num_messages": w_msgs,
                },
                "profile": {
                    "top_experts": [
                        {
                            "expert": int(e),
                            "activations": int(c),
                            "replicas": len(cur_mapping.get(int(e), [])),
                        }
                        for e, c in sorted(
                            expert_counts.items(), key=lambda x: int(x[1]), reverse=True
                        )[:profile_topk]
                    ],
                    "top_link_loads_bytes": [
                        {"edge": edge, "bytes": int(v)}
                        for edge, v in sorted(
                            link_loads.items(), key=lambda x: x[1], reverse=True
                        )[:profile_topk]
                    ],
                    "wafer_exit_bytes": {
                        str(k): int(v) for k, v in sorted(wafer_exit_load.items())
                    },
                    "edge_port_bytes": {
                        str(k): int(v) for k, v in sorted(edge_exit_bytes.items())
                    },
                    "max_rank_recv_bytes": int(
                        max(rank_recv_bytes.values()) if rank_recv_bytes else 0
                    ),
                    "actual_dispatch_bytes": int(w_total_bytes),
                    "actual_combine_bytes": int(w_combine_bytes),
                    "replicas_added": (
                        runtime_windows[w_id].get("replicas_added", [])
                        if runtime_windows and w_id < len(runtime_windows)
                        else []
                    ),
                    "replicas_removed": (
                        runtime_windows[w_id].get("replicas_removed", [])
                        if runtime_windows and w_id < len(runtime_windows)
                        else []
                    ),
                    "compute": compute_profile,
                },
                "analytical_latency_ns": {
                    "dispatch": round(dispatch_ns, 2),
                    "compute": round(compute_ns, 2),
                    "combine": round(combine_ns, 2),
                    "total": round(total_ns, 2),
                },
            }
        )

    num_windows = len(results) or 1
    total_bytes_all = summary_cross_wafer_bytes + summary_same_wafer_bytes
    avg_dispatch_ns = (
        sum(summary_dispatch_ns_list) / num_windows if summary_dispatch_ns_list else 0.0
    )
    avg_combine_ns = (
        sum(summary_combine_ns_list) / num_windows if summary_combine_ns_list else 0.0
    )
    avg_compute_ns = (
        sum(summary_compute_ns_list) / num_windows if summary_compute_ns_list else 0.0
    )

    avg_step_dispatch_us = avg_dispatch_ns / window_size / 1000.0
    avg_step_combine_us = avg_combine_ns / window_size / 1000.0
    avg_step_compute_us = avg_compute_ns / window_size / 1000.0
    avg_step_total_us = avg_step_dispatch_us + avg_step_compute_us + avg_step_combine_us

    summary = {
        "num_windows": num_windows,
        "excluded_leading_windows": int(exclude_leading_windows),
        "excluded_window_ids": excluded_window_ids,
        "evaluated_window_ids": evaluated_window_ids,
        "total_cross_wafer_bytes": summary_cross_wafer_bytes,
        "total_same_wafer_bytes": summary_same_wafer_bytes,
        "total_bytes": total_bytes_all,
        "cross_wafer_ratio": round(
            summary_cross_wafer_bytes / max(total_bytes_all, 1), 4
        ),
        "avg_hops": round(summary_total_hops / max(total_bytes_all, 1), 4),
        "max_edge_exit_bytes": summary_max_edge_exit_bytes,
        "max_rank_recv_bytes": summary_max_rank_recv_bytes,
        "max_edge_port_bytes": summary_max_edge_port_bytes,
        "actual_dispatch_bytes": summary_dispatch_actual_bytes,
        "actual_combine_bytes": summary_combine_actual_bytes,
        "total_messages": summary_total_messages,
        "avg_step_dispatch_us": round(avg_step_dispatch_us, 4),
        "avg_step_combine_us": round(avg_step_combine_us, 4),
        "avg_step_compute_us": round(avg_step_compute_us, 4),
        "avg_step_total_us": round(avg_step_total_us, 4),
        "profile": {
            "top_experts": [
                {"expert": int(e), "activations": int(c)}
                for e, c in sorted(
                    summary_expert_counts.items(), key=lambda x: x[1], reverse=True
                )[:profile_topk]
            ],
            "top_link_loads_bytes": [
                {"edge": edge, "bytes": int(v)}
                for edge, v in sorted(
                    summary_link_loads.items(), key=lambda x: x[1], reverse=True
                )[:profile_topk]
            ],
            "wafer_exit_bytes": {
                str(k): int(v) for k, v in sorted(summary_wafer_exit_load.items())
            },
            "replica_events": summary_replica_events,
        },
        "validation": {
            "sparse_p2p_no_padding": True,
            "dispatch_total_bytes": int(summary_dispatch_actual_bytes),
            "combine_total_bytes": int(summary_combine_actual_bytes),
            "dispatch_bytes_per_token": int(dispatch_bytes_per_token),
            "combine_bytes_per_token": int(combine_bytes_per_token),
        },
    }

    return {"windows": results, "summary": summary}


def analyze_routed_steps(
    route_plan: Dict,
    *,
    topo: HierarchicalTopology,
    roofline_cfg: H200RooflineConfig,
    num_ranks: int,
    include_shared_expert: bool = True,
) -> Dict:
    """Analyze one authoritative, already-routed TP2 artifact step by step."""
    path_cache: Dict[Tuple[int, int], List[Tuple]] = {}
    path_type_cache: Dict[Tuple[int, int], str] = {}
    hop_count_cache: Dict[Tuple[int, int], int] = {}
    message_latency_terms_cache: Dict[Tuple[int, int], List[Tuple[float, float]]] = {}

    def cached_path(src: int, dst: int) -> List[Tuple]:
        key = (src, dst)
        if key not in path_cache:
            path_cache[key] = topo.get_path(src, dst)
        return path_cache[key]

    def cached_path_type(src: int, dst: int) -> str:
        key = (src, dst)
        if key not in path_type_cache:
            src_w, _ = topo.die_to_wafer_local(src)
            dst_w, _ = topo.die_to_wafer_local(dst)
            path_type_cache[key] = "same_wafer" if src_w == dst_w else "cross_wafer"
        return path_type_cache[key]

    def cached_hop_count(src: int, dst: int) -> int:
        key = (src, dst)
        if key not in hop_count_cache:
            hop_count_cache[key] = len(cached_path(src, dst))
        return hop_count_cache[key]

    def cached_message_latency_ns(src: int, dst: int, nbytes: int) -> float:
        if src == dst or nbytes <= 0:
            return 0.0
        key = (src, dst)
        if key not in message_latency_terms_cache:
            terms: List[Tuple[float, float]] = []
            for edge in cached_path(src, dst):
                kind = edge[0]
                if kind == "mesh":
                    terms.append(
                        (
                            topo.cfg.mesh_latency_ns,
                            max(topo.get_link_bandwidth(edge), 1e-9),
                        )
                    )
                elif kind == "switch":
                    terms.append(
                        (
                            topo.cfg.switch_latency_ns,
                            max(topo.get_link_bandwidth(edge), 1e-9),
                        )
                    )
            message_latency_terms_cache[key] = terms
        total = 0.0
        for fixed_ns, bandwidth in message_latency_terms_cache[key]:
            total += fixed_ns + nbytes / bandwidth
        return total

    step_results: List[Dict] = []
    total_dispatch_bytes = 0
    total_combine_bytes = 0
    total_collective_bytes = 0
    total_collective_cross_wafer_bytes = 0
    total_cross_wafer_bytes = 0
    total_hop_bytes = 0
    total_routed_expert_weight_reads = 0
    total_routed_expert_weight_bytes = 0.0

    for step in route_plan.get("steps", []):
        collective_wall_ns = 0.0
        step_collective_bytes = 0
        step_collective_cross_wafer_bytes = 0
        collective_phase_results = []
        for segment in step.get("pre_dispatch_segments", []):
            messages = {
                int(src): {int(dst): int(nbytes) for dst, nbytes in dsts.items()}
                for src, dsts in segment.get("messages", {}).items()
            }
            endpoint_ns: Dict[int, float] = defaultdict(float)
            link_bytes: Dict[Tuple, int] = defaultdict(int)
            phase_bytes = 0
            phase_cross_wafer_bytes = 0
            for src, dsts in messages.items():
                for dst, nbytes in dsts.items():
                    endpoint_ns[dst] += cached_message_latency_ns(src, dst, nbytes)
                    phase_bytes += nbytes
                    if cached_path_type(src, dst) == "cross_wafer":
                        phase_cross_wafer_bytes += nbytes
                    for edge in cached_path(src, dst):
                        link_bytes[edge] += nbytes
            endpoint_wall_ns = max(endpoint_ns.values(), default=0.0)
            link_wall_ns = max(
                (
                    nbytes / max(topo.get_link_bandwidth(edge), 1e-9)
                    for edge, nbytes in link_bytes.items()
                ),
                default=0.0,
            )
            phase_wall_ns = max(endpoint_wall_ns, link_wall_ns)
            collective_wall_ns += phase_wall_ns
            step_collective_bytes += phase_bytes
            step_collective_cross_wafer_bytes += phase_cross_wafer_bytes
            collective_phase_results.append(
                {
                    "name": str(segment.get("name", "collective")),
                    "phase_wall_ns": phase_wall_ns,
                    "bytes": phase_bytes,
                    "cross_wafer_bytes": phase_cross_wafer_bytes,
                }
            )

        dispatch = {
            int(src): {int(dst): int(nbytes) for dst, nbytes in dsts.items()}
            for src, dsts in step.get("dispatch_bytes", {}).items()
        }
        combine = {
            int(src): {int(dst): int(nbytes) for dst, nbytes in dsts.items()}
            for src, dsts in step.get("combine_bytes", {}).items()
        }
        rank_expert_tokens: List[Dict[int, float]] = [{} for _ in range(num_ranks)]
        for die_str, expert_counts in step.get("rank_expert_tokens", {}).items():
            die = int(die_str)
            if 0 <= die < num_ranks:
                rank_expert_tokens[die] = {
                    int(expert): float(count) for expert, count in expert_counts.items()
                }

        roof = compute_rank_roofline_from_rank_expert_tokens(
            rank_expert_tokens, roofline_cfg
        )
        rank_compute_ns = [float(value) * 1000.0 for value in roof["rank_total_us"]]
        routed_compute_ns = max(rank_compute_ns, default=0.0)
        routed_weight_reads = sum(int(value) for value in roof["rank_active_experts"])
        routed_weight_bytes = routed_weight_reads * roofline_cfg.expert_weight_bytes

        dispatch_rank_ns: Dict[int, float] = defaultdict(float)
        dispatch_link_bytes: Dict[Tuple, int] = defaultdict(int)
        step_dispatch_bytes = 0
        step_cross_wafer_bytes = 0
        step_hop_bytes = 0
        for src, dsts in dispatch.items():
            for dst, nbytes in dsts.items():
                latency = cached_message_latency_ns(src, dst, nbytes)
                dispatch_rank_ns[dst] += latency
                step_dispatch_bytes += nbytes
                hops = cached_hop_count(src, dst)
                step_hop_bytes += hops * nbytes
                if cached_path_type(src, dst) == "cross_wafer":
                    step_cross_wafer_bytes += nbytes
                for edge in cached_path(src, dst):
                    dispatch_link_bytes[edge] += nbytes

        dispatch_endpoint_wall_ns = max(dispatch_rank_ns.values(), default=0.0)
        dispatch_link_wall_ns = max(
            (
                nbytes / max(topo.get_link_bandwidth(edge), 1e-9)
                for edge, nbytes in dispatch_link_bytes.items()
            ),
            default=0.0,
        )
        dispatch_wall_ns = max(dispatch_endpoint_wall_ns, dispatch_link_wall_ns)
        dispatch_congestion_extra_ns = max(
            0.0, dispatch_wall_ns - dispatch_endpoint_wall_ns
        )

        compute_done_ns = [0.0] * num_ranks
        for die in range(num_ranks):
            ready = dispatch_rank_ns.get(die, 0.0)
            if ready > 0.0:
                ready += dispatch_congestion_extra_ns
            compute_done_ns[die] = ready + rank_compute_ns[die]

        combine_rank_ns: Dict[int, float] = defaultdict(float)
        combine_link_bytes: Dict[Tuple, int] = defaultdict(int)
        source_done_ns: Dict[int, float] = defaultdict(float)
        step_combine_bytes = 0
        for src, dsts in combine.items():
            for dst, nbytes in dsts.items():
                latency = cached_message_latency_ns(src, dst, nbytes)
                combine_rank_ns[src] += latency
                source_done_ns[dst] = max(
                    source_done_ns[dst], compute_done_ns[src] + combine_rank_ns[src]
                )
                step_combine_bytes += nbytes
                for edge in cached_path(src, dst):
                    combine_link_bytes[edge] += nbytes

        for assignment in step.get("assignments", []):
            src = int(assignment["src_die"])
            dst = int(assignment["dst_die"])
            if src == dst:
                source_done_ns[src] = max(source_done_ns[src], compute_done_ns[dst])

        combine_endpoint_wall_ns = max(combine_rank_ns.values(), default=0.0)
        combine_link_wall_ns = max(
            (
                nbytes / max(topo.get_link_bandwidth(edge), 1e-9)
                for edge, nbytes in combine_link_bytes.items()
            ),
            default=0.0,
        )
        combine_wall_ns = max(combine_endpoint_wall_ns, combine_link_wall_ns)
        combine_congestion_extra_ns = max(
            0.0, combine_wall_ns - combine_endpoint_wall_ns
        )
        routed_overlap_ns = max(source_done_ns.values(), default=routed_compute_ns)
        if combine:
            routed_overlap_ns += combine_congestion_extra_ns

        shared_compute_ns = 0.0
        shared_weight_reads = 0
        if include_shared_expert:
            shared_rank_tokens: List[Dict[int, float]] = [{} for _ in range(num_ranks)]
            for die_str, tokens in step.get("tokens_per_die", {}).items():
                die = int(die_str)
                if 0 <= die < num_ranks and float(tokens) > 0:
                    shared_rank_tokens[die] = {-1: float(tokens)}
            shared_roof = compute_rank_roofline_from_rank_expert_tokens(
                shared_rank_tokens, roofline_cfg
            )
            shared_compute_ns = max(
                (float(value) * 1000.0 for value in shared_roof["rank_total_us"]),
                default=0.0,
            )
            shared_weight_reads = sum(
                int(value) for value in shared_roof["rank_active_experts"]
            )

        overlap_total_ns = collective_wall_ns + max(
            routed_overlap_ns, shared_compute_ns
        )
        serial_upper_ns = max(
            collective_wall_ns + dispatch_wall_ns + routed_compute_ns + combine_wall_ns,
            collective_wall_ns + shared_compute_ns,
        )
        validation = step.get("validation", {})
        step_results.append(
            {
                "step_id": int(step["step_id"]),
                "window_id": int(step["window_id"]),
                "dispatch_phase_wall_ns": dispatch_wall_ns,
                "collective_phase_wall_ns": collective_wall_ns,
                "collective_phases": collective_phase_results,
                "routed_compute_wall_ns": routed_compute_ns,
                "combine_phase_wall_ns": combine_wall_ns,
                "shared_compute_wall_ns": shared_compute_ns,
                "overlap_total_ns": overlap_total_ns,
                "serial_upper_ns": serial_upper_ns,
                "dispatch_bytes": step_dispatch_bytes,
                "collective_bytes": step_collective_bytes,
                "collective_cross_wafer_bytes": step_collective_cross_wafer_bytes,
                "combine_bytes": step_combine_bytes,
                "cross_wafer_dispatch_bytes": step_cross_wafer_bytes,
                "routed_expert_weight_reads": routed_weight_reads,
                "routed_expert_weight_bytes": routed_weight_bytes,
                "shared_expert_weight_reads": shared_weight_reads,
                "validation": validation,
            }
        )
        total_dispatch_bytes += step_dispatch_bytes
        total_combine_bytes += step_combine_bytes
        total_collective_bytes += step_collective_bytes
        total_collective_cross_wafer_bytes += step_collective_cross_wafer_bytes
        total_cross_wafer_bytes += step_cross_wafer_bytes
        total_hop_bytes += step_hop_bytes
        total_routed_expert_weight_reads += routed_weight_reads
        total_routed_expert_weight_bytes += routed_weight_bytes

    num_steps = max(len(step_results), 1)
    avg_dispatch_us = (
        sum(row["dispatch_phase_wall_ns"] for row in step_results) / num_steps / 1000.0
    )
    avg_collective_us = (
        sum(row["collective_phase_wall_ns"] for row in step_results)
        / num_steps
        / 1000.0
    )
    avg_compute_us = (
        sum(row["routed_compute_wall_ns"] for row in step_results) / num_steps / 1000.0
    )
    avg_combine_us = (
        sum(row["combine_phase_wall_ns"] for row in step_results) / num_steps / 1000.0
    )
    avg_overlap_us = (
        sum(row["overlap_total_ns"] for row in step_results) / num_steps / 1000.0
    )
    avg_serial_us = (
        sum(row["serial_upper_ns"] for row in step_results) / num_steps / 1000.0
    )
    return {
        "steps": step_results,
        "summary": {
            "num_steps": len(step_results),
            "avg_step_dispatch_us": avg_dispatch_us,
            "avg_step_collective_us": avg_collective_us,
            "avg_step_compute_us": avg_compute_us,
            "avg_step_combine_us": avg_combine_us,
            "avg_step_overlap_total_us": avg_overlap_us,
            "avg_step_serial_upper_us": avg_serial_us,
            "avg_step_total_us": avg_overlap_us,
            "total_dispatch_bytes": total_dispatch_bytes,
            "total_collective_bytes": total_collective_bytes,
            "total_collective_cross_wafer_bytes": total_collective_cross_wafer_bytes,
            "total_combine_bytes": total_combine_bytes,
            "cross_wafer_ratio": total_cross_wafer_bytes / max(total_dispatch_bytes, 1),
            "avg_hops": total_hop_bytes / max(total_dispatch_bytes, 1),
            "total_routed_expert_weight_reads": total_routed_expert_weight_reads,
            "total_routed_expert_weight_bytes": total_routed_expert_weight_bytes,
            "include_shared_expert": include_shared_expert,
            "metric_source": "step_level_die_dag",
            "validation": {
                "authoritative_route_plan_consumed": True,
                "all_steps_preserve_activations": all(
                    row.get("validation", {}).get("all_activations_routed", False)
                    and row.get("validation", {}).get(
                        "compute_matches_assignments", False
                    )
                    for row in step_results
                ),
                "overlap_bounded_by_serial_upper": all(
                    row["overlap_total_ns"] <= row["serial_upper_ns"] + 1e-9
                    for row in step_results
                ),
            },
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build window traffic matrices and compute analytical latency."
    )
    p.add_argument("--network-config", required=True)
    p.add_argument("--route-plan-json", default=None)
    p.add_argument("--mapping-json", default=None)
    p.add_argument("--window-src-expert-json", default=None)
    p.add_argument("--window-stats-json", default=None)
    p.add_argument("--window-size", type=int, default=8)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--base-num-requests", type=int, default=100)
    p.add_argument("--num-experts", type=int, default=256)
    p.add_argument("--num-ranks", type=int, default=256)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--hidden-size", type=int, default=7168)
    p.add_argument("--bytes-per-hidden-elem", type=int, default=2)
    p.add_argument("--dispatch-bytes-per-hidden-elem", type=int, default=1)
    p.add_argument("--combine-bytes-per-hidden-elem", type=int, default=2)
    p.add_argument("--expert-compute-us-per-token", type=float, default=40.0)
    p.add_argument(
        "--compute-backend",
        choices=["roofline_h200", "constant_per_token"],
        default="roofline_h200",
    )
    p.add_argument("--expert-intermediate-size", type=int, default=2048)
    p.add_argument("--h200-effective-tflops", type=float, default=1979.0)
    p.add_argument("--h200-hbm-bw-tbps", type=float, default=4.8)
    p.add_argument("--weight-bytes-per-elem", type=int, default=1)
    p.add_argument(
        "--edge-congestion-factor",
        type=float,
        default=1.0,
        help="Scale factor for edge-port congestion penalty on cross-wafer messages.",
    )
    p.add_argument(
        "--runtime-windows-json",
        default=None,
        help="Optional per-window mapping JSON from runtime_scheduler.py.",
    )
    p.add_argument("--profile-topk", type=int, default=16)
    p.add_argument(
        "--exclude-leading-windows",
        type=int,
        default=0,
        help="Exclude this many leading active runtime windows from reported averages.",
    )
    p.add_argument("--no-shared-expert", action="store_true")
    p.add_argument("--output-json", required=True)
    args = p.parse_args()

    topo = load_topology(args.network_config)
    dispatch_bytes_per_token = args.hidden_size * args.dispatch_bytes_per_hidden_elem
    combine_bytes_per_token = args.hidden_size * args.combine_bytes_per_hidden_elem
    batch_scale = max(args.batch_size / max(args.base_num_requests, 1), 1e-6)
    roofline_cfg = H200RooflineConfig(
        hidden_size=args.hidden_size,
        expert_intermediate_size=args.expert_intermediate_size,
        effective_tflops=args.h200_effective_tflops,
        hbm_bw_tbps=args.h200_hbm_bw_tbps,
        weight_bytes_per_elem=args.weight_bytes_per_elem,
        activation_bytes_per_elem=args.bytes_per_hidden_elem,
    )

    if args.route_plan_json:
        route_plan = json.loads(Path(args.route_plan_json).read_text(encoding="utf-8"))
        result = analyze_routed_steps(
            route_plan,
            topo=topo,
            roofline_cfg=roofline_cfg,
            num_ranks=args.num_ranks,
            include_shared_expert=not args.no_shared_expert,
        )
        batch_scale = 1.0
    else:
        if (
            not args.mapping_json
            or not args.window_src_expert_json
            or not args.window_stats_json
        ):
            p.error(
                "either --route-plan-json or all of --mapping-json, "
                "--window-src-expert-json, and --window-stats-json are required"
            )
        mapping_raw = json.loads(Path(args.mapping_json).read_text(encoding="utf-8"))[
            "expert_to_dies"
        ]
        mapping = {int(k): [int(x) for x in v] for k, v in mapping_raw.items()}
        window_src_expert = json.loads(
            Path(args.window_src_expert_json).read_text(encoding="utf-8")
        )
        window_stats = json.loads(
            Path(args.window_stats_json).read_text(encoding="utf-8")
        )
        runtime_windows = None
        if args.runtime_windows_json:
            runtime_windows = json.loads(
                Path(args.runtime_windows_json).read_text(encoding="utf-8")
            )
        result = build_window_matrices(
            window_src_expert_counts=window_src_expert,
            mapping=mapping,
            topo=topo,
            batch_scale=batch_scale,
            dispatch_bytes_per_token=dispatch_bytes_per_token,
            combine_bytes_per_token=combine_bytes_per_token,
            num_ranks=args.num_ranks,
            expert_compute_us_per_token=args.expert_compute_us_per_token,
            compute_backend=args.compute_backend,
            roofline_cfg=roofline_cfg,
            window_size=args.window_size,
            window_stats=window_stats,
            edge_congestion_factor=args.edge_congestion_factor,
            runtime_windows=runtime_windows,
            profile_topk=args.profile_topk,
            exclude_leading_windows=max(args.exclude_leading_windows, 0),
        )
    result["params"] = {
        "batch_size": args.batch_size,
        "window_size": args.window_size,
        "hidden_size": args.hidden_size,
        "bytes_per_hidden_elem": args.bytes_per_hidden_elem,
        "dispatch_bytes_per_hidden_elem": args.dispatch_bytes_per_hidden_elem,
        "combine_bytes_per_hidden_elem": args.combine_bytes_per_hidden_elem,
        "dispatch_bytes_per_token": dispatch_bytes_per_token,
        "combine_bytes_per_token": combine_bytes_per_token,
        "compute_backend": args.compute_backend,
        "roofline_h200": roofline_cfg.to_dict(),
        "expert_compute_us_per_token": args.expert_compute_us_per_token,
        "batch_scale": batch_scale,
        "route_plan_json": args.route_plan_json,
        "include_shared_expert": not args.no_shared_expert,
        "exclude_leading_windows": max(args.exclude_leading_windows, 0),
    }

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    s = result["summary"]
    print(
        f"[traffic_matrix] BS={args.batch_size}: "
        f"cross_wafer_ratio={s['cross_wafer_ratio']:.3f}, "
        f"avg_hops={s['avg_hops']:.2f}, "
        f"dispatch={s['avg_step_dispatch_us']:.2f}us, "
        f"compute={s['avg_step_compute_us']:.2f}us, "
        f"combine={s['avg_step_combine_us']:.2f}us, "
        f"total={s['avg_step_total_us']:.2f}us"
    )


if __name__ == "__main__":
    main()
