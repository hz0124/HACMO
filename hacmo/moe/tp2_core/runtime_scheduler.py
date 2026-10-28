#!/usr/bin/env python3
"""Runtime rebalancing engine for TP2 in-epoch adaptation.

The scheduler starts from an Epoch Base Placement that was already selected or
deployed by the serving-epoch control plane.  It implements bounded local
operations -- replicate, evict, swap -- without invoking the full static
placement solver on the request/token critical path.  The objective function
uses weighted path costs, congestion pressure, and route-aware compute/load
terms consistent with the static mapper's placement objective.

Key semantics:
- The input static mapping is fixed before the evaluated serving epoch starts.
- Window ``w`` is served by the mapping available before observing window ``w``.
- Updates triggered by window ``w`` are emitted as ``next_mapping`` and can only
  affect later windows.
- Expert migration volume is recorded for overhead analysis; the latency model
  does not retroactively credit current-window traffic with those updates.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from .compute_model import H200RooflineConfig, compute_rank_roofline_from_rank_expert_tokens
    from .route_steps import route_activation_counts
    from .utils_topology import HierarchicalTopology, load_topology
except ImportError:
    from compute_model import H200RooflineConfig, compute_rank_roofline_from_rank_expert_tokens
    from route_steps import route_activation_counts
    from utils_topology import HierarchicalTopology, load_topology


class HotnessEstimator:
    def __init__(self, experts: int, decay: float) -> None:
        self.decay = decay
        self.hotness = {e: 0.0 for e in range(experts)}

    def update(self, counts: Dict[str, int]) -> Dict[int, float]:
        for e, v in self.hotness.items():
            c = float(counts.get(str(e), 0))
            self.hotness[e] = self.decay * v + (1.0 - self.decay) * c
        return self.hotness


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = int((pct / 100.0) * (len(s) - 1))
    return s[idx]


def _die_load(mapping: Dict[int, List[int]], expert_counts: Dict[str, int]) -> Dict[int, float]:
    load: Dict[int, float] = {}
    for e, ds in mapping.items():
        c = float(expert_counts.get(str(e), 0))
        if not ds:
            continue
        share = c / len(ds)
        for d in ds:
            load[d] = load.get(d, 0.0) + share
    return load


def _path_latency_ns(topo: HierarchicalTopology, src: int, dst: int) -> float:
    if src == dst:
        return 0.0
    cache = getattr(topo, "_runtime_scheduler_latency_cache", None)
    if cache is None:
        cache = {}
        setattr(topo, "_runtime_scheduler_latency_cache", cache)
    key = (src, dst)
    if key not in cache:
        total = 0.0
        for edge in _get_path_cached(topo, src, dst):
            if edge[0] == "mesh":
                total += topo.cfg.mesh_latency_ns
            elif edge[0] == "switch":
                total += topo.cfg.switch_latency_ns
        cache[key] = total
    return cache[key]


def _get_path_cached(topo: HierarchicalTopology, src: int, dst: int):
    cache = getattr(topo, "_runtime_scheduler_path_cache", None)
    if cache is None:
        cache = {}
        setattr(topo, "_runtime_scheduler_path_cache", cache)
    key = (src, dst)
    if key not in cache:
        cache[key] = topo.get_path(src, dst)
    return cache[key]


def _get_weighted_path_profile_cached(topo: HierarchicalTopology, src: int, dst: int):
    """Return cached (edge, base_weight, inv_bandwidth) entries for a path."""
    cache = getattr(topo, "_runtime_scheduler_weighted_path_cache", None)
    if cache is None:
        cache = {}
        setattr(topo, "_runtime_scheduler_weighted_path_cache", cache)
    key = (src, dst)
    if key not in cache:
        profile = []
        for edge in _get_path_cached(topo, src, dst):
            bw = topo.get_link_bandwidth(edge)
            profile.append((edge, _edge_base_weight(topo, edge), 1.0 / max(bw, 1e-6)))
        cache[key] = profile
    return cache[key]


def _edge_base_weight(topo: HierarchicalTopology, edge: Tuple) -> float:
    if edge[0] == "mesh":
        return 1.0
    return (
        topo.cfg.mesh_bw_gbps / max(topo.cfg.switch_bw_gbps, 1e-9)
        + topo.cfg.switch_latency_ns / max(topo.cfg.mesh_latency_ns, 1e-9)
    )


def _switch_base_weight(topo: HierarchicalTopology) -> float:
    return (
        topo.cfg.mesh_bw_gbps / max(topo.cfg.switch_bw_gbps, 1e-9)
        + topo.cfg.switch_latency_ns / max(topo.cfg.mesh_latency_ns, 1e-9)
    )


def _nearest_rail_port_for_base_cost(topo: HierarchicalTopology, local_die: int) -> int:
    cache = getattr(topo, "_runtime_scheduler_nearest_rail_port_cache", None)
    if cache is None:
        cache = {}
        setattr(topo, "_runtime_scheduler_nearest_rail_port_cache", cache)
    if local_die not in cache:
        best_port = topo.cfg.rail_ports[0]
        best_hops = topo.mesh_hops(local_die, best_port)
        best_scale = topo._rail_scale(best_port)
        for port in topo.cfg.rail_ports[1:]:
            hops = topo.mesh_hops(local_die, port)
            scale = topo._rail_scale(port)
            if (
                hops < best_hops
                or (hops == best_hops and scale > best_scale)
                or (hops == best_hops and scale == best_scale and port < best_port)
            ):
                best_port = port
                best_hops = hops
                best_scale = scale
        cache[local_die] = best_port
    return cache[local_die]


def _path_base_cost_weighted(topo: HierarchicalTopology, src: int, dst: int) -> float:
    cache = getattr(topo, "_runtime_scheduler_path_base_cost_cache", None)
    if cache is None:
        cache = {}
        setattr(topo, "_runtime_scheduler_path_base_cost_cache", cache)
    key = (src, dst)
    if key not in cache:
        src_w, src_local = topo.die_to_wafer_local(src)
        dst_w, dst_local = topo.die_to_wafer_local(dst)
        if src_w == dst_w:
            cache[key] = float(topo.mesh_hops(src_local, dst_local))
        elif not topo.cfg.rail_optimized:
            cache[key] = 2.0 * _switch_base_weight(topo)
        else:
            src_port = _nearest_rail_port_for_base_cost(topo, src_local)
            dst_port = _nearest_rail_port_for_base_cost(topo, dst_local)
            cache[key] = (
                float(topo.mesh_hops(src_local, src_port))
                + 2.0 * _switch_base_weight(topo)
                + float(topo.mesh_hops(dst_port, dst_local))
            )
    return cache[key]


def _nearest_replica_by_latency(topo: HierarchicalTopology, src: int, candidate_dies: List[int]) -> int:
    if len(candidate_dies) == 1:
        return candidate_dies[0]
    cache = getattr(topo, "_runtime_scheduler_nearest_replica_cache", None)
    if cache is None:
        cache = {}
        setattr(topo, "_runtime_scheduler_nearest_replica_cache", cache)
    key = (src, tuple(candidate_dies))
    if key not in cache:
        cache[key] = min(candidate_dies, key=lambda d: _path_latency_ns(topo, src, d))
    return cache[key]


def _route_aware_die_load(
    mapping: Dict[int, List[int]],
    src_expert_counts: Dict[str, Dict[str, int]],
    topo: HierarchicalTopology,
) -> Dict[int, float]:
    load: Dict[int, float] = {}
    for src_str, e_counts in src_expert_counts.items():
        src = int(src_str)
        for e_str, cnt in e_counts.items():
            ds = mapping.get(int(e_str), [])
            if not ds:
                continue
            dst = _nearest_replica_by_latency(topo, src, ds)
            load[dst] = load.get(dst, 0.0) + float(cnt)
    return load


def _current_die_load(
    mapping: Dict[int, List[int]],
    expert_counts: Dict[str, int],
    src_expert_counts: Dict[str, Dict[str, int]],
    topo: HierarchicalTopology,
) -> Dict[int, float]:
    routed = _route_aware_die_load(mapping, src_expert_counts, topo)
    return routed if routed else _die_load(mapping, expert_counts)


def _route_aware_rank_expert_tokens(
    mapping: Dict[int, List[int]],
    src_expert_counts: Dict[str, Dict[str, int]],
    topo: HierarchicalTopology,
    batch_scale: float,
) -> List[Dict[int, float]]:
    num_ranks = len(topo.get_all_dies())
    rank_expert_tokens: List[Dict[int, float]] = [defaultdict(float) for _ in range(num_ranks)]
    for src_str, e_counts in src_expert_counts.items():
        src = int(src_str)
        for e_str, cnt in e_counts.items():
            e = int(e_str)
            ds = mapping.get(e, [])
            if not ds:
                continue
            # Match the final traffic/compute evaluator's replica assignment:
            # tokens are processed by the lowest zero-load path-latency replica.
            dst = _nearest_replica_by_latency(topo, src, ds)
            if 0 <= dst < num_ranks:
                rank_expert_tokens[dst][e] += float(cnt) * batch_scale
    return [dict(v) for v in rank_expert_tokens]


def _predicted_route_aware_compute_us(
    mapping: Dict[int, List[int]],
    src_expert_counts: Dict[str, Dict[str, int]],
    topo: HierarchicalTopology,
    roofline_cfg: H200RooflineConfig,
    batch_scale: float,
) -> float:
    if not src_expert_counts:
        return 0.0
    roof = compute_rank_roofline_from_rank_expert_tokens(
        _route_aware_rank_expert_tokens(mapping, src_expert_counts, topo, batch_scale),
        roofline_cfg,
    )
    return float(roof.get("max_rank_total_us", 0.0))


def _predicted_comm_bottleneck(
    mapping: Dict[int, List[int]],
    src_expert_counts: Dict[str, Dict[str, int]],
    topo: HierarchicalTopology,
    batch_scale: float,
    dispatch_bytes_per_token: float,
    combine_bytes_per_token: float | None = None,
) -> Dict[str, float]:
    if not src_expert_counts:
        return {
            "max_edge_port_bytes": 0.0,
            "cross_wafer_bytes": 0.0,
            "total_bytes": 0.0,
            "max_phase_service_ns": 0.0,
            "dispatch_max_link_service_ns": 0.0,
            "combine_max_link_service_ns": 0.0,
            "max_rank_send_bytes": 0.0,
            "max_rank_recv_bytes": 0.0,
        }
    combine_bytes_per_token = (
        dispatch_bytes_per_token if combine_bytes_per_token is None else combine_bytes_per_token
    )
    edge_port_bytes: Dict[int, float] = {}
    cross_wafer_bytes = 0.0
    total_bytes = 0.0
    dispatch_link_loads: Dict[Tuple, float] = {}
    combine_link_loads: Dict[Tuple, float] = {}
    rank_send_bytes: Dict[int, float] = {}
    rank_recv_bytes: Dict[int, float] = {}
    for src_str, e_counts in src_expert_counts.items():
        src = int(src_str)
        src_w, _ = topo.die_to_wafer_local(src)
        for e_str, cnt in e_counts.items():
            ds = mapping.get(int(e_str), [])
            if not ds:
                continue
            dst = _nearest_replica_by_latency(topo, src, ds)
            dispatch_b = float(cnt) * batch_scale * dispatch_bytes_per_token
            combine_b = float(cnt) * batch_scale * combine_bytes_per_token
            total_bytes += dispatch_b
            if dispatch_b > 0.0:
                rank_send_bytes[src] = rank_send_bytes.get(src, 0.0) + dispatch_b
                rank_recv_bytes[dst] = rank_recv_bytes.get(dst, 0.0) + dispatch_b
                for edge in _get_path_cached(topo, src, dst):
                    dispatch_link_loads[edge] = dispatch_link_loads.get(edge, 0.0) + dispatch_b
            if combine_b > 0.0:
                rank_send_bytes[dst] = rank_send_bytes.get(dst, 0.0) + combine_b
                rank_recv_bytes[src] = rank_recv_bytes.get(src, 0.0) + combine_b
                for edge in _get_path_cached(topo, dst, src):
                    combine_link_loads[edge] = combine_link_loads.get(edge, 0.0) + combine_b
            dst_w, _ = topo.die_to_wafer_local(dst)
            if src_w == dst_w:
                continue
            cross_wafer_bytes += dispatch_b
            edge_port_bytes[src_w] = edge_port_bytes.get(src_w, 0.0) + dispatch_b
            edge_port_bytes[dst_w] = edge_port_bytes.get(dst_w, 0.0) + dispatch_b

    def _max_link_service_ns(link_loads: Dict[Tuple, float]) -> float:
        if not link_loads:
            return 0.0
        return max(
            load / max(topo.get_link_bandwidth(edge), 1e-9)
            for edge, load in link_loads.items()
        )

    dispatch_link_ns = _max_link_service_ns(dispatch_link_loads)
    combine_link_ns = _max_link_service_ns(combine_link_loads)
    return {
        "max_edge_port_bytes": max(edge_port_bytes.values()) if edge_port_bytes else 0.0,
        "cross_wafer_bytes": cross_wafer_bytes,
        "total_bytes": total_bytes,
        "max_phase_service_ns": max(dispatch_link_ns, combine_link_ns),
        "dispatch_max_link_service_ns": dispatch_link_ns,
        "combine_max_link_service_ns": combine_link_ns,
        "max_rank_send_bytes": max(rank_send_bytes.values()) if rank_send_bytes else 0.0,
        "max_rank_recv_bytes": max(rank_recv_bytes.values()) if rank_recv_bytes else 0.0,
    }


def _cost_aware_window_evaluation(
    *,
    mapping: Dict[int, List[int]],
    src_expert_counts: Dict[str, Dict[str, int]],
    topo: HierarchicalTopology,
    omega: Dict[int, float],
    roofline_cfg: H200RooflineConfig,
    batch_scale: float,
    dispatch_bytes_per_token: float,
    combine_bytes_per_token: float,
    communication_weight: float,
    compute_load_weight: float,
    route_estimation_decisions_per_source_expert: int | None,
) -> Dict:
    route_path_cache = getattr(topo, "_runtime_scheduler_route_path_cache", None)
    if route_path_cache is None:
        route_path_cache = {}
        setattr(topo, "_runtime_scheduler_route_path_cache", route_path_cache)
    routed = route_activation_counts(
        src_expert_counts=src_expert_counts,
        mapping=mapping,
        topo=topo,
        dispatch_bytes_per_token=int(dispatch_bytes_per_token),
        routing_policy="hacmo_cost_aware",
        communication_weight=communication_weight,
        compute_load_weight=compute_load_weight,
        max_decisions_per_count=route_estimation_decisions_per_source_expert,
        path_cache=route_path_cache,
    )
    assignments = routed["assignments"]
    dies = topo.get_all_dies()
    rank_expert_tokens: List[Dict[int, float]] = [
        defaultdict(float) for _ in dies
    ]
    route_load: Dict[int, float] = dict(routed["die_load"])
    objective_link_loads: Dict[Tuple, float] = {
        edge: float(nbytes) / max(float(dispatch_bytes_per_token), 1e-12)
        for edge, nbytes in routed["dispatch_link_load_bytes"].items()
    }
    dispatch_link_loads: Dict[Tuple, float] = {
        edge: float(nbytes) * batch_scale
        for edge, nbytes in routed["dispatch_link_load_bytes"].items()
    }
    combine_link_loads: Dict[Tuple, float] = {}
    edge_port_bytes: Dict[int, float] = {}
    rank_send_bytes: Dict[int, float] = {}
    rank_recv_bytes: Dict[int, float] = {}
    comm_cost = sum(
        count * _edge_base_weight(topo, edge)
        for edge, count in objective_link_loads.items()
    )
    cross_wafer_bytes = 0.0
    total_bytes = float(routed["input_activations"]) * batch_scale * dispatch_bytes_per_token
    rank_pair_counts: Dict[Tuple[int, int], float] = {}

    for (_, src, expert, dst), raw_count in assignments.items():
        count = float(raw_count)
        scaled = count * batch_scale
        rank_expert_tokens[dst][expert] += scaled
        rank_pair_counts[(src, dst)] = rank_pair_counts.get((src, dst), 0.0) + count

    for (src, dst), count in rank_pair_counts.items():
        scaled = count * batch_scale
        dispatch_bytes = scaled * dispatch_bytes_per_token
        combine_bytes = scaled * combine_bytes_per_token
        rank_send_bytes[src] = rank_send_bytes.get(src, 0.0) + dispatch_bytes
        rank_recv_bytes[dst] = rank_recv_bytes.get(dst, 0.0) + dispatch_bytes
        rank_send_bytes[dst] = rank_send_bytes.get(dst, 0.0) + combine_bytes
        rank_recv_bytes[src] = rank_recv_bytes.get(src, 0.0) + combine_bytes
        for edge in _get_path_cached(topo, dst, src):
            combine_link_loads[edge] = (
                combine_link_loads.get(edge, 0.0) + combine_bytes
            )
        src_w, _ = topo.die_to_wafer_local(src)
        dst_w, _ = topo.die_to_wafer_local(dst)
        if src_w != dst_w:
            cross_wafer_bytes += dispatch_bytes
            edge_port_bytes[src_w] = edge_port_bytes.get(src_w, 0.0) + dispatch_bytes
            edge_port_bytes[dst_w] = edge_port_bytes.get(dst_w, 0.0) + dispatch_bytes

    def max_link_service_ns(loads: Dict[Tuple, float]) -> float:
        return max(
            (
                load / max(topo.get_link_bandwidth(edge), 1e-9)
                for edge, load in loads.items()
            ),
            default=0.0,
        )

    dispatch_link_ns = max_link_service_ns(dispatch_link_loads)
    combine_link_ns = max_link_service_ns(combine_link_loads)
    compute = compute_rank_roofline_from_rank_expert_tokens(
        [dict(tokens) for tokens in rank_expert_tokens], roofline_cfg
    )
    objective_parts = {
        "comm": comm_cost,
        "cong": _congestion_penalty(objective_link_loads, topo),
        "load": _runtime_load_penalty(route_load, omega, dies),
        "critical_load": (max(route_load.values()) if route_load else 0.0) ** 2,
    }
    return {
        "objective_parts": objective_parts,
        "compute_us": float(compute.get("max_rank_total_us", 0.0)),
        "route_load": route_load,
        "comm": {
            "max_edge_port_bytes": max(edge_port_bytes.values())
            if edge_port_bytes
            else 0.0,
            "cross_wafer_bytes": cross_wafer_bytes,
            "total_bytes": total_bytes,
            "max_phase_service_ns": max(dispatch_link_ns, combine_link_ns),
            "dispatch_max_link_service_ns": dispatch_link_ns,
            "combine_max_link_service_ns": combine_link_ns,
            "max_rank_send_bytes": max(rank_send_bytes.values())
            if rank_send_bytes
            else 0.0,
            "max_rank_recv_bytes": max(rank_recv_bytes.values())
            if rank_recv_bytes
            else 0.0,
        },
    }


def _estimated_serial_latency_us(evaluation: Dict) -> float:
    comm = evaluation["comm"]
    return float(evaluation["compute_us"]) + (
        float(comm["dispatch_max_link_service_ns"])
        + float(comm["combine_max_link_service_ns"])
    ) / 1000.0


def _mapping_cache_key(mapping: Dict[int, List[int]]) -> Tuple[Tuple[int, Tuple[int, ...]], ...]:
    return tuple((int(expert), tuple(int(die) for die in dies)) for expert, dies in sorted(mapping.items()))


def _step_level_window_latency_evaluation(
    *,
    mapping: Dict[int, List[int]],
    window_steps: List[Dict],
    topo: HierarchicalTopology,
    omega: Dict[int, float],
    roofline_cfg: H200RooflineConfig,
    batch_scale: float,
    dispatch_bytes_per_token: float,
    combine_bytes_per_token: float,
    communication_weight: float,
    compute_load_weight: float,
) -> Dict[str, float | int]:
    """Estimate one observed window with the final replay's per-step semantics."""
    if not window_steps:
        raise ValueError("step-level latency evaluation requires at least one step")
    evaluations = [
        _cost_aware_window_evaluation(
            mapping=mapping,
            src_expert_counts=step.get("src_expert_counts", {}),
            topo=topo,
            omega=omega,
            roofline_cfg=roofline_cfg,
            batch_scale=batch_scale,
            dispatch_bytes_per_token=dispatch_bytes_per_token,
            combine_bytes_per_token=combine_bytes_per_token,
            communication_weight=communication_weight,
            compute_load_weight=compute_load_weight,
            route_estimation_decisions_per_source_expert=None,
        )
        for step in window_steps
    ]
    count = len(evaluations)
    compute_us = sum(float(item["compute_us"]) for item in evaluations) / count
    dispatch_us = sum(
        float(item["comm"]["dispatch_max_link_service_ns"])
        for item in evaluations
    ) / count / 1000.0
    combine_us = sum(
        float(item["comm"]["combine_max_link_service_ns"])
        for item in evaluations
    ) / count / 1000.0
    return {
        "estimated_total_us": compute_us + dispatch_us + combine_us,
        "compute_us": compute_us,
        "dispatch_service_us": dispatch_us,
        "combine_service_us": combine_us,
        "num_steps": count,
    }


def _order_replicas_for_routing(
    mapping: Dict[int, List[int]],
    topo: HierarchicalTopology,
) -> Dict[int, List[int]]:
    ordered: Dict[int, List[int]] = {}
    for e, ds in mapping.items():
        uniq: List[int] = []
        seen: set[int] = set()
        for d in ds:
            if d not in seen:
                uniq.append(d)
                seen.add(d)
        ordered[e] = sorted(
            uniq,
            key=lambda d: (
                -topo._rail_scale(topo.die_to_wafer_local(d)[1]),
                topo.get_edge_distance(d),
                topo.die_to_wafer_local(d)[1],
                d,
            ),
        )
    return ordered


def _are_adjacent_dies(topo: HierarchicalTopology, first: int, second: int) -> bool:
    first_wafer, first_local = topo.die_to_wafer_local(first)
    second_wafer, second_local = topo.die_to_wafer_local(second)
    return (
        first_wafer == second_wafer
        and topo.mesh_hops(first_local, second_local) == 1
    )


def _bounded_replica_candidate_dies(
    *,
    expert: int,
    mapping: Dict[int, List[int]],
    src_expert_counts: Dict[str, Dict[str, int]],
    die_load: Dict[int, float],
    topo: HierarchicalTopology,
    limit: int,
) -> List[int]:
    occupied = set(mapping[expert])
    eligible = [die for die in topo.get_all_dies() if die not in occupied]
    path_cost = {
        die: sum(
            _path_base_cost_weighted(topo, int(src), die)
            * float(expert_counts.get(str(expert), 0))
            for src, expert_counts in src_expert_counts.items()
            if float(expert_counts.get(str(expert), 0)) > 0
        )
        for die in eligible
    }
    occupied_wafers = {
        topo.die_to_wafer_local(die)[0] for die in occupied
    }
    uncovered_wafer_demand: Dict[int, float] = defaultdict(float)
    for src, expert_counts in src_expert_counts.items():
        count = float(expert_counts.get(str(expert), 0))
        if count <= 0:
            continue
        wafer, _ = topo.die_to_wafer_local(int(src))
        if wafer not in occupied_wafers:
            uncovered_wafer_demand[wafer] += count

    selected: List[int] = []
    if uncovered_wafer_demand:
        target_wafer = min(
            uncovered_wafer_demand,
            key=lambda wafer: (-uncovered_wafer_demand[wafer], wafer),
        )
        wafer_candidates = [
            die
            for die in eligible
            if topo.die_to_wafer_local(die)[0] == target_wafer
        ]
        if wafer_candidates:
            selected.append(
                min(
                    wafer_candidates,
                    key=lambda die: (
                        path_cost[die],
                        die_load.get(die, 0.0),
                        topo.get_edge_distance(die),
                        die,
                    ),
                )
            )

    rankings = [
        sorted(eligible, key=lambda die: (path_cost[die], die_load.get(die, 0.0), die)),
        sorted(eligible, key=lambda die: (die_load.get(die, 0.0), path_cost[die], die)),
        sorted(
            eligible,
            key=lambda die: (
                topo.get_edge_distance(die),
                path_cost[die],
                die_load.get(die, 0.0),
                die,
            ),
        ),
    ]
    if len(selected) >= max(1, limit):
        return selected
    for rank in range(max((len(items) for items in rankings), default=0)):
        for items in rankings:
            if rank >= len(items):
                continue
            die = items[rank]
            if die not in selected:
                selected.append(die)
                if len(selected) >= max(1, limit):
                    return selected
    return selected


def _adjacent_swap_candidates(
    *,
    mapping: Dict[int, List[int]],
    hotness: Dict[int, float],
    die_load: Dict[int, float],
    topo: HierarchicalTopology,
) -> List[Dict[str, int | float]]:
    hosted: Dict[int, List[int]] = defaultdict(list)
    for expert, dies in mapping.items():
        for die in dies:
            hosted[die].append(expert)

    candidates: List[Dict[str, int | float]] = []
    rows = topo.cfg.mesh_rows
    cols = topo.cfg.mesh_cols
    for wafer in range(topo.cfg.num_wafers):
        base = wafer * topo.cfg.dies_per_wafer
        for row in range(rows):
            for col in range(cols):
                first = base + topo.rc_to_local(row, col)
                neighbors = []
                if row + 1 < rows:
                    neighbors.append(base + topo.rc_to_local(row + 1, col))
                if col + 1 < cols:
                    neighbors.append(base + topo.rc_to_local(row, col + 1))
                for second in neighbors:
                    first_depth = topo.get_edge_distance(first)
                    second_depth = topo.get_edge_distance(second)
                    if first_depth == second_depth:
                        continue
                    inner, outer = (
                        (first, second)
                        if first_depth > second_depth
                        else (second, first)
                    )
                    if not hosted[inner] or not hosted[outer]:
                        continue
                    valid_pairs = []
                    for expert_hot in hosted[inner]:
                        if outer in mapping[expert_hot]:
                            continue
                        for expert_cold in hosted[outer]:
                            if expert_hot == expert_cold or inner in mapping[expert_cold]:
                                continue
                            hotness_gap = hotness.get(
                                expert_hot, 0.0
                            ) - hotness.get(expert_cold, 0.0)
                            if hotness_gap > 0.0:
                                valid_pairs.append(
                                    (hotness_gap, expert_hot, expert_cold)
                                )
                    if not valid_pairs:
                        continue
                    hotness_gap, expert_hot, expert_cold = min(
                        valid_pairs,
                        key=lambda item: (-item[0], item[1], item[2]),
                    )
                    candidates.append(
                        {
                            "expert_hot": expert_hot,
                            "from_die": inner,
                            "to_die": outer,
                            "expert_cold": expert_cold,
                            "from_die_cold": outer,
                            "to_die_cold": inner,
                            "hotness_gap": hotness_gap,
                            "load_gap": die_load.get(inner, 0.0)
                            - die_load.get(outer, 0.0),
                        }
                    )
    return sorted(
        candidates,
        key=lambda item: (
            -float(item["hotness_gap"]),
            -float(item["load_gap"]),
            int(item["from_die"]),
            int(item["to_die"]),
        ),
    )


def build_migration_events(
    *,
    replicas_added: List[Dict],
    swaps: List[Dict],
    expert_weight_bytes: float,
    topo: HierarchicalTopology,
    observed_window: int,
) -> List[Dict]:
    raw_events = [
        (
            "replicate",
            int(entry["expert"]),
            int(entry["source_die"]),
            int(entry["die"]),
        )
        for entry in replicas_added
    ]
    for swap in swaps:
        raw_events.extend(
            [
                (
                    "swap",
                    int(swap["expert_hot"]),
                    int(swap["from_die"]),
                    int(swap["to_die"]),
                ),
                (
                    "swap",
                    int(swap["expert_cold"]),
                    int(swap["from_die_cold"]),
                    int(swap["to_die_cold"]),
                ),
            ]
        )

    events = []
    for operation, expert, source, destination in raw_events:
        path = topo.get_path(source, destination)
        latency_ns = sum(
            (
                topo.cfg.mesh_latency_ns
                if edge[0] == "mesh"
                else topo.cfg.switch_latency_ns
            )
            + expert_weight_bytes / max(topo.get_link_bandwidth(edge), 1e-9)
            for edge in path
        )
        events.append(
            {
                "observed_window": observed_window,
                "operation": operation,
                "expert": expert,
                "source_die": source,
                "destination_die": destination,
                "bytes": int(expert_weight_bytes),
                "path": [list(edge) for edge in path],
                "estimated_latency_us": latency_ns / 1000.0,
            }
        )
    return events


def _migration_activation_window(
    observed_window: int,
    *,
    migration_us: float,
    slack_us_per_window: float,
) -> int:
    if migration_us <= 0.0:
        return observed_window + 1
    if slack_us_per_window <= 0.0:
        raise ValueError("attention migration slack must be positive when migration is scheduled")
    return observed_window + max(1, math.ceil(migration_us / slack_us_per_window))


def _path_cost_weighted(topo: HierarchicalTopology, src: int, dst: int, link_loads: Dict, lambda_: float) -> float:
    """Compute weighted path cost using the same model as the static mapper.

    Switch edges have z_l >> mesh edges (z_l = mesh_bw/switch_bw +
    switch_latency/mesh_latency), so cross-wafer paths are properly
    penalised relative to same-wafer paths.
    """
    return sum(
        base + lambda_ * link_loads.get(edge, 0.0) * inv_bw
        for edge, base, inv_bw in _get_weighted_path_profile_cached(topo, src, dst)
    )


def _congestion_penalty(link_loads: Dict, topo: HierarchicalTopology) -> float:
    """Phi_cong = sum_l (L_l / B_l)^2, matching paper Eq.(5)."""
    cong = 0.0
    for edge, load in link_loads.items():
        bw = topo.get_link_bandwidth(edge)
        cong += (load / max(bw, 1e-6)) ** 2
    return cong


def _runtime_load_penalty(
    die_loads: Dict[int, float],
    omega: Dict[int, float],
    dies: List[int],
) -> float:
    qbar = sum(die_loads.get(die, 0.0) for die in dies) / max(len(dies), 1)
    return sum(
        (die_loads.get(die, 0.0) - omega.get(die, 1.0) * qbar) ** 2
        for die in dies
    )


def _window_objective_parts(
    mapping: Dict[int, List[int]],
    src_expert_counts: Dict[str, Dict[str, int]],
    expert_counts: Dict[str, int],
    topo: HierarchicalTopology,
    lambda_: float,
    omega: Dict[int, float],
) -> Dict[str, float]:
    dies = topo.get_all_dies()
    if not src_expert_counts:
        die_load_estimate = _die_load(mapping, expert_counts)
        return {
            "comm": 0.0,
            "cong": 0.0,
            "load": _runtime_load_penalty(die_load_estimate, omega, dies),
            "critical_load": (max(die_load_estimate.values()) if die_load_estimate else 0.0) ** 2,
        }

    link_loads: Dict = {}
    die_load_estimate: Dict[int, float] = {}
    comm = 0.0
    for su_str, e_counts in src_expert_counts.items():
        su = int(su_str)
        for e_str, c in e_counts.items():
            e = int(e_str)
            ds = mapping.get(e, [])
            if not ds:
                continue
            vol = float(c)
            best_d = _nearest_replica_by_latency(topo, su, ds)
            path = _get_path_cached(topo, su, best_d)
            cost = _path_base_cost_weighted(topo, su, best_d)
            comm += vol * cost
            die_load_estimate[best_d] = die_load_estimate.get(best_d, 0.0) + vol
            # Accumulate link loads for congestion.
            for edge in path:
                link_loads[edge] = link_loads.get(edge, 0.0) + vol

    cong = _congestion_penalty(link_loads, topo)

    return {
        "comm": comm,
        "cong": cong,
        "load": _runtime_load_penalty(die_load_estimate, omega, dies),
        "critical_load": (max(die_load_estimate.values()) if die_load_estimate else 0.0) ** 2,
    }


def _window_objective(
    mapping: Dict[int, List[int]],
    src_expert_counts: Dict[str, Dict[str, int]],
    expert_counts: Dict[str, int],
    topo: HierarchicalTopology,
    alpha: float,
    gamma: float,
    delta: float,
    lambda_: float,
    critical_load_weight: float,
    omega: Dict[int, float],
    normalization_refs: Dict[str, float] | None = None,
) -> float:
    """Evaluate the dimensionless runtime form of paper Eq.(6)."""
    parts = _window_objective_parts(
        mapping, src_expert_counts, expert_counts, topo, lambda_, omega
    )
    refs = normalization_refs or {name: 1.0 for name in ("comm", "cong", "load")}
    normalized = {
        name: parts[name] / max(abs(float(refs[name])), 1e-12)
        for name in ("comm", "cong", "load")
    }
    critical_ref = max(abs(float(refs.get("critical_load", 1.0))), 1e-12)
    return (
        alpha * normalized["comm"]
        + gamma * normalized["cong"]
        + delta * normalized["load"]
        + critical_load_weight * parts["critical_load"] / critical_ref
    )


def _objective_from_parts(
    parts: Dict[str, float],
    *,
    alpha: float,
    gamma: float,
    delta: float,
    critical_load_weight: float,
    normalization_refs: Dict[str, float],
) -> float:
    return (
        alpha * parts["comm"] / max(abs(normalization_refs["comm"]), 1e-12)
        + gamma * parts["cong"] / max(abs(normalization_refs["cong"]), 1e-12)
        + delta * parts["load"] / max(abs(normalization_refs["load"]), 1e-12)
        + critical_load_weight
        * parts["critical_load"]
        / max(abs(normalization_refs["critical_load"]), 1e-12)
    )


def _accumulate_link_loads_for_expert(
    link_loads: Dict,
    dst_die: int,
    src_expert_counts: Dict[str, Dict[str, int]],
    expert: int,
    topo: HierarchicalTopology,
) -> None:
    """Mutate link_loads in-place, adding traffic for *expert* -> *dst_die*."""
    for su_str, e_counts in src_expert_counts.items():
        su = int(su_str)
        vol = float(e_counts.get(str(expert), 0))
        if vol <= 0:
            continue
        for edge in _get_path_cached(topo, su, dst_die):
            link_loads[edge] = link_loads.get(edge, 0.0) + vol


def run_runtime(
    static_mapping: Dict[int, List[int]],
    window_stats: List[Dict],
    window_src_expert_counts: List[Dict] | None,
    topo: HierarchicalTopology,
    decay: float,
    hot_percentile: float,
    replica_persistence_windows: int,
    migration_budget_bytes_per_window: int,
    bytes_per_token: int,
    max_replicas_per_expert: int,
    step_inputs: List[Dict] | None = None,
    alpha: float = 1.0,
    gamma: float = 1.0,
    delta: float = 1.0,
    lambda_: float = 1.0,
    critical_load_weight: float = 0.0,
    runtime_compute_batch_scale: float = 1.0,
    dispatch_bytes_per_token: float = 7168.0,
    route_communication_weight: float = 1.0,
    route_compute_load_weight: float = 1.0,
    route_estimation_decisions_per_source_expert: int = 1,
    roofline_cfg: H200RooflineConfig | None = None,
    max_adjustments_per_window: int = 16,
    max_hot_experts_per_window: int = 8,
    max_candidate_dies_per_expert: int = 3,
    max_candidate_operations_per_window: int = 12,
    attention_migration_slack_us_per_window: float = 1000.0,
) -> List[Dict]:
    mapper = _order_replicas_for_routing(copy.deepcopy(static_mapping), topo)
    epoch_base_mapping = copy.deepcopy(mapper)
    roofline_cfg = roofline_cfg or H200RooflineConfig()
    expert_weight_bytes = roofline_cfg.expert_weight_bytes
    base_replica = {e: len(ds) for e, ds in mapper.items()}
    replica_cap = {
        e: max(max_replicas_per_expert, base_replica.get(e, 1))
        for e in mapper.keys()
    }
    estimator = HotnessEstimator(experts=len(mapper), decay=decay)
    high_count = {e: 0 for e in mapper.keys()}
    low_count = {e: 0 for e in mapper.keys()}
    omega = topo.get_outer_to_inner_weights()
    windows_out = []
    pending_mapping: Dict[int, List[int]] | None = None
    pending_activation_window: int | None = None
    pending_source_window: int | None = None
    steps_by_window: Dict[int, List[Dict]] = defaultdict(list)
    for step in step_inputs or []:
        steps_by_window[int(step["window_id"])].append(step)
    for steps in steps_by_window.values():
        steps.sort(key=lambda step: int(step["step_id"]))

    for ws in window_stats:
        w_id = int(ws["window_id"])
        activated_pending_update = False
        if (
            pending_mapping is not None
            and pending_activation_window is not None
            and w_id >= pending_activation_window
        ):
            mapper = copy.deepcopy(pending_mapping)
            pending_mapping = None
            pending_activation_window = None
            pending_source_window = None
            activated_pending_update = True
        update_blocked_by_pending_migration = pending_mapping is not None
        mapper_for_window = copy.deepcopy(mapper)
        mapper_before_update = copy.deepcopy(mapper)
        src_expert_counts: Dict[str, Dict[str, int]] = {}
        if window_src_expert_counts is not None and w_id < len(window_src_expert_counts):
            src_expert_counts = window_src_expert_counts[w_id].get("src_expert_counts", {})
        window_steps = steps_by_window.get(w_id, [])
        hotness = estimator.update(ws.get("expert_counts", {}))
        h_values = list(hotness.values())
        h_thr = _percentile(h_values, hot_percentile)
        for expert, value in hotness.items():
            high_count[expert] = (
                high_count[expert] + 1 if value >= h_thr else 0
            )
        replicas_added = []
        replicas_removed = []
        swaps_done = []
        base_rollback = None
        base_rollback_accepted = False
        migration_bytes = 0
        adjustments_done = 0

        aggregate_evaluation_cache: Dict[
            Tuple[Tuple[int, Tuple[int, ...]], ...], Dict
        ] = {}
        step_latency_cache: Dict[
            Tuple[Tuple[int, Tuple[int, ...]], ...], Dict[str, float | int]
        ] = {}

        def evaluate(candidate_mapping: Dict[int, List[int]]) -> Dict:
            key = _mapping_cache_key(candidate_mapping)
            if key in aggregate_evaluation_cache:
                return aggregate_evaluation_cache[key]
            result = _cost_aware_window_evaluation(
                mapping=candidate_mapping,
                src_expert_counts=src_expert_counts,
                topo=topo,
                omega=omega,
                roofline_cfg=roofline_cfg,
                batch_scale=runtime_compute_batch_scale,
                dispatch_bytes_per_token=dispatch_bytes_per_token,
                combine_bytes_per_token=bytes_per_token,
                communication_weight=route_communication_weight,
                compute_load_weight=route_compute_load_weight,
                route_estimation_decisions_per_source_expert=(
                    route_estimation_decisions_per_source_expert
                ),
            )
            aggregate_evaluation_cache[key] = result
            return result

        def evaluate_step_latency(
            candidate_mapping: Dict[int, List[int]]
        ) -> Dict[str, float | int]:
            key = _mapping_cache_key(candidate_mapping)
            if key in step_latency_cache:
                return step_latency_cache[key]
            result = _step_level_window_latency_evaluation(
                mapping=candidate_mapping,
                window_steps=window_steps,
                topo=topo,
                omega=omega,
                roofline_cfg=roofline_cfg,
                batch_scale=runtime_compute_batch_scale,
                dispatch_bytes_per_token=dispatch_bytes_per_token,
                combine_bytes_per_token=bytes_per_token,
                communication_weight=route_communication_weight,
                compute_load_weight=route_compute_load_weight,
            )
            step_latency_cache[key] = result
            return result

        initial_evaluation = evaluate(mapper)
        aggregate_latency_before_window = _estimated_serial_latency_us(
            initial_evaluation
        )
        if window_steps:
            step_latency_before_window = evaluate_step_latency(mapper)
        else:
            step_latency_before_window = {
                "estimated_total_us": aggregate_latency_before_window,
                "compute_us": float(initial_evaluation["compute_us"]),
                "dispatch_service_us": float(
                    initial_evaluation["comm"]["dispatch_max_link_service_ns"]
                )
                / 1000.0,
                "combine_service_us": float(
                    initial_evaluation["comm"]["combine_max_link_service_ns"]
                )
                / 1000.0,
                "num_steps": 0,
            }
        objective_reference_parts = initial_evaluation["objective_parts"]
        objective_refs = {
            name: max(abs(float(objective_reference_parts[name])), 1e-12)
            for name in ("comm", "cong", "load", "critical_load")
        }
        objective_before = _objective_from_parts(
            objective_reference_parts,
            alpha=alpha,
            gamma=gamma,
            delta=delta,
            critical_load_weight=critical_load_weight,
            normalization_refs=objective_refs,
        )
        compute_before_window = initial_evaluation["compute_us"]
        comm_before_window = initial_evaluation["comm"]
        route_load_before = initial_evaluation["route_load"]
        die_load = route_load_before
        current_objective = objective_before
        current_compute_us = compute_before_window
        current_comm = comm_before_window
        current_aggregate_latency_us = aggregate_latency_before_window
        current_step_latency = step_latency_before_window
        candidate_search = {
            "evaluated": {
                "rollback": 0,
                "replicate": 0,
                "evict": 0,
                "swap": 0,
                "total": 0,
            },
            "accepted": {
                "rollback": 0,
                "replicate": 0,
                "evict": 0,
                "swap": 0,
                "total": 0,
            },
            "rejected": {},
        }
        operation_evaluation_limits = {
            "rollback": 1,
            "replicate": max(1, math.ceil(max_candidate_operations_per_window * 2 / 3)),
            "evict": max(1, max_candidate_operations_per_window // 6),
            "swap": max_candidate_operations_per_window,
        }

        def reject(reason: str) -> None:
            rejected = candidate_search["rejected"]
            rejected[reason] = int(rejected.get(reason, 0)) + 1

        def evaluate_candidate(operation: str, candidate_mapping: Dict[int, List[int]]) -> Dict | None:
            evaluated = candidate_search["evaluated"]
            if int(evaluated["total"]) >= max_candidate_operations_per_window:
                reject("global_evaluation_budget_exhausted")
                return None
            if int(evaluated[operation]) >= operation_evaluation_limits[operation]:
                reject(f"{operation}_evaluation_budget_exhausted")
                return None
            result = evaluate(candidate_mapping)
            evaluated[operation] = int(evaluated[operation]) + 1
            evaluated["total"] = int(evaluated["total"]) + 1
            return result

        def accept(operation: str) -> None:
            accepted = candidate_search["accepted"]
            accepted[operation] = int(accepted[operation]) + 1
            accepted["total"] = int(accepted["total"]) + 1

        def candidate_latency_improves(
            operation: str,
            candidate_mapping: Dict[int, List[int]],
            candidate_evaluation: Dict,
            *,
            require_aggregate_improvement: bool = True,
        ) -> bool:
            allow_equal = operation == "evict"
            candidate_aggregate_latency_us = _estimated_serial_latency_us(
                candidate_evaluation
            )
            aggregate_rejected = (
                candidate_aggregate_latency_us > current_aggregate_latency_us + 1e-12
                if allow_equal
                else candidate_aggregate_latency_us
                >= current_aggregate_latency_us - 1e-12
            )
            if aggregate_rejected and (
                require_aggregate_improvement or not window_steps
            ):
                reject("aggregate_latency_not_improved")
                return False
            if window_steps:
                candidate_step_latency = evaluate_step_latency(candidate_mapping)
                candidate_step_total_us = float(
                    candidate_step_latency["estimated_total_us"]
                )
                current_step_total_us = float(
                    current_step_latency["estimated_total_us"]
                )
                step_rejected = (
                    candidate_step_total_us > current_step_total_us + 1e-12
                    if allow_equal
                    else candidate_step_total_us >= current_step_total_us - 1e-12
                )
                if step_rejected:
                    reject("step_latency_not_improved")
                    return False
            else:
                candidate_step_latency = {
                    "estimated_total_us": candidate_aggregate_latency_us,
                    "compute_us": float(candidate_evaluation["compute_us"]),
                    "dispatch_service_us": float(
                        candidate_evaluation["comm"]["dispatch_max_link_service_ns"]
                    )
                    / 1000.0,
                    "combine_service_us": float(
                        candidate_evaluation["comm"]["combine_max_link_service_ns"]
                    )
                    / 1000.0,
                    "num_steps": 0,
                }
            candidate_evaluation["aggregate_estimated_total_us"] = (
                candidate_aggregate_latency_us
            )
            candidate_evaluation["step_latency"] = candidate_step_latency
            return True

        # A local adjustment can become stale after the activation distribution
        # changes. Compare the active mapping against the fixed epoch-base
        # placement on the just-observed window and causally restore the base for
        # the next window when it is strictly better. Do not mix rollback with a
        # new adjustment in the same window.
        if not update_blocked_by_pending_migration and mapper != epoch_base_mapping:
            rollback_added: list[tuple[int, int, int]] = []
            rollback_removed: list[tuple[int, int]] = []
            changed_experts = 0
            for expert in sorted(mapper):
                current_dies = set(mapper[expert])
                base_dies = set(epoch_base_mapping[expert])
                if current_dies == base_dies:
                    continue
                changed_experts += 1
                for destination in sorted(base_dies - current_dies):
                    source = min(
                        mapper[expert],
                        key=lambda die: (
                            _path_latency_ns(topo, die, destination),
                            die,
                        ),
                    )
                    rollback_added.append((expert, source, destination))
                rollback_removed.extend(
                    (expert, die) for die in sorted(current_dies - base_dies)
                )
            rollback_migration_bytes = len(rollback_added) * expert_weight_bytes
            if changed_experts > max_adjustments_per_window:
                reject("rollback_adjustment_budget_exhausted")
            elif rollback_migration_bytes > migration_budget_bytes_per_window:
                reject("rollback_migration_budget_exhausted")
            else:
                candidate_mapping = copy.deepcopy(epoch_base_mapping)
                candidate_evaluation = evaluate_candidate(
                    "rollback", candidate_mapping
                )
                if candidate_evaluation is not None:
                    rollback_objective = _objective_from_parts(
                        candidate_evaluation["objective_parts"],
                        alpha=alpha,
                        gamma=gamma,
                        delta=delta,
                        critical_load_weight=critical_load_weight,
                        normalization_refs=objective_refs,
                    )
                    if candidate_latency_improves(
                        "rollback",
                        candidate_mapping,
                        candidate_evaluation,
                        require_aggregate_improvement=False,
                    ):
                        before = current_objective
                        compute_before = current_compute_us
                        comm_before = current_comm
                        compute_after = candidate_evaluation["compute_us"]
                        comm_after = candidate_evaluation["comm"]
                        for expert, source, destination in rollback_added:
                            replicas_added.append(
                                {
                                    "expert": expert,
                                    "source_die": source,
                                    "die": destination,
                                    "reason": "restore_epoch_base",
                                    "objective_before": round(before, 4),
                                    "objective_after": round(rollback_objective, 4),
                                    "compute_before_us": round(compute_before, 6),
                                    "compute_after_us": round(compute_after, 6),
                                    "max_edge_before_bytes": round(
                                        comm_before["max_edge_port_bytes"], 4
                                    ),
                                    "max_edge_after_bytes": round(
                                        comm_after["max_edge_port_bytes"], 4
                                    ),
                                    "max_phase_service_before_ns": round(
                                        comm_before["max_phase_service_ns"], 4
                                    ),
                                    "max_phase_service_after_ns": round(
                                        comm_after["max_phase_service_ns"], 4
                                    ),
                                }
                            )
                        for expert, die in rollback_removed:
                            replicas_removed.append(
                                {
                                    "expert": expert,
                                    "die": die,
                                    "reason": "restore_epoch_base",
                                    "was_static_replica": False,
                                    "objective_before": round(before, 4),
                                    "objective_after": round(rollback_objective, 4),
                                    "compute_before_us": round(compute_before, 6),
                                    "compute_after_us": round(compute_after, 6),
                                    "max_edge_before_bytes": round(
                                        comm_before["max_edge_port_bytes"], 4
                                    ),
                                    "max_edge_after_bytes": round(
                                        comm_after["max_edge_port_bytes"], 4
                                    ),
                                    "max_phase_service_before_ns": round(
                                        comm_before["max_phase_service_ns"], 4
                                    ),
                                    "max_phase_service_after_ns": round(
                                        comm_after["max_phase_service_ns"], 4
                                    ),
                                }
                            )
                        mapper = candidate_mapping
                        migration_bytes += rollback_migration_bytes
                        adjustments_done += changed_experts
                        accept("rollback")
                        base_rollback_accepted = True
                        base_rollback = {
                            "accepted": True,
                            "changed_experts": changed_experts,
                            "copies_restored": len(rollback_added),
                            "copies_removed": len(rollback_removed),
                            "objective_improved": (
                                rollback_objective < before - 1e-12
                            ),
                        }
                        current_objective = rollback_objective
                        current_compute_us = compute_after
                        current_comm = comm_after
                        current_aggregate_latency_us = candidate_evaluation[
                            "aggregate_estimated_total_us"
                        ]
                        current_step_latency = candidate_evaluation["step_latency"]
                        die_load = candidate_evaluation["route_load"]

        # --- Replicate: add replicas for hot experts ---
        hot_candidates = []
        for e, h_val in sorted(hotness.items(), key=lambda x: x[1], reverse=True):
            if h_val < h_thr:
                continue
            if high_count[e] < replica_persistence_windows:
                reject("hot_persistence_not_met")
                continue
            if len(mapper[e]) >= replica_cap.get(e, max_replicas_per_expert):
                reject("replica_cap_reached")
                continue
            hot_candidates.append((e, h_val))
            if len(hot_candidates) >= max(1, max_hot_experts_per_window):
                break
        if update_blocked_by_pending_migration or base_rollback_accepted:
            hot_candidates = []
            reject(
                "base_rollback_blocks_new_adjustments"
                if base_rollback_accepted
                else "pending_migration_blocks_search"
            )
        for e, h_val in hot_candidates:
            if adjustments_done >= max_adjustments_per_window:
                reject("adjustment_budget_exhausted")
                break
            if int(candidate_search["evaluated"]["total"]) >= max_candidate_operations_per_window:
                break
            if migration_bytes + expert_weight_bytes > migration_budget_bytes_per_window:
                reject("migration_budget_exhausted")
                continue
            candidates = _bounded_replica_candidate_dies(
                expert=e,
                mapping=mapper,
                src_expert_counts=src_expert_counts,
                die_load=die_load,
                topo=topo,
                limit=max_candidate_dies_per_expert,
            )
            before = current_objective
            improving_candidates = []
            for d in candidates:
                candidate_mapping = {k: list(v) for k, v in mapper.items()}
                candidate_mapping[e].append(d)
                candidate_evaluation = evaluate_candidate("replicate", candidate_mapping)
                if candidate_evaluation is None:
                    break
                after = _objective_from_parts(
                    candidate_evaluation["objective_parts"],
                    alpha=alpha,
                    gamma=gamma,
                    delta=delta,
                    critical_load_weight=critical_load_weight,
                    normalization_refs=objective_refs,
                )
                if after >= before - 1e-12:
                    reject("joint_objective_not_improved")
                elif candidate_latency_improves(
                    "replicate", candidate_mapping, candidate_evaluation
                ):
                    improving_candidates.append((after, d, candidate_evaluation))
            if not improving_candidates:
                continue
            improving_candidates.sort(key=lambda item: (item[0], item[1]))
            after, dst, candidate_evaluation = improving_candidates[0]
            for _ in improving_candidates[1:]:
                reject("not_best_joint_candidate")
            compute_before = current_compute_us
            comm_before = current_comm
            mapper[e].append(dst)
            compute_after = candidate_evaluation["compute_us"]
            comm_after = candidate_evaluation["comm"]
            source_die = min(
                mapper[e][:-1],
                key=lambda die: (_path_latency_ns(topo, die, dst), die),
            )
            replicas_added.append({
                "expert": e, "source_die": source_die, "die": dst,
                "objective_before": round(before, 4),
                "objective_after": round(after, 4),
                "compute_before_us": round(compute_before, 6),
                "compute_after_us": round(compute_after, 6),
                "max_edge_before_bytes": round(comm_before["max_edge_port_bytes"], 4),
                "max_edge_after_bytes": round(comm_after["max_edge_port_bytes"], 4),
                "max_phase_service_before_ns": round(comm_before["max_phase_service_ns"], 4),
                "max_phase_service_after_ns": round(comm_after["max_phase_service_ns"], 4),
            })
            migration_bytes += expert_weight_bytes
            adjustments_done += 1
            accept("replicate")
            current_objective = after
            current_compute_us = compute_after
            current_comm = comm_after
            current_aggregate_latency_us = candidate_evaluation[
                "aggregate_estimated_total_us"
            ]
            current_step_latency = candidate_evaluation["step_latency"]
            die_load = candidate_evaluation["route_load"]

        # --- Evict: remove replicas for cold experts ---
        for e, h_val in (
            []
            if update_blocked_by_pending_migration or base_rollback_accepted
            else hotness.items()
        ):
            if adjustments_done >= max_adjustments_per_window:
                reject("adjustment_budget_exhausted")
                break
            if h_val < h_thr * 0.5:
                low_count[e] += 1
            else:
                low_count[e] = 0
            if low_count[e] < replica_persistence_windows or len(mapper[e]) <= 1:
                continue
            before = current_objective
            improving_candidates = []
            for removed in list(mapper[e]):
                candidate_mapping = {k: list(v) for k, v in mapper.items()}
                candidate_mapping[e].remove(removed)
                candidate_evaluation = evaluate_candidate("evict", candidate_mapping)
                if candidate_evaluation is None:
                    break
                after = _objective_from_parts(
                    candidate_evaluation["objective_parts"],
                    alpha=alpha,
                    gamma=gamma,
                    delta=delta,
                    critical_load_weight=critical_load_weight,
                    normalization_refs=objective_refs,
                )
                if after >= before - 1e-12:
                    reject("joint_objective_not_improved")
                elif candidate_latency_improves(
                    "evict", candidate_mapping, candidate_evaluation
                ):
                    improving_candidates.append((after, removed, candidate_evaluation))
            if not improving_candidates:
                continue
            improving_candidates.sort(key=lambda item: (item[0], item[1]))
            after, removed, candidate_evaluation = improving_candidates[0]
            for _ in improving_candidates[1:]:
                reject("not_best_joint_candidate")
            compute_before = current_compute_us
            comm_before = current_comm
            mapper[e].remove(removed)
            compute_after = candidate_evaluation["compute_us"]
            comm_after = candidate_evaluation["comm"]
            replicas_removed.append({
                "expert": e, "die": removed,
                "was_static_replica": removed in static_mapping.get(e, []),
                "objective_before": round(before, 4),
                "objective_after": round(after, 4),
                "compute_before_us": round(compute_before, 6),
                "compute_after_us": round(compute_after, 6),
                "max_edge_before_bytes": round(comm_before["max_edge_port_bytes"], 4),
                "max_edge_after_bytes": round(comm_after["max_edge_port_bytes"], 4),
                "max_phase_service_before_ns": round(comm_before["max_phase_service_ns"], 4),
                "max_phase_service_after_ns": round(comm_after["max_phase_service_ns"], 4),
            })
            low_count[e] = 0
            adjustments_done += 1
            accept("evict")
            current_objective = after
            current_compute_us = compute_after
            current_comm = comm_after
            current_aggregate_latency_us = candidate_evaluation[
                "aggregate_estimated_total_us"
            ]
            current_step_latency = candidate_evaluation["step_latency"]
            die_load = candidate_evaluation["route_load"]

        # --- Swap: exchange expert positions between adjacent dies ---
        # Enumerate real mesh-neighbor pairs and move hot experts one ring
        # outward. The joint objective, rather than a per-term guard, decides
        # whether one bounded swap is accepted.
        if not update_blocked_by_pending_migration and not base_rollback_accepted:
            if adjustments_done >= max_adjustments_per_window:
                reject("adjustment_budget_exhausted")
            elif migration_bytes + 2 * expert_weight_bytes > migration_budget_bytes_per_window:
                reject("migration_budget_exhausted")
            else:
                before = current_objective
                improving_candidates = []
                for swap in _adjacent_swap_candidates(
                    mapping=mapper,
                    hotness=hotness,
                    die_load=die_load,
                    topo=topo,
                ):
                    e_hot = int(swap["expert_hot"])
                    d_inner = int(swap["from_die"])
                    d_outer = int(swap["to_die"])
                    e_cold = int(swap["expert_cold"])
                    candidate_mapping = {k: list(v) for k, v in mapper.items()}
                    candidate_mapping[e_hot] = [
                        d_outer if die == d_inner else die
                        for die in candidate_mapping[e_hot]
                    ]
                    candidate_mapping[e_cold] = [
                        d_inner if die == d_outer else die
                        for die in candidate_mapping[e_cold]
                    ]
                    candidate_evaluation = evaluate_candidate("swap", candidate_mapping)
                    if candidate_evaluation is None:
                        break
                    after = _objective_from_parts(
                        candidate_evaluation["objective_parts"],
                        alpha=alpha,
                        gamma=gamma,
                        delta=delta,
                        critical_load_weight=critical_load_weight,
                        normalization_refs=objective_refs,
                    )
                    if after >= before - 1e-12:
                        reject("joint_objective_not_improved")
                    elif candidate_latency_improves(
                        "swap", candidate_mapping, candidate_evaluation
                    ):
                        improving_candidates.append(
                            (after, d_inner, d_outer, swap, candidate_evaluation)
                        )
                if improving_candidates:
                    improving_candidates.sort(
                        key=lambda item: (item[0], item[1], item[2])
                    )
                    after, d_inner, d_outer, swap, candidate_evaluation = (
                        improving_candidates[0]
                    )
                    for _ in improving_candidates[1:]:
                        reject("not_best_joint_candidate")
                    e_hot = int(swap["expert_hot"])
                    e_cold = int(swap["expert_cold"])
                    compute_before = current_compute_us
                    comm_before = current_comm
                    mapper[e_hot] = [
                        d_outer if die == d_inner else die
                        for die in mapper[e_hot]
                    ]
                    mapper[e_cold] = [
                        d_inner if die == d_outer else die
                        for die in mapper[e_cold]
                    ]
                    compute_after = candidate_evaluation["compute_us"]
                    comm_after = candidate_evaluation["comm"]
                    swaps_done.append({
                        "expert_hot": e_hot,
                        "from_die": d_inner,
                        "to_die": d_outer,
                        "expert_cold": e_cold,
                        "from_die_cold": d_outer,
                        "to_die_cold": d_inner,
                        "hotness_gap": round(float(swap["hotness_gap"]), 4),
                        "objective_before": round(before, 4),
                        "objective_after": round(after, 4),
                        "compute_before_us": round(compute_before, 6),
                        "compute_after_us": round(compute_after, 6),
                        "max_edge_before_bytes": round(comm_before["max_edge_port_bytes"], 4),
                        "max_edge_after_bytes": round(comm_after["max_edge_port_bytes"], 4),
                        "max_phase_service_before_ns": round(comm_before["max_phase_service_ns"], 4),
                        "max_phase_service_after_ns": round(comm_after["max_phase_service_ns"], 4),
                    })
                    migration_bytes += 2 * expert_weight_bytes
                    adjustments_done += 1
                    accept("swap")
                    current_objective = after
                    current_compute_us = compute_after
                    current_comm = comm_after
                    current_aggregate_latency_us = candidate_evaluation[
                        "aggregate_estimated_total_us"
                    ]
                    current_step_latency = candidate_evaluation["step_latency"]
                    die_load = candidate_evaluation["route_load"]

        # Enforce mapping invariants after the bounded local updates.
        for e in list(mapper.keys()):
            uniq = []
            seen: set = set()
            for d in mapper[e]:
                if d not in seen:
                    uniq.append(d)
                    seen.add(d)
            if not uniq:
                raise RuntimeError(f"runtime mapping removed the final copy of expert {e}")
            if len(uniq) > replica_cap.get(e, max_replicas_per_expert):
                raise RuntimeError(f"runtime mapping exceeded replica cap for expert {e}")
            mapper[e] = uniq
        mapper = _order_replicas_for_routing(mapper, topo)

        migration_events: List[Dict] = []
        migration_total_us = 0.0
        cross_wafer_migration_us = 0.0
        intra_wafer_migration_us = 0.0

        final_evaluation = evaluate(mapper)
        objective_after = _objective_from_parts(
            final_evaluation["objective_parts"],
            alpha=alpha,
            gamma=gamma,
            delta=delta,
            critical_load_weight=critical_load_weight,
            normalization_refs=objective_refs,
        )
        compute_after_window = final_evaluation["compute_us"]
        comm_after_window = final_evaluation["comm"]
        die_load_after = final_evaluation["route_load"]
        guard_reverted = False
        if (
            objective_after > objective_before + 1e-9
            and not base_rollback_accepted
        ):
            raise RuntimeError("accepted runtime operations increased the joint objective")
        if (
            current_aggregate_latency_us > aggregate_latency_before_window + 1e-9
            and not base_rollback_accepted
        ):
            raise RuntimeError("accepted runtime operations increased aggregate estimated latency")
        if (
            float(current_step_latency["estimated_total_us"])
            > float(step_latency_before_window["estimated_total_us"]) + 1e-9
        ):
            raise RuntimeError("accepted runtime operations increased step-level estimated latency")
        if migration_bytes > migration_budget_bytes_per_window:
            raise RuntimeError("accepted runtime operations exceeded migration budget")
        migration_events = build_migration_events(
            replicas_added=replicas_added,
            swaps=swaps_done,
            expert_weight_bytes=expert_weight_bytes,
            topo=topo,
            observed_window=w_id,
        )
        migration_total_us = sum(
            float(event["estimated_latency_us"]) for event in migration_events
        )
        for event in migration_events:
            src_w, _ = topo.die_to_wafer_local(int(event["source_die"]))
            dst_w, _ = topo.die_to_wafer_local(int(event["destination_die"]))
            if src_w == dst_w:
                intra_wafer_migration_us += float(event["estimated_latency_us"])
            else:
                cross_wafer_migration_us += float(event["estimated_latency_us"])
        activation_window = _migration_activation_window(
            w_id,
            migration_us=migration_total_us,
            slack_us_per_window=attention_migration_slack_us_per_window,
        )
        planned_next_mapping = copy.deepcopy(mapper)
        if migration_events and activation_window > w_id + 1:
            pending_mapping = copy.deepcopy(planned_next_mapping)
            pending_activation_window = activation_window
            pending_source_window = w_id
            mapper = mapper_before_update
        windows_out.append({
            "window_id": ws["window_id"],
            "mapping": {str(e): list(ds) for e, ds in mapper_for_window.items()},
            "next_mapping": {str(e): list(ds) for e, ds in mapper.items()},
            "planned_next_mapping": {
                str(e): list(ds) for e, ds in planned_next_mapping.items()
            },
            "replicas_added": replicas_added,
            "replicas_removed": replicas_removed,
            "swaps": swaps_done,
            "base_rollback": base_rollback,
            "migration_bytes": migration_bytes,
            "migration_events": migration_events,
            "migration_total_us": round(migration_total_us, 4),
            "cross_wafer_migration_us": round(cross_wafer_migration_us, 4),
            "intra_wafer_migration_us": round(intra_wafer_migration_us, 4),
            "attention_migration_slack_us_per_window": attention_migration_slack_us_per_window,
            "planned_activation_window": activation_window,
            "migration_hidden_by_attention": activation_window == w_id + 1,
            "activated_pending_update": activated_pending_update,
            "update_blocked_by_pending_migration": update_blocked_by_pending_migration,
            "pending_update_source_window": pending_source_window,
            "pending_update_activation_window": pending_activation_window,
            "objective_before": round(objective_before, 4),
            "objective_after": round(objective_after, 4),
            "objective_normalization_refs": objective_refs,
            "predicted_compute_before_us": round(compute_before_window, 6),
            "predicted_compute_after_us": round(compute_after_window, 6),
            "predicted_aggregate_total_before_us": round(
                aggregate_latency_before_window, 6
            ),
            "predicted_aggregate_total_after_us": round(
                current_aggregate_latency_us, 6
            ),
            "predicted_step_total_before_us": round(
                float(step_latency_before_window["estimated_total_us"]), 6
            ),
            "predicted_step_total_after_us": round(
                float(current_step_latency["estimated_total_us"]), 6
            ),
            "predicted_step_compute_before_us": round(
                float(step_latency_before_window["compute_us"]), 6
            ),
            "predicted_step_compute_after_us": round(
                float(current_step_latency["compute_us"]), 6
            ),
            "predicted_max_edge_before_bytes": round(comm_before_window["max_edge_port_bytes"], 4),
            "predicted_max_edge_after_bytes": round(comm_after_window["max_edge_port_bytes"], 4),
            "predicted_cross_wafer_before_bytes": round(comm_before_window["cross_wafer_bytes"], 4),
            "predicted_cross_wafer_after_bytes": round(comm_after_window["cross_wafer_bytes"], 4),
            "predicted_max_phase_service_before_ns": round(comm_before_window["max_phase_service_ns"], 4),
            "predicted_max_phase_service_after_ns": round(comm_after_window["max_phase_service_ns"], 4),
            "predicted_combine_link_service_before_ns": round(comm_before_window["combine_max_link_service_ns"], 4),
            "predicted_combine_link_service_after_ns": round(comm_after_window["combine_max_link_service_ns"], 4),
            "mapping_semantics": "mapping applies to this observed window; next_mapping applies after scheduled runtime updates",
            "routing_policy": "hacmo_cost_aware",
            "route_communication_weight": route_communication_weight,
            "route_compute_load_weight": route_compute_load_weight,
            "route_estimation_granularity": (
                "aggregate_prefilter_plus_exact_decode_steps"
                if window_steps
                else "source_die_expert_aggregate_fallback"
            ),
            "route_estimation_decisions_per_source_expert": (
                route_estimation_decisions_per_source_expert
            ),
            "max_candidate_dies_per_expert": max_candidate_dies_per_expert,
            "max_candidate_operations_per_window": max_candidate_operations_per_window,
            "candidate_operation_evaluation_limits": operation_evaluation_limits,
            "candidate_search": candidate_search,
            "hotness_threshold": round(h_thr, 4),
            "guard_reverted": guard_reverted,
            "top_hot_experts": [
                {"expert": int(e), "hotness": round(float(v), 4), "replicas": len(mapper[e])}
                for e, v in sorted(hotness.items(), key=lambda x: x[1], reverse=True)[:16]
            ],
            "estimated_link_pressure": {"approx_max_die_load": round(max(die_load_after.values()) if die_load_after else 0.0, 4)},
        })
    return windows_out
def main() -> None:
    p = argparse.ArgumentParser(description="Runtime scheduler for ours_dynamic.")
    p.add_argument("--network-config", required=True)
    p.add_argument("--static-mapping-json", required=True)
    p.add_argument("--window-stats-json", required=True)
    p.add_argument("--window-src-expert-json", default=None)
    p.add_argument(
        "--step-inputs-json",
        required=True,
        help="Sampled per-decode-step inputs used by the runtime latency guard.",
    )
    p.add_argument("--ewma-decay", type=float, default=0.8)
    p.add_argument("--hot-percentile", type=float, default=90.0)
    p.add_argument("--replica-persistence-windows", type=int, default=2)
    p.add_argument("--migration-budget-bytes-per-window", type=int, default=500_000_000,
                    help="Migration budget per window in bytes. Default 500MB allows "
                         "migrating ~11 expert replicas (44MB each in FP8).")
    p.add_argument("--bytes-per-token", type=int, default=14336)
    p.add_argument("--max-replicas-per-expert", type=int, default=4)
    p.add_argument("--alpha", type=float, default=1.0, help="Communication cost weight in objective.")
    p.add_argument("--gamma", type=float, default=0.5, help="Normalized congestion penalty weight in objective.")
    p.add_argument("--delta", type=float, default=2.0, help="Normalized load imbalance penalty weight in objective.")
    p.add_argument("--lambda", dest="lambda_", type=float, default=1.0,
                    help="Congestion sensitivity in path cost model.")
    p.add_argument("--critical-load-weight", type=float, default=0.0,
                   help="Penalty weight for the maximum routed rank token load.")
    p.add_argument("--runtime-compute-batch-scale", type=float, default=1.0,
                    help="Batch scaling factor used by the route-aware H200 compute guard.")
    p.add_argument("--dispatch-bytes-per-token", type=float, default=7168.0)
    p.add_argument("--route-communication-weight", type=float, default=1.0)
    p.add_argument("--route-compute-load-weight", type=float, default=1.0)
    p.add_argument("--route-estimation-decisions-per-source-expert", type=int, default=1)
    p.add_argument("--hidden-size", type=int, default=7168)
    p.add_argument("--expert-intermediate-size", type=int, default=2048)
    p.add_argument("--h200-effective-tflops", type=float, default=1979.0)
    p.add_argument("--h200-hbm-bw-tbps", type=float, default=4.8)
    p.add_argument("--weight-bytes-per-elem", type=int, default=1)
    p.add_argument("--activation-bytes-per-elem", type=int, default=2)
    p.add_argument("--max-adjustments-per-window", type=int, default=16,
                   help="Bound runtime replica/evict/swap operations per observed window.")
    p.add_argument("--max-hot-experts-per-window", type=int, default=8,
                   help="Evaluate at most this many hottest experts for runtime replica additions per window.")
    p.add_argument("--max-candidate-dies-per-expert", type=int, default=3,
                   help="Prefilter this many light, low-path-cost dies before full cost-aware evaluation.")
    p.add_argument("--max-candidate-operations-per-window", type=int, default=12,
                   help="Bound full cost-aware mapping evaluations per runtime window.")
    p.add_argument("--attention-migration-slack-us-per-window", type=float, default=1000.0,
                   help="Communication slack available to hide expert-weight migration after each mapping window.")
    p.add_argument("--output-json", required=True)
    args = p.parse_args()

    topo = load_topology(args.network_config)
    static_mapping_raw = json.loads(Path(args.static_mapping_json).read_text(encoding="utf-8"))["expert_to_dies"]
    static_mapping = {int(k): [int(x) for x in v] for k, v in static_mapping_raw.items()}
    window_stats = json.loads(Path(args.window_stats_json).read_text(encoding="utf-8"))
    window_src_expert_counts = None
    if args.window_src_expert_json:
        window_src_expert_counts = json.loads(Path(args.window_src_expert_json).read_text(encoding="utf-8"))
    step_inputs_raw = json.loads(Path(args.step_inputs_json).read_text(encoding="utf-8"))
    step_inputs = (
        step_inputs_raw.get("steps", [])
        if isinstance(step_inputs_raw, dict)
        else step_inputs_raw
    )
    out_data = run_runtime(
        static_mapping=static_mapping,
        window_stats=window_stats,
        window_src_expert_counts=window_src_expert_counts,
        topo=topo,
        decay=args.ewma_decay,
        hot_percentile=args.hot_percentile,
        replica_persistence_windows=args.replica_persistence_windows,
        migration_budget_bytes_per_window=args.migration_budget_bytes_per_window,
        bytes_per_token=args.bytes_per_token,
        max_replicas_per_expert=args.max_replicas_per_expert,
        step_inputs=step_inputs,
        alpha=args.alpha,
        gamma=args.gamma,
        delta=args.delta,
        lambda_=args.lambda_,
        critical_load_weight=args.critical_load_weight,
        runtime_compute_batch_scale=args.runtime_compute_batch_scale,
        dispatch_bytes_per_token=args.dispatch_bytes_per_token,
        route_communication_weight=args.route_communication_weight,
        route_compute_load_weight=args.route_compute_load_weight,
        route_estimation_decisions_per_source_expert=(
            args.route_estimation_decisions_per_source_expert
        ),
        roofline_cfg=H200RooflineConfig(
            hidden_size=args.hidden_size,
            expert_intermediate_size=args.expert_intermediate_size,
            effective_tflops=args.h200_effective_tflops,
            hbm_bw_tbps=args.h200_hbm_bw_tbps,
            weight_bytes_per_elem=args.weight_bytes_per_elem,
            activation_bytes_per_elem=args.activation_bytes_per_elem,
        ),
        max_adjustments_per_window=args.max_adjustments_per_window,
        max_hot_experts_per_window=args.max_hot_experts_per_window,
        max_candidate_dies_per_expert=args.max_candidate_dies_per_expert,
        max_candidate_operations_per_window=args.max_candidate_operations_per_window,
        attention_migration_slack_us_per_window=args.attention_migration_slack_us_per_window,
    )
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(out_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote runtime windows to: {out}")


if __name__ == "__main__":
    main()
