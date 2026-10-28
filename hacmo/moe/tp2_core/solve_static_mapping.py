#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

try:
    from .utils_topology import load_topology
except ImportError:
    from utils_topology import load_topology


def normalize_objective_parts(
    parts: Dict[str, float],
    references: Dict[str, float],
) -> Dict[str, float]:
    return {
        name: float(parts[name]) / max(abs(float(references[name])), 1e-12)
        for name in ("comm", "cong", "load")
    }


def validate_mapping_capacities(
    *,
    mapping: Dict[int, List[int]],
    h: Dict[str, float],
    dies: List[int],
    expert_weight_bytes: float,
    die_memory_capacity_bytes: float,
    die_compute_capacity: float,
    routed_die_loads: Dict[int, float] | None = None,
) -> Dict:
    per_die = {
        int(die): {"memory_bytes": 0.0, "compute_requirement": 0.0, "expert_copies": 0}
        for die in dies
    }
    for expert, placed_dies in mapping.items():
        for die in placed_dies:
            if die not in per_die:
                raise ValueError(f"mapping references unknown die {die}")
            per_die[die]["memory_bytes"] += float(expert_weight_bytes)
            per_die[die]["expert_copies"] += 1

    if routed_die_loads is not None:
        total_load = sum(max(float(value), 0.0) for value in routed_die_loads.values()) or 1.0
        for die, value in routed_die_loads.items():
            if die not in per_die:
                raise ValueError(f"routed load references unknown die {die}")
            per_die[die]["compute_requirement"] += max(float(value), 0.0) / total_load
    else:
        total_h = sum(max(float(value), 0.0) for value in h.values()) or 1.0
        for expert, placed_dies in mapping.items():
            if not placed_dies:
                continue
            per_copy = (
                max(float(h.get(str(expert), 0.0)), 0.0)
                / total_h
                / len(placed_dies)
            )
            for die in placed_dies:
                per_die[die]["compute_requirement"] += per_copy

    violations = []
    for die in dies:
        usage = per_die[die]
        if usage["memory_bytes"] > die_memory_capacity_bytes + 1e-9:
            violations.append(
                {
                    "die": die,
                    "resource": "memory",
                    "used": usage["memory_bytes"],
                    "capacity": float(die_memory_capacity_bytes),
                }
            )
        if usage["compute_requirement"] > die_compute_capacity + 1e-12:
            violations.append(
                {
                    "die": die,
                    "resource": "compute",
                    "used": usage["compute_requirement"],
                    "capacity": float(die_compute_capacity),
                }
            )
    return {
        "valid": not violations,
        "expert_weight_bytes": float(expert_weight_bytes),
        "die_memory_capacity_bytes": float(die_memory_capacity_bytes),
        "die_compute_capacity": float(die_compute_capacity),
        "per_die": {str(die): per_die[die] for die in dies},
        "violations": violations,
    }


def _uniform_mapping(num_experts: int, dies: List[int]) -> Dict[int, List[int]]:
    m = {}
    n = len(dies)
    for e in range(num_experts):
        m[e] = [dies[e % n]]
    return m


def _expert_source_volumes(
    expert: int,
    p_ue: Dict[str, Dict[str, float]],
    s_u: Dict[str, int],
) -> Dict[int, float]:
    out: Dict[int, float] = {}
    key = str(expert)
    for su_str, e_probs in p_ue.items():
        prob = float(e_probs.get(key, 0.0))
        if prob <= 0:
            continue
        vol = prob * float(s_u.get(str(su_str), 0))
        if vol > 0:
            out[int(su_str)] = vol
    return out


def _route_load_for_mapping(
    mapping: Dict[int, List[int]],
    topo,
    p_ue: Dict[str, Dict[str, float]],
    s_u: Dict[str, int],
) -> Dict[int, float]:
    load: Dict[int, float] = defaultdict(float)
    for su_str, e_probs in p_ue.items():
        su = int(su_str)
        tokens = float(s_u.get(str(su), 0))
        if tokens <= 0:
            continue
        for e_str, prob in e_probs.items():
            replicas = mapping.get(int(e_str), [])
            if not replicas:
                continue
            vol = float(prob) * tokens
            if vol <= 0:
                continue
            dst = min(replicas, key=lambda d: _path_latency_ns(topo, su, d))
            load[dst] += vol
    return load


def _path_latency_ns(topo, src: int, dst: int) -> float:
    if src == dst:
        return 0.0
    total = 0.0
    for edge in topo.get_path(src, dst):
        if edge[0] == "mesh":
            total += topo.cfg.mesh_latency_ns
        else:
            total += topo.cfg.switch_latency_ns
    return total


def _comm_cost_to_die(
    die: int,
    p_ue: Dict[str, Dict[str, float]],
    s_u: Dict[str, int],
    expert: int,
    topo,
    lambda_: float,
    link_loads: Dict | None = None,
) -> float:
    """Compute communication cost from all sources to *die* for *expert*.

    If *link_loads* is provided, it reflects the cumulative traffic already
    placed on each link by previously-mapped experts.  This makes the
    congestion penalty (lambda_) meaningful: a link already under load
    receives a higher marginal cost, discouraging new experts from
    routing through congested paths.

    When *link_loads* is ``None`` (legacy mode), a fresh empty dict is used
    each call, so only the current expert's own traffic contributes to
    congestion -- equivalent to lambda_ having no cross-expert effect.
    """
    if link_loads is None:
        link_loads = {}
    c = 0.0
    for su_str, e_probs in p_ue.items():
        su = int(su_str)
        prob = float(e_probs.get(str(expert), 0.0))
        if prob <= 0:
            continue
        vol = prob * float(s_u.get(str(su), 0))
        c += vol * topo.get_path_cost(su, die, link_loads, lambda_)
    return c


def _path_cost_estimate(topo, src: int, dst: int, link_loads: Dict, lambda_: float) -> float:
    """Estimate per-unit-volume path cost from src to dst.

    Uses the same weighted model as _comm_cost_to_die: switch edges are
    penalised relative to mesh edges via z_l, and congestion adds
    lambda_ * (load/bw) per edge.
    """
    cost = 0.0
    for edge in topo.get_path(src, dst):
        kind = edge[0]
        bw = topo.get_link_bandwidth(edge)
        load = link_loads.get(edge, 0.0)
        if kind == "mesh":
            z_l = 1.0
        else:
            z_l = (topo.cfg.mesh_bw_gbps / max(topo.cfg.switch_bw_gbps, 1e-9)
                   + topo.cfg.switch_latency_ns / max(topo.cfg.mesh_latency_ns, 1e-9))
        cost += z_l + lambda_ * (load / max(bw, 1e-6))
    return cost


def _path_cost_from_edges(topo, edges: List[Tuple[str, int, int]], link_loads: Dict, lambda_: float) -> float:
    cost = 0.0
    for edge in edges:
        kind = edge[0]
        bw = topo.get_link_bandwidth(edge)
        load = link_loads.get(edge, 0.0)
        if kind == "mesh":
            z_l = 1.0
        else:
            z_l = (topo.cfg.mesh_bw_gbps / max(topo.cfg.switch_bw_gbps, 1e-9)
                   + topo.cfg.switch_latency_ns / max(topo.cfg.mesh_latency_ns, 1e-9))
        cost += z_l + lambda_ * (load / max(bw, 1e-6))
    return cost


def _source_agnostic_comm_cost_to_die(
    die: int,
    expert_weight: float,
    dies: List[int],
    topo,
    lambda_: float,
    link_loads: Dict,
    path_cache: Dict[Tuple[int, int], List[Tuple[str, int, int]]],
) -> float:
    return expert_weight * sum(
        _path_cost_from_edges(topo, path_cache[(src, die)], link_loads, lambda_)
        for src in dies
    )


def _source_agnostic_accumulate_link_loads(
    link_loads: Dict,
    dst_die: int,
    expert_weight: float,
    dies: List[int],
    path_cache: Dict[Tuple[int, int], List[Tuple[str, int, int]]],
) -> None:
    if expert_weight <= 0:
        return
    for src in dies:
        for edge in path_cache[(src, dst_die)]:
            link_loads[edge] = link_loads.get(edge, 0.0) + expert_weight


def _source_agnostic_project_link_loads(
    link_loads: Dict,
    dst_die: int,
    expert_weight: float,
    dies: List[int],
    path_cache: Dict[Tuple[int, int], List[Tuple[str, int, int]]],
) -> Dict:
    projected = dict(link_loads)
    _source_agnostic_accumulate_link_loads(projected, dst_die, expert_weight, dies, path_cache)
    return projected


def _routing_cost_and_link_loads(
    mapping: Dict[int, List[int]],
    p_ue: Dict[str, Dict[str, float]],
    s_u: Dict[str, int],
    topo,
    lambda_: float,
) -> Tuple[float, Dict, Dict[int, float]]:
    """Route all expert traffic to nearest/current-lowest-cost replicas."""
    link_loads: Dict = {}
    die_loads: Dict[int, float] = defaultdict(float)
    comm = 0.0
    for su_str, e_probs in p_ue.items():
        su = int(su_str)
        tokens = float(s_u.get(str(su), 0))
        if tokens <= 0:
            continue
        for e_str, prob in e_probs.items():
            replicas = mapping.get(int(e_str), [])
            if not replicas:
                continue
            vol = float(prob) * tokens
            if vol <= 0:
                continue
            dst = min(replicas, key=lambda d: _path_cost_estimate(topo, su, d, link_loads, lambda_))
            comm += vol * _path_cost_estimate(topo, su, dst, link_loads, lambda_)
            die_loads[dst] += vol
            for edge in topo.get_path(su, dst):
                link_loads[edge] = link_loads.get(edge, 0.0) + vol
    return comm, link_loads, die_loads


def _source_agnostic_routing_cost_and_link_loads(
    mapping: Dict[int, List[int]],
    h: Dict[str, float],
    dies: List[int],
    topo,
    lambda_: float,
    path_cache: Dict[Tuple[int, int], List[Tuple[str, int, int]]],
) -> Tuple[float, Dict, Dict[int, float]]:
    link_loads: Dict = {}
    die_loads: Dict[int, float] = defaultdict(float)
    comm = 0.0
    source_normalizer = max(len(dies), 1)
    for e, replicas in mapping.items():
        if not replicas:
            continue
        vol = float(h.get(str(e), 0.0))
        if vol <= 0:
            continue
        for src in dies:
            dst = min(
                replicas,
                key=lambda d: _path_cost_from_edges(topo, path_cache[(src, d)], link_loads, lambda_),
            )
            comm += vol * _path_cost_from_edges(topo, path_cache[(src, dst)], link_loads, lambda_)
            die_loads[dst] += vol / source_normalizer
            for edge in path_cache[(src, dst)]:
                link_loads[edge] = link_loads.get(edge, 0.0) + vol
    return comm, link_loads, die_loads


def _source_agnostic_replica_projection(
    mapping: Dict[int, List[int]],
    expert: int,
    candidate_die: int,
    link_loads: Dict,
    h: Dict[str, float],
    dies: List[int],
    topo,
    lambda_: float,
    path_cache: Dict[Tuple[int, int], List[Tuple[str, int, int]]],
) -> Tuple[float, float, Dict]:
    """Project one source-agnostic replica insertion by rerouting one expert."""
    vol = float(h.get(str(expert), 0.0))
    projected = dict(link_loads)
    if vol <= 0:
        return 0.0, 0.0, projected

    old_replicas = mapping[expert]
    new_replicas = old_replicas + [candidate_die]
    before_comm = 0.0
    after_comm = 0.0

    old_routes: List[Tuple[int, List[Tuple[str, int, int]]]] = []
    for src in dies:
        old_dst = min(
            old_replicas,
            key=lambda d: _path_cost_from_edges(topo, path_cache[(src, d)], link_loads, lambda_),
        )
        old_edges = path_cache[(src, old_dst)]
        before_comm += vol * _path_cost_from_edges(topo, old_edges, link_loads, lambda_)
        old_routes.append((src, old_edges))

    for _, old_edges in old_routes:
        for edge in old_edges:
            projected[edge] = projected.get(edge, 0.0) - vol
            if projected[edge] <= 1e-15:
                projected.pop(edge, None)

    for src, _ in old_routes:
        new_dst = min(
            new_replicas,
            key=lambda d: _path_cost_from_edges(topo, path_cache[(src, d)], projected, lambda_),
        )
        new_edges = path_cache[(src, new_dst)]
        after_comm += vol * _path_cost_from_edges(topo, new_edges, projected, lambda_)
        for edge in new_edges:
            projected[edge] = projected.get(edge, 0.0) + vol

    return before_comm, after_comm, projected


def _mapping_objective(
    mapping: Dict[int, List[int]],
    p_ue: Dict[str, Dict[str, float]],
    s_u: Dict[str, int],
    h: Dict[str, float],
    dies: List[int],
    topo,
    alpha: float,
    gamma: float,
    delta: float,
    lambda_: float,
    source_model: str,
    path_cache: Dict[Tuple[int, int], List[Tuple[str, int, int]]] | None = None,
    normalization_refs: Dict[str, float] | None = None,
) -> Tuple[float, Dict]:
    """Evaluate the full Eq.(6) placement objective for a complete mapping."""
    if source_model == "source_agnostic":
        if path_cache is None:
            path_cache = {
                (src, dst): topo.get_path(src, dst)
                for src in dies
                for dst in dies
            }
        comm, link_loads, die_loads = _source_agnostic_routing_cost_and_link_loads(
            mapping, h, dies, topo, lambda_, path_cache
        )
    else:
        comm, link_loads, die_loads = _routing_cost_and_link_loads(
            mapping, p_ue, s_u, topo, lambda_
        )
    omega = topo.get_outer_to_inner_weights()
    cong = _congestion_penalty(link_loads, topo)
    load = _load_imbalance_penalty(die_loads, omega, dies)
    raw_parts = {"comm": comm, "cong": cong, "load": load}
    normalized = (
        normalize_objective_parts(raw_parts, normalization_refs)
        if normalization_refs is not None
        else raw_parts
    )
    total = (
        alpha * normalized["comm"]
        + gamma * normalized["cong"]
        + delta * normalized["load"]
    )
    return total, {
        "comm": comm,
        "cong": cong,
        "load": load,
        "normalized": normalized,
        "normalization_refs": normalization_refs,
        "link_loads": link_loads,
        "die_load": dict(die_loads),
    }


def _primary_die(mapping: Dict[int, List[int]], expert: int) -> int:
    return mapping[expert][0]


def _order_replicas_for_routing(mapping: Dict[int, List[int]], topo) -> Dict[int, List[int]]:
    """Order equal-latency replicas by physical egress quality.

    Downstream trace generation chooses the first replica when multiple dies
    have the same zero-load route latency.  Keep that tie-break topology-aware:
    higher-bandwidth rail ports should receive remote traffic before ordinary
    edge ports, otherwise hot experts can be placed on valid replicas but routed
    through a weaker 1x rail endpoint.
    """
    ordered: Dict[int, List[int]] = {}
    for e, ds in mapping.items():
        uniq = []
        seen: Set[int] = set()
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


def _local_refine_primary_mapping(
    mapping: Dict[int, List[int]],
    p_ue: Dict[str, Dict[str, float]],
    s_u: Dict[str, int],
    h: Dict[str, float],
    dies: List[int],
    topo,
    alpha: float,
    gamma: float,
    delta: float,
    lambda_: float,
    source_model: str,
    path_cache: Dict[Tuple[int, int], List[Tuple[str, int, int]]] | None,
    normalization_refs: Dict[str, float],
    topk_hot: int,
    max_passes: int = 16,
) -> Tuple[Dict[int, List[int]], Dict]:
    """Apply bounded swap/move refinements accepted only by Eq.(6).

    The paper describes a lightweight solver that starts from a heuristic
    placement and then applies local refinements.  This routine keeps the
    search bounded: hot experts are considered first, each expert only tests a
    small local candidate set, and every accepted operation must lower the full
    Phi_place objective.
    """
    current_obj, current_parts = _mapping_objective(
        mapping, p_ue, s_u, h, dies, topo, alpha, gamma, delta, lambda_, source_model,
        path_cache, normalization_refs
    )
    refinement_log = {
        "objective_initial": current_obj,
        "objective_final": current_obj,
        "accepted_swaps": 0,
        "accepted_moves": 0,
    }
    max_hot = min(len(mapping), max(1, topk_hot if topk_hot > 0 else 32))
    local_candidate_limit = 8
    experts_by_hotness = sorted(mapping, key=lambda e: float(h.get(str(e), 0.0)), reverse=True)

    for _ in range(max_passes):
        improved = False
        primary_used = {_primary_die(mapping, e) for e in mapping}
        for e in experts_by_hotness[:max_hot]:
            if e not in mapping or len(mapping[e]) != 1:
                continue
            d_e = _primary_die(mapping, e)
            w_e, _ = topo.die_to_wafer_local(d_e)

            # Move into an empty die on the same wafer when the problem has
            # slack capacity.  Full 256-expert/256-die TP2 runs usually have
            # no empty primary slots, but this keeps the solver faithful for
            # smaller or future settings.
            wafer_dies = [d for d in dies if topo.die_to_wafer_local(d)[0] == w_e]
            move_candidates = sorted(
                [d for d in wafer_dies if d not in primary_used],
                key=lambda d: (
                    topo.get_edge_distance(d),
                    _path_cost_estimate(topo, d_e, d, current_parts["link_loads"], lambda_),
                ),
            )[:local_candidate_limit]
            for cand in move_candidates:
                candidate = {k: list(v) for k, v in mapping.items()}
                candidate[e] = [cand]
                obj, parts = _mapping_objective(
                    candidate, p_ue, s_u, h, dies, topo, alpha, gamma, delta, lambda_,
                    source_model, path_cache, normalization_refs
                )
                if obj < current_obj - 1e-12:
                    mapping = candidate
                    current_obj = obj
                    current_parts = parts
                    refinement_log["accepted_moves"] += 1
                    improved = True
                    break
            if improved:
                break

            # Swap with a small same-wafer candidate set.  This preserves the
            # Phase-1 wafer assignment while letting Eq.(6) correct edge/inner
            # die choices and local congestion hot spots.
            same_wafer_experts = [
                e2 for e2 in mapping
                if e2 != e
                and len(mapping[e2]) == 1
                and topo.die_to_wafer_local(_primary_die(mapping, e2))[0] == w_e
            ]
            same_wafer_experts = sorted(
                same_wafer_experts,
                key=lambda e2: (
                    abs(topo.get_edge_distance(_primary_die(mapping, e2)) - topo.get_edge_distance(d_e)),
                    -abs(float(h.get(str(e), 0.0)) - float(h.get(str(e2), 0.0))),
                ),
                reverse=True,
            )[:local_candidate_limit]
            for e2 in same_wafer_experts:
                d2 = _primary_die(mapping, e2)
                candidate = {k: list(v) for k, v in mapping.items()}
                candidate[e] = [d2]
                candidate[e2] = [d_e]
                obj, parts = _mapping_objective(
                    candidate, p_ue, s_u, h, dies, topo, alpha, gamma, delta, lambda_,
                    source_model, path_cache, normalization_refs
                )
                if obj < current_obj - 1e-12:
                    mapping = candidate
                    current_obj = obj
                    current_parts = parts
                    refinement_log["accepted_swaps"] += 1
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    refinement_log["objective_final"] = current_obj
    refinement_log.update({
        "comm_final": current_parts["comm"],
        "cong_final": current_parts["cong"],
        "load_final": current_parts["load"],
        "normalized_final": current_parts["normalized"],
        "max_passes": max_passes,
        "hot_experts_considered": max_hot,
        "candidate_limit_per_expert": local_candidate_limit,
    })
    return mapping, refinement_log


def _ours_static_mapping(
    num_experts: int,
    dies: List[int],
    p_ue: Dict[str, Dict[str, float]],
    h: Dict[str, float],
    s_u: Dict[str, int],
    topo,
    alpha: float,
    gamma: float,
    delta: float,
    lambda_: float,
    topk_hot: int,
    max_replicas: int = 1,
    source_model: str = "source_agnostic",
    return_stats: bool = False,
    expert_weight_bytes: float = 3 * 7168 * 2048,
    die_memory_capacity_bytes: float = 141 * 1024**3,
    die_compute_capacity: float = 1.0,
    refinement_max_passes: int = 16,
) -> Dict[int, List[int]] | Tuple[Dict[int, List[int]], Dict]:
    """Three-phase solver implementing the paper's Eq.(6) objective:

    Phi_place(X) = alpha * Phi_comm(X) + gamma * Phi_cong(X) + delta * Phi_load(X)

    Phase 1: Demand-aware wafer assignment minimising cross-wafer traffic.
    Phase 2: Communication-cost-aware die placement within each wafer,
              with cumulative link-load tracking so gamma*Phi_cong properly
              penalises congested links across all already-placed experts.
    Phase 3: Cross-wafer replication of hot experts.  For the top-k hottest
              experts with significant demand on multiple wafers, place
              replicas on edge dies of those demand-heavy wafers.  Each
              replica reduces cross-wafer dispatch traffic and distributes
              compute, consistent with the paper's static + dynamic approach.
    """

    num_wafers = topo.cfg.num_wafers
    dpw = topo.cfg.dies_per_wafer
    omega = topo.get_outer_to_inner_weights()

    if source_model not in {"source_agnostic", "projected"}:
        raise ValueError(f"unknown placement source model: {source_model}")

    # The real H200 trace provides expert activation, but its rank IDs do not
    # define source dies in the target 256-die system.  For paper-facing static
    # placement, use a source-agnostic expected traffic model: every target die
    # may source tokens with the profiled expert hotness.  The old projected
    # source model is kept only for diagnostics where request-to-die placement
    # is intentionally fixed before expert placement.
    path_cache = {
        (src, dst): topo.get_path(src, dst)
        for src in dies
        for dst in dies
    } if source_model == "source_agnostic" else {}

    if source_model == "projected":
        placement_s_u = s_u
        placement_p_ue = p_ue
    else:
        placement_s_u = {}
        placement_p_ue = {}

    reference_mapping = _uniform_mapping(num_experts, dies)
    _, reference_parts = _mapping_objective(
        reference_mapping,
        placement_p_ue,
        placement_s_u,
        h,
        dies,
        topo,
        alpha,
        gamma,
        delta,
        lambda_,
        source_model,
        path_cache if source_model == "source_agnostic" else None,
    )
    normalization_refs = {
        name: max(abs(float(reference_parts[name])), 1e-12)
        for name in ("comm", "cong", "load")
    }

    # --- Phase 1: Wafer assignment ---
    wafer_demand: Dict[int, Dict[int, float]] = {e: defaultdict(float) for e in range(num_experts)}
    total_demand: Dict[int, float] = defaultdict(float)
    if source_model == "source_agnostic":
        for e in range(num_experts):
            weight = float(h.get(str(e), 0.0))
            total_demand[e] = weight * len(dies)
            for w in range(num_wafers):
                wafer_demand[e][w] = weight * dpw
    else:
        for su_str, e_probs in placement_p_ue.items():
            su = int(su_str)
            tokens = float(placement_s_u.get(str(su), 0))
            if tokens <= 0:
                continue
            w = su // dpw
            for e_str, prob in e_probs.items():
                pf = float(prob)
                if pf <= 0:
                    continue
                e = int(e_str)
                vol = pf * tokens
                wafer_demand[e][w] += vol
                total_demand[e] += vol

    def _best_wafer(e: int) -> List[Tuple[int, float]]:
        return sorted(wafer_demand[e].items(), key=lambda x: x[1], reverse=True)

    wafer_cap: Dict[int, int] = {w: dpw for w in range(num_wafers)}
    expert_wafer: Dict[int, int] = {}
    if source_model == "source_agnostic":
        sorted_experts = sorted(
            range(num_experts),
            key=lambda e: (float(h.get(str(e), 0.0)), total_demand[e]),
            reverse=True,
        )
        # Without a known source-wafer distribution, wafer assignment should
        # not invent locality.  Spread hot experts across wafers first, then
        # fill remaining capacity, so every wafer receives a balanced hot/cold
        # mix for its intra-wafer edge placement.
        next_w = 0
        for e in sorted_experts:
            for off in range(num_wafers):
                w = (next_w + off) % num_wafers
                if wafer_cap[w] > 0:
                    expert_wafer[e] = w
                    wafer_cap[w] -= 1
                    next_w = (w + 1) % num_wafers
                    break
    else:
        # Priority: experts whose best wafer has highest absolute demand first
        # (saves the most cross-wafer traffic), tie-break by total volume.
        sorted_experts = sorted(
            range(num_experts),
            key=lambda e: (max(wafer_demand[e].values()) if wafer_demand[e] else 0, total_demand[e]),
            reverse=True,
        )
        for e in sorted_experts:
            placed = False
            for w, _ in _best_wafer(e):
                if wafer_cap[w] > 0:
                    expert_wafer[e] = w
                    wafer_cap[w] -= 1
                    placed = True
                    break
            if not placed:
                for w in range(num_wafers):
                    if wafer_cap[w] > 0:
                        expert_wafer[e] = w
                        wafer_cap[w] -= 1
                        break

        _improve_wafer_assignment(expert_wafer, wafer_demand)

    # --- Phase 2: Die placement within each wafer ---
    # Accumulate link loads across *all* experts so the congestion penalty
    # (gamma * Phi_cong) is aware of traffic already routed through links.
    link_loads: Dict = {}
    # Also track per-die compute load for Phi_load.
    die_compute_load: Dict[int, float] = defaultdict(float)

    wafer_experts: Dict[int, List[int]] = defaultdict(list)
    for e, w in expert_wafer.items():
        wafer_experts[w].append(e)

    mapping: Dict[int, List[int]] = {}
    for w in range(num_wafers):
        experts = wafer_experts.get(w, [])
        if not experts:
            continue
        wafer_dies = [w * dpw + i for i in range(dpw)]
        used: Set[int] = set()
        # Place hot/high-traffic experts first.  This is the heuristic
        # initialization described in the paper: hot experts get first choice
        # of edge/low-cost dies before colder experts fill inner slots.
        experts_sorted = sorted(
            experts,
            key=lambda e: (float(h.get(str(e), 0.0)), total_demand[e]),
            reverse=True,
        )
        for e in experts_sorted:
            best_d = None
            best_cost = float("inf")
            candidate_dies = [d for d in wafer_dies if d not in used]
            if source_model == "source_agnostic":
                by_distance: Dict[int, List[int]] = defaultdict(list)
                for d in candidate_dies:
                    by_distance[topo.get_edge_distance(d)].append(d)
                shortlisted: List[int] = []
                for _, ds in sorted(by_distance.items()):
                    shortlisted.extend(sorted(ds)[:4])
                candidate_dies = shortlisted or candidate_dies
            for d in candidate_dies:
                # Phi_comm: communication cost with cumulative congestion
                # penalty from all previously-placed experts.
                if source_model == "source_agnostic":
                    expert_weight = float(h.get(str(e), 0.0))
                    comm_cost = _source_agnostic_comm_cost_to_die(
                        d, expert_weight, dies, topo, lambda_, link_loads, path_cache
                    )
                else:
                    comm_cost = _comm_cost_to_die(d, placement_p_ue, placement_s_u, e, topo, lambda_, link_loads)
                # Phi_cong: marginal placement quality.  This must be
                # evaluated after hypothetically routing the current expert;
                # otherwise gamma multiplies a constant across all candidate
                # dies and cannot affect the die choice.
                if source_model == "source_agnostic":
                    projected_link_loads = _source_agnostic_project_link_loads(
                        link_loads, d, expert_weight, dies, path_cache
                    )
                else:
                    projected_link_loads = _project_link_loads(link_loads, d, placement_p_ue, placement_s_u, e, topo)
                cong_cost = _congestion_penalty(projected_link_loads, topo)
                # Phi_load: projected all-die load-gradient objective from
                # the paper.  omega[d] is a target-load multiplier: edge dies
                # have omega > 1, center dies have omega < 1.
                projected_die_load = dict(die_compute_load)
                projected_die_load[d] = projected_die_load.get(d, 0.0) + float(h.get(str(e), 0.0))
                load_penalty = _load_imbalance_penalty(projected_die_load, omega, dies)
                normalized = normalize_objective_parts(
                    {"comm": comm_cost, "cong": cong_cost, "load": load_penalty},
                    normalization_refs,
                )
                cost = (
                    alpha * normalized["comm"]
                    + gamma * normalized["cong"]
                    + delta * normalized["load"]
                )
                if cost < best_cost:
                    best_cost = cost
                    best_d = d
            if best_d is None:
                best_d = wafer_dies[0]
            mapping[e] = [best_d]
            used.add(best_d)
            # Update link loads with traffic for this expert going to best_d.
            if source_model == "source_agnostic":
                _source_agnostic_accumulate_link_loads(
                    link_loads, best_d, float(h.get(str(e), 0.0)), dies, path_cache
                )
            else:
                _accumulate_link_loads(link_loads, best_d, placement_p_ue, placement_s_u, e, topo)
            die_compute_load[best_d] += float(h.get(str(e), 0.0))

    refinement_log: Dict = {}
    mapping, refinement_log = _local_refine_primary_mapping(
        mapping=mapping,
        p_ue=placement_p_ue,
        s_u=placement_s_u,
        h=h,
        dies=dies,
        topo=topo,
        alpha=alpha,
        gamma=gamma,
        delta=delta,
        lambda_=lambda_,
        source_model=source_model,
        path_cache=path_cache if source_model == "source_agnostic" else None,
        normalization_refs=normalization_refs,
        topk_hot=topk_hot,
        max_passes=refinement_max_passes,
    )
    objective_before_replicas, objective_parts = _mapping_objective(
        mapping, placement_p_ue, placement_s_u, h, dies, topo,
        alpha, gamma, delta, lambda_, source_model,
        path_cache if source_model == "source_agnostic" else None,
        normalization_refs,
    )
    link_loads = objective_parts["link_loads"]
    current_cong_penalty = objective_parts["cong"]
    current_load_penalty = objective_parts["load"]
    current_objective = objective_before_replicas

    # --- Optional Phase 3: Cross-wafer replication of hot experts ---
    # The default static-only policy keeps one copy per expert.  Set
    # --max-replicas > 1 only for ablations that intentionally allow offline
    # replicas; the production dynamic method performs hot-replica adjustment
    # in runtime_scheduler.py.
    if max_replicas > 1:
        # Determine which experts benefit most from replication.
        # Criterion: significant cross-wafer demand (tokens arriving from
        # wafers other than where the expert is already placed).
        cross_wafer_demand: Dict[int, float] = {}
        for e in range(num_experts):
            if e not in mapping:
                continue
            primary_w = expert_wafer.get(e, 0)
            cross_demand = 0.0
            for w, dem in wafer_demand[e].items():
                if w != primary_w:
                    cross_demand += dem
            cross_wafer_demand[e] = cross_demand

        # Select top-k experts by cross-wafer demand for replication.
        hot_for_replica = sorted(
            [e for e in cross_wafer_demand if cross_wafer_demand[e] > 0],
            key=lambda e: cross_wafer_demand[e],
            reverse=True,
        )[:topk_hot]

        # Track die load for load-balancing replicas.  Multiple experts
        # can share a die; we just want to spread replicas across lightly-loaded dies.
        die_load: Dict[int, float] = defaultdict(float)
        for e, ds in mapping.items():
            for d in ds:
                die_load[d] += float(h.get(str(e), 0.0))
        omega = topo.get_outer_to_inner_weights()

        for e in hot_for_replica:
            while len(mapping[e]) < max_replicas:
                covered_wafers = set()
                for d in mapping[e]:
                    w, _ = topo.die_to_wafer_local(d)
                    covered_wafers.add(w)

                candidate_dies: List[int] = []
                already_has = set(mapping[e])
                for w, dem in wafer_demand[e].items():
                    if w in covered_wafers or dem <= 0:
                        continue
                    wafer_dies = [w * dpw + i for i in range(dpw)]
                    edge_dies_on_wafer = [d for d in wafer_dies if topo.get_edge_distance(d) == 0]
                    wafer_candidates = sorted(edge_dies_on_wafer or wafer_dies, key=lambda d: die_load.get(d, 0.0))[:4]
                    for d in wafer_candidates:
                        if d not in already_has:
                            candidate_dies.append(d)
                candidate_dies = sorted(
                    candidate_dies,
                    key=lambda d: (die_load.get(d, 0.0), topo.get_edge_distance(d), d),
                )[:8]
                if not candidate_dies:
                    break

                before_cost = None
                best_dst = None
                best_after_cost = float("inf")
                best_after_parts = None

                for dst in candidate_dies:
                    candidate_mapping = {k: list(v) for k, v in mapping.items()}
                    candidate_mapping[e].append(dst)
                    before_cost = current_objective
                    after_cost, after_parts = _mapping_objective(
                        candidate_mapping,
                        placement_p_ue,
                        placement_s_u,
                        h,
                        dies,
                        topo,
                        alpha,
                        gamma,
                        delta,
                        lambda_,
                        source_model,
                        path_cache if source_model == "source_agnostic" else None,
                        normalization_refs,
                    )
                    capacity = validate_mapping_capacities(
                        mapping=candidate_mapping,
                        h=h,
                        dies=dies,
                        expert_weight_bytes=expert_weight_bytes,
                        die_memory_capacity_bytes=die_memory_capacity_bytes,
                        die_compute_capacity=die_compute_capacity,
                        routed_die_loads=after_parts["die_load"],
                    )
                    if not capacity["valid"]:
                        continue
                    # Tie-break toward lighter dies so equal-cost source-agnostic
                    # replicas do not all pile onto the same edge coordinates.
                    if (
                        after_cost < best_after_cost - 1e-12
                        or (
                            abs(after_cost - best_after_cost) <= 1e-12
                            and best_dst is not None
                            and die_load.get(dst, 0.0) < die_load.get(best_dst, 0.0)
                        )
                    ):
                        best_dst = dst
                        best_after_cost = after_cost
                        best_after_parts = after_parts

                if best_dst is None or best_after_parts is None or before_cost is None:
                    break
                if best_after_cost < before_cost - 1e-12:
                    mapping[e].append(best_dst)
                    current_objective = best_after_cost
                    link_loads = best_after_parts["link_loads"]
                    current_cong_penalty = best_after_parts.get("cong", _congestion_penalty(link_loads, topo))
                    current_load_penalty = best_after_parts.get("load", current_load_penalty)
                    if "die_load" in best_after_parts:
                        die_load = best_after_parts["die_load"]
                    else:
                        die_load[best_dst] += float(h.get(str(e), 0.0))
                else:
                    break

    objective_final, final_parts = _mapping_objective(
        mapping, placement_p_ue, placement_s_u, h, dies, topo,
        alpha, gamma, delta, lambda_, source_model,
        path_cache if source_model == "source_agnostic" else None,
        normalization_refs,
    )
    mapping = _order_replicas_for_routing(mapping, topo)
    capacity_report = validate_mapping_capacities(
        mapping=mapping,
        h=h,
        dies=dies,
        expert_weight_bytes=expert_weight_bytes,
        die_memory_capacity_bytes=die_memory_capacity_bytes,
        die_compute_capacity=die_compute_capacity,
        routed_die_loads=final_parts["die_load"],
    )
    if not capacity_report["valid"]:
        raise ValueError(f"Eq.(9)-(10) capacity validation failed: {capacity_report['violations'][:4]}")
    stats = {
        "phase2_refinement": refinement_log,
        "objective_before_replicas": objective_before_replicas,
        "objective_final": objective_final,
        "phi_comm": final_parts["comm"],
        "phi_cong": final_parts["cong"],
        "phi_load": final_parts["load"],
        "normalized_phi_comm": final_parts["normalized"]["comm"],
        "normalized_phi_cong": final_parts["normalized"]["cong"],
        "normalized_phi_load": final_parts["normalized"]["load"],
        "objective_normalization_refs": normalization_refs,
        "capacity_validation": capacity_report,
        "extra_replicas": sum(max(0, len(v) - 1) for v in mapping.values()),
    }
    if return_stats:
        return mapping, stats
    return mapping


def _improve_wafer_assignment(
    expert_wafer: Dict[int, int],
    wafer_demand: Dict[int, Dict[int, float]],
    max_passes: int = 8,
) -> None:
    """Capacity-preserving local search for lower cross-wafer demand."""
    experts = sorted(expert_wafer)
    eps = 1e-12
    for _ in range(max_passes):
        improved = False
        for i, e1 in enumerate(experts):
            w1 = expert_wafer[e1]
            for e2 in experts[i + 1:]:
                w2 = expert_wafer[e2]
                if w1 == w2:
                    continue
                cur = wafer_demand[e1].get(w1, 0.0) + wafer_demand[e2].get(w2, 0.0)
                swapped = wafer_demand[e1].get(w2, 0.0) + wafer_demand[e2].get(w1, 0.0)
                if swapped > cur + eps:
                    expert_wafer[e1], expert_wafer[e2] = w2, w1
                    w1 = w2
                    improved = True
        if not improved:
            break


def _congestion_penalty(link_loads: Dict, topo) -> float:
    """Compute Phi_cong = sum_l (L_l / B_l)^2 per the paper's Eq.(5).

    The squared penalty provides smooth gradients and heavily penalises
    severe traffic hotspots.
    """
    cong = 0.0
    for edge, load in link_loads.items():
        bw = topo.get_link_bandwidth(edge)
        cong += (load / max(bw, 1e-6)) ** 2
    return cong


def _load_imbalance_penalty(
    die_loads: Dict[int, float],
    omega: Dict[int, float],
    dies: List[int],
) -> float:
    """Compute Phi_load = sum_d (q_d - omega_d * qbar)^2."""
    qbar = sum(die_loads.get(d, 0.0) for d in dies) / max(len(dies), 1)
    penalty = 0.0
    for d in dies:
        target = omega.get(d, 1.0) * qbar
        diff = die_loads.get(d, 0.0) - target
        penalty += diff * diff
    return penalty


def _accumulate_link_loads(
    link_loads: Dict,
    dst_die: int,
    p_ue: Dict[str, Dict[str, float]],
    s_u: Dict[str, int],
    expert: int,
    topo,
) -> None:
    """Add the traffic volume for *expert* routed to *dst_die* into *link_loads*.

    This mutates *link_loads* in-place so subsequent expert placements see
    the accumulated congestion.
    """
    for su_str, e_probs in p_ue.items():
        su = int(su_str)
        prob = float(e_probs.get(str(expert), 0.0))
        if prob <= 0:
            continue
        vol = prob * float(s_u.get(str(su), 0))
        for edge in topo.get_path(su, dst_die):
            link_loads[edge] = link_loads.get(edge, 0.0) + vol


def _project_link_loads(
    link_loads: Dict,
    dst_die: int,
    p_ue: Dict[str, Dict[str, float]],
    s_u: Dict[str, int],
    expert: int,
    topo,
) -> Dict:
    """Return link loads after hypothetically placing *expert* on *dst_die*."""
    projected = dict(link_loads)
    _accumulate_link_loads(projected, dst_die, p_ue, s_u, expert, topo)
    return projected


def _validate_mapping(mapping: Dict[int, List[int]], num_experts: int) -> None:
    if len(mapping) != num_experts:
        raise ValueError("mapping sanity failed: not all experts are placed")
    for e in range(num_experts):
        if e not in mapping:
            raise ValueError(f"expert {e} missing from mapping")
        if not mapping[e]:
            raise ValueError(f"expert {e} has empty die assignment")


def main() -> None:
    p = argparse.ArgumentParser(description="Solve uniform or HACMO static mappings.")
    p.add_argument("--method", choices=["uniform", "ours_static"], required=True)
    p.add_argument("--network-config", required=True)
    p.add_argument("--num-experts", type=int, required=True)
    p.add_argument("--p-ue-json", required=True)
    p.add_argument("--h-json", required=True)
    p.add_argument("--s-u-json", required=True)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--delta", type=float, default=2.0)
    p.add_argument("--lambda", dest="lambda_", type=float, default=1.0)
    p.add_argument("--topk-hot-experts-for-ilp", type=int, default=64)
    p.add_argument("--max-replicas", type=int, default=1,
                   help="Max replicas per expert in static mapping. Default 1 keeps static and dynamic replica policies separate.")
    p.add_argument("--placement-source-model", choices=["source_agnostic", "projected"], default="source_agnostic",
                   help="source_agnostic uses only target-system expected source traffic; projected uses preassigned request-to-die sources for diagnostics.")
    p.add_argument("--expert-weight-bytes", type=float, default=3 * 7168 * 2048)
    p.add_argument("--die-memory-capacity-gib", type=float, default=141.0)
    p.add_argument("--die-compute-capacity", type=float, default=1.0)
    p.add_argument("--refinement-max-passes", type=int, default=16)
    p.add_argument("--output-json", required=True)
    args = p.parse_args()

    topo = load_topology(args.network_config)
    dies = topo.get_all_dies()
    p_ue = json.loads(Path(args.p_ue_json).read_text(encoding="utf-8"))
    h = json.loads(Path(args.h_json).read_text(encoding="utf-8"))
    s_u = json.loads(Path(args.s_u_json).read_text(encoding="utf-8"))

    if args.method == "uniform":
        mapping = _uniform_mapping(args.num_experts, dies)
        solver_stats = {}
    else:
        mapping, solver_stats = _ours_static_mapping(
            num_experts=args.num_experts,
            dies=dies,
            p_ue=p_ue,
            h=h,
            s_u=s_u,
            topo=topo,
            alpha=args.alpha,
            gamma=args.gamma,
            delta=args.delta,
            lambda_=args.lambda_,
            topk_hot=args.topk_hot_experts_for_ilp,
            max_replicas=args.max_replicas,
            source_model=args.placement_source_model,
            return_stats=True,
            expert_weight_bytes=args.expert_weight_bytes,
            die_memory_capacity_bytes=args.die_memory_capacity_gib * 1024**3,
            die_compute_capacity=args.die_compute_capacity,
            refinement_max_passes=args.refinement_max_passes,
        )

    _validate_mapping(mapping, args.num_experts)
    output = {
        "method": args.method,
        "expert_to_dies": {str(k): v for k, v in sorted(mapping.items(), key=lambda x: x[0])},
        "metadata": {
            "alpha": args.alpha,
            "gamma": args.gamma,
            "delta": args.delta,
            "lambda": args.lambda_,
            "placement_source_model": args.placement_source_model,
            "topk_hot_experts_for_ilp": args.topk_hot_experts_for_ilp,
            "max_replicas": args.max_replicas,
            "expert_weight_bytes": args.expert_weight_bytes,
            "die_memory_capacity_gib": args.die_memory_capacity_gib,
            "die_compute_capacity": args.die_compute_capacity,
            "refinement_max_passes": args.refinement_max_passes,
            "solver_stats": solver_stats,
        },
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote mapping to: {out}")


if __name__ == "__main__":
    main()
