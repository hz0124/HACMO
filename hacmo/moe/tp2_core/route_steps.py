#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List

try:
    from .utils_topology import HierarchicalTopology
except ImportError:
    from utils_topology import HierarchicalTopology


ROUTING_POLICIES = {"nearest", "moentwine_er_nearest", "hacmo_cost_aware"}


def _normalize_mapping(mapping: Dict[int | str, List[int]]) -> Dict[int, List[int]]:
    return {int(expert): [int(die) for die in dies] for expert, dies in mapping.items()}


def _window_mapping_table(
    window_mappings: list[dict] | None,
) -> dict[int, Dict[int, List[int]]]:
    if window_mappings is None:
        return {}
    return {
        int(item["window_id"]): _normalize_mapping(item.get("mapping", item))
        for item in window_mappings
    }


def _path_latency_ns(topo: HierarchicalTopology, path: list[tuple]) -> float:
    return sum(
        topo.cfg.mesh_latency_ns if edge[0] == "mesh" else topo.cfg.switch_latency_ns
        for edge in path
    )


def route_activation_counts(
    *,
    src_expert_counts: Dict[int | str, Dict[int | str, int]],
    mapping: Dict[int | str, List[int]],
    topo: HierarchicalTopology,
    dispatch_bytes_per_token: int,
    routing_policy: str,
    source_domains: Dict[int | str, List[int]] | None = None,
    communication_weight: float = 1.0,
    compute_load_weight: float = 1.0,
    path_cache: dict[tuple[int, int], list[tuple]] | None = None,
    max_decisions_per_count: int | None = None,
    moentwine_her_wafer_local: bool = False,
) -> dict:
    """Route one step or one scheduler window with the authoritative policy."""
    if routing_policy not in ROUTING_POLICIES:
        raise ValueError(f"unsupported routing policy: {routing_policy}")
    if dispatch_bytes_per_token <= 0:
        raise ValueError("dispatch bytes per token must be positive")
    if max_decisions_per_count is not None and max_decisions_per_count <= 0:
        raise ValueError("max decisions per count must be positive when provided")

    normalized_mapping = _normalize_mapping(mapping)
    normalized_source_domains = (
        {int(src): [int(die) for die in dies] for src, dies in source_domains.items()}
        if source_domains is not None
        else {}
    )
    cache = path_cache if path_cache is not None else {}

    def path(src: int, dst: int) -> list[tuple]:
        key = (src, dst)
        if key not in cache:
            cache[key] = topo.get_path(src, dst)
        return cache[key]

    latency_cache = getattr(topo, "_route_steps_path_latency_cache", None)
    if latency_cache is None:
        latency_cache = {}
        setattr(topo, "_route_steps_path_latency_cache", latency_cache)
    profile_cache = getattr(topo, "_route_steps_cost_profile_cache", None)
    if profile_cache is None:
        profile_cache = {}
        setattr(topo, "_route_steps_cost_profile_cache", profile_cache)

    def path_latency(src: int, dst: int) -> float:
        key = (src, dst)
        if key not in latency_cache:
            latency_cache[key] = _path_latency_ns(topo, path(src, dst))
        return latency_cache[key]

    def cost_profile(src: int, dst: int) -> tuple:
        key = (src, dst)
        if key not in profile_cache:
            route = path(src, dst)
            edge_inv_bandwidths = [
                (edge, 1.0 / max(topo.get_link_bandwidth(edge), 1e-9))
                for edge in route
            ]
            profile_cache[key] = (
                route,
                path_latency(src, dst),
                edge_inv_bandwidths,
                sum(inv_bw for _, inv_bw in edge_inv_bandwidths),
            )
        return profile_cache[key]

    input_activations = sum(
        int(count)
        for counts in src_expert_counts.values()
        for count in counts.values()
    )
    ideal_die_load = max(input_activations / max(topo.total_dies, 1), 1.0)
    die_load: Counter[int] = Counter()
    link_load_bytes: Counter[tuple] = Counter()
    assignments: Counter[tuple[int, int, int, int]] = Counter()

    for src_raw, expert_counts in sorted(
        src_expert_counts.items(), key=lambda item: int(item[0])
    ):
        src = int(src_raw)
        for expert_raw, count in sorted(
            expert_counts.items(), key=lambda item: int(item[0])
        ):
            expert = int(expert_raw)
            candidates = normalized_mapping.get(expert, [])
            if not candidates:
                raise ValueError(f"mapping has no placement for expert {expert}")
            count = int(count)
            if routing_policy in {"nearest", "hacmo_cost_aware"} and len(candidates) == 1:
                actual_src = src
                dst = candidates[0]
                assignments[(src, actual_src, expert, dst)] += count
                die_load[dst] += count
                for edge in path(actual_src, dst):
                    link_load_bytes[edge] += count * dispatch_bytes_per_token
                continue
            if routing_policy == "nearest":
                actual_src = src
                dst = min(
                    candidates,
                    key=lambda die: (path_latency(src, die), die),
                )
                assignments[(src, actual_src, expert, dst)] += count
                die_load[dst] += count
                for edge in path(actual_src, dst):
                    link_load_bytes[edge] += count * dispatch_bytes_per_token
                continue
            if routing_policy == "moentwine_er_nearest":
                source_candidates = normalized_source_domains.get(src)
                if not source_candidates:
                    raise ValueError(f"MoEntwine ER source domain missing for die {src}")
                candidate_pairs = [
                    (candidate_src, candidate_dst)
                    for candidate_src in source_candidates
                    for candidate_dst in candidates
                    if not moentwine_her_wafer_local
                    or topo.die_to_wafer_local(candidate_src)[0]
                    == topo.die_to_wafer_local(candidate_dst)[0]
                ]
                if not candidate_pairs:
                    raise ValueError(
                        f"MoEntwine HER has no wafer-local source for expert {expert}"
                    )
                actual_src, dst = min(
                    candidate_pairs,
                    key=lambda pair: (
                        path_latency(pair[0], pair[1]),
                        pair[0],
                        pair[1],
                    ),
                )
                assignments[(src, actual_src, expert, dst)] += count
                die_load[dst] += count
                for edge in path(actual_src, dst):
                    link_load_bytes[edge] += count * dispatch_bytes_per_token
                continue

            candidate_profiles = {}
            for die in candidates:
                candidate_profiles[die] = cost_profile(src, die)
            candidate_base_comm_costs = {
                die: profile[1]
                + sum(link_load_bytes[edge] * inv_bw for edge, inv_bw in profile[2])
                for die, profile in candidate_profiles.items()
            }
            affected_candidates_by_edge: dict[tuple, list[tuple[int, float]]] = (
                defaultdict(list)
            )
            for die, profile in candidate_profiles.items():
                for edge, inv_bw in profile[2]:
                    affected_candidates_by_edge[edge].append((die, inv_bw))
            decision_count = (
                count
                if max_decisions_per_count is None
                else min(count, max_decisions_per_count)
            )
            base_chunk, extra = divmod(count, decision_count)
            for decision in range(decision_count):
                chunk_size = base_chunk + (1 if decision < extra else 0)
                chunk_bytes = chunk_size * dispatch_bytes_per_token
                comm_costs = {
                    die: candidate_base_comm_costs[die]
                    + chunk_bytes * profile[3]
                    for die, profile in candidate_profiles.items()
                }
                comm_reference = max(max(comm_costs.values(), default=0.0), 1.0)
                dst = min(
                    candidates,
                    key=lambda die: (
                        communication_weight * comm_costs[die] / comm_reference
                        + compute_load_weight
                        * (die_load[die] + chunk_size)
                        / ideal_die_load,
                        die,
                    ),
                )
                assignments[(src, src, expert, dst)] += chunk_size
                die_load[dst] += chunk_size
                for edge in candidate_profiles[dst][0]:
                    link_load_bytes[edge] += chunk_bytes
                    for affected_die, inv_bw in affected_candidates_by_edge.get(
                        edge, []
                    ):
                        candidate_base_comm_costs[affected_die] += (
                            chunk_bytes * inv_bw
                        )

    return {
        "assignments": assignments,
        "die_load": die_load,
        "dispatch_link_load_bytes": link_load_bytes,
        "input_activations": input_activations,
    }


def build_routed_steps(
    *,
    steps: list[dict],
    static_mapping: Dict[int | str, List[int]],
    window_mappings: list[dict] | None,
    topo: HierarchicalTopology,
    dispatch_bytes_per_token: int,
    combine_bytes_per_token: int,
    routing_policy: str,
    source_domains: Dict[int | str, List[int]] | None = None,
    communication_weight: float = 1.0,
    compute_load_weight: float = 1.0,
    moentwine_her_wafer_local: bool = False,
) -> dict:
    if routing_policy not in ROUTING_POLICIES:
        raise ValueError(f"unsupported routing policy: {routing_policy}")
    if dispatch_bytes_per_token <= 0 or combine_bytes_per_token <= 0:
        raise ValueError("dispatch and combine bytes per token must be positive")

    base_mapping = _normalize_mapping(static_mapping)
    mappings_by_window = _window_mapping_table(window_mappings)
    path_cache: dict[tuple[int, int], list[tuple]] = {}

    def path(src: int, dst: int) -> list[tuple]:
        key = (src, dst)
        if key not in path_cache:
            path_cache[key] = topo.get_path(src, dst)
        return path_cache[key]

    routed_steps: list[dict] = []
    mapping_copy_provenance: list[dict] = []
    for step in steps:
        window_id = int(step["window_id"])
        mapping = mappings_by_window.get(window_id, base_mapping)
        routed = route_activation_counts(
            src_expert_counts=step.get("src_expert_counts", {}),
            mapping=mapping,
            topo=topo,
            dispatch_bytes_per_token=dispatch_bytes_per_token,
            routing_policy=routing_policy,
            source_domains=source_domains,
            communication_weight=communication_weight,
            compute_load_weight=compute_load_weight,
            path_cache=path_cache,
            moentwine_her_wafer_local=moentwine_her_wafer_local,
        )
        assignments = routed["assignments"]
        input_activations = int(routed["input_activations"])

        dispatch: dict[int, Counter[int]] = defaultdict(Counter)
        combine: dict[int, Counter[int]] = defaultdict(Counter)
        rank_expert_tokens: dict[int, Counter[int]] = defaultdict(Counter)
        for (_, src, expert, dst), count in assignments.items():
            rank_expert_tokens[dst][expert] += count
            if src == dst:
                continue
            dispatch[src][dst] += count * dispatch_bytes_per_token
            combine[dst][src] += count * combine_bytes_per_token

        routed_activations = sum(assignments.values())
        compute_activations = sum(
            count
            for expert_counts in rank_expert_tokens.values()
            for count in expert_counts.values()
        )
        routed_steps.append(
            {
                "step_id": int(step["step_id"]),
                "token_idx": int(step.get("token_idx", int(step["step_id"]) + 1)),
                "window_id": window_id,
                "active": bool(step.get("active", input_activations > 0)),
                "tokens_per_die": {
                    str(src): int(count)
                    for src, count in sorted(
                        step.get("tokens_per_die", {}).items(),
                        key=lambda item: int(item[0]),
                    )
                },
                "assignments": [
                    {
                        **(
                            {"nominal_src_die": nominal_src}
                            if nominal_src != src
                            else {}
                        ),
                        "src_die": src,
                        "expert": expert,
                        "dst_die": dst,
                        "activations": count,
                    }
                    for (nominal_src, src, expert, dst), count in sorted(
                        assignments.items()
                    )
                ],
                "dispatch_bytes": {
                    str(src): {
                        str(dst): int(count) for dst, count in sorted(dsts.items())
                    }
                    for src, dsts in sorted(dispatch.items())
                },
                "combine_bytes": {
                    str(src): {
                        str(dst): int(count) for dst, count in sorted(dsts.items())
                    }
                    for src, dsts in sorted(combine.items())
                },
                "rank_expert_tokens": {
                    str(die): {
                        str(expert): int(count)
                        for expert, count in sorted(expert_counts.items())
                    }
                    for die, expert_counts in sorted(rank_expert_tokens.items())
                },
                "validation": {
                    "input_activations": input_activations,
                    "routed_activations": routed_activations,
                    "compute_activations": compute_activations,
                    "all_activations_routed": routed_activations == input_activations,
                    "compute_matches_assignments": compute_activations
                    == routed_activations,
                },
            }
        )
        mapping_copy_provenance.append(
            {
                "step_id": int(step["step_id"]),
                "window_id": window_id,
                "total_copies": sum(len(dies) for dies in mapping.values()),
                "extra_replicas": sum(
                    max(len(dies) - 1, 0) for dies in mapping.values()
                ),
            }
        )

    return {
        "metadata": {
            "artifact": "tp2_step_route_plan",
            "routing_policy": routing_policy,
            "source_domain_policy": (
                "moentwine_her_wafer_local_after_inter_wafer_all_gather"
                if moentwine_her_wafer_local
                else "moentwine_er_tp_group_after_all_gather"
                if routing_policy == "moentwine_er_nearest"
                else "fixed_source_die"
            ),
            "communication_weight": communication_weight,
            "compute_load_weight": compute_load_weight,
            "dispatch_bytes_per_token": dispatch_bytes_per_token,
            "combine_bytes_per_token": combine_bytes_per_token,
            "num_steps": len(routed_steps),
            "batch_scale": 1.0,
            "mapping_copy_provenance": {
                "source": "effective_mapping_used_for_each_step",
                "per_step": mapping_copy_provenance,
            },
        },
        "steps": routed_steps,
    }
