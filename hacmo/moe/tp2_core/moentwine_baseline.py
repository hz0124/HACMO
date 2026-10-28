#!/usr/bin/env python3
"""Independent MoEntwine ER/HER and causal NI-Balancer model."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List

try:
    from .utils_topology import HierarchicalTopology
except ImportError:
    from utils_topology import HierarchicalTopology


def valid_attention_tp_degrees(num_query_heads: int, num_dies: int) -> list[int]:
    """Return TP degrees supported by both query-head and device divisibility."""
    if num_query_heads <= 0 or num_dies <= 0:
        raise ValueError("num_query_heads and num_dies must be positive")
    return [
        degree
        for degree in range(1, min(num_query_heads, num_dies) + 1)
        if num_query_heads % degree == 0 and num_dies % degree == 0
    ]


def _local_tp_groups(
    topo: HierarchicalTopology, tp_rows: int, tp_cols: int
) -> list[list[int]]:
    rows = topo.cfg.mesh_rows
    cols = topo.cfg.mesh_cols
    if tp_rows <= 0 or tp_cols <= 0:
        raise ValueError("TP dimensions must be positive")
    if rows % tp_rows != 0 or cols % tp_cols != 0:
        raise ValueError(
            f"TP=({tp_rows},{tp_cols}) does not divide mesh=({rows},{cols})"
        )
    ftd_rows = rows // tp_rows
    ftd_cols = cols // tp_cols
    return [
        [
            topo.rc_to_local(row, col)
            for row in range(rows)
            for col in range(cols)
            if row % ftd_rows == residue_row and col % ftd_cols == residue_col
        ]
        for residue_row in range(ftd_rows)
        for residue_col in range(ftd_cols)
    ]


def build_er_source_domains(
    topo: HierarchicalTopology,
    tp_rows: int,
    tp_cols: int,
    *,
    hierarchical: bool,
) -> dict[int, list[int]]:
    """Return source alternatives created by paper-accurate ER/HER collectives.

    ER uses coordinate-residue TP groups. HER omits the local all-gather and
    instead all-gathers each reduce-scattered local partition across wafers, so
    each partition has one source copy at the same local coordinate per wafer.
    """
    local_groups = _local_tp_groups(topo, tp_rows, tp_cols)
    if hierarchical:
        return {
            wafer * topo.cfg.dies_per_wafer
            + local: [
                peer_wafer * topo.cfg.dies_per_wafer + local
                for peer_wafer in range(topo.cfg.num_wafers)
            ]
            for wafer in range(topo.cfg.num_wafers)
            for local in range(topo.cfg.dies_per_wafer)
        }

    domains: dict[int, list[int]] = {}
    for wafer in range(topo.cfg.num_wafers):
        wafer_base = wafer * topo.cfg.dies_per_wafer
        for local_group in local_groups:
            group = [wafer_base + local for local in local_group]
            for die in group:
                domains[die] = group
    return domains


def _append_ring_phase(
    phases: list[dict],
    *,
    name: str,
    rings: list[tuple[list[int], list[int]]],
    phase_index: int,
) -> None:
    messages: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for ranks, owner_bytes in rings:
        size = len(ranks)
        if size <= 1:
            continue
        for rank_index, src in enumerate(ranks):
            owner = (rank_index - phase_index) % size
            nbytes = int(owner_bytes[owner])
            if nbytes > 0:
                dst = ranks[(rank_index + 1) % size]
                messages[src][dst] += nbytes
    phases.append(
        {
            "name": name,
            "messages": {
                str(src): {
                    str(dst): int(nbytes) for dst, nbytes in sorted(dsts.items())
                }
                for src, dsts in sorted(messages.items())
            },
        }
    )


def _ring_ordered_local_group(
    topo: HierarchicalTopology,
    local_group: list[int],
    tp_rows: int,
    tp_cols: int,
) -> list[int]:
    """Order the fixed 2x2 TP group around its perimeter."""
    if tp_rows != 2 or tp_cols != 2:
        return list(local_group)
    by_coordinate = {topo.local_to_rc(local): local for local in local_group}
    rows = sorted({row for row, _ in by_coordinate})
    cols = sorted({col for _, col in by_coordinate})
    if len(rows) != 2 or len(cols) != 2:
        raise ValueError("TP=(2,2) group does not form a rectangle")
    return [
        by_coordinate[(rows[0], cols[0])],
        by_coordinate[(rows[0], cols[1])],
        by_coordinate[(rows[1], cols[1])],
        by_coordinate[(rows[1], cols[0])],
    ]


def _build_local_tp_rings(
    *,
    tokens_per_die: Dict[int | str, int],
    topo: HierarchicalTopology,
    hidden_size: int,
    bytes_per_hidden_elem: int,
    tp_rows: int,
    tp_cols: int,
) -> tuple[list[tuple[list[int], list[int]]], dict[tuple[int, int], int]]:
    local_groups = _local_tp_groups(topo, tp_rows, tp_cols)
    tp_degree = tp_rows * tp_cols
    if any(len(group) != tp_degree for group in local_groups):
        raise ValueError("each ER TP group must contain exactly TP dies")
    normalized_tokens = {int(die): int(count) for die, count in tokens_per_die.items()}

    rings: list[tuple[list[int], list[int]]] = []
    local_partition_bytes: dict[tuple[int, int], int] = {}
    for wafer in range(topo.cfg.num_wafers):
        wafer_base = wafer * topo.cfg.dies_per_wafer
        for local_group in local_groups:
            ordered_group = _ring_ordered_local_group(
                topo, local_group, tp_rows, tp_cols
            )
            ranks = [wafer_base + local for local in ordered_group]
            group_tokens = sum(normalized_tokens.get(die, 0) for die in ranks)
            tensor_bytes = group_tokens * hidden_size * bytes_per_hidden_elem
            partition_bytes = math.ceil(tensor_bytes / tp_degree)
            rings.append((ranks, [partition_bytes for _ in ranks]))
            for local in local_group:
                local_partition_bytes[(wafer, local)] = partition_bytes
    return rings, local_partition_bytes


def build_conventional_tp_collective_segments(
    *,
    tokens_per_die: Dict[int | str, int],
    topo: HierarchicalTopology,
    hidden_size: int,
    bytes_per_hidden_elem: int,
    tp_rows: int,
    tp_cols: int,
) -> list[dict]:
    """Build the conventional wafer-local TP ring all-reduce."""
    if hidden_size <= 0 or bytes_per_hidden_elem <= 0:
        raise ValueError("attention tensor dimensions must be positive")
    rings, _ = _build_local_tp_rings(
        tokens_per_die=tokens_per_die,
        topo=topo,
        hidden_size=hidden_size,
        bytes_per_hidden_elem=bytes_per_hidden_elem,
        tp_rows=tp_rows,
        tp_cols=tp_cols,
    )
    tp_degree = tp_rows * tp_cols
    phases: list[dict] = []
    for phase in range(tp_degree - 1):
        _append_ring_phase(
            phases,
            name=f"conventional_tp_reduce_scatter_{phase}",
            rings=rings,
            phase_index=phase,
        )
    for phase in range(tp_degree - 1):
        _append_ring_phase(
            phases,
            name=f"conventional_tp_all_gather_{phase}",
            rings=rings,
            phase_index=phase,
        )
    return phases


def build_moentwine_collective_segments(
    *,
    tokens_per_die: Dict[int | str, int],
    topo: HierarchicalTopology,
    hidden_size: int,
    bytes_per_hidden_elem: int,
    tp_rows: int,
    tp_cols: int,
) -> list[dict]:
    """Build the full sequential ER/HER attention collective for one step.

    Single-wafer ER is a ring all-reduce (reduce-scatter plus all-gather).
    Multi-wafer HER replaces the local all-gather with the paper's inter-wafer
    all-gather, but the complete enabling collective is still charged.
    """
    if hidden_size <= 0 or bytes_per_hidden_elem <= 0:
        raise ValueError("attention tensor dimensions must be positive")
    tp_degree = tp_rows * tp_cols
    reduce_scatter_rings, local_partition_bytes = _build_local_tp_rings(
        tokens_per_die=tokens_per_die,
        topo=topo,
        hidden_size=hidden_size,
        bytes_per_hidden_elem=bytes_per_hidden_elem,
        tp_rows=tp_rows,
        tp_cols=tp_cols,
    )

    phases: list[dict] = []
    reduce_scatter_prefix = (
        "her_intra_wafer" if topo.cfg.num_wafers > 1 else "er_intra_wafer"
    )
    for phase in range(tp_degree - 1):
        _append_ring_phase(
            phases,
            name=f"{reduce_scatter_prefix}_reduce_scatter_{phase}",
            rings=reduce_scatter_rings,
            phase_index=phase,
        )

    if topo.cfg.num_wafers > 1:
        all_gather_rings = []
        for local in range(topo.cfg.dies_per_wafer):
            ranks = [
                wafer * topo.cfg.dies_per_wafer + local
                for wafer in range(topo.cfg.num_wafers)
            ]
            owner_bytes = [
                local_partition_bytes[(wafer, local)]
                for wafer in range(topo.cfg.num_wafers)
            ]
            all_gather_rings.append((ranks, owner_bytes))
        for phase in range(topo.cfg.num_wafers - 1):
            _append_ring_phase(
                phases,
                name=f"her_inter_wafer_all_gather_{phase}",
                rings=all_gather_rings,
                phase_index=phase,
            )
    else:
        for phase in range(tp_degree - 1):
            _append_ring_phase(
                phases,
                name=f"er_intra_wafer_all_gather_{phase}",
                rings=reduce_scatter_rings,
                phase_index=phase,
            )
    return phases


def incremental_collective_accounting(
    *,
    no_collective_total_us: float,
    conventional_tp_collective_us: float,
    full_her_collective_us: float,
) -> dict[str, float]:
    """Charge HER only for cost beyond the conventional TP collective."""
    incremental_us = max(
        0.0, float(full_her_collective_us) - float(conventional_tp_collective_us)
    )
    return {
        "no_collective_total_us": float(no_collective_total_us),
        "conventional_tp_collective_us": float(conventional_tp_collective_us),
        "full_her_collective_us": float(full_her_collective_us),
        "incremental_collective_us": incremental_us,
        "charged_total_us": float(no_collective_total_us) + incremental_us,
    }


def _copy_mapping(mapping: Dict[int | str, List[int]]) -> dict[int, list[int]]:
    return {int(expert): [int(die) for die in dies] for expert, dies in mapping.items()}


def _path_latency_ns(topo: HierarchicalTopology, src: int, dst: int) -> float:
    return sum(
        topo.cfg.mesh_latency_ns if edge[0] == "mesh" else topo.cfg.switch_latency_ns
        for edge in topo.get_path(src, dst)
    )


def _device_heat(
    mapping: dict[int, list[int]],
    expert_load: dict[int, float],
    num_dies: int,
) -> tuple[dict[int, float], dict[int, list[int]]]:
    heat = {die: 0.0 for die in range(num_dies)}
    hosted: dict[int, list[int]] = defaultdict(list)
    for expert, dies in mapping.items():
        share = expert_load.get(expert, 0.0) / max(len(dies), 1)
        for die in dies:
            heat[die] += share
            hosted[die].append(expert)
    return heat, hosted


def _balance_from_load(
    *,
    base_mapping: dict[int, list[int]],
    expert_load: dict[int, float],
    topo: HierarchicalTopology,
    max_replicas_per_expert: int,
    shadow_slots_per_die: int,
    max_adjustments: int,
) -> dict[int, list[int]]:
    """Implement MoEntwine Algorithm 1 without HACMO placement objectives."""
    mapping = _copy_mapping(base_mapping)
    primary_slots = defaultdict(int)
    for dies in base_mapping.values():
        for die in dies:
            primary_slots[die] += 1

    for _ in range(max_adjustments):
        heat, hosted = _device_heat(mapping, expert_load, topo.total_dies)
        hottest_die = max(heat, key=lambda die: (heat[die], -die))
        source_candidates = [
            expert
            for expert in hosted[hottest_die]
            if expert_load.get(expert, 0.0) > 0
            and len(mapping[expert]) < max_replicas_per_expert
        ]
        if not source_candidates:
            break
        source_expert = max(
            source_candidates,
            key=lambda expert: (
                expert_load[expert] / max(len(mapping[expert]), 1),
                -expert,
            ),
        )
        replicated_share = expert_load[source_expert] / (
            len(mapping[source_expert]) + 1
        )
        cold_dies = []
        for die in range(topo.total_dies):
            if die in mapping[source_expert]:
                continue
            slot_capacity = primary_slots[die] + shadow_slots_per_die
            if len(hosted[die]) >= slot_capacity:
                continue
            if heat[die] + replicated_share < heat[hottest_die] - 1e-12:
                cold_dies.append(die)
        if not cold_dies:
            break
        destination = min(
            cold_dies,
            key=lambda die: (_path_latency_ns(topo, hottest_die, die), die),
        )
        mapping[source_expert].append(destination)
    return mapping


def build_causal_ni_windows(
    *,
    base_mapping: Dict[int | str, List[int]],
    window_stats: list[dict],
    topo: HierarchicalTopology,
    expert_weight_bytes: int,
    max_replicas_per_expert: int = 2,
    shadow_slots_per_die: int = 1,
    max_adjustments_per_window: int = 256,
    ewma_decay: float = 0.8,
) -> list[dict]:
    """Return mappings where window w observations affect only window w+1 onward."""
    if not 0.0 <= ewma_decay < 1.0:
        raise ValueError("ewma_decay must be in [0, 1)")
    if max_replicas_per_expert < 1 or shadow_slots_per_die < 0:
        raise ValueError("replica and shadow-slot limits are invalid")

    primary_mapping = _copy_mapping(base_mapping)
    current_mapping = _copy_mapping(base_mapping)
    historical_load: dict[int, float] = {}
    windows: list[dict] = []
    for window in window_stats:
        window_id = int(window["window_id"])
        observed = {
            int(expert): float(count)
            for expert, count in window.get("expert_counts", {}).items()
        }
        experts = set(historical_load) | set(observed) | set(primary_mapping)
        historical_load = {
            expert: (
                ewma_decay * historical_load.get(expert, observed.get(expert, 0.0))
                + (1.0 - ewma_decay) * observed.get(expert, 0.0)
            )
            for expert in experts
        }
        planned = _balance_from_load(
            base_mapping=primary_mapping,
            expert_load=historical_load,
            topo=topo,
            max_replicas_per_expert=max_replicas_per_expert,
            shadow_slots_per_die=shadow_slots_per_die,
            max_adjustments=max_adjustments_per_window,
        )

        added = []
        removed = []
        migrations = []
        for expert in sorted(primary_mapping):
            old = set(current_mapping[expert])
            new = set(planned[expert])
            for destination in sorted(new - old):
                source = min(
                    current_mapping[expert],
                    key=lambda die: (_path_latency_ns(topo, die, destination), die),
                )
                path = topo.get_path(source, destination)
                added.append(
                    {"expert": expert, "die": destination, "source_die": source}
                )
                migrations.append(
                    {
                        "expert": expert,
                        "source_die": source,
                        "destination_die": destination,
                        "bytes": int(expert_weight_bytes),
                        "path": [list(edge) for edge in path],
                        "path_latency_ns": _path_latency_ns(topo, source, destination),
                    }
                )
            for die in sorted(old - new):
                removed.append({"expert": expert, "die": die})

        windows.append(
            {
                "window_id": window_id,
                "mapping": {
                    str(e): list(ds) for e, ds in sorted(current_mapping.items())
                },
                "next_mapping": {str(e): list(ds) for e, ds in sorted(planned.items())},
                "replicas_added": added,
                "replicas_removed": removed,
                "migration_events": migrations,
                "migration_bytes": sum(event["bytes"] for event in migrations),
                "observation_semantics": "window_w_updates_window_w_plus_1",
                "load_estimator": "causal_ewma",
            }
        )
        current_mapping = planned
    return windows
