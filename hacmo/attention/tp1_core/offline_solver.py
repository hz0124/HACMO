"""Paper Eq. (1)-style offline mapping search, isolated from simulator output."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Sequence

from .mapping import MappingPlan, build_mapping_plan
from .routing import (
    ResidencyState,
    commit_hbm_fills,
    hbm_port_core,
    manhattan_hops,
    route_window,
    seed_static_cache,
)
from .workload import RequestLane, WorkloadWindow


@dataclass(frozen=True)
class OfflineCostModel:
    aggregate_hbm_gbps: float
    hbm_ports: int
    hbm_internal_ns: float
    hbm_concurrency_per_port: int
    noc_link_gbps: float
    gamma: float
    die_compute_tflops: float
    flops_per_selected_kv: float
    cacheline_bytes: int
    kv_entry_bytes: int
    max_context: int
    n_layers: int


@dataclass(frozen=True)
class CandidateScore:
    compute_gradient: float
    cache_gradient: float
    mean_window_us: float
    mean_compute_us: float
    mean_hbm_us: float
    mean_noc_us: float
    mean_bottleneck_link_us: float


class _XYLinkLoads:
    """Exact O(1)-per-flow load accumulator for directed X-then-Y routes."""

    def __init__(self, rows: int, cols: int) -> None:
        self.rows = rows
        self.cols = cols
        self.horizontal_stride = cols + 1
        self.vertical_stride = rows + 1
        self.east = [0] * (rows * self.horizontal_stride)
        self.west = [0] * (rows * self.horizontal_stride)
        self.south = [0] * (cols * self.vertical_stride)
        self.north = [0] * (cols * self.vertical_stride)

    def add(self, source: int, target: int, size_bytes: int) -> None:
        if source == target or size_bytes <= 0:
            return
        row, col = divmod(source, self.cols)
        target_row, target_col = divmod(target, self.cols)
        horizontal_offset = row * self.horizontal_stride
        if target_col > col:
            self.east[horizontal_offset + col] += size_bytes
            self.east[horizontal_offset + target_col] -= size_bytes
        elif target_col < col:
            self.west[horizontal_offset + target_col] += size_bytes
            self.west[horizontal_offset + col] -= size_bytes

        vertical_offset = target_col * self.vertical_stride
        if target_row > row:
            self.south[vertical_offset + row] += size_bytes
            self.south[vertical_offset + target_row] -= size_bytes
        elif target_row < row:
            self.north[vertical_offset + target_row] += size_bytes
            self.north[vertical_offset + row] -= size_bytes

    def peak_bytes(self) -> int:
        peak = 0
        for row in range(self.rows):
            offset = row * self.horizontal_stride
            east = 0
            west = 0
            for col in range(self.cols - 1):
                east += self.east[offset + col]
                west += self.west[offset + col]
                peak = max(peak, east, west)
        for col in range(self.cols):
            offset = col * self.vertical_stride
            south = 0
            north = 0
            for row in range(self.rows - 1):
                south += self.south[offset + row]
                north += self.north[offset + row]
                peak = max(peak, south, north)
        return peak


def _window_cost_us(
    plan: MappingPlan,
    accesses,
    model: OfflineCostModel,
) -> tuple[float, float, float, float, float]:
    per_core = [0] * plan.num_cores
    requests_per_port = [0] * model.hbm_ports
    bytes_per_port = [0] * model.hbm_ports
    noc_byte_hops = [0] * plan.num_cores
    link_loads = _XYLinkLoads(plan.rows, plan.cols)
    for access in accesses:
        core = access.consumer_core
        per_core[core] += 1
        if access.source == "remote":
            assert access.source_core is not None
            hops = manhattan_hops(
                access.source_core, access.consumer_core, plan.cols
            )
            noc_byte_hops[core] += access.size_bytes * hops
            link_loads.add(
                access.source_core,
                access.consumer_core,
                access.size_bytes,
            )
        elif access.source == "hbm":
            assert access.hbm_port is not None
            assert access.hbm_address is not None
            first = access.hbm_address // model.cacheline_bytes
            last = (
                access.hbm_address + access.size_bytes - 1
            ) // model.cacheline_bytes
            transactions = last - first + 1
            requests_per_port[access.hbm_port] += transactions
            bytes_per_port[access.hbm_port] += transactions * model.cacheline_bytes
            port_core = hbm_port_core(access.hbm_port, plan.rows, plan.cols, model.hbm_ports)
            wire_bytes = transactions * model.cacheline_bytes
            noc_byte_hops[core] += wire_bytes * manhattan_hops(
                port_core, access.consumer_core, plan.cols
            )
            link_loads.add(
                port_core,
                access.consumer_core,
                wire_bytes,
            )
            target = access.cache_target_core
            if target is not None and target != access.consumer_core and plan.total_cache_entries:
                noc_byte_hops[core] += access.size_bytes * manhattan_hops(
                    access.consumer_core, target, plan.cols
                )
                link_loads.add(
                    access.consumer_core,
                    target,
                    access.size_bytes,
                )

    per_core_tflops = model.die_compute_tflops / plan.num_cores
    compute_by_core = [
        count
        * model.flops_per_selected_kv
        / max(per_core_tflops * 1e12, 1.0)
        * 1e6
        for count in per_core
    ]
    per_port_gbps = model.aggregate_hbm_gbps / model.hbm_ports
    # Every consumer shares the same physical queue at a given HBM port.
    # Compute placement can change the following NoC route, but it cannot
    # create an independent copy of that port's service bandwidth.
    hbm_service_us = max(
        (
            math.ceil(requests / max(model.hbm_concurrency_per_port, 1))
            * model.hbm_internal_ns
            / 1000.0
            + bytes_count / max(per_port_gbps * 1e9, 1.0) * 1e6
        )
        for requests, bytes_count in zip(requests_per_port, bytes_per_port)
    )
    noc_by_core = [
        byte_hops
        / max(model.noc_link_gbps * 1e9, 1.0)
        * model.gamma
        * 1e6
        for byte_hops in noc_byte_hops
    ]
    bottleneck_link_us = (
        link_loads.peak_bytes()
        / max(model.noc_link_gbps * 1e9, 1.0)
        * model.gamma
        * 1e6
    )
    finish_by_core = [
        compute + hbm_service_us + noc
        for compute, noc in zip(compute_by_core, noc_by_core)
    ]
    max_core_noc_us = max(noc_by_core)
    # Per-core byte-hop cost misses queueing shared by flows from many cores.
    # Add only the service-time excess to avoid double-counting the path term.
    shared_link_excess_us = max(bottleneck_link_us - max_core_noc_us, 0.0)
    return (
        max(finish_by_core) + shared_link_excess_us,
        max(compute_by_core),
        hbm_service_us,
        max(max_core_noc_us, bottleneck_link_us),
        bottleneck_link_us,
    )


def score_mapping_plan(
    plan: MappingPlan,
    lanes: Sequence[RequestLane],
    layer_ids: Sequence[int],
    decode_steps: Sequence[int],
    model: OfflineCostModel,
    residency_layer_ids: Sequence[int] | None = None,
) -> CandidateScore:
    state = ResidencyState(plan)
    cache_layers = tuple(residency_layer_ids or layer_ids)
    missing = sorted(set(layer_ids) - set(cache_layers))
    if missing:
        raise ValueError(f"scored layers are absent from cache population: {missing}")
    # Representative layers are scored analytically, but their initial cache
    # occupancy must reflect all physical layers sharing the same SRAM.
    seed_static_cache(state, lanes, cache_layers)
    totals = [0.0, 0.0, 0.0, 0.0, 0.0]
    windows = 0
    for layer_id in layer_ids:
        for decode_step in decode_steps:
            accesses, _ = route_window(
                window=WorkloadWindow(layer_id, decode_step, tuple(lanes)),
                plan=plan,
                state=state,
                kv_entry_bytes=model.kv_entry_bytes,
                cacheline_bytes=model.cacheline_bytes,
                num_hbm_ports=model.hbm_ports,
                max_context=model.max_context,
                n_layers=model.n_layers,
            )
            values = _window_cost_us(plan, accesses, model)
            for index, value in enumerate(values):
                totals[index] += value
            commit_hbm_fills(state, accesses)
            windows += 1
    if not windows:
        raise ValueError("offline solver requires at least one complete window")
    return CandidateScore(
        compute_gradient=plan.compute_gradient,
        cache_gradient=plan.cache_gradient,
        mean_window_us=totals[0] / windows,
        mean_compute_us=totals[1] / windows,
        mean_hbm_us=totals[2] / windows,
        mean_noc_us=totals[3] / windows,
        mean_bottleneck_link_us=totals[4] / windows,
    )


def solve_nonuniform_mapping(
    *,
    rows: int,
    cols: int,
    batch_size: int,
    topk: int,
    kv_entry_bytes: int,
    total_sram_bytes: int,
    cache_ratio: float,
    lanes: Sequence[RequestLane],
    layer_ids: Sequence[int],
    decode_steps: Sequence[int],
    model: OfflineCostModel,
    residency_layer_ids: Sequence[int] | None = None,
    compute_gradients: Sequence[float] = (0.0, 0.5, 1.0, 1.5),
    cache_gradients: Sequence[float] = (0.0, 0.5, 1.0, 1.5),
) -> tuple[MappingPlan, tuple[CandidateScore, ...]]:
    if not lanes:
        raise ValueError("offline solver requires training lanes")
    scores: list[CandidateScore] = []
    plans: list[MappingPlan] = []
    for compute_gradient in compute_gradients:
        for cache_gradient in cache_gradients:
            plan = build_mapping_plan(
                policy="S",
                rows=rows,
                cols=cols,
                batch_size=batch_size,
                topk=topk,
                kv_entry_bytes=kv_entry_bytes,
                total_sram_bytes=total_sram_bytes,
                cache_ratio=cache_ratio,
                compute_gradient=compute_gradient,
                cache_gradient=cache_gradient,
                offline_gamma=model.gamma,
            )
            score = score_mapping_plan(
                plan,
                lanes,
                layer_ids,
                decode_steps,
                model,
                residency_layer_ids=residency_layer_ids,
            )
            plans.append(plan)
            scores.append(score)
    eligible = [
        index
        for index, score in enumerate(scores)
        if score.compute_gradient > 0.0 and score.cache_gradient > 0.0
    ]
    if not eligible:
        raise ValueError(
            "non-uniform mapping requires at least one positive compute/cache gradient pair"
        )
    best_index = min(
        eligible,
        key=lambda index: (
            scores[index].mean_window_us,
            scores[index].compute_gradient + scores[index].cache_gradient,
        ),
    )
    best = replace(plans[best_index], offline_score_us=scores[best_index].mean_window_us)
    return best, tuple(scores)
