"""Frozen per-core TP1 mapping plans produced without simulator feedback."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class CoreMapping:
    core_id: int
    row: int
    col: int
    cp_group: int
    dp_group: int
    compute_quota_entries: int
    cache_capacity_entries: int
    compute_memory_bytes: int


@dataclass(frozen=True)
class MappingPlan:
    policy: str
    rows: int
    cols: int
    batch_size: int
    topk: int
    kv_entry_bytes: int
    total_sram_bytes: int
    cache_budget_bytes: int
    compute_gradient: float
    cache_gradient: float
    offline_gamma: float
    offline_score_us: float | None
    cores: tuple[CoreMapping, ...]

    @property
    def num_cores(self) -> int:
        return self.rows * self.cols

    @cached_property
    def total_compute_entries(self) -> int:
        return sum(core.compute_quota_entries for core in self.cores)

    @cached_property
    def total_cache_entries(self) -> int:
        return sum(core.cache_capacity_entries for core in self.cores)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def write_json(self, path: str | Path) -> None:
        output = Path(path).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


def _largest_remainder(total: int, weights: Sequence[float]) -> list[int]:
    if total < 0:
        raise ValueError("total must be non-negative")
    if not weights or any(weight < 0 for weight in weights):
        raise ValueError("weights must be a non-empty non-negative sequence")
    weight_sum = sum(weights)
    normalized = [weight / weight_sum for weight in weights] if weight_sum else [1.0 / len(weights)] * len(weights)
    exact = [total * weight for weight in normalized]
    allocated = [math.floor(value) for value in exact]
    remaining = total - sum(allocated)
    order = sorted(range(len(weights)), key=lambda index: (-(exact[index] - allocated[index]), index))
    for index in order[:remaining]:
        allocated[index] += 1
    return allocated


def _bounded_largest_remainder(total: int, weights: Sequence[float], capacities: Sequence[int]) -> list[int]:
    if len(weights) != len(capacities) or any(capacity < 0 for capacity in capacities):
        raise ValueError("weights and non-negative capacities must have equal length")
    if total > sum(capacities):
        raise ValueError("bounded allocation exceeds aggregate capacity")
    allocated = [0] * len(weights)
    active = {index for index, capacity in enumerate(capacities) if capacity > 0}
    remaining = total
    while remaining > 0:
        if not active:
            raise RuntimeError("bounded allocation ran out of capacity")
        proposal = _largest_remainder(remaining, [weights[index] for index in sorted(active)])
        accepted = 0
        for local_index, index in enumerate(sorted(active)):
            room = capacities[index] - allocated[index]
            take = min(proposal[local_index], room)
            allocated[index] += take
            accepted += take
        if accepted == 0:
            index = min(active)
            allocated[index] += 1
            accepted = 1
        remaining -= accepted
        active = {index for index in active if allocated[index] < capacities[index]}
    return allocated


def gradient_weights(rows: int, cols: int, compute_gradient: float, cache_gradient: float) -> tuple[list[float], list[float]]:
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")
    center = (rows - 1) / 2.0
    denominator = max(center, 1.0)
    compute: list[float] = []
    cache: list[float] = []
    for row in range(rows):
        edge = abs(row - center) / denominator
        for _ in range(cols):
            compute.append(1.0 + compute_gradient * edge)
            cache.append(1.0 + cache_gradient * (1.0 - edge))
    return compute, cache


def build_mapping_plan(
    *,
    policy: str,
    rows: int,
    cols: int,
    batch_size: int,
    topk: int,
    kv_entry_bytes: int,
    total_sram_bytes: int,
    cache_ratio: float,
    compute_gradient: float = 0.0,
    cache_gradient: float = 0.0,
    offline_gamma: float = 1.0,
    offline_score_us: float | None = None,
) -> MappingPlan:
    if policy not in {"B", "C", "S", "O"}:
        raise ValueError(f"Unsupported TP1 policy: {policy}")
    if batch_size <= 0 or topk <= 0 or kv_entry_bytes <= 0:
        raise ValueError("batch_size, topk, and kv_entry_bytes must be positive")
    if not 0.0 <= cache_ratio <= 1.0:
        raise ValueError("cache_ratio must be between zero and one")
    if policy in {"B", "C"} and (compute_gradient != 0.0 or cache_gradient != 0.0):
        raise ValueError(f"{policy} must use uniform mapping")
    if policy == "B":
        cache_ratio = 0.0

    compute_weights, cache_weights = gradient_weights(rows, cols, compute_gradient, cache_gradient)
    compute_quotas = _largest_remainder(batch_size * topk, compute_weights)
    per_core_sram = _largest_remainder(total_sram_bytes, [1.0] * (rows * cols))
    cache_budget_bytes = int(total_sram_bytes * cache_ratio)
    cache_entry_caps = [core_bytes // kv_entry_bytes for core_bytes in per_core_sram]
    total_cache_entries = min(cache_budget_bytes // kv_entry_bytes, sum(cache_entry_caps))
    cache_entries = _bounded_largest_remainder(
        total_cache_entries,
        cache_weights,
        cache_entry_caps,
    )
    compute_memory = [
        core_bytes - entries * kv_entry_bytes
        for core_bytes, entries in zip(per_core_sram, cache_entries)
    ]
    cores = tuple(
        CoreMapping(
            core_id=core_id,
            row=core_id // cols,
            col=core_id % cols,
            cp_group=core_id // cols,
            dp_group=core_id % cols,
            compute_quota_entries=compute_quotas[core_id],
            cache_capacity_entries=cache_entries[core_id],
            compute_memory_bytes=compute_memory[core_id],
        )
        for core_id in range(rows * cols)
    )
    plan = MappingPlan(
        policy=policy,
        rows=rows,
        cols=cols,
        batch_size=batch_size,
        topk=topk,
        kv_entry_bytes=kv_entry_bytes,
        total_sram_bytes=total_sram_bytes,
        cache_budget_bytes=cache_budget_bytes,
        compute_gradient=compute_gradient,
        cache_gradient=cache_gradient,
        offline_gamma=offline_gamma,
        offline_score_us=offline_score_us,
        cores=cores,
    )
    validate_mapping_plan(plan)
    return plan


def validate_mapping_plan(plan: MappingPlan) -> None:
    if len(plan.cores) != plan.num_cores:
        raise ValueError("mapping must contain every physical core")
    if [core.core_id for core in plan.cores] != list(range(plan.num_cores)):
        raise ValueError("core IDs must be contiguous and row-major")
    if plan.total_compute_entries != plan.batch_size * plan.topk:
        raise ValueError("per-core compute quotas do not conserve selected-KV work")
    if plan.total_cache_entries * plan.kv_entry_bytes > plan.cache_budget_bytes:
        raise ValueError("per-core cache capacities exceed the mapping cache budget")
    accounted = sum(
        core.compute_memory_bytes + core.cache_capacity_entries * plan.kv_entry_bytes
        for core in plan.cores
    )
    if accounted > plan.total_sram_bytes:
        raise ValueError("per-core compute and KV reservations exceed total SRAM")
    per_core_sram = _largest_remainder(plan.total_sram_bytes, [1.0] * plan.num_cores)
    for core, physical_bytes in zip(plan.cores, per_core_sram):
        if core.compute_memory_bytes + core.cache_capacity_entries * plan.kv_entry_bytes > physical_bytes:
            raise ValueError(f"core {core.core_id} exceeds its physical SRAM allocation")


def consumer_schedule(plan: MappingPlan, logical_accesses: int) -> tuple[int, ...]:
    quotas = _largest_remainder(
        logical_accesses,
        [float(core.compute_quota_entries) for core in plan.cores],
    )
    remaining = quotas[:]
    schedule: list[int] = []
    while len(schedule) < logical_accesses:
        progressed = False
        for core_id, count in enumerate(remaining):
            if count > 0:
                schedule.append(core_id)
                remaining[core_id] -= 1
                progressed = True
        if not progressed:
            raise RuntimeError("failed to construct a work-conserving consumer schedule")
    return tuple(schedule)
