#!/usr/bin/env python3
"""Shared wafer-core NoC movement model.

This module models movement across the many-core array mesh. It intentionally
does not model HBM DRAM service; callers should add a separate HBM backend or
service curve for the memory-controller side.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from .utils import avg_remote_hops, near_sram_hops


@dataclass(frozen=True)
class WaferCoreNoCConfig:
    rows: int
    cols: int
    base_noc_bw_tbps: float = 5.0
    gamma: float = 1.0
    noc_bw_scale_mode: str = "fixed"
    hbm_edges: tuple[str, ...] = ("top", "bottom")
    near_sram_hops: float = 0.0


@dataclass(frozen=True)
class WaferCoreNoCTraffic:
    hbm_bytes: float = 0.0
    near_sram_bytes: float = 0.0
    remote_sram_bytes: float = 0.0
    avg_hbm_hops: float | None = None
    near_sram_hops: float | None = None
    avg_remote_hops: float | None = None


@dataclass(frozen=True)
class WaferCoreNoCResult:
    hbm_us: float
    near_sram_us: float
    remote_sram_us: float
    total_us: float
    avg_hbm_hops: float
    near_sram_hops: float
    avg_remote_hops: float
    noc_bw_scale: float
    effective_noc_bw_tbps: float
    hbm_byte_hops: float
    near_sram_byte_hops: float
    remote_sram_byte_hops: float
    gamma: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _normalize_edges(edges: Iterable[str]) -> tuple[str, ...]:
    normalized = tuple(str(edge).lower() for edge in edges)
    if not normalized:
        raise ValueError("hbm_edges must contain at least one edge")
    valid = {"top", "bottom", "left", "right"}
    unknown = sorted(set(normalized) - valid)
    if unknown:
        raise ValueError(f"Unsupported HBM edge(s): {unknown}")
    return normalized


def avg_hbm_edge_hops(rows: int, cols: int, hbm_edges: Iterable[str] = ("top", "bottom")) -> float:
    """Average Manhattan distance from each core to its nearest HBM edge."""
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")
    edges = _normalize_edges(hbm_edges)
    total = 0.0
    for r in range(rows):
        for c in range(cols):
            distances: list[int] = []
            if "top" in edges:
                distances.append(r)
            if "bottom" in edges:
                distances.append(rows - 1 - r)
            if "left" in edges:
                distances.append(c)
            if "right" in edges:
                distances.append(cols - 1 - c)
            total += min(distances)
    return total / float(rows * cols)


def noc_bw_scale(rows: int, cols: int, mode: str) -> float:
    if mode == "fixed":
        return 1.0
    if mode == "cols":
        return max(cols / 20.0, 1.0)
    if mode == "sqrt_area":
        return max(((rows * cols) / 400.0) ** 0.5, 1.0)
    raise ValueError(f"Unsupported noc_bw_scale_mode: {mode}")


def estimate_wafer_core_noc(config: WaferCoreNoCConfig, traffic: WaferCoreNoCTraffic) -> WaferCoreNoCResult:
    if config.rows <= 0 or config.cols <= 0:
        raise ValueError("rows and cols must be positive")
    scale = noc_bw_scale(config.rows, config.cols, config.noc_bw_scale_mode)
    effective_noc_bw_tbps = config.base_noc_bw_tbps * scale
    noc_bw_bps = max(effective_noc_bw_tbps * 1e12, 1.0)

    hbm_hops = (
        float(traffic.avg_hbm_hops)
        if traffic.avg_hbm_hops is not None
        else avg_hbm_edge_hops(config.rows, config.cols, config.hbm_edges)
    )
    near_hops = (
        float(traffic.near_sram_hops)
        if traffic.near_sram_hops is not None
        else near_sram_hops(config.cols, config.near_sram_hops)
    )
    remote_hops = (
        float(traffic.avg_remote_hops)
        if traffic.avg_remote_hops is not None
        else avg_remote_hops(config.rows, config.cols)
    )

    hbm_byte_hops = max(float(traffic.hbm_bytes), 0.0) * hbm_hops
    near_byte_hops = max(float(traffic.near_sram_bytes), 0.0) * near_hops
    remote_byte_hops = max(float(traffic.remote_sram_bytes), 0.0) * remote_hops

    hbm_us = (hbm_byte_hops / noc_bw_bps) * config.gamma * 1e6
    near_us = (near_byte_hops / noc_bw_bps) * config.gamma * 1e6
    remote_us = (remote_byte_hops / noc_bw_bps) * config.gamma * 1e6

    return WaferCoreNoCResult(
        hbm_us=hbm_us,
        near_sram_us=near_us,
        remote_sram_us=remote_us,
        total_us=hbm_us + near_us + remote_us,
        avg_hbm_hops=hbm_hops,
        near_sram_hops=near_hops,
        avg_remote_hops=remote_hops,
        noc_bw_scale=scale,
        effective_noc_bw_tbps=effective_noc_bw_tbps,
        hbm_byte_hops=hbm_byte_hops,
        near_sram_byte_hops=near_byte_hops,
        remote_sram_byte_hops=remote_byte_hops,
        gamma=config.gamma,
    )
