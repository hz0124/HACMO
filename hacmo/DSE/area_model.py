#!/usr/bin/env python3
"""Area filtering for HACMO DSE cases."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ArchCase:
    case_id: str
    mode: str
    label: str
    sram_per_core_mb: float
    compute_per_core_tflops: float
    array_rows: int
    array_cols: int
    hbm_stacks_per_die: int
    wafer_side: int
    num_wafers: int
    wafer_diameter_mm: float
    role: str = ""
    inter_wafer_bw_gbps: float | None = None
    mesh_bw_gbps: float | None = None
    switch_latency_ns: float | None = None
    crossbar_latency_ns: float | None = None
    crossbar_beta: float = 0.5
    crossbar_rho: float = 0.4
    batch_size_override: int | None = None

    @property
    def num_cores(self) -> int:
        return int(self.array_rows * self.array_cols)

    @property
    def dies_per_wafer(self) -> int:
        return int(self.wafer_side * self.wafer_side)

    @property
    def active_die_count(self) -> int:
        return int(self.num_wafers * self.dies_per_wafer)

    @property
    def total_sram_mb(self) -> float:
        return float(self.num_cores * self.sram_per_core_mb)

    @property
    def compute_peak_tflops(self) -> float:
        return float(self.num_cores * self.compute_per_core_tflops)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.update(
            {
                "num_cores": self.num_cores,
                "dies_per_wafer": self.dies_per_wafer,
                "active_die_count": self.active_die_count,
                "total_sram_mb": self.total_sram_mb,
                "compute_peak_tflops": self.compute_peak_tflops,
            }
        )
        return d


def area_breakdown(case: ArchCase, const: dict[str, Any]) -> dict[str, float]:
    a = const["area"]
    hw = const["hardware"]
    n_ref = float(a["n_core_ref"])
    die_ref = float(a["die_area_ref_mm2"])
    compute_ref = float(a["compute_ref_tflops_per_die"])
    sram_ref = die_ref * float(a["sram_area_share"])
    compute_area_per_tflop = float(a["compute_logic_area_share"]) * die_ref / compute_ref
    sram_area_per_mb = sram_ref / 440.0
    noc_ctrl_area = float(a["noc_ctrl_area_share"]) * die_ref * math.sqrt(case.num_cores / n_ref)
    hbm_ctrl_area = 0.0

    compute_area = case.compute_peak_tflops * compute_area_per_tflop
    sram_area = case.total_sram_mb * sram_area_per_mb
    die_area = compute_area + sram_area + noc_ctrl_area + hbm_ctrl_area
    hbm_stack_area = float(a["hbm_stack_raw_area_mm2"]) * float(a["hbm_keepout_factor"])
    hbm_area_per_die = case.hbm_stacks_per_die * hbm_stack_area
    raw_wafer_area = math.pi * (case.wafer_diameter_mm / 2.0) ** 2
    usable_wafer_area = float(a["wafer_utilization_factor"]) * raw_wafer_area
    edge_xbar = float(a["edge_xbar_area_mm2"])
    edge_io = float(a["edge_io_area_fraction_of_raw_wafer"]) * raw_wafer_area
    required = case.dies_per_wafer * (die_area + hbm_area_per_die) + edge_xbar + edge_io
    hbm_capacity_gb = case.hbm_stacks_per_die * float(hw["hbm_capacity_per_stack_gb"])

    return {
        "compute_area_mm2": compute_area,
        "sram_area_mm2": sram_area,
        "noc_ctrl_area_mm2": noc_ctrl_area,
        "die_area_mm2": die_area,
        "hbm_area_per_die_mm2": hbm_area_per_die,
        "edge_xbar_area_mm2": edge_xbar,
        "edge_io_area_mm2": edge_io,
        "wafer_required_area_mm2": required,
        "wafer_budget_area_mm2": usable_wafer_area,
        "wafer_area_utilization": required / max(usable_wafer_area, 1e-9),
        "wafer_area_headroom": usable_wafer_area - required,
        "wafer_area_margin_mm2": usable_wafer_area - required,
        "system_required_area_mm2": required * case.num_wafers,
        "hbm_capacity_gb_per_die": hbm_capacity_gb,
    }


def validate_case(case: ArchCase, const: dict[str, Any]) -> tuple[bool, str, dict[str, float]]:
    b = area_breakdown(case, const)
    reasons: list[str] = []
    if b["wafer_required_area_mm2"] > b["wafer_budget_area_mm2"]:
        reasons.append("wafer_area_exceeded")
    min_hbm = float(const["hardware"]["min_hbm_capacity_gb_per_die"])
    if b["hbm_capacity_gb_per_die"] < min_hbm:
        reasons.append("insufficient_hbm_capacity")
    if case.hbm_stacks_per_die <= 0:
        reasons.append("no_local_hbm_bandwidth")
    return (not reasons), "|".join(reasons), b
