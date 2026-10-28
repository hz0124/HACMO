#!/usr/bin/env python3
"""Candidate architecture generation for HACMO DSE."""

from __future__ import annotations

import math
from typing import Any

from area_model import ArchCase


def _base_kwargs(const: dict[str, Any]) -> dict:
    hw = const["hardware"]
    area = const["area"]
    side = int(hw["base_wafer_side"])
    return {
        "sram_per_core_mb": float(hw["base_sram_per_core_mb"]),
        "compute_per_core_tflops": float(hw["base_compute_per_core_tflops"]),
        "array_rows": int(hw["base_array_rows"]),
        "array_cols": int(hw["base_array_cols"]),
        "hbm_stacks_per_die": int(hw["base_hbm_stacks_per_die"]),
        "wafer_side": side,
        "num_wafers": int(int(hw["target_total_dies"]) / (side * side)),
        "wafer_diameter_mm": float(area["wafer_diameter_mm"]),
    }


def generate_cases(mode: str, const: dict[str, Any]) -> list[ArchCase]:
    if mode == "all":
        modes = ["sram_compute", "core_hbm", "wafer_scale", "batch_sweep", "edge_bandwidth"]
    elif mode == "pareto":
        modes = ["sram_compute", "core_hbm", "wafer_scale", "batch_sweep", "edge_bandwidth"]
    else:
        modes = [mode]
    out: list[ArchCase] = []
    seen: set[str] = set()
    for m in modes:
        for case in _generate_mode(m, const):
            if case.case_id not in seen:
                seen.add(case.case_id)
                out.append(case)
    return out


def _generate_mode(mode: str, const: dict[str, Any]) -> list[ArchCase]:
    if mode == "sram_compute":
        return sram_compute_cases(const)
    if mode == "core_hbm":
        return core_hbm_cases(const)
    if mode == "wafer_scale":
        return wafer_scale_cases(const)
    if mode == "edge_bandwidth":
        return edge_bandwidth_cases(const)
    if mode == "edge_sensitivity":
        return edge_sensitivity_cases(const)
    if mode == "batch_sweep":
        return batch_sweep_cases(const)
    raise ValueError(f"unsupported DSE mode: {mode}")


def sram_compute_cases(const: dict[str, Any]) -> list[ArchCase]:
    base = _base_kwargs(const)
    rows = 16
    items = [
        ("sram_small_compute_heavy", "1MB/1.024T", 1.0, 1.024, "compute-heavy"),
        ("balanced_low", "2MB/0.768T", 2.0, 0.768, "balanced-low"),
        ("balanced_mid", "4MB/0.512T", 4.0, 0.512, "balanced-mid"),
        ("sram_large", "8MB/0.256T", 8.0, 0.256, "sram-large"),
        ("sram_max", "16MB/0.128T", 16.0, 0.128, "sram-max"),
    ]
    return [
        ArchCase(
            case_id=f"sram_{key}",
            mode="sram_compute",
            label=label,
            sram_per_core_mb=sram,
            compute_per_core_tflops=compute,
            array_rows=rows,
            array_cols=rows,
            hbm_stacks_per_die=base["hbm_stacks_per_die"],
            wafer_side=base["wafer_side"],
            num_wafers=base["num_wafers"],
            wafer_diameter_mm=base["wafer_diameter_mm"],
            role=role,
        )
        for key, label, sram, compute, role in items
    ]


def core_hbm_cases(const: dict[str, Any]) -> list[ArchCase]:
    base = _base_kwargs(const)
    arrays = [16, 24, 32, 48, 64]
    hbms = [0, 2, 4, 8]
    out = []
    for arr in arrays:
        for hbm in hbms:
            out.append(
                ArchCase(
                    case_id=f"core_hbm_a{arr}_h{hbm}",
                    mode="core_hbm",
                    label=f"{arr}x{arr}/H{hbm}",
                    sram_per_core_mb=2.0,
                    compute_per_core_tflops=0.512,
                    array_rows=arr,
                    array_cols=arr,
                    hbm_stacks_per_die=hbm,
                    wafer_side=base["wafer_side"],
                    num_wafers=base["num_wafers"],
                    wafer_diameter_mm=base["wafer_diameter_mm"],
                    role="core-hbm-grid",
                )
            )
    return out


def wafer_scale_cases(const: dict[str, Any]) -> list[ArchCase]:
    hw = const["hardware"]
    area = const["area"]
    target = int(hw["target_total_dies"])
    out = []
    for side in [2, 4, 6, 8]:
        dies_per_wafer = side * side
        wafers = max(1, int(round(target / dies_per_wafer)))
        if side == 6:
            wafers = 7
        out.append(
            ArchCase(
                case_id=f"wafer_{wafers}w_{side}x{side}",
                mode="wafer_scale",
                label=f"{wafers}W {side}x{side}",
                sram_per_core_mb=2.0,
                compute_per_core_tflops=0.512,
                array_rows=16,
                array_cols=16,
                hbm_stacks_per_die=2,
                wafer_side=side,
                num_wafers=wafers,
                wafer_diameter_mm=float(area["wafer_diameter_mm"]),
                role="wafer-scale",
            )
        )
    return out


def edge_bandwidth_cases(const: dict[str, Any]) -> list[ArchCase]:
    """Rack-wafer external edge-port bandwidth sensitivity for a fixed shape."""
    base = _base_kwargs(const)
    side = int(base["wafer_side"])
    wafers = int(base["num_wafers"])
    mesh_bw = float(const["topology"]["mesh_bandwidth_gbps"])
    items = [
        ("ext100", "100G/2T", 100.0, "external-bandwidth-100"),
        ("ext250", "250G/2T", 250.0, "external-bandwidth-250"),
        ("base", "500G/2T", 500.0, "paper-16w4x4"),
        ("ext700", "700G/2T", 700.0, "external-bandwidth-700"),
        ("ext800", "800G/2T", 800.0, "external-bandwidth-800"),
        ("ext900", "900G/2T", 900.0, "external-bandwidth-900"),
    ]
    return [
        ArchCase(
            case_id=f"edge_bw_{key}",
            mode="edge_bandwidth",
            label=label,
            sram_per_core_mb=base["sram_per_core_mb"],
            compute_per_core_tflops=base["compute_per_core_tflops"],
            array_rows=base["array_rows"],
            array_cols=base["array_cols"],
            hbm_stacks_per_die=base["hbm_stacks_per_die"],
            wafer_side=side,
            num_wafers=wafers,
            wafer_diameter_mm=base["wafer_diameter_mm"],
            role=role,
            inter_wafer_bw_gbps=inter_bw,
            mesh_bw_gbps=mesh_bw,
        )
        for key, label, inter_bw, role in items
    ]


def edge_sensitivity_cases(const: dict[str, Any]) -> list[ArchCase]:
    """Dense diagnostic sweep over smaller edge bandwidths and switch latency."""
    base = _base_kwargs(const)
    side = 4
    wafers = 16
    mesh_bw = float(const["topology"]["mesh_bandwidth_gbps"])
    bandwidths = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 250.0, 500.0]
    latencies = [10.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2000.0]
    out: list[ArchCase] = []
    for bw in bandwidths:
        for lat in latencies:
            bw_tag = int(bw) if float(bw).is_integer() else bw
            lat_tag = int(lat) if float(lat).is_integer() else lat
            out.append(
                ArchCase(
                    case_id=f"edge_diag_b{int(round(bw)):03d}_l{int(round(lat)):03d}",
                    mode="edge_sensitivity",
                    label=f"{bw_tag}G/{lat_tag}ns",
                    sram_per_core_mb=base["sram_per_core_mb"],
                    compute_per_core_tflops=base["compute_per_core_tflops"],
                    array_rows=base["array_rows"],
                    array_cols=base["array_cols"],
                    hbm_stacks_per_die=base["hbm_stacks_per_die"],
                    wafer_side=side,
                    num_wafers=wafers,
                    wafer_diameter_mm=base["wafer_diameter_mm"],
                    role="edge-sensitivity",
                    inter_wafer_bw_gbps=bw,
                    mesh_bw_gbps=mesh_bw,
                    switch_latency_ns=lat,
                )
            )
    return out


def batch_sweep_cases(const: dict[str, Any]) -> list[ArchCase]:
    """Per-die batch sweep on the paper baseline shape (16W 4x4)."""
    base = _base_kwargs(const)
    batches = [16, 32, 64, 128, 256, 512]
    out: list[ArchCase] = []
    for batch in batches:
        out.append(
            ArchCase(
                case_id=f"batch_b{batch:04d}",
                mode="batch_sweep",
                label=f"B/die={batch}",
                sram_per_core_mb=base["sram_per_core_mb"],
                compute_per_core_tflops=base["compute_per_core_tflops"],
                array_rows=base["array_rows"],
                array_cols=base["array_cols"],
                hbm_stacks_per_die=base["hbm_stacks_per_die"],
                wafer_side=base["wafer_side"],
                num_wafers=base["num_wafers"],
                wafer_diameter_mm=base["wafer_diameter_mm"],
                role="batch-sweep",
                batch_size_override=batch,
            )
        )
    return out
