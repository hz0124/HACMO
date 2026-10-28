#!/usr/bin/env python3
"""Energy model for HACMO DSE."""

from __future__ import annotations

from typing import Any

from area_model import ArchCase


def pj_to_j(value_pj: float) -> float:
    return value_pj * 1e-12


def estimate_energy(case: ArchCase, counters: dict[str, float], const: dict[str, Any]) -> dict[str, float]:
    e = const["energy"]
    sp = const["static_power"]
    area = const["area"]

    compute_j = pj_to_j(counters["flops_total"] * float(e["compute_pj_per_flop"]))
    sram_j = pj_to_j(
        8.0
        * (
            counters["sram_read_bytes"] * float(e["sram_read_pj_per_bit"])
            + counters["sram_write_bytes"] * float(e["sram_write_pj_per_bit"])
        )
    )
    noc_j = pj_to_j(8.0 * counters["noc_byte_hops"] * float(e["noc_pj_per_bit_per_hop"]))
    hbm_j = pj_to_j(
        8.0
        * (counters["hbm_read_bytes_actual"] + counters["hbm_write_bytes_actual"])
        * float(e["hbm_pj_per_bit"])
    )
    d2d_j = pj_to_j(8.0 * counters["d2d_byte_hops"] * float(e["d2d_pj_per_bit_per_hop"]))
    inter_elec_j = pj_to_j(
        8.0 * counters["inter_wafer_bytes_electrical"] * float(e["inter_wafer_electrical_pj_per_bit"])
    )
    inter_opt_j = pj_to_j(
        8.0 * counters["inter_wafer_bytes_optical"] * float(e["inter_wafer_optical_pj_per_bit"])
    )
    switch_j = pj_to_j(8.0 * counters["switch_bytes"] * float(e["switch_pj_per_bit"]))

    compute_per_core_ref = float(area["compute_ref_tflops_per_die"]) / float(area["n_core_ref"])
    die_peak_w = (
        float(sp["reference_die_peak_w"])
        * (case.num_cores / float(area["n_core_ref"]))
        * (case.compute_per_core_tflops / compute_per_core_ref)
    )
    die_peak_w = max(die_peak_w, float(sp["min_die_peak_w"]))
    static_power_w = (
        counters["active_die_count"] * die_peak_w * float(sp["die_static_power_ratio"])
        + counters["active_hbm_stack_count"] * float(sp["hbm_stack_idle_w"])
        + counters["active_switch_count"] * float(sp["switch_idle_w"])
    )
    static_j = counters["latency_s"] * static_power_w

    total_j = compute_j + sram_j + noc_j + hbm_j + d2d_j + inter_elec_j + inter_opt_j + switch_j + static_j
    tokens = max(counters["num_output_tokens"], 1.0)
    return {
        "energy_j": total_j,
        "energy_per_token_j": total_j / tokens,
        "energy_efficiency_tokens_per_j": tokens / max(total_j, 1e-30),
        "compute_energy_j": compute_j,
        "sram_energy_j": sram_j,
        "noc_energy_j": noc_j,
        "hbm_energy_j": hbm_j,
        "d2d_energy_j": d2d_j,
        "inter_wafer_energy_j": inter_elec_j + inter_opt_j,
        "switch_energy_j": switch_j,
        "static_energy_j": static_j,
        "die_peak_power_w": die_peak_w,
        "static_power_w": static_power_w,
    }
