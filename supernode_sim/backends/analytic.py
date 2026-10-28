#!/usr/bin/env python3
"""Analytic backend for SupernodeSim v0.

This backend is a fast reference model. It is not the final evidence source;
it validates the public API and gives cross-check values while Astra/Ramulator
backends are brought online.
"""

from __future__ import annotations

from .utils import compute_attention_flops
from .wafer_core_noc import WaferCoreNoCConfig, WaferCoreNoCTraffic, estimate_wafer_core_noc
from ..types import SimulationConfig, SimulationResult


def _hit_breakdown(config: SimulationConfig) -> tuple[float, float, float, float]:
    """Return hbm/local/near/remote fractions for selected KV bytes."""
    policy = config.mapping.cache_policy
    if policy == "none":
        hit = 0.0
    elif policy == "naive_all_layer":
        hit = 0.02
    elif policy == "static_only":
        hit = 0.02
    elif policy == "layer_prefetch":
        hit = 0.90
    else:
        hit = 0.0

    hbm = 1.0 - hit
    if config.mapping.dynamic_prefetch:
        local = hit * 0.75
        near = hit * 0.20
        remote = hit * 0.05
    else:
        local = hit * 0.30
        near = 0.0
        remote = hit * 0.70
    return hbm, local, near, remote


def simulate(config: SimulationConfig) -> SimulationResult:
    hw = config.hardware
    wl = config.workload
    par = config.parallel

    selected_bytes = 2.0 * wl.batch_size * wl.selected_vector_bytes * wl.topk
    hbm_frac, local_frac, near_frac, remote_frac = _hit_breakdown(config)
    hbm_bytes = selected_bytes * hbm_frac
    local_bytes = selected_bytes * local_frac
    near_bytes = selected_bytes * near_frac
    remote_bytes = selected_bytes * remote_frac

    flops = compute_attention_flops(wl.batch_size, wl.context_len, wl.hidden_size, wl.n_heads, wl.head_dim)
    compute_efficiency = 1.0
    if par.batch_coalesce:
        compute_efficiency *= 1.10
    if par.horizontal_strategy == "tp":
        compute_efficiency *= min(1.20, 1.0 + 0.04 * max(par.tp_degree_horizontal - 1, 0))
    compute_us = flops / max(hw.compute_peak_tflops * 1e12 * compute_efficiency, 1.0) * 1e6

    hbm_bw_bps = hw.hbm_peak_tbps * 1e12 * 0.21
    hbm_us = hbm_bytes / max(hbm_bw_bps, 1.0) * 1e6
    noc = estimate_wafer_core_noc(
        WaferCoreNoCConfig(
            rows=hw.array_rows,
            cols=hw.array_cols,
            base_noc_bw_tbps=hw.noc_bw_tbps,
            gamma=1.0,
            noc_bw_scale_mode="fixed",
            hbm_edges=tuple(hw.hbm_edges),
            near_sram_hops=hw.near_sram_hops,
        ),
        WaferCoreNoCTraffic(
            hbm_bytes=hbm_bytes,
            near_sram_bytes=near_bytes,
            remote_sram_bytes=remote_bytes,
        ),
    )
    hbm_noc_us = noc.hbm_us
    sram_noc_us = noc.near_sram_us + noc.remote_sram_us

    horizontal_bytes = 0.0
    if par.horizontal_strategy == "tp":
        horizontal_bytes += selected_bytes * (par.tp_degree_horizontal - 1) / max(par.tp_degree_horizontal, 1)
    if par.owner_compute:
        horizontal_bytes += selected_bytes * 0.25
    vertical_bytes = selected_bytes * (par.cp_degree - 1) / max(par.cp_degree, 1)
    communication_us = (horizontal_bytes + vertical_bytes) / max(hw.noc_bw_tbps * 1e12, 1.0) * 1e6

    prefetch_time_us = (local_bytes + near_bytes + remote_bytes) / max(hw.hbm_peak_tbps * 1e12 * 0.21, 1.0) * 1e6
    prefetch_exposed_us = max(prefetch_time_us - config.mapping.prefetch_overlap_us, 0.0) if config.mapping.dynamic_prefetch else 0.0

    total = compute_us + hbm_us + hbm_noc_us + sram_noc_us + communication_us + prefetch_exposed_us
    return SimulationResult(
        backend="analytic",
        total_latency_us=total,
        compute_us=compute_us,
        hbm_us=hbm_us + hbm_noc_us,
        sram_noc_us=sram_noc_us,
        communication_us=communication_us,
        prefetch_exposed_us=prefetch_exposed_us,
        hbm_bytes=hbm_bytes,
        sram_local_bytes=local_bytes,
        sram_near_bytes=near_bytes,
        sram_remote_bytes=remote_bytes,
        horizontal_link_bytes=horizontal_bytes,
        vertical_link_bytes=vertical_bytes,
        metadata={
            "hbm_fraction": hbm_frac,
            "local_fraction": local_frac,
            "near_fraction": near_frac,
            "remote_fraction": remote_frac,
            "wafer_core_noc": noc.to_dict(),
            "horizontal_strategy": par.horizontal_strategy,
        },
    )
