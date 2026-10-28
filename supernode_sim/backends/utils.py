#!/usr/bin/env python3
"""Shared backend math utilities."""

from __future__ import annotations


def avg_hbm_hops(rows: int, cols: int) -> float:
    return sum(min(r, rows - 1 - r) for r in range(rows)) / float(rows)


def avg_remote_hops(rows: int, cols: int) -> float:
    n = rows * cols
    if n <= 1:
        return 0.0
    exp_row = (rows * rows - 1) / (3.0 * rows)
    exp_col = (cols * cols - 1) / (3.0 * cols)
    return (exp_row + exp_col) * n / (n - 1)


def near_sram_hops(cols: int, override: float = 0.0) -> float:
    if override > 0:
        return override
    return max(1.0, min(4.0, cols / 16.0))


def compute_attention_flops(batch: int, context: int, hidden: int, n_heads: int, head_dim: int) -> float:
    flops_proj = 3.0 * 2.0 * batch * hidden * hidden
    flops_qk = 2.0 * 2.0 * batch * n_heads * head_dim * context
    flops_mix = 2.0 * batch * n_heads * head_dim * hidden
    return flops_proj + flops_qk + flops_mix
