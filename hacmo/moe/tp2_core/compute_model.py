#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List


@dataclass(frozen=True)
class H200RooflineConfig:
    hidden_size: int = 7168
    expert_intermediate_size: int = 2048
    effective_tflops: float = 1979.0
    hbm_bw_tbps: float = 4.8
    weight_bytes_per_elem: int = 1
    activation_bytes_per_elem: int = 2

    @property
    def expert_ffn_flops_per_token(self) -> float:
        # Gate, up and down projections. Each matmul multiply-add is 2 FLOPs.
        return float(6 * self.hidden_size * self.expert_intermediate_size)

    @property
    def expert_weight_bytes(self) -> float:
        return float(3 * self.hidden_size * self.expert_intermediate_size * self.weight_bytes_per_elem)

    @property
    def activation_bytes_per_expert_token(self) -> float:
        # Read input, write output and account for a small intermediate stream.
        return float(3 * self.hidden_size * self.activation_bytes_per_elem)

    def to_dict(self) -> Dict[str, float]:
        d = asdict(self)
        d["expert_ffn_flops_per_token"] = self.expert_ffn_flops_per_token
        d["expert_weight_bytes"] = self.expert_weight_bytes
        d["activation_bytes_per_expert_token"] = self.activation_bytes_per_expert_token
        return d


def compute_rank_roofline_us(
    expert_counts: Dict[str, int],
    mapping: Dict[int, List[int]],
    batch_scale: float,
    num_ranks: int,
    cfg: H200RooflineConfig,
    route_aware: bool = False,
    dispatch_matrix: Dict | None = None,
    bytes_per_token: float = 7168.0,
) -> Dict:
    """Compute per-rank expert compute time using a roofline model.

    If route_aware=True and dispatch_matrix is provided, uses the actual
    dispatch traffic to determine how many tokens each rank processes.
    This is consistent with the nearest-replica routing used for communication:
    tokens are only dispatched to the nearest replica, so only that replica
    die actually does the compute.

    If route_aware=False (legacy), distributes tokens uniformly across all
    replica dies for each expert.
    """
    rank_expert_tokens: List[Dict[int, float]] = [{} for _ in range(num_ranks)]

    if route_aware and dispatch_matrix is not None and bytes_per_token > 0:
        # Route-aware: derive token counts from actual dispatch traffic.
        # dispatch_matrix[src][dst] = bytes = token_count * batch_scale * bytes_per_token
        # We want scaled_token_count = dispatch_bytes / bytes_per_token
        # (batch_scale is already embedded in dispatch bytes).
        for src, dsts in dispatch_matrix.items():
            for dst, byte_vol in dsts.items():
                tokens = float(byte_vol) / max(bytes_per_token, 1e-9)
                if dst < 0 or dst >= num_ranks:
                    continue
                rank_expert_tokens[dst][0] = rank_expert_tokens[dst].get(0, 0.0) + tokens
    else:
        # Legacy: uniform distribution across replicas.
        for e_str, raw_count in expert_counts.items():
            e = int(e_str)
            ds = mapping.get(e, [])
            if not ds:
                continue
            replicas = max(len(ds), 1)
            tokens_per_replica = float(raw_count) * batch_scale / replicas
            for d in ds:
                if d < 0 or d >= num_ranks:
                    continue
                rank_expert_tokens[d][e] = rank_expert_tokens[d].get(e, 0.0) + tokens_per_replica

    return compute_rank_roofline_from_rank_expert_tokens(rank_expert_tokens, cfg)


def compute_rank_roofline_from_rank_tokens(rank_expert_tokens: List[float], cfg: H200RooflineConfig) -> Dict:
    """Compute per-rank roofline time from already-routed expert-token counts.

    ``rank_expert_tokens`` must include both remote-dispatched and local tokens.
    This is the preferred interface for route-aware simulators because a sparse
    P2P dispatch matrix intentionally omits local ``src == dst`` traffic.
    """
    return compute_rank_roofline_from_rank_expert_tokens(
        [{0: float(tokens)} if float(tokens) > 0 else {} for tokens in rank_expert_tokens],
        cfg,
    )


def compute_rank_roofline_from_rank_expert_tokens(rank_expert_tokens: List[Dict[int, float]], cfg: H200RooflineConfig) -> Dict:
    """Compute per-rank roofline time from routed token counts by expert.

    Each active expert's weights are charged once per rank-local expert batch.
    FLOPs and activation traffic still scale with routed token count.
    """
    num_ranks = len(rank_expert_tokens)
    rank_total_tokens = [sum(float(v) for v in expert_tokens.values()) for expert_tokens in rank_expert_tokens]
    rank_active_experts = [sum(1 for v in expert_tokens.values() if float(v) > 0) for expert_tokens in rank_expert_tokens]
    rank_flops = [tokens * cfg.expert_ffn_flops_per_token for tokens in rank_total_tokens]
    rank_bytes = [
        active_experts * cfg.expert_weight_bytes + tokens * cfg.activation_bytes_per_expert_token
        for tokens, active_experts in zip(rank_total_tokens, rank_active_experts)
    ]
    tflops_per_us = cfg.effective_tflops * 1e6
    bytes_per_us = cfg.hbm_bw_tbps * 1e6
    rank_compute_us = [f / max(tflops_per_us, 1e-9) for f in rank_flops]
    rank_memory_us = [b / max(bytes_per_us, 1e-9) for b in rank_bytes]
    rank_total_us = [max(c, m) for c, m in zip(rank_compute_us, rank_memory_us)]
    max_rank = max(range(num_ranks), key=lambda r: rank_total_us[r]) if num_ranks > 0 else 0
    return {
        "rank_compute_us": rank_compute_us,
        "rank_memory_us": rank_memory_us,
        "rank_total_us": rank_total_us,
        "rank_flops": rank_flops,
        "rank_bytes": rank_bytes,
        "rank_expert_tokens": rank_total_tokens,
        "rank_expert_token_breakdown": rank_expert_tokens,
        "rank_active_experts": rank_active_experts,
        "max_rank": int(max_rank),
        "max_rank_expert_tokens": float(rank_total_tokens[max_rank]) if num_ranks > 0 else 0.0,
        "max_rank_active_experts": int(rank_active_experts[max_rank]) if num_ranks > 0 else 0,
        "max_rank_flops": float(rank_flops[max_rank]) if num_ranks > 0 else 0.0,
        "max_rank_bytes": float(rank_bytes[max_rank]) if num_ranks > 0 else 0.0,
        "max_rank_compute_us": float(rank_compute_us[max_rank]) if num_ranks > 0 else 0.0,
        "max_rank_memory_us": float(rank_memory_us[max_rank]) if num_ranks > 0 else 0.0,
        "max_rank_total_us": float(rank_total_us[max_rank]) if num_ranks > 0 else 0.0,
    }
