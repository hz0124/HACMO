#!/usr/bin/env python3
"""Die-level MoE trace planning for the upper switch-mesh supernode levels."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import yaml

from ..types import SimulationConfig


def _zipf_dist(n: int, alpha: float = 1.2) -> list[float]:
    raw = [1.0 / ((i + 1) ** alpha) for i in range(n)]
    s = sum(raw) or 1.0
    return [v / s for v in raw]


def _load_freq(config: SimulationConfig, repo_root: Path) -> list[float]:
    if config.moe.real_freq_json:
        p = Path(config.moe.real_freq_json)
        if not p.is_absolute():
            p = repo_root / p
        raw = json.loads(p.read_text(encoding="utf-8"))
        if len(raw) != config.moe.num_experts:
            raise ValueError(
                f"real_freq_json has {len(raw)} entries, expected {config.moe.num_experts}"
            )
        s = sum(float(x) for x in raw) or 1.0
        return [float(x) / s for x in raw]
    return _zipf_dist(config.moe.num_experts)


def _biased_dist(
    base: list[float], src_wafer: int, config: SimulationConfig
) -> list[float]:
    factor = config.moe.locality_factor
    if factor <= 0:
        return base
    biased = base[:]
    nw = config.supernode.num_wafers
    shift = config.moe.locality_wafer_shift
    for e in range(config.moe.num_experts):
        natural_w = (e * 7) % nw
        if (natural_w + shift) % nw == src_wafer:
            biased[e] *= 1.0 + factor
    s = sum(biased) or 1.0
    return [x / s for x in biased]


def _sample_topk(dist: list[float], topk: int, rng: random.Random) -> list[int]:
    """Sample topk distinct experts from a probability distribution.

    Uses sampling-without-replacement: each selected expert is removed
    from the candidate pool before the next draw, preventing the same
    expert from being picked twice (matching MoE gating semantics).
    """
    ids = list(range(len(dist)))
    remaining_dist = list(dist)
    out = []
    for _ in range(topk):
        if not ids:
            break
        s = sum(remaining_dist) or 1.0
        x = rng.random() * s
        acc = 0.0
        pick = ids[-1]
        pick_idx = len(ids) - 1
        for idx, i in enumerate(ids):
            acc += remaining_dist[idx]
            if x <= acc:
                pick = i
                pick_idx = idx
                break
        out.append(pick)
        ids.pop(pick_idx)
        remaining_dist.pop(pick_idx)
    return out


def build_window_counts(
    config: SimulationConfig, repo_root: Path
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(config.moe.seed)
    base = _load_freq(config, repo_root)
    num_dies = config.supernode.num_wafers * config.supernode.dies_per_wafer
    window_size = max(config.moe.window_size, 1)
    num_windows = (config.workload.decode_len + window_size - 1) // window_size
    window_stats = []
    window_src_expert = []

    for w in range(num_windows):
        expert_counts: Counter[int] = Counter()
        tokens_per_die: Counter[int] = Counter()
        src_expert = {u: Counter() for u in range(num_dies)}
        st = w * window_size
        ed = min(config.workload.decode_len, (w + 1) * window_size)
        for req in range(config.moe.num_requests):
            src = _source_die(req, config.moe.num_requests, num_dies)
            src_w = src // config.supernode.dies_per_wafer
            dist = _biased_dist(base, src_w, config)
            for _tok in range(st, ed):
                eids = _sample_topk(dist, config.moe.topk, rng)
                tokens_per_die[src] += 1
                for e in eids:
                    expert_counts[e] += 1
                    src_expert[src][e] += 1
        window_stats.append(
            {
                "window_id": w,
                "start_token_idx": st + 1,
                "end_token_idx": ed,
                "expert_counts": {str(k): int(v) for k, v in expert_counts.items()},
                "tokens_per_die": {str(k): int(v) for k, v in tokens_per_die.items()},
            }
        )
        window_src_expert.append(
            {
                "window_id": w,
                "src_expert_counts": {
                    str(u): {str(k): int(v) for k, v in c.items()}
                    for u, c in src_expert.items()
                    if c
                },
            }
        )
    return window_stats, window_src_expert


def _source_die(req_idx: int, num_requests: int, num_dies: int) -> int:
    """Spread sampled requests over all dies when requests are sparse."""
    if num_dies <= 0:
        return 0
    if num_requests <= 0:
        return req_idx % num_dies
    if num_requests < num_dies:
        return min((req_idx * num_dies) // num_requests, num_dies - 1)
    return req_idx % num_dies


def solve_mapping(
    config: SimulationConfig, window_src_expert: list[dict]
) -> dict[int, list[int]]:
    num_dies = config.supernode.num_wafers * config.supernode.dies_per_wafer
    dpw = config.supernode.dies_per_wafer
    method = _canonical_method(config.moe.mapping_method)
    if method == "vllm":
        return {e: [e % num_dies] for e in range(config.moe.num_experts)}

    demand: dict[int, Counter[int]] = {
        e: Counter() for e in range(config.moe.num_experts)
    }
    total: Counter[int] = Counter()
    for wdata in window_src_expert:
        for src_str, e_counts in wdata["src_expert_counts"].items():
            sw = int(src_str) // dpw
            for e_str, c in e_counts.items():
                e = int(e_str)
                demand[e][sw] += int(c)
                total[e] += int(c)

    wafer_slots: dict[int, list[int]] = {
        w: [] for w in range(config.supernode.num_wafers)
    }
    for e in sorted(
        range(config.moe.num_experts), key=lambda x: total[x], reverse=True
    ):
        candidates = [w for w, _ in demand[e].most_common()]
        candidates.extend(
            w for w in range(config.supernode.num_wafers) if w not in candidates
        )
        for w in candidates:
            if len(wafer_slots[w]) < dpw:
                wafer_slots[w].append(e)
                break

    mapping: dict[int, list[int]] = {}
    for w, experts in wafer_slots.items():
        if method in {"ours_static", "ours_dynamic"}:
            locals_order = _edge_first_locals(dpw)
        elif method == "moentwine_static":
            locals_order = list(range(dpw))
        else:
            raise ValueError(f"Unsupported moe.mapping_method: {method}")
        for e, local in zip(experts, locals_order):
            mapping[e] = [w * dpw + local]
    return mapping


def _canonical_method(method: str) -> str:
    aliases = {
        "uniform": "vllm",
        "vllm": "vllm",
        "wafer_local": "moentwine_static",
        "moentwine": "moentwine_static",
        "moentwine_static": "moentwine_static",
        "edge_aware": "ours_static",
        "ours_static": "ours_static",
        "ours_dynamic": "ours_dynamic",
    }
    return aliases.get(method, method)


def _edge_first_locals(dpw: int) -> list[int]:
    side = int(dpw**0.5)
    if side * side != dpw:
        return list(range(dpw))
    return sorted(
        range(dpw),
        key=lambda x: min(
            x // side, x % side, side - 1 - x // side, side - 1 - x % side
        ),
    )


def build_window_mappings(
    config: SimulationConfig,
    window_stats: list[dict],
    window_src_expert: list[dict],
    static_mapping: dict[int, list[int]],
) -> list[dict[int, list[int]]]:
    """Return per-window mappings.

    Static methods reuse one mapping. `ours_dynamic` starts from `ours_static`
    and adds bounded hot-expert replicas on demand-heavy remote wafers.
    """
    if _canonical_method(config.moe.mapping_method) != "ours_dynamic":
        return [static_mapping for _ in window_stats]

    dpw = config.supernode.dies_per_wafer
    max_replicas = 3
    hot_count = max(1, config.moe.num_experts // 10)
    edge_locals = _edge_first_locals(dpw)
    current = {e: list(ds) for e, ds in static_mapping.items()}
    out = []
    for ws, wsrc in zip(window_stats, window_src_expert):
        expert_counts = {int(k): int(v) for k, v in ws.get("expert_counts", {}).items()}
        hot_experts = [
            e
            for e, _ in sorted(expert_counts.items(), key=lambda x: x[1], reverse=True)[
                :hot_count
            ]
        ]
        src_expert_counts = wsrc.get("src_expert_counts", {})
        die_load = _die_compute_load(config, expert_counts, current)
        for e in hot_experts:
            replicas = current.get(e, [])
            old_replicas = list(replicas)
            if len(replicas) >= max_replicas:
                continue
            covered = {d // dpw for d in replicas}
            wafer_demand: dict[int, int] = {}
            for src_str, e_counts in src_expert_counts.items():
                cnt = int(e_counts.get(str(e), 0))
                if cnt <= 0:
                    continue
                wafer = int(src_str) // dpw
                if wafer not in covered:
                    wafer_demand[wafer] = wafer_demand.get(wafer, 0) + cnt
            if not wafer_demand:
                continue
            target_w = max(wafer_demand.items(), key=lambda x: x[1])[0]
            occupied = set(replicas)
            expert_work = (
                float(expert_counts.get(e, 0))
                * config.workload.batch_size
                / max(config.moe.num_requests, 1)
            )
            if expert_work <= 0:
                continue
            old_share = expert_work / max(len(old_replicas), 1)
            new_share = expert_work / (len(old_replicas) + 1)
            current_max = max(die_load.values()) if die_load else 0.0
            new_die = None
            best_max = current_max
            for local in edge_locals:
                cand = target_w * dpw + local
                if cand not in occupied:
                    projected = dict(die_load)
                    for d in replicas:
                        projected[d] = projected.get(d, 0.0) - old_share + new_share
                    projected[cand] = projected.get(cand, 0.0) + new_share
                    projected_max = max(projected.values()) if projected else 0.0
                    if projected_max < best_max:
                        best_max = projected_max
                        new_die = cand
            if new_die is not None:
                for d in old_replicas:
                    die_load[d] = die_load.get(d, 0.0) - old_share + new_share
                die_load[new_die] = die_load.get(new_die, 0.0) + new_share
                current[e].append(new_die)
        out.append({e: list(ds) for e, ds in current.items()})
    return out


def _die_compute_load(
    config: SimulationConfig,
    expert_counts: dict[int, int],
    mapping: dict[int, list[int]],
) -> dict[int, float]:
    batch_scale = config.workload.batch_size / max(config.moe.num_requests, 1)
    out: dict[int, float] = {}
    for e, ds in mapping.items():
        if not ds:
            continue
        share = float(expert_counts.get(e, 0)) * batch_scale / len(ds)
        for d in ds:
            out[d] = out.get(d, 0.0) + share
    return out


def build_dispatch_matrices(
    config: SimulationConfig,
    window_src_expert: list[dict],
    mapping: dict[int, list[int]],
    window_mappings: list[dict[int, list[int]]] | None = None,
) -> list[dict[int, dict[int, int]]]:
    bytes_per_token = dispatch_bytes_per_token(config)
    batch_scale = config.workload.batch_size / max(config.moe.num_requests, 1)
    matrices = []
    for w_idx, wdata in enumerate(window_src_expert):
        cur_mapping = window_mappings[w_idx] if window_mappings is not None else mapping
        mat: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        for src_str, e_counts in wdata["src_expert_counts"].items():
            src = int(src_str)
            for e_str, c in e_counts.items():
                replicas = cur_mapping[int(e_str)]
                dst = min(replicas, key=lambda d: _approx_route_cost_ns(config, src, d))
                if src != dst:
                    size = int(round(int(c) * batch_scale * bytes_per_token))
                    if size > 0:
                        mat[src][dst] += size
        matrices.append({s: dict(dsts) for s, dsts in mat.items()})
    return matrices


def build_combine_matrices(
    config: SimulationConfig,
    dispatch_matrices: list[dict[int, dict[int, int]]],
) -> list[dict[int, dict[int, int]]]:
    dispatch_bpt = max(dispatch_bytes_per_token(config), 1)
    combine_bpt = combine_bytes_per_token(config)
    out = []
    for mat in dispatch_matrices:
        comb: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        for src, dsts in mat.items():
            for dst, b in dsts.items():
                comb[dst][src] += int(round(b * combine_bpt / dispatch_bpt))
        out.append({s: dict(dsts) for s, dsts in comb.items()})
    return out


def route_plan_to_simulator_inputs(
    route_plan: dict,
    *,
    num_ranks: int,
) -> tuple[
    list[dict],
    list[dict[int, dict[int, int]]],
    list[dict[int, dict[int, int]]],
    list[list[dict[int, float]]],
]:
    """Convert an authoritative route plan without making policy decisions."""
    if float(route_plan.get("metadata", {}).get("batch_scale", 1.0)) != 1.0:
        raise ValueError(
            "route plan must contain already-expanded traffic with batch_scale=1"
        )
    window_stats: list[dict] = []
    dispatch: list[dict[int, dict[int, int]]] = []
    combine: list[dict[int, dict[int, int]]] = []
    rank_expert_steps: list[list[dict[int, float]]] = []
    for index, step in enumerate(route_plan.get("steps", [])):
        validation = step.get("validation", {})
        if not validation.get("all_activations_routed", False):
            raise ValueError(f"route-plan step {index} does not preserve activations")
        if not validation.get("compute_matches_assignments", False):
            raise ValueError(
                f"route-plan step {index} compute assignment is inconsistent"
            )

        expert_counts: Counter[int] = Counter()
        for assignment in step.get("assignments", []):
            expert_counts[int(assignment["expert"])] += int(assignment["activations"])
        window_stats.append(
            {
                "window_id": int(step.get("window_id", index)),
                "step_id": int(step.get("step_id", index)),
                "start_token_idx": int(step.get("token_idx", index + 1)),
                "end_token_idx": int(step.get("token_idx", index + 1)),
                "tokens_per_die": {
                    str(src): int(count)
                    for src, count in step.get("tokens_per_die", {}).items()
                },
                "expert_counts": {
                    str(expert): int(count)
                    for expert, count in sorted(expert_counts.items())
                },
                "active": bool(step.get("active", True)),
            }
        )
        dispatch.append(
            {
                int(src): {int(dst): int(nbytes) for dst, nbytes in dsts.items()}
                for src, dsts in step.get("dispatch_bytes", {}).items()
            }
        )
        combine.append(
            {
                int(src): {int(dst): int(nbytes) for dst, nbytes in dsts.items()}
                for src, dsts in step.get("combine_bytes", {}).items()
            }
        )
        rank_experts: list[dict[int, float]] = [{} for _ in range(num_ranks)]
        for die_str, counts in step.get("rank_expert_tokens", {}).items():
            die = int(die_str)
            if not 0 <= die < num_ranks:
                raise ValueError(
                    f"route-plan step {index} references invalid rank {die}"
                )
            rank_experts[die] = {
                int(expert): float(count) for expert, count in counts.items()
            }
        rank_expert_steps.append(rank_experts)
    if not window_stats:
        raise ValueError("route plan has no steps")
    return window_stats, dispatch, combine, rank_expert_steps


def build_shared_rank_compute_us(
    config: SimulationConfig,
    window_stats: list[dict],
) -> list[list[float]]:
    """Compute source-local shared-expert work for every decode step."""
    num_ranks = config.supernode.num_wafers * config.supernode.dies_per_wafer
    out: list[list[float]] = []
    for step in window_stats:
        per_rank = []
        tokens_per_die = step.get("tokens_per_die", {})
        for rank in range(num_ranks):
            tokens = float(tokens_per_die.get(str(rank), tokens_per_die.get(rank, 0.0)))
            per_rank.append(
                _compute_us_for_rank_experts(config, {-1: tokens})
                if tokens > 0
                else 0.0
            )
        out.append(per_rank)
    return out


def build_rank_compute_tokens(
    config: SimulationConfig,
    window_src_expert: list[dict],
    mapping: dict[int, list[int]],
    window_mappings: list[dict[int, list[int]]] | None = None,
) -> list[list[float]]:
    """Return batch-scaled expert-token counts processed by each rank.

    Sparse dispatch matrices omit local ``src == dst`` transfers.  Compute
    injection must still charge those tokens to the selected expert die, so
    this helper follows the same nearest-replica routing as
    ``build_dispatch_matrices`` and records token work directly.
    """
    rank_expert_tokens = build_rank_expert_compute_tokens(
        config,
        window_src_expert,
        mapping,
        window_mappings=window_mappings,
    )
    return [
        [sum(expert_tokens.values()) for expert_tokens in window_tokens]
        for window_tokens in rank_expert_tokens
    ]


def build_rank_expert_compute_tokens(
    config: SimulationConfig,
    window_src_expert: list[dict],
    mapping: dict[int, list[int]],
    window_mappings: list[dict[int, list[int]]] | None = None,
) -> list[list[dict[int, float]]]:
    """Return batch-scaled per-rank, per-expert token counts.

    Expert GEMMs reuse weights across the token batch for each active expert.
    Keeping the expert dimension prevents the compute model from charging one
    expert's weights once per token.
    """
    num_ranks = config.supernode.num_wafers * config.supernode.dies_per_wafer
    batch_scale = config.workload.batch_size / max(config.moe.num_requests, 1)
    out: list[list[dict[int, float]]] = []
    for w_idx, wdata in enumerate(window_src_expert):
        cur_mapping = window_mappings[w_idx] if window_mappings is not None else mapping
        rank_expert_tokens: list[dict[int, float]] = [
            defaultdict(float) for _ in range(num_ranks)
        ]
        for src_str, e_counts in wdata["src_expert_counts"].items():
            src = int(src_str)
            for e_str, c in e_counts.items():
                expert = int(e_str)
                replicas = cur_mapping[expert]
                dst = min(replicas, key=lambda d: _approx_route_cost_ns(config, src, d))
                if 0 <= dst < num_ranks:
                    rank_expert_tokens[dst][expert] += float(c) * batch_scale
        out.append([dict(expert_tokens) for expert_tokens in rank_expert_tokens])
    return out


def build_rank_compute_us(
    config: SimulationConfig,
    rank_compute_tokens: list[list[float]],
    rank_expert_compute_tokens: list[list[dict[int, float]]] | None = None,
) -> list[list[float]]:
    if rank_expert_compute_tokens is not None:
        return [
            [
                _compute_us_for_rank_experts(config, expert_tokens)
                for expert_tokens in window_tokens
            ]
            for window_tokens in rank_expert_compute_tokens
        ]
    return [
        [_compute_us_for_tokens(config, tokens) for tokens in window_tokens]
        for window_tokens in rank_compute_tokens
    ]


def dispatch_bytes_per_token(config: SimulationConfig) -> int:
    elem = config.moe.dispatch_bytes_per_hidden_elem or config.moe.bytes_per_hidden_elem
    return int(config.moe.hidden_size * elem)


def combine_bytes_per_token(config: SimulationConfig) -> int:
    elem = config.moe.combine_bytes_per_hidden_elem or config.moe.bytes_per_hidden_elem
    return int(config.moe.hidden_size * elem)


def _compute_us_for_tokens(config: SimulationConfig, tokens: float) -> float:
    if tokens <= 0:
        return 0.0
    if config.moe.compute_backend == "constant_per_token":
        return float(tokens) * config.moe.expert_compute_us_per_token
    if config.moe.compute_backend == "roofline_h200":
        flops_per_token = _expert_ffn_flops_per_token(config)
        weight_bytes_per_token = _expert_weight_bytes(config)
        activation_bytes_per_token = _activation_bytes_per_expert_token(config)
        flops = float(tokens) * flops_per_token
        bytes_ = float(tokens) * (weight_bytes_per_token + activation_bytes_per_token)
        compute_us = flops / max(config.moe.h200_effective_tflops * 1e6, 1e-9)
        memory_us = bytes_ / max(config.moe.h200_hbm_bw_tbps * 1e6, 1e-9)
        return max(compute_us, memory_us)
    raise ValueError(f"Unsupported moe.compute_backend: {config.moe.compute_backend}")


def _compute_us_for_rank_experts(
    config: SimulationConfig, expert_tokens: dict[int, float]
) -> float:
    active = {
        e: float(tokens) for e, tokens in expert_tokens.items() if float(tokens) > 0
    }
    total_tokens = sum(active.values())
    if total_tokens <= 0:
        return 0.0
    if config.moe.compute_backend == "constant_per_token":
        return total_tokens * config.moe.expert_compute_us_per_token
    if config.moe.compute_backend == "roofline_h200":
        flops = total_tokens * _expert_ffn_flops_per_token(config)
        bytes_ = len(active) * _expert_weight_bytes(
            config
        ) + total_tokens * _activation_bytes_per_expert_token(config)
        compute_us = flops / max(config.moe.h200_effective_tflops * 1e6, 1e-9)
        memory_us = bytes_ / max(config.moe.h200_hbm_bw_tbps * 1e6, 1e-9)
        return max(compute_us, memory_us)
    raise ValueError(f"Unsupported moe.compute_backend: {config.moe.compute_backend}")


def _expert_ffn_flops_per_token(config: SimulationConfig) -> float:
    return 6.0 * config.moe.hidden_size * config.moe.expert_intermediate_size


def _expert_weight_bytes(config: SimulationConfig) -> float:
    return (
        3.0
        * config.moe.hidden_size
        * config.moe.expert_intermediate_size
        * config.moe.weight_bytes_per_elem
    )


def _activation_bytes_per_expert_token(config: SimulationConfig) -> float:
    return 3.0 * config.moe.hidden_size * config.moe.activation_bytes_per_elem


def analysis_compute_us_per_step(
    config: SimulationConfig,
    window_stats: list[dict],
    mapping: dict[int, list[int]],
    window_mappings: list[dict[int, list[int]]] | None = None,
    window_src_expert: list[dict] | None = None,
) -> float:
    """MoE compute model aligned with old Ablation2 for single-replica mappings.

    For dynamic replicas, compute is charged to the same nearest replica used by
    dispatch trace generation, so the compute breakdown matches the ASTRA trace.
    """
    per_window_us = []
    for w_idx, ws in enumerate(window_stats):
        cur_mapping = window_mappings[w_idx] if window_mappings is not None else mapping
        if window_src_expert is not None:
            rank_expert_tokens = build_rank_expert_compute_tokens(
                config,
                [window_src_expert[w_idx]],
                mapping,
                window_mappings=[cur_mapping],
            )[0]
            rank_compute = [
                _compute_us_for_rank_experts(config, expert_tokens)
                for expert_tokens in rank_expert_tokens
            ]
        else:
            num_ranks = config.supernode.num_wafers * config.supernode.dies_per_wafer
            batch_scale = config.workload.batch_size / max(config.moe.num_requests, 1)
            rank_expert_tokens: list[dict[int, float]] = [
                defaultdict(float) for _ in range(num_ranks)
            ]
            # No source breakdown is available, so fall back to even split across
            # replicas for this diagnostic path only.
            for e_str, cnt in ws.get("expert_counts", {}).items():
                e = int(e_str)
                replicas = cur_mapping.get(e, [])
                if not replicas:
                    continue
                work = float(cnt) * batch_scale / len(replicas)
                for d in replicas:
                    rank_expert_tokens[d][e] += work
            rank_compute = [
                _compute_us_for_rank_experts(config, dict(expert_tokens))
                for expert_tokens in rank_expert_tokens
            ]
        per_window_us.append(max(rank_compute) if rank_compute else 0.0)
    avg_window_us = sum(per_window_us) / max(len(per_window_us), 1)
    return avg_window_us / max(config.moe.window_size, 1)


@lru_cache(maxsize=64)
def _network_latencies_ns(network_config: str) -> tuple[float, float]:
    """Return (switch_latency_ns, mesh_latency_ns) from the ASTRA YAML."""
    if not network_config:
        return 2000.0, 500.0
    path = Path(network_config)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        vals = list(raw.get("latency", [2000.0, 500.0]))
        if len(vals) >= 2:
            return float(vals[0]), float(vals[1])
    except Exception:
        pass
    return 2000.0, 500.0


def _approx_route_cost_ns(config: SimulationConfig, src: int, dst: int) -> float:
    if src == dst:
        return 0.0
    dpw = config.supernode.dies_per_wafer
    side = int(dpw**0.5)
    switch_latency_ns, mesh_latency_ns = _network_latencies_ns(
        config.supernode.network_config
    )
    if side * side != dpw:
        if src // dpw == dst // dpw:
            return abs(src % dpw - dst % dpw) * mesh_latency_ns
        return 2.0 * switch_latency_ns + mesh_latency_ns
    sw, sl = divmod(src, dpw)
    dw, dl = divmod(dst, dpw)
    sr, sc = divmod(sl, side)
    dr, dc = divmod(dl, side)
    if sw == dw:
        return (abs(sr - dr) + abs(sc - dc)) * mesh_latency_ns
    src_edge = min(sr, sc, side - 1 - sr, side - 1 - sc)
    dst_edge = min(dr, dc, side - 1 - dr, side - 1 - dc)
    # Cross-wafer routes traverse two switch/rail links in RailOptimizedSwitchMesh:
    # source edge->switch and switch->destination edge.
    return (src_edge + dst_edge) * mesh_latency_ns + 2.0 * switch_latency_ns
