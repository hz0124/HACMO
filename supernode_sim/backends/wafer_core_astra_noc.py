#!/usr/bin/env python3
"""ASTRA-backed wafer-core NoC movement evaluator.

This backend models the many-core array as ASTRA ranks connected by an
analytical Mesh network.  It intentionally covers only on-wafer movement:

- HBM-edge resource -> core data movement;
- nearby SRAM owner core -> consumer core movement;
- remote SRAM owner core -> consumer core movement.

HBM DRAM service is still modeled by the caller's HBM backend/Ramulator path.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from .astra_wafer_core import (
    AstraWaferCoreCase,
    AstraWaferCoreConfig,
    AstraWaferCoreFlow,
    run_cases as run_astra_cases,
)
from .wafer_core_noc import WaferCoreNoCConfig, WaferCoreNoCTraffic, estimate_wafer_core_noc


@dataclass(frozen=True)
class WaferCoreAstraNoCConfig:
    rows: int
    cols: int
    astra_sim_dir: Path
    output_root: Path
    system_config: Path
    remote_memory_config: Path
    astra_network_backend: str = "congestion_aware"
    run_astra: bool = True
    base_noc_bw_tbps: float = 5.0
    noc_bw_scale_mode: str = "fixed"
    hop_latency_ns: float = 2.0
    gamma: float = 1.0
    hbm_edges: tuple[str, ...] = ("top", "bottom")
    near_sram_hops: float = 1.0
    max_flows_per_component: int = 256
    trace_prefix: str = "wafer_core_noc"


@dataclass(frozen=True)
class WaferCoreAstraNoCTraffic:
    hbm_bytes: float = 0.0
    near_sram_bytes: float = 0.0
    remote_sram_bytes: float = 0.0
    avg_hbm_hops: float = 0.0
    near_sram_hops: float = 1.0
    avg_remote_hops: float = 0.0
    hbm_dst_row_weights: tuple[float, ...] | None = None
    remote_src_row_weights: tuple[float, ...] | None = None
    remote_dst_row_weights: tuple[float, ...] | None = None


@dataclass(frozen=True)
class WaferCoreAstraNoCResult:
    hbm_us: float
    near_sram_us: float
    remote_sram_us: float
    total_us: float
    analytic_hbm_us: float
    analytic_near_sram_us: float
    analytic_remote_sram_us: float
    output_root: str
    network_config: str
    component_rows: tuple[dict[str, object], ...]
    flow_summary: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def estimate_wafer_core_astra_noc(
    config: WaferCoreAstraNoCConfig,
    traffic: WaferCoreAstraNoCTraffic,
) -> WaferCoreAstraNoCResult:
    """Run ASTRA cases for wafer-core NoC movement components."""
    if config.rows <= 0 or config.cols <= 0:
        raise ValueError("rows and cols must be positive")
    if config.rows != config.cols:
        raise ValueError("ASTRA analytical Mesh path currently expects a square core array")

    output_root = config.output_root / _case_hash(config, traffic)
    cached = output_root / "wafer_core_astra_noc_result.json"
    if cached.exists():
        data = json.loads(cached.read_text(encoding="utf-8"))
        return WaferCoreAstraNoCResult(
            hbm_us=float(data["hbm_us"]),
            near_sram_us=float(data["near_sram_us"]),
            remote_sram_us=float(data["remote_sram_us"]),
            total_us=float(data["total_us"]),
            analytic_hbm_us=float(data["analytic_hbm_us"]),
            analytic_near_sram_us=float(data["analytic_near_sram_us"]),
            analytic_remote_sram_us=float(data["analytic_remote_sram_us"]),
            output_root=str(data["output_root"]),
            network_config=str(data["network_config"]),
            component_rows=tuple(data.get("component_rows", [])),
            flow_summary=dict(data.get("flow_summary", {})),
        )
    output_root.mkdir(parents=True, exist_ok=True)
    network_config = output_root / "network.yaml"
    effective_bw_tbps = _effective_noc_bw_tbps(config)
    _write_network_config(
        network_config,
        ranks=config.rows * config.cols,
        bandwidth_gb_per_s=effective_bw_tbps * 1000.0,
        latency_ns=config.hop_latency_ns,
    )

    analytic = estimate_wafer_core_noc(
        WaferCoreNoCConfig(
            rows=config.rows,
            cols=config.cols,
            base_noc_bw_tbps=config.base_noc_bw_tbps,
            gamma=config.gamma,
            noc_bw_scale_mode=config.noc_bw_scale_mode,
            hbm_edges=config.hbm_edges,
            near_sram_hops=config.near_sram_hops,
        ),
        WaferCoreNoCTraffic(
            hbm_bytes=traffic.hbm_bytes,
            near_sram_bytes=traffic.near_sram_bytes,
            remote_sram_bytes=traffic.remote_sram_bytes,
            avg_hbm_hops=traffic.avg_hbm_hops,
            near_sram_hops=traffic.near_sram_hops,
            avg_remote_hops=traffic.avg_remote_hops,
        ),
    )

    cases = _build_cases(config, traffic, effective_bw_tbps)
    if cases:
        rows = run_astra_cases(
            AstraWaferCoreConfig(
                astra_sim_dir=config.astra_sim_dir,
                output_root=output_root,
                network_config=network_config,
                system_config=config.system_config,
                remote_memory_config=config.remote_memory_config,
                num_ranks=config.rows * config.cols,
                active_ranks=config.rows * config.cols,
                astra_network_backend=config.astra_network_backend,
                run_astra=config.run_astra,
                trace_prefix=config.trace_prefix,
            ),
            cases,
        )
    else:
        rows = []

    by_name = {str(row["name"]): row for row in rows}
    hbm_us = _component_us(by_name.get("hbm"), analytic.hbm_us, config.gamma)
    near_us = _component_us(by_name.get("near_sram"), analytic.near_sram_us, config.gamma)
    remote_us = _component_us(by_name.get("remote_sram"), analytic.remote_sram_us, config.gamma)
    result = WaferCoreAstraNoCResult(
        hbm_us=hbm_us,
        near_sram_us=near_us,
        remote_sram_us=remote_us,
        total_us=hbm_us + near_us + remote_us,
        analytic_hbm_us=analytic.hbm_us,
        analytic_near_sram_us=analytic.near_sram_us,
        analytic_remote_sram_us=analytic.remote_sram_us,
        output_root=str(output_root),
        network_config=str(network_config),
        component_rows=tuple(rows),
        flow_summary={
            "rows": config.rows,
            "cols": config.cols,
            "effective_noc_bw_tbps": effective_bw_tbps,
            "hop_latency_ns": config.hop_latency_ns,
            "gamma": config.gamma,
            "max_flows_per_component": config.max_flows_per_component,
            "run_astra": config.run_astra,
            "num_cases": len(cases),
        },
    )
    (output_root / "wafer_core_astra_noc_result.json").write_text(
        json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result


def _effective_noc_bw_tbps(config: WaferCoreAstraNoCConfig) -> float:
    if config.noc_bw_scale_mode == "fixed":
        scale = 1.0
    elif config.noc_bw_scale_mode == "cols":
        scale = max(config.cols / 20.0, 1.0)
    elif config.noc_bw_scale_mode == "sqrt_area":
        scale = max(math.sqrt((config.rows * config.cols) / 400.0), 1.0)
    else:
        raise ValueError(f"Unsupported noc_bw_scale_mode: {config.noc_bw_scale_mode}")
    return config.base_noc_bw_tbps * scale


def _write_network_config(path: Path, ranks: int, bandwidth_gb_per_s: float, latency_ns: float) -> None:
    path.write_text(
        "\n".join(
            [
                "topology: [ Mesh ]",
                f"npus_count: [ {ranks} ]",
                f"bandwidth: [ {bandwidth_gb_per_s} ]",
                f"latency: [ {latency_ns} ]",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _build_cases(
    config: WaferCoreAstraNoCConfig,
    traffic: WaferCoreAstraNoCTraffic,
    effective_bw_tbps: float,
) -> list[AstraWaferCoreCase]:
    cases: list[AstraWaferCoreCase] = []
    if traffic.hbm_bytes > 0.0:
        flows = _hbm_flows(config, traffic.hbm_bytes, traffic.hbm_dst_row_weights)
        if flows:
            cases.append(
                AstraWaferCoreCase(
                    name="hbm",
                    rows=config.rows,
                    cols=config.cols,
                    flows=flows,
                    hbm_bytes=traffic.hbm_bytes,
                    avg_hbm_hops=traffic.avg_hbm_hops,
                    noc_bw_tbps=effective_bw_tbps,
                    gamma=config.gamma,
                    hbm_edges=config.hbm_edges,
                )
            )
    if traffic.near_sram_bytes > 0.0:
        flows = _core_to_core_flows(config, traffic.near_sram_bytes, traffic.near_sram_hops, "near")
        if flows:
            cases.append(
                AstraWaferCoreCase(
                    name="near_sram",
                    rows=config.rows,
                    cols=config.cols,
                    flows=flows,
                    near_sram_bytes=traffic.near_sram_bytes,
                    near_sram_hops=traffic.near_sram_hops,
                    noc_bw_tbps=effective_bw_tbps,
                    gamma=config.gamma,
                )
            )
    if traffic.remote_sram_bytes > 0.0:
        flows = _core_to_core_flows(
            config,
            traffic.remote_sram_bytes,
            traffic.avg_remote_hops,
            "remote",
            src_row_weights=traffic.remote_src_row_weights,
            dst_row_weights=traffic.remote_dst_row_weights,
        )
        if flows:
            cases.append(
                AstraWaferCoreCase(
                    name="remote_sram",
                    rows=config.rows,
                    cols=config.cols,
                    flows=flows,
                    remote_sram_bytes=traffic.remote_sram_bytes,
                    avg_remote_hops=traffic.avg_remote_hops,
                    noc_bw_tbps=effective_bw_tbps,
                    gamma=config.gamma,
                )
            )
    return cases


def _hbm_flows(
    config: WaferCoreAstraNoCConfig,
    total_bytes: float,
    dst_row_weights: tuple[float, ...] | None = None,
) -> tuple[AstraWaferCoreFlow, ...]:
    destinations = _rank_samples(config, dst_row_weights)
    moving: list[tuple[int, int, float]] = []
    for dst, share in destinations:
        row, col = divmod(dst, config.cols)
        src = _nearest_hbm_edge_rank(row, col, config)
        if src != dst:
            moving.append((src, dst, share))
    return _pack_weighted_flows(moving, total_bytes, tag_base=10_000)


def _core_to_core_flows(
    config: WaferCoreAstraNoCConfig,
    total_bytes: float,
    target_hops: float,
    family: str,
    src_row_weights: tuple[float, ...] | None = None,
    dst_row_weights: tuple[float, ...] | None = None,
) -> tuple[AstraWaferCoreFlow, ...]:
    if src_row_weights is not None or dst_row_weights is not None:
        return _weighted_core_to_core_flows(
            config,
            total_bytes,
            family,
            src_row_weights=src_row_weights,
            dst_row_weights=dst_row_weights,
        )
    destinations = _selected_nodes(config)
    pairs: list[tuple[int, int]] = []
    for dst in destinations:
        dst_row, dst_col = divmod(dst, config.cols)
        src = _source_at_distance(config, dst_row, dst_col, target_hops)
        if src != dst:
            pairs.append((src, dst))
    tag_base = 20_000 if family == "near" else 30_000
    return _pack_flows(pairs, total_bytes, tag_base=tag_base)


def _weighted_core_to_core_flows(
    config: WaferCoreAstraNoCConfig,
    total_bytes: float,
    family: str,
    src_row_weights: tuple[float, ...] | None,
    dst_row_weights: tuple[float, ...] | None,
) -> tuple[AstraWaferCoreFlow, ...]:
    limit = _sample_limit(config)
    if limit <= 0 or total_bytes <= 0.0:
        return ()
    weighted_pairs: list[tuple[int, int, float]] = []
    for idx in range(limit):
        src_q = ((idx * 37) % limit + 0.5) / limit
        dst_q = (idx + 0.5) / limit
        src = _rank_at_quantile(config, src_row_weights, src_q)
        dst = _rank_at_quantile(config, dst_row_weights, dst_q)
        if src != dst:
            weighted_pairs.append((src, dst, 1.0 / limit))
    tag_base = 20_000 if family == "near" else 30_000
    return _pack_weighted_flows(weighted_pairs, total_bytes, tag_base=tag_base)


def _pack_flows(pairs: list[tuple[int, int]], total_bytes: float, tag_base: int) -> tuple[AstraWaferCoreFlow, ...]:
    if not pairs or total_bytes <= 0.0:
        return ()
    bytes_per_flow = max(1, int(math.ceil(total_bytes / len(pairs))))
    return tuple(
        AstraWaferCoreFlow(src=src, dst=dst, bytes=bytes_per_flow, tag=tag_base + idx)
        for idx, (src, dst) in enumerate(pairs)
    )


def _pack_weighted_flows(
    weighted_pairs: list[tuple[int, int, float]],
    total_bytes: float,
    tag_base: int,
) -> tuple[AstraWaferCoreFlow, ...]:
    if not weighted_pairs or total_bytes <= 0.0:
        return ()
    merged: dict[tuple[int, int], float] = {}
    for src, dst, share in weighted_pairs:
        if share <= 0.0:
            continue
        merged[(src, dst)] = merged.get((src, dst), 0.0) + share
    flows = []
    for idx, ((src, dst), share) in enumerate(sorted(merged.items())):
        bytes_for_flow = int(round(total_bytes * share))
        if bytes_for_flow <= 0:
            continue
        flows.append(AstraWaferCoreFlow(src=src, dst=dst, bytes=bytes_for_flow, tag=tag_base + idx))
    return tuple(flows)


def _selected_nodes(config: WaferCoreAstraNoCConfig) -> list[int]:
    total = config.rows * config.cols
    limit = int(config.max_flows_per_component)
    if limit <= 0 or limit >= total:
        return list(range(total))
    if limit == 1:
        return [total // 2]
    # Deterministic stratified sample over rank order. This keeps all runs
    # reproducible while covering edge, middle, and far positions.
    return sorted({round(i * (total - 1) / (limit - 1)) for i in range(limit)})


def _sample_limit(config: WaferCoreAstraNoCConfig) -> int:
    total = config.rows * config.cols
    limit = int(config.max_flows_per_component)
    if limit <= 0:
        return 0
    return min(limit, total)


def _rank_samples(
    config: WaferCoreAstraNoCConfig,
    row_weights: tuple[float, ...] | None,
) -> list[tuple[int, float]]:
    limit = _sample_limit(config)
    if limit <= 0:
        return []
    if row_weights is None:
        nodes = _selected_nodes(config)
        share = 1.0 / len(nodes) if nodes else 0.0
        return [(rank, share) for rank in nodes]
    return [(rank, 1.0 / limit) for rank in (_rank_at_quantile(config, row_weights, (i + 0.5) / limit) for i in range(limit))]


def _rank_at_quantile(
    config: WaferCoreAstraNoCConfig,
    row_weights: tuple[float, ...] | None,
    q: float,
) -> int:
    if row_weights is None:
        total = config.rows * config.cols
        return min(total - 1, max(0, int(q * total)))
    rows = _normalize_row_weights(row_weights, config.rows)
    q = min(max(q, 0.0), 1.0 - 1e-15)
    acc = 0.0
    row = config.rows - 1
    prev = 0.0
    for idx, weight in enumerate(rows):
        prev = acc
        acc += weight
        if q < acc:
            row = idx
            break
    row_mass = max(rows[row], 1e-15)
    within_row = min(max((q - prev) / row_mass, 0.0), 1.0 - 1e-15)
    col = min(config.cols - 1, max(0, int(within_row * config.cols)))
    return _rank(row, col, config.cols)


def _normalize_row_weights(row_weights: tuple[float, ...], rows: int) -> tuple[float, ...]:
    if len(row_weights) != rows:
        raise ValueError(f"row weight length mismatch: got {len(row_weights)}, expected {rows}")
    cleaned = [max(float(x), 0.0) for x in row_weights]
    total = sum(cleaned)
    if total <= 0.0:
        return tuple(1.0 / rows for _ in range(rows))
    return tuple(x / total for x in cleaned)


def _nearest_hbm_edge_rank(row: int, col: int, config: WaferCoreAstraNoCConfig) -> int:
    candidates: list[tuple[int, int]] = []
    edges = tuple(edge.lower() for edge in config.hbm_edges)
    if "top" in edges:
        candidates.append((row, _rank(0, col, config.cols)))
    if "bottom" in edges:
        candidates.append((config.rows - 1 - row, _rank(config.rows - 1, col, config.cols)))
    if "left" in edges:
        candidates.append((col, _rank(row, 0, config.cols)))
    if "right" in edges:
        candidates.append((config.cols - 1 - col, _rank(row, config.cols - 1, config.cols)))
    if not candidates:
        raise ValueError("hbm_edges must contain at least one supported edge")
    return min(candidates, key=lambda item: item[0])[1]


def _source_at_distance(config: WaferCoreAstraNoCConfig, dst_row: int, dst_col: int, target_hops: float) -> int:
    distance = max(1, min(int(round(target_hops)), config.rows + config.cols - 2))
    candidates = (
        (dst_row - distance, dst_col),
        (dst_row + distance, dst_col),
        (dst_row, dst_col - distance),
        (dst_row, dst_col + distance),
    )
    valid = [(r, c) for r, c in candidates if 0 <= r < config.rows and 0 <= c < config.cols]
    if valid:
        # Pick the first valid direction in a deterministic order.
        return _rank(valid[0][0], valid[0][1], config.cols)

    # If an exact Manhattan shell is clipped by array boundaries, use the
    # farthest corner. This preserves "remote" behavior for edge destinations.
    corners = [(0, 0), (0, config.cols - 1), (config.rows - 1, 0), (config.rows - 1, config.cols - 1)]
    farthest = max(corners, key=lambda rc: abs(rc[0] - dst_row) + abs(rc[1] - dst_col))
    return _rank(farthest[0], farthest[1], config.cols)


def _rank(row: int, col: int, cols: int) -> int:
    return row * cols + col


def _component_us(row: dict[str, object] | None, fallback_us: float, gamma: float) -> float:
    if row is None:
        return 0.0 if fallback_us <= 0.0 else fallback_us
    comm = row.get("astra_comm_us")
    wall = row.get("astra_wall_us")
    value = comm if comm is not None and comm != "" else wall
    if value is None or value == "":
        return fallback_us
    return float(value) * gamma


def _case_hash(config: WaferCoreAstraNoCConfig, traffic: WaferCoreAstraNoCTraffic) -> str:
    payload = {
        "config": {
            "rows": config.rows,
            "cols": config.cols,
            "backend": config.astra_network_backend,
            "run_astra": config.run_astra,
            "base_noc_bw_tbps": config.base_noc_bw_tbps,
            "noc_bw_scale_mode": config.noc_bw_scale_mode,
            "hop_latency_ns": config.hop_latency_ns,
            "gamma": config.gamma,
            "hbm_edges": config.hbm_edges,
            "near_sram_hops": config.near_sram_hops,
            "max_flows_per_component": config.max_flows_per_component,
        },
        "traffic": asdict(traffic),
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return f"{config.rows}x{config.cols}_{digest}"
