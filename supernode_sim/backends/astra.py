#!/usr/bin/env python3
"""ASTRA-Sim backend for die-level switch-mesh supernode simulation."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import yaml

from ..tracegen.chakra import write_moe_p2p_et
from ..tracegen.moe import (
    analysis_compute_us_per_step,
    build_combine_matrices,
    build_dispatch_matrices,
    build_rank_expert_compute_tokens,
    build_rank_compute_tokens,
    build_rank_compute_us,
    build_shared_rank_compute_us,
    build_window_counts,
    build_window_mappings,
    route_plan_to_simulator_inputs,
    solve_mapping,
)
from ..types import SimulationConfig, SimulationResult

WALL_RE = re.compile(r"sys\[(\d+)\],\s*Wall time:\s*(\d+)")
COMM_RE = re.compile(r"sys\[(\d+)\],\s*Comm time:\s*(\d+)")
FINISHED_RE = re.compile(r"sys\[(\d+)\]\s*finished,\s*(\d+)\s*cycles")


def simulate(config: SimulationConfig) -> SimulationResult:
    if config.workload.layer_type != "moe":
        raise NotImplementedError(
            "The ASTRA backend currently supports workload.layer_type='moe'."
        )

    repo_root = Path(__file__).resolve().parents[2]
    sn = config.supernode
    astra_dir = _resolve(repo_root, sn.astra_sim_dir)
    backend_name = sn.astra_network_backend.lower()
    if backend_name in {"congestion_aware", "aware"}:
        astra_bin = (
            astra_dir
            / "build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware"
        )
        backend_name = "congestion_aware"
    elif backend_name in {"congestion_unaware", "unaware"}:
        astra_bin = (
            astra_dir
            / "build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware"
        )
        backend_name = "congestion_unaware"
    else:
        raise ValueError(
            "supernode.astra_network_backend must be 'congestion_aware' "
            f"or 'congestion_unaware', got {sn.astra_network_backend!r}"
        )
    if sn.run_astra and not astra_bin.exists():
        raise FileNotFoundError(f"ASTRA-Sim analytical binary not found: {astra_bin}")

    output_root = _resolve(repo_root, sn.output_root)
    trace_dir = output_root / "traces"
    log_dir = output_root / "logs"
    for p in [trace_dir, log_dir]:
        p.mkdir(parents=True, exist_ok=True)

    num_ranks = sn.num_wafers * sn.dies_per_wafer
    route_plan_mode = bool(config.moe.route_plan_json)
    route_plan_pre_dispatch: list[list[dict[int, dict[int, int]]]] = []
    route_plan_replica_counts: list[int] | None = None
    exclude_leading_windows = max(
        int(getattr(config.moe, "exclude_leading_windows", 0) or 0), 0
    )
    if route_plan_mode:
        route_plan = _load_json(repo_root, config.moe.route_plan_json)
        provenance_rows = (
            route_plan.get("metadata", {})
            .get("mapping_copy_provenance", {})
            .get("per_step")
        )
        if provenance_rows is not None:
            if len(provenance_rows) != len(route_plan.get("steps", [])):
                raise ValueError(
                    "route-plan mapping-copy provenance must have one row per step"
                )
            route_plan_replica_counts = [
                int(row["extra_replicas"]) for row in provenance_rows
            ]
        route_plan_pre_dispatch = [
            [
                {
                    int(src): {int(dst): int(nbytes) for dst, nbytes in dsts.items()}
                    for src, dsts in segment.get("messages", {}).items()
                }
                for segment in step.get("pre_dispatch_segments", [])
            ]
            for step in route_plan.get("steps", [])
        ]
        window_stats, dispatch, combine, rank_expert_compute_tokens = (
            route_plan_to_simulator_inputs(route_plan, num_ranks=num_ranks)
        )
        if config.moe.expert_mapping:
            mapping = {int(k): list(v) for k, v in config.moe.expert_mapping.items()}
        elif config.moe.expert_mapping_json:
            mapping = _load_mapping(repo_root, config.moe.expert_mapping_json)
        else:
            mapping = {}
        window_mappings = [mapping for _ in window_stats]
        if exclude_leading_windows:
            kept = _kept_active_indices(window_stats, exclude_leading_windows)
            excluded_window_ids = [
                int(ws.get("step_id", index))
                for index, ws in enumerate(window_stats)
                if index not in kept
            ]
            window_stats = [window_stats[index] for index in kept]
            dispatch = [dispatch[index] for index in kept]
            combine = [combine[index] for index in kept]
            rank_expert_compute_tokens = [
                rank_expert_compute_tokens[index] for index in kept
            ]
            route_plan_pre_dispatch = [route_plan_pre_dispatch[index] for index in kept]
            if route_plan_replica_counts is not None:
                route_plan_replica_counts = [
                    route_plan_replica_counts[index] for index in kept
                ]
            window_mappings = [window_mappings[index] for index in kept]
        else:
            excluded_window_ids = []
        window_src_expert = None
    else:
        if config.moe.window_stats_json or config.moe.window_src_expert_json:
            if not (config.moe.window_stats_json and config.moe.window_src_expert_json):
                raise ValueError(
                    "Both moe.window_stats_json and moe.window_src_expert_json must be provided."
                )
            window_stats = _load_json(repo_root, config.moe.window_stats_json)
            window_src_expert = _load_json(repo_root, config.moe.window_src_expert_json)
        else:
            window_stats, window_src_expert = build_window_counts(config, repo_root)

        if config.moe.expert_mapping:
            mapping = {int(k): list(v) for k, v in config.moe.expert_mapping.items()}
        elif config.moe.expert_mapping_json:
            mapping = _load_mapping(repo_root, config.moe.expert_mapping_json)
        else:
            mapping = solve_mapping(config, window_src_expert)

        external_window_mappings = (
            _load_window_mappings(repo_root, config.moe.window_mappings_json)
            if config.moe.window_mappings_json
            else None
        )
        if external_window_mappings is not None:
            window_mappings = external_window_mappings
            if len(window_mappings) != len(window_stats):
                raise ValueError(
                    f"moe.window_mappings_json has {len(window_mappings)} windows, "
                    f"but trace has {len(window_stats)} windows"
                )
            if not mapping and window_mappings:
                mapping = window_mappings[0]
        elif config.moe.expert_mapping or config.moe.expert_mapping_json:
            window_mappings = [mapping for _ in window_stats]
        else:
            window_mappings = build_window_mappings(
                config, window_stats, window_src_expert, mapping
            )

        if exclude_leading_windows:
            window_stats, window_src_expert, window_mappings, excluded_window_ids = (
                _exclude_leading_active_windows(
                    window_stats,
                    window_src_expert,
                    window_mappings,
                    exclude_leading_windows,
                )
            )
        else:
            excluded_window_ids = []
        dispatch = build_dispatch_matrices(
            config, window_src_expert, mapping, window_mappings=window_mappings
        )
        combine = build_combine_matrices(config, dispatch)
        rank_expert_compute_tokens = build_rank_expert_compute_tokens(
            config,
            window_src_expert,
            mapping,
            window_mappings=window_mappings,
        )

    trace_mode = getattr(config.moe, "astra_trace_mode", "full")
    if trace_mode not in {
        "full",
        "communication_only",
        "dispatch_only",
        "combine_only",
    }:
        raise ValueError(f"Unsupported moe.astra_trace_mode: {trace_mode}")
    if trace_mode == "dispatch_only":
        combine_for_trace = [{} for _ in dispatch]
    else:
        combine_for_trace = combine
    if trace_mode == "combine_only":
        dispatch_for_trace = [{} for _ in dispatch]
    else:
        dispatch_for_trace = dispatch
    dispatch_segments = None
    combine_segments = None
    include_pre_dispatch = route_plan_mode and trace_mode != "combine_only"
    logical_dispatch_segments = [
        [
            *(route_plan_pre_dispatch[index] if include_pre_dispatch else []),
            dispatch_for_trace[index],
        ]
        for index in range(len(dispatch_for_trace))
    ]
    if config.moe.route_via_edge_ports:
        rail_ports = _load_rail_ports(
            _resolve(repo_root, sn.network_config), sn.dies_per_wafer
        )
        dispatch_segments = [
            [
                segment
                for matrix in step_segments
                for segment in _edge_routed_segments(config, [matrix], rail_ports)[0]
            ]
            for step_segments in logical_dispatch_segments
        ]
        combine_segments = _edge_routed_segments(config, combine_for_trace, rail_ports)
    elif include_pre_dispatch:
        dispatch_segments = logical_dispatch_segments
    rank_compute_tokens = [
        [sum(expert_tokens.values()) for expert_tokens in window_tokens]
        for window_tokens in rank_expert_compute_tokens
    ]
    routed_rank_compute_us = build_rank_compute_us(
        config,
        rank_compute_tokens,
        rank_expert_compute_tokens=rank_expert_compute_tokens,
    )
    shared_rank_compute_us = build_shared_rank_compute_us(config, window_stats)
    total_rank_compute_us = [
        [max(routed, shared) for routed, shared in zip(routed_step, shared_step)]
        for routed_step, shared_step in zip(
            routed_rank_compute_us, shared_rank_compute_us
        )
    ]
    if trace_mode == "full":
        rank_compute_us = total_rank_compute_us
    else:
        rank_compute_us = [
            [0.0 for _ in window_tokens] for window_tokens in rank_compute_tokens
        ]
    trace_prefix = "supernode_moe"

    workload_prefix = write_moe_p2p_et(
        astra_sim_dir=astra_dir,
        output_dir=trace_dir,
        trace_prefix=trace_prefix,
        num_ranks=num_ranks,
        dispatch_matrices=dispatch_for_trace,
        combine_matrices=combine_for_trace,
        dispatch_segments=dispatch_segments,
        combine_segments=combine_segments,
        window_stats=window_stats,
        mapping=mapping,
        expert_compute_us_per_token=config.moe.expert_compute_us_per_token,
        attention_us_per_die=config.moe.attention_us_per_die,
        batch_scale=config.workload.batch_size / max(config.moe.num_requests, 1),
        window_mappings=window_mappings,
        rank_compute_tokens=rank_compute_tokens,
        rank_compute_us=rank_compute_us,
    )

    profile = _profile(
        config,
        window_stats,
        dispatch,
        combine,
        mapping,
        window_mappings,
        authoritative_replica_counts=route_plan_replica_counts,
    )
    profile["excluded_leading_windows"] = exclude_leading_windows
    profile["excluded_window_ids"] = excluded_window_ids
    profile["route_plan_consumed"] = route_plan_mode
    profile["route_plan_json"] = config.moe.route_plan_json if route_plan_mode else ""
    profile["total_pre_dispatch_collective_bytes"] = sum(
        int(nbytes)
        for step_segments in route_plan_pre_dispatch
        for segment in step_segments
        for dsts in segment.values()
        for nbytes in dsts.values()
    )
    profile["pre_dispatch_collective_phase_instances"] = sum(
        len(step_segments) for step_segments in route_plan_pre_dispatch
    )
    profile["analysis_compute_us_per_step"] = (
        sum(max(step, default=0.0) for step in total_rank_compute_us)
        / max(len(total_rank_compute_us), 1)
        if route_plan_mode
        else analysis_compute_us_per_step(
            config=config,
            window_stats=window_stats,
            mapping=mapping,
            window_mappings=window_mappings,
            window_src_expert=window_src_expert,
        )
    )
    (output_root / "profile.json").write_text(
        json.dumps(profile, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    if not sn.run_astra:
        return SimulationResult(
            backend="astra_tracegen",
            total_latency_us=0.0,
            compute_us=0.0,
            hbm_us=0.0,
            sram_noc_us=0.0,
            communication_us=0.0,
            prefetch_exposed_us=0.0,
            hbm_bytes=0.0,
            sram_local_bytes=0.0,
            sram_near_bytes=0.0,
            sram_remote_bytes=0.0,
            horizontal_link_bytes=profile["total_dispatch_bytes"]
            + profile["total_combine_bytes"],
            vertical_link_bytes=0.0,
            metadata={"output_root": str(output_root), "profile": profile},
        )

    log_path = log_dir / "astra.log"
    # Note: cycle-to-us conversion assumes 1 GHz clock (1 cycle = 1 ns = 0.001 us).
    # If the ASTRA-Sim system config specifies a different clock frequency,
    # this conversion will be incorrect.  TODO: read cycle_freq from system config.
    cmd = [
        str(astra_bin),
        f"--workload-configuration={workload_prefix}",
        f"--system-configuration={_resolve(repo_root, sn.system_config)}",
        f"--remote-memory-configuration={_resolve(repo_root, sn.remote_memory_config)}",
        f"--network-configuration={_resolve(repo_root, sn.network_config)}",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{astra_dir}:{env.get('PYTHONPATH', '')}"
        if env.get("PYTHONPATH")
        else str(astra_dir)
    )
    with log_path.open("w", encoding="utf-8") as f:
        subprocess.run(
            cmd,
            cwd=str(output_root),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
        )

    parsed = parse_log(log_path.read_text(encoding="utf-8", errors="ignore"))
    control_prefix = write_moe_p2p_et(
        astra_sim_dir=astra_dir,
        output_dir=trace_dir,
        trace_prefix="supernode_moe_step_wall_control",
        num_ranks=num_ranks,
        dispatch_matrices=[{} for _ in window_stats],
        combine_matrices=[{} for _ in window_stats],
        window_stats=window_stats,
        mapping=mapping,
        expert_compute_us_per_token=0.0,
        attention_us_per_die=0.0,
        batch_scale=1.0,
        window_mappings=window_mappings,
        rank_compute_us=[[0.0 for _ in range(num_ranks)] for _ in window_stats],
    )
    control_log_path = log_dir / "astra_step_wall_control.log"
    control_cmd = [
        str(astra_bin),
        f"--workload-configuration={control_prefix}",
        f"--system-configuration={_resolve(repo_root, sn.system_config)}",
        f"--remote-memory-configuration={_resolve(repo_root, sn.remote_memory_config)}",
        f"--network-configuration={_resolve(repo_root, sn.network_config)}",
    ]
    with control_log_path.open("w", encoding="utf-8") as f:
        subprocess.run(
            control_cmd,
            cwd=str(output_root),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
        )
    control_parsed = parse_log(
        control_log_path.read_text(encoding="utf-8", errors="ignore")
    )
    (output_root / "astra_summary.json").write_text(
        json.dumps(
            {
                "raw": parsed,
                "step_wall_control": control_parsed,
                "correction": "subtract measured zero-byte ALL_TO_ALL step-wall control cycles",
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    wall_cycles_raw = parsed["summary"].get("wall_cycles_max")
    finished_cycles_raw = parsed["summary"].get("finished_cycles_max")
    # Use explicit None-check instead of truthy `or`, because 0 is a valid cycle count.
    wall_cycles = (
        wall_cycles_raw
        if wall_cycles_raw is not None
        else (finished_cycles_raw if finished_cycles_raw is not None else 0)
    )
    comm_cycles_raw = parsed["summary"].get("comm_cycles_max")
    comm_cycles = comm_cycles_raw if comm_cycles_raw is not None else 0
    control_wall_raw = control_parsed["summary"].get("wall_cycles_max")
    control_comm_raw = control_parsed["summary"].get("comm_cycles_max")
    control_wall_cycles = control_wall_raw if control_wall_raw is not None else 0
    control_comm_cycles = control_comm_raw if control_comm_raw is not None else 0
    corrected_wall_cycles = max(wall_cycles - control_wall_cycles, 0)
    corrected_comm_cycles = max(comm_cycles - control_comm_cycles, 0)
    total_us = corrected_wall_cycles / 1000.0
    comm_us = corrected_comm_cycles / 1000.0
    compute_us = max(total_us - comm_us, 0.0)
    profile["astra_step_wall_control"] = {
        "mechanism": "zero_byte_all_to_all",
        "num_step_walls": len(window_stats),
        "raw_wall_cycles": wall_cycles,
        "raw_comm_cycles": comm_cycles,
        "control_wall_cycles": control_wall_cycles,
        "control_comm_cycles": control_comm_cycles,
        "corrected_wall_cycles": corrected_wall_cycles,
        "corrected_comm_cycles": corrected_comm_cycles,
    }
    (output_root / "profile.json").write_text(
        json.dumps(profile, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    return SimulationResult(
        backend="astra",
        total_latency_us=total_us,
        compute_us=compute_us,
        hbm_us=0.0,
        sram_noc_us=0.0,
        communication_us=comm_us,
        prefetch_exposed_us=0.0,
        hbm_bytes=0.0,
        sram_local_bytes=0.0,
        sram_near_bytes=0.0,
        sram_remote_bytes=0.0,
        horizontal_link_bytes=profile["total_dispatch_bytes"]
        + profile["total_combine_bytes"],
        vertical_link_bytes=0.0,
        metadata={
            "output_root": str(output_root),
            "network_config": str(_resolve(repo_root, sn.network_config)),
            "astra_network_backend": backend_name,
            "mapping_method": config.moe.mapping_method,
            "profile": profile,
            "astra_summary": parsed["summary"],
            "astra_step_wall_control_summary": control_parsed["summary"],
            "astra_corrected_summary": {
                "wall_cycles": corrected_wall_cycles,
                "comm_cycles": corrected_comm_cycles,
            },
        },
    )


def parse_log(text: str) -> dict:
    wall = {int(m.group(1)): int(m.group(2)) for m in WALL_RE.finditer(text)}
    comm = {int(m.group(1)): int(m.group(2)) for m in COMM_RE.finditer(text)}
    finished = {int(m.group(1)): int(m.group(2)) for m in FINISHED_RE.finditer(text)}
    ranks = sorted(set(wall) | set(comm) | set(finished))
    per_rank = [
        {
            "rank": r,
            "wall_cycles": wall.get(r),
            "comm_cycles": comm.get(r),
            "finished_cycles": finished.get(r),
        }
        for r in ranks
    ]
    return {
        "summary": {
            "num_ranks_with_wall": len(wall),
            "num_ranks_with_comm": len(comm),
            "num_ranks_with_finished": len(finished),
            "wall_cycles_max": max(wall.values()) if wall else None,
            "comm_cycles_max": max(comm.values()) if comm else None,
            "finished_cycles_max": max(finished.values()) if finished else None,
        },
        "per_rank": per_rank,
    }


def _profile(
    config: SimulationConfig,
    window_stats: list[dict],
    dispatch: list[dict[int, dict[int, int]]],
    combine: list[dict[int, dict[int, int]]],
    mapping: dict[int, list[int]],
    window_mappings: list[dict[int, list[int]]] | None = None,
    authoritative_replica_counts: list[int] | None = None,
) -> dict:
    dpw = config.supernode.dies_per_wafer
    total_dispatch = 0
    total_combine = 0
    cross = 0
    same = 0
    max_rank_recv = 0
    for mat, combine_mat in zip(dispatch, combine):
        recv = {}
        for src, dsts in mat.items():
            for dst, b in dsts.items():
                total_dispatch += b
                recv[dst] = recv.get(dst, 0) + b
                if src // dpw == dst // dpw:
                    same += b
                else:
                    cross += b
        if recv:
            max_rank_recv = max(max_rank_recv, max(recv.values()))
        total_combine += sum(b for dsts in combine_mat.values() for b in dsts.values())
    expert_counts = {}
    for ws in window_stats:
        for e, c in ws["expert_counts"].items():
            expert_counts[int(e)] = expert_counts.get(int(e), 0) + int(c)
    replica_counts = list(authoritative_replica_counts or [])
    replica_count_source = (
        "route_plan_effective_mapping"
        if authoritative_replica_counts is not None
        else "window_mapping"
    )
    if authoritative_replica_counts is None and window_mappings:
        for wm in window_mappings:
            replica_counts.append(sum(max(len(ds) - 1, 0) for ds in wm.values()))
    return {
        "total_dispatch_bytes": total_dispatch,
        "total_combine_bytes": total_combine,
        "total_moe_bytes": total_dispatch + total_combine,
        "cross_wafer_bytes": cross,
        "same_wafer_bytes": same,
        "cross_wafer_ratio": cross / max(total_dispatch, 1),
        "max_rank_recv_bytes": max_rank_recv,
        "top_experts": [
            {
                "expert": int(e),
                "activations": int(c),
                "dies": (window_mappings[-1] if window_mappings else mapping).get(
                    int(e), []
                ),
            }
            for e, c in sorted(expert_counts.items(), key=lambda x: x[1], reverse=True)[
                :16
            ]
        ],
        "avg_extra_replicas": (
            sum(replica_counts) / max(len(replica_counts), 1) if replica_counts else 0.0
        ),
        "max_extra_replicas": max(replica_counts) if replica_counts else 0,
        "replica_count_source": replica_count_source,
    }


def _is_active_window(ws: dict, wsrc: dict) -> bool:
    tokens = sum(int(v) for v in (ws.get("tokens_per_die") or {}).values())
    activations = sum(int(v) for v in (ws.get("expert_counts") or {}).values())
    if tokens > 0 or activations > 0:
        return True
    return any(
        (e_counts or {}) for e_counts in (wsrc.get("src_expert_counts") or {}).values()
    )


def _kept_active_indices(window_stats: list[dict], count: int) -> list[int]:
    kept = []
    active_seen = 0
    for index, step in enumerate(window_stats):
        active = bool(step.get("active", True))
        if active and active_seen < count:
            active_seen += 1
            continue
        if active:
            active_seen += 1
        kept.append(index)
    if not kept:
        raise ValueError(
            f"exclude_leading_windows={count} removed all route-plan steps"
        )
    return kept


def _exclude_leading_active_windows(
    window_stats: list[dict],
    window_src_expert: list[dict],
    window_mappings: list[dict[int, list[int]]],
    count: int,
) -> tuple[list[dict], list[dict], list[dict[int, list[int]]], list[int]]:
    kept_stats: list[dict] = []
    kept_src: list[dict] = []
    kept_mappings: list[dict[int, list[int]]] = []
    excluded_ids: list[int] = []
    active_seen = 0
    for idx, (ws, wsrc, wm) in enumerate(
        zip(window_stats, window_src_expert, window_mappings)
    ):
        if _is_active_window(ws, wsrc):
            if active_seen < count:
                excluded_ids.append(
                    int(ws.get("window_id", wsrc.get("window_id", idx)))
                )
                active_seen += 1
                continue
            active_seen += 1
        kept_stats.append(ws)
        kept_src.append(wsrc)
        kept_mappings.append(wm)
    if not kept_stats:
        raise ValueError(f"exclude_leading_windows={count} removed all runtime windows")
    return kept_stats, kept_src, kept_mappings, excluded_ids


def _resolve(repo_root: Path, path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else repo_root / p


def _load_json(repo_root: Path, path: str | Path):
    return json.loads(_resolve(repo_root, path).read_text(encoding="utf-8"))


def _load_mapping(repo_root: Path, path: str | Path) -> dict[int, list[int]]:
    raw = _load_json(repo_root, path)
    payload = raw.get("expert_to_dies", raw)
    return {int(k): [int(x) for x in v] for k, v in payload.items()}


def _load_window_mappings(
    repo_root: Path, path: str | Path
) -> list[dict[int, list[int]]]:
    raw = _load_json(repo_root, path)
    out = []
    for entry in raw:
        payload = entry.get("mapping", entry)
        out.append({int(k): [int(x) for x in v] for k, v in payload.items()})
    return out


def _load_rail_ports(network_config: Path, dies_per_wafer: int) -> list[int]:
    try:
        raw = yaml.safe_load(network_config.read_text(encoding="utf-8")) or {}
        ports = raw.get("rail-port-nodes")
        if ports:
            return [int(p) for p in ports]
    except Exception:
        pass
    side = int(dies_per_wafer**0.5)
    if side * side != dies_per_wafer:
        return list(range(dies_per_wafer))
    ports = []
    for r in range(side):
        for c in range(side):
            if r == 0 or c == 0 or r == side - 1 or c == side - 1:
                ports.append(r * side + c)
    return ports


def _edge_routed_segments(
    config: SimulationConfig,
    matrices: list[dict[int, dict[int, int]]],
    rail_ports: list[int],
) -> list[list[dict[int, dict[int, int]]]]:
    """Split cross-wafer P2P messages through wafer edge I/O dies.

    Segment 0: source die -> source wafer edge port.
    Segment 1: source edge port -> destination edge port through switch.
    Segment 2: destination edge port -> destination die.
    Same-wafer traffic stays in segment 0.
    """
    dpw = config.supernode.dies_per_wafer
    out = []
    for mat in matrices:
        segs: list[dict[int, dict[int, int]]] = [{}, {}, {}]
        for src, dsts in mat.items():
            for dst, size in dsts.items():
                if src // dpw == dst // dpw:
                    _add_edge(segs[0], src, dst, size)
                    continue
                src_port, dst_port = _select_edge_ports(config, src, dst, rail_ports)
                if src != src_port:
                    _add_edge(segs[0], src, src_port, size)
                _add_edge(segs[1], src_port, dst_port, size)
                if dst_port != dst:
                    _add_edge(segs[2], dst_port, dst, size)
        out.append(segs)
    return out


def _select_edge_ports(
    config: SimulationConfig, src: int, dst: int, rail_ports: list[int]
) -> tuple[int, int]:
    dpw = config.supernode.dies_per_wafer
    sw, sl = divmod(src, dpw)
    dw, dl = divmod(dst, dpw)
    src_port = min(rail_ports, key=lambda p: (_mesh_hops(dpw, sl, p), p))
    dst_port = min(rail_ports, key=lambda p: (_mesh_hops(dpw, dl, p), p))
    return sw * dpw + src_port, dw * dpw + dst_port


def _mesh_hops(dies_per_wafer: int, a: int, b: int) -> int:
    side = int(dies_per_wafer**0.5)
    if side * side != dies_per_wafer:
        return abs(a - b)
    ar, ac = divmod(a, side)
    br, bc = divmod(b, side)
    return abs(ar - br) + abs(ac - bc)


def _add_edge(mat: dict[int, dict[int, int]], src: int, dst: int, size: int) -> None:
    if src == dst or size <= 0:
        return
    mat.setdefault(src, {})
    mat[src][dst] = mat[src].get(dst, 0) + int(size)
