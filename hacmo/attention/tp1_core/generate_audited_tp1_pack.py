#!/usr/bin/env python3
"""Generate the trace-driven TP1 ASTRA-Sim + Ramulator paper-facing pack."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from supernode_sim.backends.ramulator import (
    TimedHBMRequest,
    _resolve_ramulator_exec,
    run_timed_hbm_trace,
)
from supernode_sim.backends.tp1_astra import (
    TP1AstraAccess,
    TP1AstraConfig,
    TP1AstraPrefetch,
    TP1AstraWindow,
    run_tp1_astra_sequence,
)

from .mapping import MappingPlan, build_mapping_plan
from .offline_solver import OfflineCostModel, solve_nonuniform_mapping
from .routing import (
    KVBlock,
    PrefetchTransfer,
    ReplaySummary,
    ResidencyState,
    RoutedAccess,
    advance_prefetch_residency,
    advance_residency,
    commit_hbm_fills,
    commit_prefetch_fills,
    hbm_port_core,
    plan_layer_prefetch,
    route_window,
    seed_static_cache,
)
from .workload import (
    RequestLane,
    WorkloadWindow,
    build_lanes,
    common_layer_ids,
    evaluation_length_groups,
    load_profiles,
    split_middle_train,
    valid_decode_steps,
)


@dataclass(frozen=True)
class HardwareConfig:
    total_sram_bytes: int = 440 * 1024 * 1024
    cache_ratio: float = 0.20
    kv_entry_bytes: int = 656
    cacheline_bytes: int = 128
    hbm_ports: int = 8
    aggregate_hbm_gb_per_s: float = 4800.0
    noc_link_gb_per_s: float = 256.0
    noc_link_latency_ns: float = 2.0
    die_compute_tflops: float = 2250.0
    flops_per_selected_kv: float = 294_912.0
    topk: int = 2048
    n_layers: int = 61
    max_context: int = 18_000
    ramulator_channels_per_port: int = 16
    ramulator_channel_width_bits: int = 128
    ramulator_data_rate_mtps: int = 2000
    ramulator_tck_ns: float = 1.0
    ramulator_issue_width: int = 16
    ramulator_source_issue_width: int = 0
    ramulator_internal_prefetch_size: int = 2

    @property
    def hbm_port_gb_per_s(self) -> float:
        return self.aggregate_hbm_gb_per_s / self.hbm_ports


@dataclass
class WindowReplay:
    window_id: str
    layer_id: int
    decode_step: int
    accesses: tuple[RoutedAccess, ...]
    prefetches: tuple[PrefetchTransfer, ...]
    runtime_source: dict[str, str]
    prefetch_by_block: dict[KVBlock, PrefetchTransfer]
    summary: ReplaySummary


@dataclass
class HBMEvent:
    request_id: str
    window_index: int
    kind: str
    port: int
    release_local_ns: float
    offset_ns: float
    access_ids: list[str]
    transfer_id: str | None = None
    completion_local_ns: float | None = None
    transaction_count: int = 0
    dram_bytes: int = 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_seed(*parts: object) -> int:
    payload = ":".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & 0x7FFFFFFF


def _mapping_fingerprint(plan: MappingPlan) -> str:
    payload = {
        "rows": plan.rows,
        "cols": plan.cols,
        "batch_size": plan.batch_size,
        "topk": plan.topk,
        "kv_entry_bytes": plan.kv_entry_bytes,
        "total_sram_bytes": plan.total_sram_bytes,
        "cache_budget_bytes": plan.cache_budget_bytes,
        "compute_gradient": plan.compute_gradient,
        "cache_gradient": plan.cache_gradient,
        "cores": [asdict(core) for core in plan.cores],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _hardware_fingerprint(hardware: HardwareConfig, rows: int, cols: int) -> str:
    payload = {**asdict(hardware), "rows": rows, "cols": cols}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_preceding_phase_us(
    csv_path: Path, *, topology: str, method: str, batches: Sequence[int]
) -> dict[int, float]:
    selected: dict[int, float] = {}
    with csv_path.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["topology"] != topology or row["method"] != method:
                continue
            batch = int(round(float(row["batch_per_die"])))
            if batch in batches:
                selected[batch] = float(row["total_us"])
    missing = sorted(set(batches) - set(selected))
    if missing:
        raise ValueError(
            f"preceding-phase CSV lacks {topology}/{method} batches {missing}"
        )
    return selected


def _build_plans(
    *,
    rows: int,
    cols: int,
    batch_size: int,
    training_lanes: Sequence[RequestLane],
    hardware: HardwareConfig,
    output_dir: Path,
    offline_gamma: float,
    offline_compute_gradients: Sequence[float],
    offline_cache_gradients: Sequence[float],
    offline_layer: int,
    offline_steps: Sequence[int],
) -> dict[str, MappingPlan]:
    model = OfflineCostModel(
        aggregate_hbm_gbps=hardware.aggregate_hbm_gb_per_s,
        hbm_ports=hardware.hbm_ports,
        hbm_internal_ns=200.0,
        hbm_concurrency_per_port=8,
        noc_link_gbps=hardware.noc_link_gb_per_s,
        gamma=offline_gamma,
        die_compute_tflops=hardware.die_compute_tflops,
        flops_per_selected_kv=hardware.flops_per_selected_kv,
        cacheline_bytes=hardware.cacheline_bytes,
        kv_entry_bytes=hardware.kv_entry_bytes,
        max_context=hardware.max_context,
        n_layers=hardware.n_layers,
    )
    best, scores = solve_nonuniform_mapping(
        rows=rows,
        cols=cols,
        batch_size=batch_size,
        topk=hardware.topk,
        kv_entry_bytes=hardware.kv_entry_bytes,
        total_sram_bytes=hardware.total_sram_bytes,
        cache_ratio=hardware.cache_ratio,
        lanes=training_lanes,
        layer_ids=(offline_layer,),
        decode_steps=tuple(offline_steps),
        model=model,
        residency_layer_ids=tuple(range(hardware.n_layers)),
        compute_gradients=tuple(offline_compute_gradients),
        cache_gradients=tuple(offline_cache_gradients),
    )
    plans = {
        "B": build_mapping_plan(
            policy="B",
            rows=rows,
            cols=cols,
            batch_size=batch_size,
            topk=hardware.topk,
            kv_entry_bytes=hardware.kv_entry_bytes,
            total_sram_bytes=hardware.total_sram_bytes,
            cache_ratio=0.0,
        ),
        "C": build_mapping_plan(
            policy="C",
            rows=rows,
            cols=cols,
            batch_size=batch_size,
            topk=hardware.topk,
            kv_entry_bytes=hardware.kv_entry_bytes,
            total_sram_bytes=hardware.total_sram_bytes,
            cache_ratio=hardware.cache_ratio,
        ),
        "S": best,
        "O": replace(best, policy="O"),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for method, plan in plans.items():
        plan.write_json(output_dir / f"mapping_{method}.json")
    (output_dir / "offline_scores.json").write_text(
        json.dumps([asdict(score) for score in scores], indent=2) + "\n",
        encoding="utf-8",
    )
    return plans


def _replay_policy(
    *,
    method: str,
    plan: MappingPlan,
    lanes: Sequence[RequestLane],
    layers: Sequence[int],
    decode_steps: Sequence[int],
    all_layer_ids: Sequence[int],
    hardware: HardwareConfig,
) -> list[WindowReplay]:
    state = ResidencyState(plan)
    seed_static_cache(state, lanes, all_layer_ids)
    history: dict[tuple[str, int], dict[int, int]] = {}
    history_tick = 0
    replayed: list[WindowReplay] = []
    sampled_layers = set(layers)
    last_sampled_layer = max(sampled_layers)
    for step_index, decode_step in enumerate(decode_steps):
        if method == "B":
            replay_layers = tuple(layer for layer in all_layer_ids if layer in sampled_layers)
        else:
            replay_layers = (
                all_layer_ids
                if step_index + 1 < len(decode_steps)
                else tuple(layer for layer in all_layer_ids if layer <= last_sampled_layer)
            )
        for layer_id in replay_layers:
            window = WorkloadWindow(layer_id, decode_step, tuple(lanes))
            sampled = layer_id in sampled_layers
            prefetches: tuple[PrefetchTransfer, ...] = ()
            if method == "O":
                if sampled:
                    prefetches = plan_layer_prefetch(
                        state=state,
                        lanes=lanes,
                        layer_id=layer_id,
                        decode_step=decode_step,
                        history=history,
                        kv_entry_bytes=hardware.kv_entry_bytes,
                        cacheline_bytes=hardware.cacheline_bytes,
                        num_hbm_ports=hardware.hbm_ports,
                        max_context=hardware.max_context,
                        n_layers=hardware.n_layers,
                    )
                else:
                    advance_prefetch_residency(state, lanes, layer_id, history)
            if not sampled:
                blocks = advance_residency(
                    state,
                    window,
                    kv_entry_bytes=hardware.kv_entry_bytes,
                    num_hbm_ports=hardware.hbm_ports,
                    max_context=hardware.max_context,
                    n_layers=hardware.n_layers,
                    collect_blocks=(method == "O"),
                )
                if method == "O":
                    for block in blocks:
                        history.setdefault(
                            (block.request_id, block.layer_id), {}
                        )[block.token_id] = history_tick
                        history_tick += 1
                continue
            prefetch_by_block = {transfer.block: transfer for transfer in prefetches}
            accesses, summary = route_window(
                window=window,
                plan=plan,
                state=state,
                kv_entry_bytes=hardware.kv_entry_bytes,
                cacheline_bytes=hardware.cacheline_bytes,
                num_hbm_ports=hardware.hbm_ports,
                max_context=hardware.max_context,
                n_layers=hardware.n_layers,
            )
            runtime_source = {
                access.access_id: (
                    "prefetch"
                    if access.source == "hbm" and access.block in prefetch_by_block
                    else access.source
                )
                for access in accesses
            }
            replayed.append(
                WindowReplay(
                    window_id=f"L{layer_id}-D{decode_step}",
                    layer_id=layer_id,
                    decode_step=decode_step,
                    accesses=accesses,
                    prefetches=prefetches,
                    runtime_source=runtime_source,
                    prefetch_by_block=prefetch_by_block,
                    summary=summary,
                )
            )
            commit_prefetch_fills(state, prefetches)
            misplaced_prefetches = [
                transfer.transfer_id
                for transfer in prefetches
                if state.owner(transfer.block) != transfer.target_core
            ]
            if misplaced_prefetches:
                raise RuntimeError(
                    "planned layer prefetches exceeded physical per-core capacity: "
                    f"{misplaced_prefetches[:8]}"
                )
            commit_hbm_fills(state, accesses)
            if method == "O":
                for access in accesses:
                    history.setdefault(
                        (access.block.request_id, access.block.layer_id), {}
                    )[access.block.token_id] = history_tick
                    history_tick += 1
    return replayed


def _write_canonical_trace(output_dir: Path, windows: Sequence[WindowReplay]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_dir / "per_core_kv_accesses.jsonl.gz", "wt", encoding="utf-8") as stream:
        for window in windows:
            for access in window.accesses:
                stream.write(
                    json.dumps(
                        {
                            "window_id": window.window_id,
                            **access.to_dict(),
                            "runtime_source": window.runtime_source[access.access_id],
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
    with gzip.open(output_dir / "layer_prefetches.jsonl.gz", "wt", encoding="utf-8") as stream:
        for window in windows:
            for transfer in window.prefetches:
                stream.write(
                    json.dumps(
                        {"window_id": window.window_id, **transfer.to_dict()},
                        separators=(",", ":"),
                    )
                    + "\n"
                )


def _build_hbm_events(
    *,
    windows: Sequence[WindowReplay],
    preceding_phase_us: float,
    window_gap_ns: float,
    hardware: HardwareConfig,
) -> tuple[dict[int, list[TimedHBMRequest]], dict[str, HBMEvent], dict[str, str]]:
    requests: dict[int, list[TimedHBMRequest]] = {
        port: [] for port in range(hardware.hbm_ports)
    }
    events: dict[str, HBMEvent] = {}
    event_by_access: dict[str, str] = {}
    for window_index, window in enumerate(windows):
        offset = window_index * window_gap_ns
        for transfer in window.prefetches:
            request_id = f"W{window_index}:P:{transfer.transfer_id}"
            event = HBMEvent(
                request_id=request_id,
                window_index=window_index,
                kind="prefetch",
                port=transfer.hbm_port,
                release_local_ns=0.0,
                offset_ns=offset,
                access_ids=[],
                transfer_id=transfer.transfer_id,
            )
            events[request_id] = event
            requests[transfer.hbm_port].append(
                TimedHBMRequest(
                    request_id=request_id,
                    source_core=transfer.target_core,
                    release_time_ns=offset,
                    address=transfer.hbm_address,
                    size_bytes=transfer.size_bytes,
                )
            )

        demand_by_block: dict[KVBlock, list[RoutedAccess]] = defaultdict(list)
        for access in window.accesses:
            if window.runtime_source[access.access_id] == "hbm":
                demand_by_block[access.block].append(access)
        for demand_index, block in enumerate(sorted(demand_by_block)):
            grouped = demand_by_block[block]
            primary = grouped[0]
            assert primary.hbm_port is not None
            assert primary.hbm_address is not None
            request_id = f"W{window_index}:D:{demand_index}:{primary.access_id}"
            event = HBMEvent(
                request_id=request_id,
                window_index=window_index,
                kind="demand",
                port=primary.hbm_port,
                release_local_ns=preceding_phase_us * 1000.0,
                offset_ns=offset,
                access_ids=[access.access_id for access in grouped],
            )
            events[request_id] = event
            for access in grouped:
                event_by_access[access.access_id] = request_id
            requests[primary.hbm_port].append(
                TimedHBMRequest(
                    request_id=request_id,
                    source_core=primary.consumer_core,
                    release_time_ns=offset + preceding_phase_us * 1000.0,
                    address=primary.hbm_address,
                    size_bytes=primary.size_bytes,
                )
            )
    return requests, events, event_by_access


def _run_ramulator_sequence(
    *,
    requests_by_port: Mapping[int, Sequence[TimedHBMRequest]],
    events: dict[str, HBMEvent],
    output_dir: Path,
    hardware: HardwareConfig,
    ramulator_exec: str,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    requested_bytes = 0
    dram_bytes = 0
    completed = 0
    port_rows: list[dict[str, object]] = []
    requests_per_port = {
        port: tuple(requests_by_port.get(port, ()))
        for port in range(hardware.hbm_ports)
    }
    request_by_id = {
        request.request_id: request
        for requests in requests_per_port.values()
        for request in requests
    }
    if len(request_by_id) != sum(len(requests) for requests in requests_per_port.values()):
        raise ValueError("HBM request IDs must be globally unique across ports")

    def run_port(port: int):
        requests = requests_per_port[port]
        return run_timed_hbm_trace(
            requests,
            output_dir / f"port_{port}",
            ramulator_exec=ramulator_exec,
            cacheline_bytes=hardware.cacheline_bytes,
            channels=hardware.ramulator_channels_per_port,
            channel_width_bits=hardware.ramulator_channel_width_bits,
            data_rate_mtps=hardware.ramulator_data_rate_mtps,
            target_peak_gbps=hardware.hbm_port_gb_per_s,
            tck_ns=hardware.ramulator_tck_ns,
            issue_width=hardware.ramulator_issue_width,
            source_issue_width=hardware.ramulator_source_issue_width,
            dram_internal_prefetch_size=hardware.ramulator_internal_prefetch_size,
            retain_transaction_artifacts=False,
        )

    active_ports = [port for port, requests in requests_per_port.items() if requests]
    with ThreadPoolExecutor(max_workers=max(len(active_ports), 1)) as executor:
        port_results = dict(zip(active_ports, executor.map(run_port, active_ports)))

    for port in range(hardware.hbm_ports):
        requests = requests_per_port[port]
        if not requests:
            port_rows.append({"port": port, "logical_requests": 0})
            continue
        result = port_results[port]
        requested_bytes += result.requested_bytes
        dram_bytes += result.dram_transaction_bytes
        completed += len(result.completions)
        for completion in result.completions:
            event = events[completion.request_id]
            event.completion_local_ns = completion.completion_time_ns - event.offset_ns
            event.transaction_count = completion.transaction_count
            event.dram_bytes = (
                completion.transaction_count
                * result.dram_transaction_size_bytes
            )
            request = request_by_id[completion.request_id]
            first_line = request.address // hardware.cacheline_bytes
            last_line = (
                request.address + request.size_bytes - 1
            ) // hardware.cacheline_bytes
            expected_dram_bytes = (
                last_line - first_line + 1
            ) * hardware.cacheline_bytes
            if event.dram_bytes != expected_dram_bytes:
                raise RuntimeError(
                    "Ramulator physical bytes do not cover the logical cachelines: "
                    f"{completion.request_id}: {event.dram_bytes} != "
                    f"{expected_dram_bytes}"
                )
            if event.completion_local_ns + 1e-9 < event.release_local_ns:
                raise RuntimeError(
                    f"Ramulator completion precedes release: {completion.request_id}"
                )
        port_rows.append(
            {
                "port": port,
                "logical_requests": len(requests),
                "requested_bytes": result.requested_bytes,
                "dram_transaction_bytes": result.dram_transaction_bytes,
                "target_peak_gb_per_s": result.target_peak_gbps,
                "raw_peak_gb_per_s": result.raw_peak_gbps,
                "dram_transaction_size_bytes": result.dram_transaction_size_bytes,
                "completion_csv": result.completion_csv,
                "input_trace_sha256": result.trace_sha256,
                "transaction_completion_sha256": result.transaction_completion_sha256,
                "transaction_artifacts_retained": result.transaction_artifacts_retained,
            }
        )
    if completed != len(events):
        raise RuntimeError(f"Ramulator logical completion mismatch: {completed} != {len(events)}")
    summary = {
        "logical_requests": len(events),
        "completed_requests": completed,
        "requested_bytes": requested_bytes,
        "dram_transaction_bytes": dram_bytes,
        "logical_cacheline_bytes": hardware.cacheline_bytes,
        "dram_transaction_size_bytes": (
            hardware.ramulator_internal_prefetch_size
            * hardware.ramulator_channel_width_bits
            // 8
        ),
        "issue_width": hardware.ramulator_issue_width,
        "source_issue_width": hardware.ramulator_source_issue_width,
        "bandwidth_unit": "GB/s",
        "ports": port_rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _astra_windows(
    *,
    windows: Sequence[WindowReplay],
    events: Mapping[str, HBMEvent],
    event_by_access: Mapping[str, str],
    hardware: HardwareConfig,
    rows: int,
    cols: int,
) -> tuple[TP1AstraWindow, ...]:
    event_by_transfer = {
        event.transfer_id: event
        for event in events.values()
        if event.transfer_id is not None
    }
    astra_windows: list[TP1AstraWindow] = []
    for window in windows:
        prefetches: list[TP1AstraPrefetch] = []
        for transfer in window.prefetches:
            event = event_by_transfer[transfer.transfer_id]
            assert event.completion_local_ns is not None
            prefetches.append(
                TP1AstraPrefetch(
                    transfer_id=transfer.transfer_id,
                    target_core=transfer.target_core,
                    hbm_port_core=hbm_port_core(
                        transfer.hbm_port, rows, cols, hardware.hbm_ports
                    ),
                    size_bytes=transfer.size_bytes,
                    hbm_wire_bytes=event.dram_bytes,
                    hbm_completion_ns=event.completion_local_ns,
                )
            )
        astra_accesses: list[TP1AstraAccess] = []
        for access in window.accesses:
            source = window.runtime_source[access.access_id]
            if source == "hbm":
                event = events[event_by_access[access.access_id]]
                assert access.hbm_port is not None
                assert event.completion_local_ns is not None
                astra_accesses.append(
                    TP1AstraAccess(
                        access_id=access.access_id,
                        consumer_core=access.consumer_core,
                        source="hbm",
                        source_core=None,
                        cache_target_core=access.cache_target_core,
                        size_bytes=access.size_bytes,
                        hbm_port_core=hbm_port_core(
                            access.hbm_port, rows, cols, hardware.hbm_ports
                        ),
                        hbm_wire_bytes=event.dram_bytes,
                        hbm_completion_ns=event.completion_local_ns,
                    )
                )
            elif source == "prefetch":
                transfer = window.prefetch_by_block[access.block]
                astra_accesses.append(
                    TP1AstraAccess(
                        access_id=access.access_id,
                        consumer_core=access.consumer_core,
                        source="prefetch",
                        source_core=transfer.target_core,
                        cache_target_core=transfer.target_core,
                        size_bytes=access.size_bytes,
                        prefetch_transfer_id=transfer.transfer_id,
                    )
                )
            else:
                astra_accesses.append(
                    TP1AstraAccess(
                        access_id=access.access_id,
                        consumer_core=access.consumer_core,
                        source=source,
                        source_core=access.source_core,
                        cache_target_core=access.cache_target_core,
                        size_bytes=access.size_bytes,
                    )
                )
        astra_windows.append(
            TP1AstraWindow(
                window_id=window.window_id,
                accesses=tuple(astra_accesses),
                prefetches=tuple(prefetches),
            )
        )
    return tuple(astra_windows)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _validate_pack(
    rows: Sequence[dict[str, object]],
    window_rows: Sequence[dict[str, object]],
    core_rows: Sequence[dict[str, object]],
    case_records: Sequence[dict[str, object]],
    *,
    expected_groups: int,
    expected_windows_per_case: int,
) -> dict[str, object]:
    errors: list[str] = []
    warnings: list[str] = []
    expected_cases = expected_groups * 4
    if len(rows) != expected_cases:
        errors.append(f"metric row count is {len(rows)}, expected {expected_cases}")
    if len(case_records) != expected_cases:
        errors.append(
            f"case record count is {len(case_records)}, expected {expected_cases}"
        )
    expected_window_rows = expected_cases * expected_windows_per_case
    if len(window_rows) != expected_window_rows:
        errors.append(
            f"window row count is {len(window_rows)}, expected {expected_window_rows}"
        )
    expected_core_rows = sum(int(row["physical_cores"]) for row in rows)
    if len(core_rows) != expected_core_rows:
        errors.append(
            f"core row count is {len(core_rows)}, expected {expected_core_rows}"
        )
    core_keys = {
        (
            row["context_group"],
            row["array_rows"],
            row["batch_size"],
            row["method"],
            row["core_id"],
        )
        for row in core_rows
    }
    if len(core_keys) != len(core_rows):
        errors.append("core metrics contain duplicate case/core rows")
    by_case: dict[tuple[object, ...], dict[str, dict[str, object]]] = defaultdict(dict)
    for row in rows:
        key = (row["context_group"], row["array_rows"], row["batch_size"])
        by_case[key][str(row["method"])] = row
    if len(by_case) != expected_groups:
        errors.append(
            f"scenario group count is {len(by_case)}, expected {expected_groups}"
        )
    for key, methods in by_case.items():
        if set(methods) != {"B", "C", "S", "O"}:
            errors.append(f"{key}: methods are {sorted(methods)}")
            continue
        logical = {int(row["logical_accesses"]) for row in methods.values()}
        hardware = {str(row["hardware_fingerprint"]) for row in methods.values()}
        if len(logical) != 1:
            errors.append(f"{key}: methods do not conserve identical logical work")
        if len(hardware) != 1:
            errors.append(f"{key}: methods do not use identical hardware")
        if methods["S"]["mapping_fingerprint"] != methods["O"]["mapping_fingerprint"]:
            errors.append(f"{key}: S/O mapping fingerprints differ")
        if int(methods["B"]["cache_entries"]) != 0:
            errors.append(f"{key}: B has persistent cache entries")
        if len(
            {
                int(methods[method]["cache_entries"])
                for method in ("C", "S", "O")
            }
        ) != 1:
            errors.append(f"{key}: C/S/O do not use the same cache budget")
        if int(methods["C"]["prefetch_transfers"]) or int(methods["S"]["prefetch_transfers"]):
            errors.append(f"{key}: C/S unexpectedly contain prefetch transfers")
        for method, row in methods.items():
            sources = sum(
                int(row[name])
                for name in (
                    "local_accesses",
                    "remote_accesses",
                    "prefetched_accesses",
                    "hbm_demand_accesses",
                )
            )
            if sources != int(row["logical_accesses"]):
                errors.append(f"{key}/{method}: routed sources do not conserve work")
            if int(row["hbm_requests"]) != int(row["hbm_completed_requests"]):
                errors.append(f"{key}/{method}: incomplete Ramulator requests")
            if int(row["hbm_dram_bytes"]) < int(row["hbm_requested_bytes"]):
                errors.append(f"{key}/{method}: HBM physical bytes omit requested data")
            if int(row["physical_cores_reported"]) != int(row["physical_cores"]):
                errors.append(f"{key}/{method}: incomplete ASTRA core coverage")
        if float(methods["S"]["mean_exposed_us"]) >= float(methods["C"]["mean_exposed_us"]):
            warnings.append(f"{key}: S does not improve over C")
        if float(methods["O"]["mean_exposed_us"]) >= float(methods["S"]["mean_exposed_us"]):
            warnings.append(f"{key}: O does not improve over S")

    for record in case_records:
        if int(record["compute_entries"]) != int(record["expected_compute_entries"]):
            errors.append(f"{record['case_id']}: mapping compute work is not conserved")
        if int(record["sram_accounted_bytes"]) > int(record["total_sram_bytes"]):
            errors.append(f"{record['case_id']}: SRAM capacity exceeded")
        if record["runtime_gamma_present"]:
            errors.append(f"{record['case_id']}: gamma leaked into runtime simulator input")
        astra_record = dict(record["astra_result"])
        if astra_record.get("execution_traces_retained"):
            errors.append(f"{record['case_id']}: disposable ASTRA traces were retained")
        if len(str(astra_record.get("trace_sha256", ""))) != 64:
            errors.append(f"{record['case_id']}: ASTRA trace hash is missing")
        for port in record["hbm_summary"]["ports"]:
            if not int(port.get("logical_requests", 0)):
                continue
            completion_path = Path(str(port.get("completion_csv", "")))
            if not completion_path.is_file():
                errors.append(
                    f"{record['case_id']}/port{port['port']}: logical completions are missing"
                )
            if port.get("transaction_artifacts_retained"):
                errors.append(
                    f"{record['case_id']}/port{port['port']}: disposable Ramulator transactions were retained"
                )
            if len(str(port.get("input_trace_sha256", ""))) != 64 or len(
                str(port.get("transaction_completion_sha256", ""))
            ) != 64:
                errors.append(
                    f"{record['case_id']}/port{port['port']}: Ramulator transaction hash is missing"
                )

    by_window: dict[tuple[object, ...], dict[str, dict[str, object]]] = defaultdict(dict)
    for row in window_rows:
        key = (
            row["context_group"],
            row["array_rows"],
            row["batch_size"],
            row["window_id"],
        )
        by_window[key][str(row["method"])] = row
    expected_window_groups = expected_groups * expected_windows_per_case
    if len(by_window) != expected_window_groups:
        errors.append(
            f"window group count is {len(by_window)}, expected {expected_window_groups}"
        )
    for key, methods in by_window.items():
        if set(methods) != {"B", "C", "S", "O"}:
            errors.append(f"{key}: window methods are {sorted(methods)}")
            continue
        if len({int(row["logical_accesses"]) for row in methods.values()}) != 1:
            errors.append(f"{key}: window logical work differs across methods")
        for method, row in methods.items():
            routed = sum(
                int(row[field])
                for field in (
                    "local_accesses",
                    "remote_accesses",
                    "prefetched_accesses",
                    "hbm_demand_accesses",
                )
            )
            if routed != int(row["logical_accesses"]):
                errors.append(f"{key}/{method}: window routed work is not conserved")
            requested = int(row["hbm_prefetch_requests"]) + int(
                row["hbm_demand_requests"]
            )
            if requested != int(row["hbm_completed_requests"]):
                errors.append(f"{key}/{method}: window HBM requests did not drain")
    return {
        "status": "ok" if not errors else "error",
        "errors": errors,
        "warnings": warnings,
        "groups_checked": len(by_case),
        "case_records_checked": len(case_records),
        "windows_checked": len(by_window),
        "metric_rows_checked": len(rows),
        "window_rows_checked": len(window_rows),
        "core_rows_checked": len(core_rows),
    }


def generate_pack(args: argparse.Namespace) -> Path:
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    profile_root = Path(args.profile_dir).resolve()
    profiles = load_profiles(profile_root)
    training_profiles, evaluation_profiles = split_middle_train(profiles)
    groups = evaluation_length_groups(evaluation_profiles)
    requested_groups = tuple(args.context_groups)
    missing_groups = sorted(set(requested_groups) - set(groups))
    if missing_groups:
        raise ValueError(f"missing measured evaluation groups: {missing_groups}")
    if {profile.sha256 for profile in training_profiles} & {
        profile.sha256 for profile in evaluation_profiles
    }:
        raise ValueError("offline mapping and heldout profiles overlap")

    hardware = HardwareConfig(
        total_sram_bytes=int(args.total_sram_mb * 1024 * 1024),
        cache_ratio=args.cache_ratio,
        kv_entry_bytes=args.kv_entry_bytes,
        cacheline_bytes=args.cacheline_bytes,
        hbm_ports=args.hbm_ports,
        aggregate_hbm_gb_per_s=args.aggregate_hbm_gb_per_s,
        noc_link_gb_per_s=args.noc_link_gb_per_s,
        noc_link_latency_ns=args.noc_link_latency_ns,
        die_compute_tflops=args.die_compute_tflops,
        flops_per_selected_kv=args.flops_per_selected_kv,
        topk=args.topk,
        n_layers=args.n_layers,
        max_context=args.max_context,
        ramulator_channels_per_port=args.ramulator_channels_per_port,
        ramulator_channel_width_bits=args.ramulator_channel_width_bits,
        ramulator_data_rate_mtps=args.ramulator_data_rate_mtps,
        ramulator_tck_ns=args.ramulator_tck_ns,
        ramulator_issue_width=args.ramulator_issue_width,
        ramulator_source_issue_width=args.ramulator_source_issue_width,
        ramulator_internal_prefetch_size=args.ramulator_internal_prefetch_size,
    )
    batches = tuple(args.batch_sizes)
    tp2_csv = Path(args.preceding_phase_csv).resolve()
    preceding_us = _load_preceding_phase_us(
        tp2_csv,
        topology=args.preceding_phase_topology,
        method=args.preceding_phase_method,
        batches=batches,
    )
    astra_dir = Path(args.astra_sim_dir).resolve()
    system_config = Path(args.astra_system_config).resolve()
    remote_config = Path(args.astra_remote_memory_config).resolve()
    ramulator_binary = Path(_resolve_ramulator_exec(args.ramulator_exec))
    astra_binary = (
        astra_dir
        / "build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware"
    ).resolve()
    metrics: list[dict[str, object]] = []
    window_metrics: list[dict[str, object]] = []
    core_metrics: list[dict[str, object]] = []
    case_records: list[dict[str, object]] = []
    plan_cache: dict[tuple[int, int], dict[str, MappingPlan]] = {}

    for side in args.array_sides:
        for batch in batches:
            print(f"[TP1] offline mapping: {side}x{side}, batch={batch}", flush=True)
            train_lanes = build_lanes(
                training_profiles,
                batch,
                _stable_seed(args.seed, "offline", side, batch),
            )
            offline_steps = valid_decode_steps(
                train_lanes,
                (args.offline_layer,),
                tuple(args.offline_decode_steps),
            )
            if len(offline_steps) != len(args.offline_decode_steps):
                raise ValueError("training profiles do not cover offline decode steps")
            plans = _build_plans(
                rows=side,
                cols=side,
                batch_size=batch,
                training_lanes=train_lanes,
                hardware=hardware,
                output_dir=output_root / "offline_mapping" / f"{side}x{side}_b{batch}",
                offline_gamma=args.offline_gamma,
                offline_compute_gradients=tuple(args.offline_compute_gradients),
                offline_cache_gradients=tuple(args.offline_cache_gradients),
                offline_layer=args.offline_layer,
                offline_steps=offline_steps,
            )
            plan_cache[(side, batch)] = plans

    for context_group in requested_groups:
        group_profiles = groups[context_group]
        for side in args.array_sides:
            for batch in batches:
                lanes = build_lanes(
                    group_profiles,
                    batch,
                    _stable_seed(args.seed, "heldout", context_group, side, batch),
                )
                layers = common_layer_ids(group_profiles, tuple(args.layers))
                if len(layers) != len(args.layers):
                    raise ValueError(f"{context_group} profiles do not cover requested layers")
                decode_steps = valid_decode_steps(
                    lanes, layers, tuple(args.decode_steps)
                )
                if len(decode_steps) != len(args.decode_steps):
                    raise ValueError(
                        f"{context_group} profiles do not cover requested decode steps"
                    )
                all_layers = common_layer_ids(group_profiles, tuple(range(hardware.n_layers)))
                if len(all_layers) != hardware.n_layers:
                    raise ValueError(f"{context_group} profiles lack complete layer coverage")
                for method in ("B", "C", "S", "O"):
                    case_id = f"{context_group}_{side}x{side}_b{batch}_{method}"
                    print(f"[TP1] replay/simulate: {case_id}", flush=True)
                    plan = plan_cache[(side, batch)][method]
                    case_root = output_root / "cases" / case_id
                    windows = _replay_policy(
                        method=method,
                        plan=plan,
                        lanes=lanes,
                        layers=layers,
                        decode_steps=decode_steps,
                        all_layer_ids=all_layers,
                        hardware=hardware,
                    )
                    _write_canonical_trace(case_root / "trace", windows)
                    requests, events, event_by_access = _build_hbm_events(
                        windows=windows,
                        preceding_phase_us=preceding_us[batch],
                        window_gap_ns=args.ramulator_window_gap_ns,
                        hardware=hardware,
                    )
                    hbm_summary = _run_ramulator_sequence(
                        requests_by_port=requests,
                        events=events,
                        output_dir=case_root / "ramulator",
                        hardware=hardware,
                        ramulator_exec=args.ramulator_exec,
                    )
                    astra_input = _astra_windows(
                        windows=windows,
                        events=events,
                        event_by_access=event_by_access,
                        hardware=hardware,
                        rows=side,
                        cols=side,
                    )
                    astra_result = run_tp1_astra_sequence(
                        TP1AstraConfig(
                            rows=side,
                            cols=side,
                            astra_sim_dir=astra_dir,
                            output_dir=case_root / "astra",
                            system_config=system_config,
                            remote_memory_config=remote_config,
                            link_bandwidth_gb_per_s=hardware.noc_link_gb_per_s,
                            link_latency_ns=hardware.noc_link_latency_ns,
                            die_compute_tflops=hardware.die_compute_tflops,
                            flops_per_access=hardware.flops_per_selected_kv,
                            preceding_phase_us=preceding_us[batch],
                            time_scale=args.astra_time_scale,
                            max_items_per_flow=args.astra_max_items_per_flow,
                            retain_execution_traces=False,
                        ),
                        astra_input,
                    )
                    counts = Counter(
                        window.runtime_source[access.access_id]
                        for window in windows
                        for access in window.accesses
                    )
                    logical = sum(len(window.accesses) for window in windows)
                    prefetch_count = sum(len(window.prefetches) for window in windows)
                    remote_byte_hops = sum(
                        window.summary.remote_byte_hops for window in windows
                    )
                    hbm_byte_hops = sum(
                        window.summary.hbm_byte_hops for window in windows
                    )
                    mapping_fp = _mapping_fingerprint(plan)
                    hardware_fp = _hardware_fingerprint(hardware, side, side)
                    row = {
                        "context_group": context_group,
                        "prompt_len_min": min(profile.prompt_len for profile in group_profiles),
                        "prompt_len_max": max(profile.prompt_len for profile in group_profiles),
                        "array_rows": side,
                        "array_cols": side,
                        "batch_size": batch,
                        "method": method,
                        "sampled_windows": len(windows),
                        "logical_accesses": logical,
                        "local_accesses": counts["local"],
                        "remote_accesses": counts["remote"],
                        "prefetched_accesses": counts["prefetch"],
                        "hbm_demand_accesses": counts["hbm"],
                        "static_hit_rate": (counts["local"] + counts["remote"])
                        / max(logical, 1),
                        "prefetch_covered_rate": counts["prefetch"] / max(logical, 1),
                        "prefetch_transfers": prefetch_count,
                        "remote_byte_hops": remote_byte_hops,
                        "hbm_byte_hops": hbm_byte_hops,
                        "cache_entries": plan.total_cache_entries,
                        "compute_gradient": plan.compute_gradient,
                        "cache_gradient": plan.cache_gradient,
                        "mapping_fingerprint": mapping_fp,
                        "hardware_fingerprint": hardware_fp,
                        "hbm_requests": hbm_summary["logical_requests"],
                        "hbm_completed_requests": hbm_summary["completed_requests"],
                        "hbm_requested_bytes": hbm_summary["requested_bytes"],
                        "hbm_dram_bytes": hbm_summary["dram_transaction_bytes"],
                        "astra_flow_count": astra_result.flow_count,
                        "astra_communication_bytes": astra_result.communication_bytes,
                        "preceding_phase_us_per_window": preceding_us[batch],
                        "sequence_wall_us": astra_result.wall_us,
                        "sequence_exposed_us": astra_result.exposed_us,
                        "mean_exposed_us": astra_result.exposed_us / len(windows),
                        "compute_busy_us_sum": astra_result.compute_busy_us,
                        "physical_cores": side * side,
                        "physical_cores_reported": len(astra_result.per_core),
                        "ramulator_backend": "ramulator2_hbm3_timed_trace",
                        "noc_backend": astra_result.backend,
                        "mapping_role": "offline_technique_output",
                        "runtime_role": "simulator_validation",
                    }
                    metrics.append(row)
                    for core_row in astra_result.per_core:
                        post_phase_finish = float(
                            core_row["post_phase_finish_us"]
                        )
                        compute_busy = float(core_row["compute_busy_us"])
                        core_metrics.append(
                            {
                                "context_group": context_group,
                                "array_rows": side,
                                "array_cols": side,
                                "batch_size": batch,
                                "method": method,
                                "core_id": core_row["core_id"],
                                "row": core_row["row"],
                                "col": core_row["col"],
                                "logical_accesses": core_row["logical_accesses"],
                                "compute_busy_us": compute_busy,
                                "astra_comm_us": core_row["astra_comm_us"],
                                "astra_finish_us": core_row["astra_finish_us"],
                                "post_phase_finish_us": post_phase_finish,
                                "exposed_noncompute_us": max(
                                    post_phase_finish - compute_busy, 0.0
                                ),
                            }
                        )
                    for window_index, window in enumerate(windows):
                        window_counts = Counter(
                            window.runtime_source[access.access_id]
                            for access in window.accesses
                        )
                        window_events = [
                            event
                            for event in events.values()
                            if event.window_index == window_index
                        ]
                        window_metrics.append(
                            {
                                "context_group": context_group,
                                "array_rows": side,
                                "array_cols": side,
                                "batch_size": batch,
                                "method": method,
                                "window_id": window.window_id,
                                "layer_id": window.layer_id,
                                "decode_step": window.decode_step,
                                "logical_accesses": len(window.accesses),
                                "local_accesses": window_counts["local"],
                                "remote_accesses": window_counts["remote"],
                                "prefetched_accesses": window_counts["prefetch"],
                                "hbm_demand_accesses": window_counts["hbm"],
                                "prefetch_transfers": len(window.prefetches),
                                "remote_byte_hops": window.summary.remote_byte_hops,
                                "hbm_byte_hops": window.summary.hbm_byte_hops,
                                "hbm_prefetch_requests": sum(
                                    event.kind == "prefetch" for event in window_events
                                ),
                                "hbm_demand_requests": sum(
                                    event.kind == "demand" for event in window_events
                                ),
                                "hbm_completed_requests": sum(
                                    event.completion_local_ns is not None
                                    for event in window_events
                                ),
                            }
                        )
                    sram_accounted = sum(
                        core.compute_memory_bytes
                        + core.cache_capacity_entries * plan.kv_entry_bytes
                        for core in plan.cores
                    )
                    astra_network_text = Path(astra_result.network_config).read_text(
                        encoding="utf-8"
                    )
                    projection_text = Path(astra_result.projection_summary).read_text(
                        encoding="utf-8"
                    )
                    case_records.append(
                        {
                            "case_id": case_id,
                            "method": method,
                            "mapping_fingerprint": mapping_fp,
                            "hardware_fingerprint": hardware_fp,
                            "compute_entries": plan.total_compute_entries,
                            "expected_compute_entries": batch * hardware.topk,
                            "sram_accounted_bytes": sram_accounted,
                            "total_sram_bytes": hardware.total_sram_bytes,
                            "runtime_gamma_present": "gamma" in astra_network_text.lower()
                            or "gamma" in projection_text.lower(),
                            "profile_hashes": [lane.profile.sha256 for lane in lanes],
                            "astra_result": astra_result.to_dict(),
                            "hbm_summary": hbm_summary,
                        }
                    )
                    (case_root / "case_record.json").write_text(
                        json.dumps(case_records[-1], indent=2) + "\n",
                        encoding="utf-8",
                    )

    grouped_rows: dict[tuple[object, ...], dict[str, dict[str, object]]] = defaultdict(dict)
    for row in metrics:
        key = (row["context_group"], row["array_rows"], row["batch_size"])
        grouped_rows[key][str(row["method"])] = row
    for methods in grouped_rows.values():
        baseline_b = float(methods["B"]["mean_exposed_us"])
        baseline_c = float(methods["C"]["mean_exposed_us"])
        for row in methods.values():
            latency = float(row["mean_exposed_us"])
            row["speedup_over_B"] = baseline_b / max(latency, 1e-12)
            row["speedup_over_C"] = baseline_c / max(latency, 1e-12)

    _write_csv(output_root / "csv" / "tp1_audited_metrics.csv", metrics)
    _write_csv(
        output_root / "csv" / "tp1_audited_window_metrics.csv",
        window_metrics,
    )
    _write_csv(
        output_root / "csv" / "tp1_audited_core_metrics.csv",
        core_metrics,
    )
    validation = _validate_pack(
        metrics,
        window_metrics,
        core_metrics,
        case_records,
        expected_groups=len(requested_groups) * len(args.array_sides) * len(batches),
        expected_windows_per_case=len(args.layers) * len(args.decode_steps),
    )
    (output_root / "validation.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )
    metadata = {
        "status": "current" if validation["status"] == "ok" else "invalid",
        "scope": "non-CP TP1 selected-KV core attention",
        "methods": {
            "B": "uniform mapping, no persistent KV cache",
            "C": "uniform compute/cache mapping, no layer prefetch",
            "S": "frozen offline non-uniform compute/cache mapping only",
            "O": "identical S mapping plus causal layer prefetch",
        },
        "hardware": asdict(hardware),
        "noc_link_setting": {
            "scope": "per directed physical Mesh2D link",
            "configured_gb_per_s": hardware.noc_link_gb_per_s,
            "nominal_derivation": "2048 bit/cycle * 1.0 GHz / 8 = 256 GB/s",
            "paper_range": "128-2048 bit/cycle; frequency is independently configurable in the paper design space",
            "legacy_5tbps_interpretation": "shared 20-column analytical bandwidth, not per-link bandwidth",
        },
        "compute_scope": {
            "included": "selected-KV QK and SV arithmetic",
            "flops_per_selected_kv_formula": "2 operations * 2 matmuls * 128 heads * 576 dimensions",
            "excluded_common_work": "QKV/output projections are method-identical and outside this selected-KV core pack",
        },
        "runtime_backends": {
            "hbm": "request-aware Ramulator2 HBM3 with full drain",
            "noc": "ASTRA-Sim congestion-aware Mesh2D with deterministic XY routing",
            "gamma_used": False,
            "ramulator_binary": str(ramulator_binary),
            "ramulator_binary_sha256": _sha256(ramulator_binary),
            "astra_binary": str(astra_binary),
            "astra_binary_sha256": _sha256(astra_binary),
            "astra_system_config_sha256": _sha256(system_config),
            "astra_remote_memory_config_sha256": _sha256(remote_config),
        },
        "offline_solver": {
            "equation": "paper Eq. (1)-style candidate scoring only",
            "gamma": args.offline_gamma,
            "compute_gradients": args.offline_compute_gradients,
            "cache_gradients": args.offline_cache_gradients,
            "selection_constraint": "S requires positive compute and cache gradients; zero-gradient scores are retained as controls",
            "training_profile_hashes": [profile.sha256 for profile in training_profiles],
            "layer": args.offline_layer,
            "decode_steps": args.offline_decode_steps,
            "cache_population_layers": list(range(hardware.n_layers)),
        },
        "heldout_profiles": {
            name: [
                {
                    "prompt_id": profile.prompt_id,
                    "task": profile.task,
                    "prompt_len": profile.prompt_len,
                    "sha256": profile.sha256,
                    "source_path": profile.source_path,
                }
                for profile in group
            ]
            for name, group in groups.items()
            if name in requested_groups
        },
        "sampled_layers": args.layers,
        "sampled_decode_steps": args.decode_steps,
        "sample_order": "decode step major, then layer; cache/LRU state replays every layer while only requested layers enter the timed simulator",
        "preceding_phase": {
            "role": "identical frozen overlap window for every TP1 method",
            "source_csv": str(tp2_csv),
            "source_sha256": _sha256(tp2_csv),
            "topology": args.preceding_phase_topology,
            "method": args.preceding_phase_method,
            "duration_us_by_batch": preceding_us,
        },
        "excluded_components": {
            "indexer": "analytical boundary, not included in this core-attention pack",
            "cp_communication": "may be reported separately as analytical; not included here",
            "runtime_closed_loop_source_selection": "future optional mechanism",
        },
        "trace_contract": {
            "canonical_input": "per-core KV access routing trace",
            "hbm_miss_source": "causal residency replay",
            "baseline_compute_assignment": "B/C use a deterministic topology-oblivious uniform quota schedule for each decode window",
            "nonuniform_compute_assignment": "S/O exactly preserve the offline per-core quotas while assigning each window's selected KV work from its fixed address-derived HBM port to the nearest core with remaining quota",
            "assignment_isolation": "the per-window assignment is completed before source classification and reads only the frozen mapping plus the current selected KV identities and addresses; it does not read cache state, runtime congestion, Ramulator completion, or ASTRA output",
            "source_selection": "after the per-window compute assignment is frozen, explicit persistent single-owner residency classifies each access as local, remote, or HBM",
            "demand_fill": "consumer-local, capacity-accounted LRU insertion after HBM arrival",
            "sampled_state_equivalence": "untimed layers reuse the same compute assignment, demand-fill, prefetch-target, and LRU rules as timed layers",
            "hbm_port_mapping": "KV block round-robin across ports; dense port-local block address inside each Ramulator instance",
            "hbm_transaction_projection": "each logical cacheline expands into HBM3 internal-prefetch transactions; ASTRA receives their conserved total bytes",
            "layer_prefetch_capacity": "min(per-request physical cache share, topk) KV entries per request",
            "flow_packetization": args.astra_max_items_per_flow,
            "flow_packetization_semantics": "lossless byte/work aggregation, no core or access sampling",
            "hbm_readiness_quantum_ns": 1000.0 / args.astra_time_scale,
            "hbm_packetization_rule": "same endpoints and same represented readiness tick only",
            "ramulator_window_gap_ns": args.ramulator_window_gap_ns,
        },
        "validation": validation,
        "implementation_sha256": {
            str(path): _sha256(path)
            for path in (
                Path(__file__).resolve(),
                Path(__file__).with_name("workload.py").resolve(),
                Path(__file__).with_name("mapping.py").resolve(),
                Path(__file__).with_name("routing.py").resolve(),
                Path(__file__).with_name("offline_solver.py").resolve(),
                Path("supernode_sim/backends/ramulator.py").resolve(),
                Path("supernode_sim/backends/tp1_astra.py").resolve(),
                Path("supernode_sim/tools/ramulator2_timed_trace.cpp").resolve(),
                Path("supernode_sim/tools/astra_mesh2d.cpp").resolve(),
                Path("supernode_sim/tools/astra_mesh2d.h").resolve(),
                Path("supernode_sim/tools/astra_statistics_cpu_timer.patch").resolve(),
            )
        },
    }
    (output_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    if validation["status"] != "ok":
        raise RuntimeError(f"TP1 pack validation failed: {validation['errors']}")
    return output_root


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile-dir",
        default="rebuttal/active/sparse_sensitivity/results/pkl_longbench_tp1_15_top2048_compact",
    )
    parser.add_argument(
        "--output-root",
        default="hacmo/attention/outputs/technical_point1_final_pack_astra_ramulator_trace_driven_20260724",
    )
    parser.add_argument("--context-groups", nargs="+", default=["8k", "16k"])
    parser.add_argument("--array-sides", nargs="+", type=int, default=[20, 68])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--layers", nargs="+", type=int, default=[0, 30, 60])
    parser.add_argument("--decode-steps", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--offline-layer", type=int, default=30)
    parser.add_argument("--offline-decode-steps", nargs="+", type=int, default=[0, 1])
    parser.add_argument(
        "--offline-compute-gradients",
        nargs="+",
        type=float,
        default=[0.0, 0.05, 0.1, 0.125, 0.25, 0.5],
    )
    parser.add_argument(
        "--offline-cache-gradients",
        nargs="+",
        type=float,
        default=[0.0, 0.5, 1.0, 1.5, 2.25, 3.0],
    )
    parser.add_argument("--offline-gamma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--total-sram-mb", type=float, default=440.0)
    parser.add_argument("--cache-ratio", type=float, default=0.20)
    parser.add_argument("--kv-entry-bytes", type=int, default=656)
    parser.add_argument("--cacheline-bytes", type=int, default=128)
    parser.add_argument("--hbm-ports", type=int, default=8)
    parser.add_argument("--aggregate-hbm-gb-per-s", type=float, default=4800.0)
    parser.add_argument("--noc-link-gb-per-s", type=float, default=256.0)
    parser.add_argument("--noc-link-latency-ns", type=float, default=2.0)
    parser.add_argument("--die-compute-tflops", type=float, default=2250.0)
    parser.add_argument("--flops-per-selected-kv", type=float, default=294_912.0)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--n-layers", type=int, default=61)
    parser.add_argument("--max-context", type=int, default=18_000)
    parser.add_argument("--ramulator-channels-per-port", type=int, default=16)
    parser.add_argument("--ramulator-channel-width-bits", type=int, default=128)
    parser.add_argument("--ramulator-data-rate-mtps", type=int, default=2000)
    parser.add_argument("--ramulator-tck-ns", type=float, default=1.0)
    parser.add_argument("--ramulator-issue-width", type=int, default=16)
    parser.add_argument("--ramulator-source-issue-width", type=int, default=0)
    parser.add_argument("--ramulator-internal-prefetch-size", type=int, default=2)
    parser.add_argument("--ramulator-exec", default="")
    parser.add_argument("--ramulator-window-gap-ns", type=float, default=1_000_000.0)
    parser.add_argument("--astra-sim-dir", default="astra-sim")
    parser.add_argument(
        "--astra-system-config", default="supernode_sim/configs/astra/system/p2p_only.json"
    )
    parser.add_argument(
        "--astra-remote-memory-config",
        default="supernode_sim/configs/astra/remote_memory/no_memory_expansion.json",
    )
    parser.add_argument("--astra-time-scale", type=int, default=100)
    parser.add_argument("--astra-max-items-per-flow", type=int, default=32)
    parser.add_argument(
        "--preceding-phase-csv",
        default="hacmo/moe/outputs/tp2_serving_epoch_paper_pack_20260831/csv/tp2_serving_epoch_astra_comm_only_summary.csv",
    )
    parser.add_argument("--preceding-phase-topology", default="16w_4x4_rail_500")
    parser.add_argument("--preceding-phase-method", default="ours_dynamic")
    return parser


def main() -> None:
    output = generate_pack(_parser().parse_args())
    print(f"[TP1] validated current paper-facing pack: {output}")


if __name__ == "__main__":
    main()
