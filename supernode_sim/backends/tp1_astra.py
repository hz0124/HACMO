#!/usr/bin/env python3
"""Congestion-aware ASTRA-Sim replay for TP1 per-core KV routing traces."""

from __future__ import annotations

import json
import hashlib
import math
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .astra_wafer_core import parse_log


@dataclass(frozen=True)
class TP1AstraAccess:
    access_id: str
    consumer_core: int
    source: str
    source_core: int | None
    cache_target_core: int | None
    size_bytes: int
    hbm_port_core: int | None = None
    hbm_wire_bytes: int = 0
    hbm_completion_ns: float | None = None
    prefetch_transfer_id: str | None = None


@dataclass(frozen=True)
class TP1AstraPrefetch:
    transfer_id: str
    target_core: int
    hbm_port_core: int
    size_bytes: int
    hbm_wire_bytes: int
    hbm_completion_ns: float


@dataclass(frozen=True)
class TP1AstraWindow:
    window_id: str
    accesses: tuple[TP1AstraAccess, ...]
    prefetches: tuple[TP1AstraPrefetch, ...] = ()


@dataclass(frozen=True)
class TP1AstraConfig:
    rows: int
    cols: int
    astra_sim_dir: Path
    output_dir: Path
    system_config: Path
    remote_memory_config: Path
    link_bandwidth_gb_per_s: float
    link_latency_ns: float
    die_compute_tflops: float
    flops_per_access: float
    preceding_phase_us: float = 0.0
    time_scale: int = 100
    max_items_per_flow: int = 32
    retain_execution_traces: bool = True
    trace_prefix: str = "tp1_window"

    @property
    def num_cores(self) -> int:
        return self.rows * self.cols


@dataclass(frozen=True)
class TP1AstraResult:
    wall_us: float
    exposed_us: float
    preceding_phase_us: float
    compute_busy_us: float
    communication_bytes: int
    flow_count: int
    per_core: tuple[dict[str, object], ...]
    trace_dir: str
    trace_sha256: str
    execution_traces_retained: bool
    network_config: str
    log: str
    parsed_log: str
    projection_summary: str
    backend: str = "astra_sim_congestion_aware_mesh2d"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class _Node:
    kind: str
    name: str
    deps: tuple[int, ...] = ()
    duration_us: int = 0
    src: int | None = None
    dst: int | None = None
    tag: int | None = None
    size_bytes: int = 0


@dataclass(frozen=True)
class _Flow:
    family: str
    src: int
    dst: int
    tag: int
    size_bytes: int
    item_count: int
    send_node: int
    recv_node: int


class _Graph:
    def __init__(self, num_ranks: int) -> None:
        self.nodes: list[list[_Node]] = [[] for _ in range(num_ranks)]
        self.flows: list[_Flow] = []
        self.next_tag = 1

    def add_node(self, rank: int, node: _Node) -> int:
        node_id = len(self.nodes[rank])
        self.nodes[rank].append(node)
        return node_id

    def add_flow(
        self,
        *,
        family: str,
        src: int,
        dst: int,
        size_bytes: int,
        item_count: int,
        send_deps: Iterable[int] = (),
        recv_deps: Iterable[int] = (),
    ) -> tuple[int, int]:
        if src == dst:
            raise ValueError("ASTRA flow source and destination must differ")
        tag = self.next_tag
        self.next_tag += 1
        send = self.add_node(
            src,
            _Node(
                "send",
                f"{family}_send_{src}_{dst}_{tag}",
                tuple(dict.fromkeys(send_deps)),
                src=src,
                dst=dst,
                tag=tag,
                size_bytes=size_bytes,
            ),
        )
        recv = self.add_node(
            dst,
            _Node(
                "recv",
                f"{family}_recv_{src}_{dst}_{tag}",
                tuple(dict.fromkeys(recv_deps)),
                src=src,
                dst=dst,
                tag=tag,
                size_bytes=size_bytes,
            ),
        )
        self.flows.append(
            _Flow(family, src, dst, tag, size_bytes, item_count, send, recv)
        )
        return send, recv


def _directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(file for file in path.rglob("*") if file.is_file()):
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _chunks(values: Sequence[object], size: int) -> Iterable[Sequence[object]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _scaled_duration_us(real_us: float, scale: int) -> int:
    if real_us <= 0:
        return 0
    return max(int(round(real_us * scale)), 1)


def _ready_bucket_ns(completion_ns: float, scale: int) -> float:
    """Round readiness up to the simulator's represented time quantum."""
    quantum_ns = 1000.0 / scale
    tick = math.ceil(completion_ns / quantum_ns - 1e-12)
    return tick * quantum_ns


def _validate_inputs(
    config: TP1AstraConfig,
    accesses: Sequence[TP1AstraAccess],
    prefetches: Sequence[TP1AstraPrefetch],
) -> None:
    if config.rows <= 1 or config.rows != config.cols:
        raise ValueError("TP1 ASTRA requires a square 2-D core mesh")
    if config.link_bandwidth_gb_per_s <= 0 or config.link_latency_ns < 0:
        raise ValueError("NoC bandwidth must be positive and latency non-negative")
    if config.die_compute_tflops <= 0 or config.flops_per_access <= 0:
        raise ValueError("fixed die compute and per-access FLOPs must be positive")
    if config.time_scale <= 0 or config.max_items_per_flow <= 0:
        raise ValueError("time_scale and max_items_per_flow must be positive")
    for access in accesses:
        if not 0 <= access.consumer_core < config.num_cores:
            raise ValueError(f"consumer core is outside the physical mesh: {access}")
        if access.source not in {"local", "remote", "hbm", "prefetch"}:
            raise ValueError(f"unsupported TP1 access source: {access.source}")
        if access.size_bytes <= 0:
            raise ValueError(f"access size must be positive: {access}")
        if access.source == "remote" and access.source_core is None:
            raise ValueError(f"remote access lacks source core: {access}")
        if access.source == "hbm" and (
            access.hbm_port_core is None or access.hbm_completion_ns is None
        ):
            raise ValueError(f"HBM access lacks port/completion: {access}")
        if access.source == "prefetch" and not access.prefetch_transfer_id:
            raise ValueError(f"prefetched access lacks transfer identity: {access}")
    transfer_ids = [item.transfer_id for item in prefetches]
    if len(set(transfer_ids)) != len(transfer_ids):
        raise ValueError("prefetch transfer IDs must be unique")


def _build_graph(
    config: TP1AstraConfig,
    accesses: Sequence[TP1AstraAccess],
    prefetches: Sequence[TP1AstraPrefetch],
) -> tuple[_Graph, dict[str, object]]:
    _validate_inputs(config, accesses, prefetches)
    graph = _Graph(config.num_cores)
    phase_nodes = [
        graph.add_node(
            rank,
            _Node(
                "compute",
                "preceding_phase",
                duration_us=_scaled_duration_us(
                    config.preceding_phase_us, config.time_scale
                ),
            ),
        )
        for rank in range(config.num_cores)
    ]

    # Prefetch and demand packets share one monotonic timer chain per physical
    # HBM port. Timer nodes use ASTRA's CPU resource, so HBM service readiness
    # advances concurrently with the core's GPU/compute pipeline.
    prefetch_by_endpoints: dict[
        tuple[int, int, float], list[TP1AstraPrefetch]
    ] = defaultdict(list)
    for item in prefetches:
        ready_ns = _ready_bucket_ns(item.hbm_completion_ns, config.time_scale)
        prefetch_by_endpoints[
            (item.hbm_port_core, item.target_core, ready_ns)
        ].append(item)
    prefetch_packets: list[
        tuple[str, int, int, Sequence[TP1AstraPrefetch]]
    ] = []
    for packet_index, ((port_core, target, _ready_ns), items) in enumerate(
        sorted(prefetch_by_endpoints.items())
    ):
        ordered = sorted(
            items, key=lambda item: (item.hbm_completion_ns, item.transfer_id)
        )
        for chunk_index, packet in enumerate(
            _chunks(ordered, config.max_items_per_flow)
        ):
            key = f"prefetch:{packet_index}:{chunk_index}"
            prefetch_packets.append((key, port_core, target, packet))

    hbm_by_endpoints: dict[
        tuple[int, int, float], list[TP1AstraAccess]
    ] = defaultdict(list)
    for access in accesses:
        if access.source == "hbm":
            assert access.hbm_port_core is not None
            assert access.hbm_completion_ns is not None
            ready_ns = _ready_bucket_ns(
                access.hbm_completion_ns, config.time_scale
            )
            hbm_by_endpoints[
                (access.hbm_port_core, access.consumer_core, ready_ns)
            ].append(access)
    demand_packets: list[tuple[str, int, int, Sequence[TP1AstraAccess]]] = []
    for packet_index, ((port_core, consumer, _ready_ns), items) in enumerate(
        sorted(hbm_by_endpoints.items())
    ):
        ordered = sorted(
            items,
            key=lambda item: (float(item.hbm_completion_ns or 0.0), item.access_id),
        )
        for chunk_index, packet in enumerate(
            _chunks(ordered, config.max_items_per_flow)
        ):
            key = f"demand:{packet_index}:{chunk_index}"
            demand_packets.append((key, port_core, consumer, packet))

    ready_events: dict[int, dict[float, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for key, port_core, _, packet in prefetch_packets:
        ready_events[port_core][
            _ready_bucket_ns(
                max(item.hbm_completion_ns for item in packet),
                config.time_scale,
            )
        ].append(key)
    for key, port_core, _, packet in demand_packets:
        ready_events[port_core][
            _ready_bucket_ns(
                max(float(item.hbm_completion_ns or 0.0) for item in packet),
                config.time_scale,
            )
        ].append(key)
    timer_node_by_packet: dict[str, int] = {}
    for port_core, events in sorted(ready_events.items()):
        previous: int | None = None
        previous_ns = 0.0
        for ready_ns, keys in sorted(events.items()):
            delta_ns = ready_ns - previous_ns
            if delta_ns < 0:
                raise RuntimeError("HBM completion timer moved backwards")
            timer = graph.add_node(
                port_core,
                _Node(
                    "timer",
                    f"hbm_ready_{keys[0]}",
                    () if previous is None else (previous,),
                    _scaled_duration_us(delta_ns / 1000.0, config.time_scale),
                ),
            )
            for key in keys:
                timer_node_by_packet[key] = timer
            previous = timer
            previous_ns = ready_ns

    # Transfers sharing physical endpoints are packetized without dropping any
    # item or byte.
    prefetch_data_node: dict[str, int] = {}
    for key, port_core, target, packet in prefetch_packets:
        wait = timer_node_by_packet[key]
        wire_bytes = sum(item.hbm_wire_bytes for item in packet)
        if port_core == target:
            recv = wait
        else:
            _, recv = graph.add_flow(
                family="prefetch",
                src=port_core,
                dst=target,
                size_bytes=wire_bytes,
                item_count=len(packet),
                send_deps=(wait,),
            )
        for item in packet:
            prefetch_data_node[item.transfer_id] = recv

    data_node_by_access: dict[str, int] = {}
    compute_family_by_access: dict[str, str] = {}

    local = [access for access in accesses if access.source == "local"]
    for access in local:
        data_node_by_access[access.access_id] = phase_nodes[access.consumer_core]
        compute_family_by_access[access.access_id] = "local"

    remote_by_endpoints: dict[tuple[int, int], list[TP1AstraAccess]] = defaultdict(list)
    for access in accesses:
        if access.source == "remote":
            assert access.source_core is not None
            remote_by_endpoints[(access.source_core, access.consumer_core)].append(access)
    for (source, consumer), items in sorted(remote_by_endpoints.items()):
        for packet in _chunks(items, config.max_items_per_flow):
            if source == consumer:
                recv = phase_nodes[consumer]
            else:
                _, recv = graph.add_flow(
                    family="remote",
                    src=source,
                    dst=consumer,
                    size_bytes=sum(item.size_bytes for item in packet),
                    item_count=len(packet),
                    send_deps=(phase_nodes[source],),
                    recv_deps=(phase_nodes[consumer],),
                )
            for item in packet:
                data_node_by_access[item.access_id] = recv
                compute_family_by_access[item.access_id] = "remote"

    for key, port_core, consumer, packet in demand_packets:
        wait = timer_node_by_packet[key]
        if port_core == consumer:
            recv = wait
        else:
            _, recv = graph.add_flow(
                family="hbm_demand",
                src=port_core,
                dst=consumer,
                size_bytes=sum(item.hbm_wire_bytes for item in packet),
                item_count=len(packet),
                send_deps=(wait,),
                recv_deps=(phase_nodes[consumer],),
            )
        for item in packet:
            data_node_by_access[item.access_id] = recv
            compute_family_by_access[item.access_id] = "hbm"

        fills: dict[int, list[TP1AstraAccess]] = defaultdict(list)
        for item in packet:
            target = item.cache_target_core
            if target is not None and target != consumer:
                fills[target].append(item)
        for target, fill_items in sorted(fills.items()):
            graph.add_flow(
                family="demand_fill",
                src=consumer,
                dst=target,
                size_bytes=sum(item.size_bytes for item in fill_items),
                item_count=len(fill_items),
                send_deps=(recv,),
            )

    prefetched_by_path: dict[tuple[int, int, int], list[TP1AstraAccess]] = defaultdict(list)
    for access in accesses:
        if access.source != "prefetch":
            continue
        assert access.prefetch_transfer_id is not None
        transfer_node = prefetch_data_node.get(access.prefetch_transfer_id)
        if transfer_node is None:
            raise ValueError(
                f"access references missing prefetch transfer: {access.prefetch_transfer_id}"
            )
        source = access.source_core
        if source is None:
            raise ValueError(f"prefetched access lacks target owner: {access}")
        prefetched_by_path[(source, access.consumer_core, transfer_node)].append(access)
    for (source, consumer, transfer_node), items in sorted(prefetched_by_path.items()):
        if source == consumer:
            recv = transfer_node
        else:
            barrier = graph.add_node(
                source,
                _Node(
                    "compute",
                    f"prefetch_available_{items[0].prefetch_transfer_id}",
                    tuple(dict.fromkeys((phase_nodes[source], transfer_node))),
                ),
            )
            _, recv = graph.add_flow(
                family="prefetch_remote",
                src=source,
                dst=consumer,
                size_bytes=sum(item.size_bytes for item in items),
                item_count=len(items),
                send_deps=(barrier,),
                recv_deps=(phase_nodes[consumer],),
            )
        for item in items:
            data_node_by_access[item.access_id] = recv
            compute_family_by_access[item.access_id] = "prefetch"

    # Local-ready work forms a fixed prefix. Non-local compute packets then
    # become independently ready from their own data events; ASTRA's single
    # GPU compute resource per rank serializes them in actual arrival order.
    family_order = {"local": 0, "remote": 1, "prefetch": 2, "hbm": 3}
    per_core_accesses: dict[int, list[TP1AstraAccess]] = defaultdict(list)
    for access in accesses:
        per_core_accesses[access.consumer_core].append(access)
    compute_busy = [0.0] * config.num_cores
    access_counts = [0] * config.num_cores
    per_core_tflops = config.die_compute_tflops / config.num_cores
    for core in range(config.num_cores):
        ordered = sorted(
            per_core_accesses.get(core, []),
            key=lambda item: (
                family_order[compute_family_by_access[item.access_id]],
                item.access_id,
            ),
        )
        local_tail = phase_nodes[core]
        local_items = [
            item
            for item in ordered
            if compute_family_by_access[item.access_id] == "local"
        ]
        for packet in _chunks(local_items, config.max_items_per_flow):
            duration = (
                len(packet)
                * config.flops_per_access
                / (per_core_tflops * 1e12)
                * 1e6
            )
            local_tail = graph.add_node(
                core,
                _Node(
                    "compute",
                    f"local_compute_{packet[0].access_id}",
                    (local_tail,),
                    _scaled_duration_us(duration, config.time_scale),
                ),
            )
            compute_busy[core] += duration
            access_counts[core] += len(packet)

        ready_groups: dict[tuple[str, int], list[TP1AstraAccess]] = defaultdict(list)
        for item in ordered:
            family = compute_family_by_access[item.access_id]
            if family == "local":
                continue
            ready_groups[(family, data_node_by_access[item.access_id])].append(item)
        for (family, data_node), items in sorted(
            ready_groups.items(),
            key=lambda entry: (
                family_order[entry[0][0]],
                entry[0][1],
                entry[1][0].access_id,
            ),
        ):
            for packet in _chunks(items, config.max_items_per_flow):
                duration = (
                    len(packet)
                    * config.flops_per_access
                    / (per_core_tflops * 1e12)
                    * 1e6
                )
                graph.add_node(
                    core,
                    _Node(
                        "compute",
                        f"{family}_compute_{packet[0].access_id}",
                        tuple(dict.fromkeys((local_tail, data_node))),
                        _scaled_duration_us(duration, config.time_scale),
                    ),
                )
                compute_busy[core] += duration
                access_counts[core] += len(packet)

    if sum(access_counts) != len(accesses):
        raise RuntimeError("ASTRA compute projection dropped routed KV accesses")
    summary: dict[str, object] = {
        "num_cores": config.num_cores,
        "logical_accesses": len(accesses),
        "prefetch_transfers": len(prefetches),
        "flow_count": len(graph.flows),
        "communication_bytes": sum(flow.size_bytes for flow in graph.flows),
        "compute_busy_us": compute_busy,
        "access_counts": access_counts,
        "flows": [asdict(flow) for flow in graph.flows],
        "time_scale": config.time_scale,
        "max_items_per_flow": config.max_items_per_flow,
        "hbm_readiness_quantum_ns": 1000.0 / config.time_scale,
        "compute_scheduling": "local prefix, then ASTRA arrival-ready on one GPU compute resource per core",
        "source_counts": {
            source: sum(1 for access in accesses if access.source == source)
            for source in ("local", "remote", "hbm", "prefetch")
        },
    }
    return graph, summary


def _write_network_config(config: TP1AstraConfig, path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "topology: [ Mesh ]",
                f"npus_count: [ {config.num_cores} ]",
                f"bandwidth: [ {config.link_bandwidth_gb_per_s / config.time_scale:.12g} ]",
                f"latency: [ {config.link_latency_ns * config.time_scale:.12g} ]",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _load_chakra(astra_sim_dir: Path):
    astra_path = str(astra_sim_dir)
    if astra_path not in sys.path:
        sys.path.insert(0, astra_path)
    from extern.graph_frontend.chakra.schema.protobuf.et_def_pb2 import (  # type: ignore
        COMM_RECV_NODE,
        COMM_SEND_NODE,
        COMP_NODE,
        GlobalMetadata,
    )
    from extern.graph_frontend.chakra.schema.protobuf.et_def_pb2 import AttributeProto as ChakraAttr  # type: ignore
    from extern.graph_frontend.chakra.schema.protobuf.et_def_pb2 import Node as ChakraNode  # type: ignore
    from extern.graph_frontend.chakra.src.third_party.utils.protolib import encodeMessage as encode_message  # type: ignore

    return COMM_RECV_NODE, COMM_SEND_NODE, COMP_NODE, GlobalMetadata, ChakraAttr, ChakraNode, encode_message


def _write_traces(config: TP1AstraConfig, graph: _Graph, trace_dir: Path) -> None:
    trace_dir.mkdir(parents=True, exist_ok=True)
    (
        comm_recv,
        comm_send,
        comp,
        metadata,
        attr,
        node_cls,
        encode,
    ) = _load_chakra(config.astra_sim_dir)
    for rank, nodes in enumerate(graph.nodes):
        with (trace_dir / f"{config.trace_prefix}.{rank}.et").open("wb") as stream:
            encode(stream, metadata(version="0.0.4"))
            for node_id, spec in enumerate(nodes):
                node = node_cls()
                node.id = node_id
                node.name = spec.name
                if spec.kind in {"compute", "timer"}:
                    node.type = comp
                    node.duration_micros = spec.duration_us
                    node.attr.append(attr(name="num_ops", int64_val=0))
                    node.attr.append(attr(name="tensor_size", uint64_val=0))
                else:
                    node.type = comm_send if spec.kind == "send" else comm_recv
                    node.attr.append(attr(name="comm_src", uint32_val=int(spec.src)))
                    node.attr.append(attr(name="comm_dst", uint32_val=int(spec.dst)))
                    node.attr.append(attr(name="comm_tag", uint32_val=int(spec.tag)))
                    node.attr.append(
                        attr(name="comm_size", int64_val=max(spec.size_bytes, 1))
                    )
                node.attr.append(
                    attr(name="is_cpu_op", bool_val=(spec.kind == "timer"))
                )
                for dep in spec.deps:
                    node.data_deps.append(int(dep))
                encode(stream, node)


def _merge_window_graphs(
    config: TP1AstraConfig,
    windows: Sequence[TP1AstraWindow],
) -> tuple[_Graph, dict[str, object]]:
    if not windows:
        raise ValueError("TP1 ASTRA sequence requires at least one window")
    merged = _Graph(config.num_cores)
    previous_done: list[int | None] = [None] * config.num_cores
    window_summaries: list[dict[str, object]] = []
    total_compute = [0.0] * config.num_cores
    total_accesses = [0] * config.num_cores
    source_counts = {source: 0 for source in ("local", "remote", "hbm", "prefetch")}

    for window in windows:
        graph, summary = _build_graph(config, window.accesses, window.prefetches)
        rank_offsets = [len(merged.nodes[rank]) for rank in range(config.num_cores)]
        tag_offset = merged.next_tag - 1
        for rank in range(config.num_cores):
            segment = graph.nodes[rank]
            child_ids = {dep for node in segment for dep in node.deps}
            leaves = [index for index in range(len(segment)) if index not in child_ids]
            for spec in segment:
                deps = tuple(rank_offsets[rank] + dep for dep in spec.deps)
                if not deps and previous_done[rank] is not None:
                    deps = (int(previous_done[rank]),)
                merged.nodes[rank].append(
                    _Node(
                        kind=spec.kind,
                        name=f"{window.window_id}:{spec.name}",
                        deps=deps,
                        duration_us=spec.duration_us,
                        src=spec.src,
                        dst=spec.dst,
                        tag=(spec.tag + tag_offset) if spec.tag is not None else None,
                        size_bytes=spec.size_bytes,
                    )
                )
            done_deps = tuple(rank_offsets[rank] + leaf for leaf in leaves)
            done = merged.add_node(
                rank,
                _Node(
                    "compute",
                    f"{window.window_id}:window_done",
                    done_deps,
                ),
            )
            previous_done[rank] = done

        for flow in graph.flows:
            merged.flows.append(
                _Flow(
                    family=flow.family,
                    src=flow.src,
                    dst=flow.dst,
                    tag=flow.tag + tag_offset,
                    size_bytes=flow.size_bytes,
                    item_count=flow.item_count,
                    send_node=rank_offsets[flow.src] + flow.send_node,
                    recv_node=rank_offsets[flow.dst] + flow.recv_node,
                )
            )
        merged.next_tag += graph.next_tag - 1
        for core in range(config.num_cores):
            total_compute[core] += float(summary["compute_busy_us"][core])
            total_accesses[core] += int(summary["access_counts"][core])
        for source, count in summary["source_counts"].items():
            source_counts[str(source)] += int(count)
        window_summaries.append(
            {
                "window_id": window.window_id,
                "logical_accesses": int(summary["logical_accesses"]),
                "prefetch_transfers": int(summary["prefetch_transfers"]),
                "flow_count": int(summary["flow_count"]),
                "communication_bytes": int(summary["communication_bytes"]),
                "source_counts": summary["source_counts"],
            }
        )

    summary = {
        "num_cores": config.num_cores,
        "num_windows": len(windows),
        "logical_accesses": sum(total_accesses),
        "prefetch_transfers": sum(len(window.prefetches) for window in windows),
        "flow_count": len(merged.flows),
        "communication_bytes": sum(flow.size_bytes for flow in merged.flows),
        "compute_busy_us": total_compute,
        "access_counts": total_accesses,
        "flows": [asdict(flow) for flow in merged.flows],
        "time_scale": config.time_scale,
        "max_items_per_flow": config.max_items_per_flow,
        "hbm_readiness_quantum_ns": 1000.0 / config.time_scale,
        "source_counts": source_counts,
        "windows": window_summaries,
    }
    return merged, summary


def _astra_binary(astra_sim_dir: Path) -> Path:
    path = (
        astra_sim_dir
        / "build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"ASTRA-Sim congestion-aware binary not found: {path}; "
            "run scripts/build_astra_analytical.sh"
        )
    return path


def _run_graph(
    config: TP1AstraConfig,
    graph: _Graph,
    projection: dict[str, object],
    *,
    preceding_phase_us: float,
) -> TP1AstraResult:
    output = config.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    trace_dir = output / "traces"
    network = output / "mesh2d.yaml"
    log = output / "astra.log"
    parsed_path = output / "astra_parsed.json"
    projection_path = output / "projection_summary.json"
    _write_network_config(config, network)
    _write_traces(config, graph, trace_dir)
    projection_path.write_text(
        json.dumps(projection, indent=2) + "\n", encoding="utf-8"
    )

    command = [
        str(_astra_binary(config.astra_sim_dir.resolve())),
        f"--workload-configuration={trace_dir / config.trace_prefix}",
        f"--system-configuration={config.system_config.resolve()}",
        f"--remote-memory-configuration={config.remote_memory_config.resolve()}",
        f"--network-configuration={network}",
    ]
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    root = str(config.astra_sim_dir.resolve())
    env["PYTHONPATH"] = f"{root}:{existing}" if existing else root
    with log.open("w", encoding="utf-8") as stream:
        proc = subprocess.run(
            command,
            cwd=root,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ASTRA-Sim TP1 Mesh run failed ({proc.returncode}); log={log}"
        )
    parsed = parse_log(log.read_text(encoding="utf-8", errors="ignore"))
    parsed_path.write_text(json.dumps(parsed, indent=2) + "\n", encoding="utf-8")
    per_rank = {int(row["rank"]): row for row in parsed["per_rank"]}
    per_core_rows: list[dict[str, object]] = []
    for core in range(config.num_cores):
        row = per_rank.get(core, {})
        wall_cycles = row.get("wall_cycles") or row.get("finished_cycles") or 0
        comm_cycles = row.get("comm_cycles") or 0
        finish_us = float(wall_cycles) / 1000.0 / config.time_scale
        comm_us = float(comm_cycles) / 1000.0 / config.time_scale
        busy_us = float(projection["compute_busy_us"][core])
        per_core_rows.append(
            {
                "core_id": core,
                "row": core // config.cols,
                "col": core % config.cols,
                "logical_accesses": int(projection["access_counts"][core]),
                "compute_busy_us": busy_us,
                "astra_comm_us": comm_us,
                "astra_finish_us": finish_us,
                "post_phase_finish_us": max(
                    finish_us - preceding_phase_us, 0.0
                ),
            }
        )
    wall_us = max(
        (float(row["astra_finish_us"]) for row in per_core_rows), default=0.0
    )
    trace_sha256 = _directory_sha256(trace_dir)
    retained_trace_dir = str(trace_dir)
    if not config.retain_execution_traces:
        shutil.rmtree(trace_dir)
        retained_trace_dir = ""
    result = TP1AstraResult(
        wall_us=wall_us,
        exposed_us=max(wall_us - preceding_phase_us, 0.0),
        preceding_phase_us=preceding_phase_us,
        compute_busy_us=sum(float(value) for value in projection["compute_busy_us"]),
        communication_bytes=int(projection["communication_bytes"]),
        flow_count=int(projection["flow_count"]),
        per_core=tuple(per_core_rows),
        trace_dir=retained_trace_dir,
        trace_sha256=trace_sha256,
        execution_traces_retained=config.retain_execution_traces,
        network_config=str(network),
        log=str(log),
        parsed_log=str(parsed_path),
        projection_summary=str(projection_path),
    )
    (output / "result.json").write_text(
        json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    return result


def run_tp1_astra_window(
    config: TP1AstraConfig,
    accesses: Sequence[TP1AstraAccess],
    prefetches: Sequence[TP1AstraPrefetch] = (),
) -> TP1AstraResult:
    graph, projection = _build_graph(config, accesses, prefetches)
    return _run_graph(
        config,
        graph,
        projection,
        preceding_phase_us=config.preceding_phase_us,
    )


def run_tp1_astra_sequence(
    config: TP1AstraConfig,
    windows: Sequence[TP1AstraWindow],
) -> TP1AstraResult:
    graph, projection = _merge_window_graphs(config, windows)
    return _run_graph(
        config,
        graph,
        projection,
        preceding_phase_us=config.preceding_phase_us * len(windows),
    )
