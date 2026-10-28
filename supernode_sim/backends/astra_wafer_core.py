#!/usr/bin/env python3
"""ASTRA-Sim wafer-core NoC movement runner for attention microbenches."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .wafer_core_noc import WaferCoreNoCConfig, WaferCoreNoCTraffic, estimate_wafer_core_noc


WALL_RE = re.compile(r"sys\[(\d+)\],\s*Wall time:\s*(\d+)")
COMM_RE = re.compile(r"sys\[(\d+)\],\s*Comm time:\s*(\d+)")
FINISHED_RE = re.compile(r"sys\[(\d+)\]\s*finished,\s*(\d+)\s*cycles")
REPORT_RE = re.compile(r"sys\[(\d+)\]\s*report called at[^0-9]*(\d+)\s*cycles")


@dataclass(frozen=True)
class AstraWaferCoreFlow:
    src: int
    dst: int
    bytes: int
    tag: int | None = None


@dataclass(frozen=True)
class AstraWaferCoreCase:
    name: str
    rows: int
    cols: int
    flows: tuple[AstraWaferCoreFlow, ...] = ()
    hbm_bytes: float = 0.0
    near_sram_bytes: float = 0.0
    remote_sram_bytes: float = 0.0
    avg_hbm_hops: float | None = None
    near_sram_hops: float | None = None
    avg_remote_hops: float | None = None
    noc_bw_tbps: float = 5.0
    gamma: float = 1.0
    noc_bw_scale_mode: str = "fixed"
    hbm_edges: tuple[str, ...] = ("top", "bottom")


@dataclass(frozen=True)
class AstraWaferCoreConfig:
    astra_sim_dir: Path
    output_root: Path
    network_config: Path
    system_config: Path
    remote_memory_config: Path
    num_ranks: int
    active_ranks: int
    astra_network_backend: str = "congestion_unaware"
    run_astra: bool = True
    trace_prefix: str = "wafer_core_noc"


def run_cases(config: AstraWaferCoreConfig, cases: list[AstraWaferCoreCase]) -> list[dict[str, object]]:
    config.output_root.mkdir(parents=True, exist_ok=True)
    trace_dir = config.output_root / "traces"
    log_dir = config.output_root / "logs"
    parsed_dir = config.output_root / "parsed"
    for path in (trace_dir, log_dir, parsed_dir):
        path.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, object]] = []
    astra_bin = _astra_binary(config.astra_sim_dir, config.astra_network_backend) if config.run_astra else None
    for case in cases:
        case_dir = trace_dir / case.name
        _write_case_traces(config, case, case_dir)
        analytic = estimate_wafer_core_noc(
            WaferCoreNoCConfig(
                rows=case.rows,
                cols=case.cols,
                base_noc_bw_tbps=case.noc_bw_tbps,
                gamma=case.gamma,
                noc_bw_scale_mode=case.noc_bw_scale_mode,
                hbm_edges=case.hbm_edges,
                near_sram_hops=case.near_sram_hops or 0.0,
            ),
            WaferCoreNoCTraffic(
                hbm_bytes=case.hbm_bytes,
                near_sram_bytes=case.near_sram_bytes,
                remote_sram_bytes=case.remote_sram_bytes,
                avg_hbm_hops=case.avg_hbm_hops,
                near_sram_hops=case.near_sram_hops,
                avg_remote_hops=case.avg_remote_hops,
            ),
        )
        row: dict[str, object] = {
            **asdict(case),
            "trace_dir": str(case_dir),
            "analytic_hbm_us": analytic.hbm_us,
            "analytic_near_sram_us": analytic.near_sram_us,
            "analytic_remote_sram_us": analytic.remote_sram_us,
            "analytic_total_us": analytic.total_us,
            "analytic_effective_noc_bw_tbps": analytic.effective_noc_bw_tbps,
            "analytic_avg_hbm_hops": analytic.avg_hbm_hops,
            "analytic_near_sram_hops": analytic.near_sram_hops,
            "analytic_avg_remote_hops": analytic.avg_remote_hops,
        }
        if config.run_astra:
            assert astra_bin is not None
            log_path = log_dir / f"{case.name}.log"
            cmd = [
                str(astra_bin),
                f"--workload-configuration={case_dir / config.trace_prefix}",
                f"--system-configuration={config.system_config}",
                f"--remote-memory-configuration={config.remote_memory_config}",
                f"--network-configuration={config.network_config}",
            ]
            _run(cmd, config.astra_sim_dir, log_path)
            parsed = parse_log(log_path.read_text(encoding="utf-8", errors="ignore"))
            parsed_path = parsed_dir / f"{case.name}.json"
            parsed_path.write_text(json.dumps(parsed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            row.update(_summary_to_row(parsed))
            row["astra_log"] = str(log_path)
            row["astra_parsed_json"] = str(parsed_path)
        results.append(row)

    (config.output_root / "wafer_core_astra_summary.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return results


def parse_log(text: str) -> dict[str, object]:
    wall = {int(m.group(1)): int(m.group(2)) for m in WALL_RE.finditer(text)}
    comm = {int(m.group(1)): int(m.group(2)) for m in COMM_RE.finditer(text)}
    finished = {int(m.group(1)): int(m.group(2)) for m in FINISHED_RE.finditer(text)}
    report = {int(m.group(1)): int(m.group(2)) for m in REPORT_RE.finditer(text)}
    ranks = sorted(set(wall) | set(comm) | set(finished) | set(report))

    def summary(values: Iterable[int]) -> dict[str, float | None]:
        xs = list(values)
        if not xs:
            return {"max": None, "avg": None}
        return {"max": max(xs), "avg": sum(xs) / len(xs)}

    return {
        "summary": {
            "num_ranks": len(ranks),
            "wall_cycles": summary(wall.values()),
            "comm_cycles": summary(comm.values()),
            "finished_cycles": summary(finished.values()),
            "report_tick_cycles": summary(report.values()),
        },
        "per_rank": [
            {
                "rank": rank,
                "wall_cycles": wall.get(rank),
                "comm_cycles": comm.get(rank),
                "finished_cycles": finished.get(rank),
                "report_tick_cycles": report.get(rank),
            }
            for rank in ranks
        ],
    }


def _summary_to_row(parsed: dict[str, object]) -> dict[str, object]:
    summary = parsed["summary"]
    assert isinstance(summary, dict)
    wall = _pick_cycles(summary, "wall_cycles", "finished_cycles", "report_tick_cycles")
    comm = _pick_cycles(summary, "comm_cycles")
    return {
        "astra_wall_cycles": wall,
        "astra_comm_cycles": comm,
        "astra_wall_us": wall / 1000.0 if wall is not None else None,
        "astra_comm_us": comm / 1000.0 if comm is not None else None,
    }


def _pick_cycles(summary: dict[str, object], *keys: str) -> int | None:
    for key in keys:
        value = summary.get(key)
        if isinstance(value, dict) and value.get("max") is not None:
            return int(value["max"])
    return None


def _astra_binary(astra_sim_dir: Path, backend: str) -> Path:
    name = backend.lower()
    if name in {"congestion_aware", "aware"}:
        path = astra_sim_dir / "build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware"
    elif name in {"congestion_unaware", "unaware"}:
        path = astra_sim_dir / "build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware"
    else:
        raise ValueError(f"Unsupported ASTRA network backend: {backend}")
    if not path.exists():
        raise FileNotFoundError(f"ASTRA-Sim analytical binary not found: {path}")
    return path


def _write_case_traces(config: AstraWaferCoreConfig, case: AstraWaferCoreCase, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _load_chakra(config.astra_sim_dir)
    from extern.graph_frontend.chakra.schema.protobuf.et_def_pb2 import (  # type: ignore
        COMM_RECV_NODE,
        COMM_SEND_NODE,
        COMP_NODE,
        GlobalMetadata,
    )
    from extern.graph_frontend.chakra.schema.protobuf.et_def_pb2 import AttributeProto as ChakraAttr  # type: ignore
    from extern.graph_frontend.chakra.schema.protobuf.et_def_pb2 import Node as ChakraNode  # type: ignore
    from extern.graph_frontend.chakra.src.third_party.utils.protolib import encodeMessage as encode_message  # type: ignore

    flows = _case_flows(config, case)
    sends_by_rank, recvs_by_rank = _index_flows(config.num_ranks, flows)
    trace_summary: dict[str, object] = {
        "case": case.name,
        "num_ranks": config.num_ranks,
        "num_flows": len(flows),
        "total_flow_bytes": sum(flow.bytes for flow in flows),
        "flows": [asdict(flow) for flow in flows],
        "per_rank": [],
    }

    for rank in range(config.num_ranks):
        rank_file = output_dir / f"{config.trace_prefix}.{rank}.et"
        with rank_file.open("wb") as et:
            encode_message(et, GlobalMetadata(version="0.0.4"))
            node_id = 0
            start = _make_comp_node(ChakraNode, ChakraAttr, COMP_NODE, node_id, "wafer_core_start", None)
            encode_message(et, start)
            prev = node_id
            node_id += 1
            deps: list[int] = []

            for flow in sends_by_rank[rank]:
                node = _make_comm_node(
                    ChakraNode,
                    ChakraAttr,
                    COMM_SEND_NODE,
                    node_id,
                    "send",
                    flow,
                    dep_id=prev,
                )
                encode_message(et, node)
                deps.append(node_id)
                node_id += 1

            for flow in recvs_by_rank[rank]:
                node = _make_comm_node(
                    ChakraNode,
                    ChakraAttr,
                    COMM_RECV_NODE,
                    node_id,
                    "recv",
                    flow,
                    dep_id=prev,
                )
                encode_message(et, node)
                deps.append(node_id)
                node_id += 1

            if deps:
                done = _make_comp_node(ChakraNode, ChakraAttr, COMP_NODE, node_id, "wafer_core_done", deps)
                encode_message(et, done)
                node_id += 1

        trace_summary["per_rank"].append(
            {
                "rank": rank,
                "send_count": len(sends_by_rank[rank]),
                "recv_count": len(recvs_by_rank[rank]),
                "send_bytes": sum(flow.bytes for flow in sends_by_rank[rank]),
                "recv_bytes": sum(flow.bytes for flow in recvs_by_rank[rank]),
                "nodes_written": node_id,
            }
        )
    (output_dir / f"{config.trace_prefix}.summary.json").write_text(
        json.dumps(trace_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _case_flows(config: AstraWaferCoreConfig, case: AstraWaferCoreCase) -> tuple[AstraWaferCoreFlow, ...]:
    if case.flows:
        flows = tuple(
            AstraWaferCoreFlow(
                src=int(flow.src),
                dst=int(flow.dst),
                bytes=max(int(flow.bytes), 1),
                tag=int(flow.tag) if flow.tag is not None else idx,
            )
            for idx, flow in enumerate(case.flows)
        )
    else:
        total_bytes = int(round(case.hbm_bytes + case.near_sram_bytes + case.remote_sram_bytes))
        if total_bytes <= 0:
            flows = ()
        else:
            per_rank_bytes = max(1, int(round(total_bytes / max(config.active_ranks, 1))))
            flows = tuple(
                AstraWaferCoreFlow(
                    src=rank,
                    dst=(rank + 1) % config.num_ranks,
                    bytes=per_rank_bytes,
                    tag=rank,
                )
                for rank in range(config.active_ranks)
            )
    for flow in flows:
        if flow.src < 0 or flow.src >= config.num_ranks:
            raise ValueError(f"Flow source rank out of range: {flow}")
        if flow.dst < 0 or flow.dst >= config.num_ranks:
            raise ValueError(f"Flow destination rank out of range: {flow}")
        if flow.src == flow.dst:
            raise ValueError(f"Flow source and destination must differ: {flow}")
        if flow.bytes <= 0:
            raise ValueError(f"Flow bytes must be positive: {flow}")
        if flow.tag is None or flow.tag < 0:
            raise ValueError(f"Flow tag must be a non-negative integer: {flow}")
    return flows


def _index_flows(
    num_ranks: int,
    flows: tuple[AstraWaferCoreFlow, ...],
) -> tuple[list[list[AstraWaferCoreFlow]], list[list[AstraWaferCoreFlow]]]:
    sends_by_rank: list[list[AstraWaferCoreFlow]] = [[] for _ in range(num_ranks)]
    recvs_by_rank: list[list[AstraWaferCoreFlow]] = [[] for _ in range(num_ranks)]
    seen: set[tuple[int, int, int]] = set()
    for flow in flows:
        key = (flow.src, flow.dst, int(flow.tag))
        if key in seen:
            raise ValueError(f"Duplicate ASTRA P2P flow tag for src/dst/tag: {key}")
        seen.add(key)
        sends_by_rank[flow.src].append(flow)
        recvs_by_rank[flow.dst].append(flow)
    return sends_by_rank, recvs_by_rank


def _make_comp_node(
    chakra_node_cls,
    chakra_attr_cls,
    comp_node_type: int,
    node_id: int,
    name: str,
    dep_ids: list[int] | None,
):
    node = chakra_node_cls()
    node.id = node_id
    node.name = name
    node.type = comp_node_type
    node.duration_micros = 1
    node.attr.append(chakra_attr_cls(name="is_cpu_op", bool_val=False))
    node.attr.append(chakra_attr_cls(name="num_ops", int64_val=1))
    node.attr.append(chakra_attr_cls(name="tensor_size", uint64_val=1))
    for dep in dep_ids or []:
        node.data_deps.append(int(dep))
    return node


def _make_comm_node(
    chakra_node_cls,
    chakra_attr_cls,
    node_type: int,
    node_id: int,
    direction: str,
    flow: AstraWaferCoreFlow,
    dep_id: int | None,
):
    node = chakra_node_cls()
    node.id = node_id
    node.name = f"wafer_core_noc_{direction}_{flow.src}_{flow.dst}_{flow.tag}"
    node.type = node_type
    node.attr.append(chakra_attr_cls(name="is_cpu_op", bool_val=False))
    node.attr.append(chakra_attr_cls(name="comm_src", uint32_val=int(flow.src)))
    node.attr.append(chakra_attr_cls(name="comm_dst", uint32_val=int(flow.dst)))
    node.attr.append(chakra_attr_cls(name="comm_tag", uint32_val=int(flow.tag)))
    node.attr.append(chakra_attr_cls(name="comm_size", int64_val=max(int(flow.bytes), 1)))
    if dep_id is not None:
        node.data_deps.append(int(dep_id))
    return node


def _load_chakra(astra_sim_dir: Path) -> None:
    astra_path = str(astra_sim_dir)
    if astra_path not in sys.path:
        sys.path.insert(0, astra_path)


def _run(cmd: list[str], cwd: Path, log_path: Path) -> None:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{cwd}:{existing}" if existing else str(cwd)
    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=f, stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}; log={log_path}")
