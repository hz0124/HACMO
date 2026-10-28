#!/usr/bin/env python3
"""Small Chakra protobuf writer used by the ASTRA backend."""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def _load_chakra(astra_sim_dir: Path):
    if str(astra_sim_dir) not in sys.path:
        sys.path.insert(0, str(astra_sim_dir))
    from extern.graph_frontend.chakra.schema.protobuf.et_def_pb2 import (  # type: ignore
        ALL_TO_ALL,
        COMM_COLL_NODE,
        COMM_RECV_NODE,
        COMM_SEND_NODE,
        COMP_NODE,
        GlobalMetadata,
    )
    from extern.graph_frontend.chakra.schema.protobuf.et_def_pb2 import AttributeProto as ChakraAttr  # type: ignore
    from extern.graph_frontend.chakra.schema.protobuf.et_def_pb2 import Node as ChakraNode  # type: ignore
    from extern.graph_frontend.chakra.src.third_party.utils.protolib import encodeMessage as encode_message  # type: ignore

    return (
        ALL_TO_ALL,
        COMM_COLL_NODE,
        COMM_RECV_NODE,
        COMM_SEND_NODE,
        COMP_NODE,
        GlobalMetadata,
        ChakraAttr,
        ChakraNode,
        encode_message,
    )


def write_moe_p2p_et(
    *,
    astra_sim_dir: Path,
    output_dir: Path,
    trace_prefix: str,
    num_ranks: int,
    dispatch_matrices: list[dict[int, dict[int, int]]],
    combine_matrices: list[dict[int, dict[int, int]]] | None = None,
    dispatch_segments: list[list[dict[int, dict[int, int]]]] | None = None,
    combine_segments: list[list[dict[int, dict[int, int]]]] | None = None,
    window_stats: list[dict],
    mapping: dict[int, list[int]],
    expert_compute_us_per_token: float,
    attention_us_per_die: float,
    batch_scale: float = 1.0,
    window_mappings: list[dict[int, list[int]]] | None = None,
    rank_compute_tokens: list[list[float]] | None = None,
    rank_compute_us: list[list[float]] | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    (
        ALL_TO_ALL,
        COMM_COLL_NODE,
        COMM_RECV_NODE,
        COMM_SEND_NODE,
        COMP_NODE,
        GlobalMetadata,
        ChakraAttr,
        ChakraNode,
        encode_message,
    ) = _load_chakra(astra_sim_dir)

    if dispatch_segments is None:
        dispatch_segments = [[m] for m in dispatch_matrices]
    if combine_segments is None:
        if combine_matrices is not None:
            combine_segments = [[m] for m in combine_matrices]
        else:
            combine_segments = [[m] for m in dispatch_matrices]
    comm_cache = _build_segment_comm_cache(
        dispatch_segments, combine_segments, num_ranks
    )

    def append_comp(
        et, node_id: int, name: str, runtime_us: float, dep_id: int | None
    ) -> int:
        node = ChakraNode()
        node.id = node_id
        node.name = name
        node.type = COMP_NODE
        node.duration_micros = max(int(round(runtime_us)), 1)
        node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
        node.attr.append(
            ChakraAttr(
                name="num_ops", int64_val=max(int(round(runtime_us)), 1) * 10_000_000
            )
        )
        node.attr.append(
            ChakraAttr(
                name="tensor_size",
                uint64_val=max(int(round(runtime_us)), 1) * 1_000_000,
            )
        )
        if dep_id is not None:
            node.data_deps.append(dep_id)
        encode_message(et, node)
        return node.id

    def append_barrier(et, node_id: int, name: str, dep_ids: list[int]) -> int:
        node = ChakraNode()
        node.id = node_id
        node.name = name
        node.type = COMP_NODE
        node.duration_micros = 0
        node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
        node.attr.append(ChakraAttr(name="num_ops", int64_val=0))
        node.attr.append(ChakraAttr(name="tensor_size", uint64_val=0))
        for dep in dep_ids:
            node.data_deps.append(int(dep))
        encode_message(et, node)
        return node.id

    def append_step_wall(et, node_id: int, name: str, dep_id: int | None) -> int:
        node = ChakraNode()
        node.id = node_id
        node.name = name
        node.type = COMM_COLL_NODE
        node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
        node.attr.append(ChakraAttr(name="comm_type", int64_val=ALL_TO_ALL))
        node.attr.append(ChakraAttr(name="comm_size", int64_val=0))
        if dep_id is not None:
            node.data_deps.append(dep_id)
        encode_message(et, node)
        return node.id

    def append_send(
        et, node_id: int, src: int, dst: int, tag: int, size: int, dep_id: int | None
    ) -> int:
        node = ChakraNode()
        node.id = node_id
        node.name = f"send_{src}_{dst}_{tag}"
        node.type = COMM_SEND_NODE
        node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
        node.attr.append(ChakraAttr(name="comm_src", uint32_val=int(src)))
        node.attr.append(ChakraAttr(name="comm_dst", uint32_val=int(dst)))
        node.attr.append(ChakraAttr(name="comm_tag", uint32_val=int(tag)))
        node.attr.append(ChakraAttr(name="comm_size", int64_val=max(int(size), 1)))
        if dep_id is not None:
            node.data_deps.append(dep_id)
        encode_message(et, node)
        return node.id

    def append_recv(
        et, node_id: int, src: int, dst: int, tag: int, size: int, dep_id: int | None
    ) -> int:
        node = ChakraNode()
        node.id = node_id
        node.name = f"recv_{src}_{dst}_{tag}"
        node.type = COMM_RECV_NODE
        node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
        node.attr.append(ChakraAttr(name="comm_src", uint32_val=int(src)))
        node.attr.append(ChakraAttr(name="comm_dst", uint32_val=int(dst)))
        node.attr.append(ChakraAttr(name="comm_tag", uint32_val=int(tag)))
        node.attr.append(ChakraAttr(name="comm_size", int64_val=max(int(size), 1)))
        if dep_id is not None:
            node.data_deps.append(dep_id)
        encode_message(et, node)
        return node.id

    for rank in range(num_ranks):
        with (output_dir / f"{trace_prefix}.{rank}.et").open("wb") as et:
            encode_message(et, GlobalMetadata(version="0.0.4"))
            node_id = 0
            prev = None
            for w_id, ws in enumerate(window_stats):
                cur_mapping = (
                    window_mappings[w_id] if window_mappings is not None else mapping
                )
                die_to_experts: dict[int, list[int]] = {}
                for e, ds in cur_mapping.items():
                    for d in ds:
                        die_to_experts.setdefault(d, []).append(e)

                if attention_us_per_die > 0:
                    prev = append_comp(
                        et, node_id, f"w{w_id}_attention", attention_us_per_die, prev
                    )
                    node_id += 1

                for seg_id, seg in enumerate(comm_cache[w_id]["dispatch"]):
                    deps = []
                    for dst, tag, size in seg["send"][rank]:
                        deps.append(
                            append_send(et, node_id, rank, dst, tag, size, prev)
                        )
                        node_id += 1
                    for src, tag, size in seg["recv"][rank]:
                        deps.append(
                            append_recv(et, node_id, src, rank, tag, size, prev)
                        )
                        node_id += 1
                    if deps:
                        prev = append_barrier(
                            et, node_id, f"w{w_id}_dispatch_s{seg_id}_done", deps
                        )
                        node_id += 1

                if rank_compute_us is not None:
                    runtime_us = float(rank_compute_us[w_id][rank])
                elif rank_compute_tokens is not None:
                    expert_tokens = float(rank_compute_tokens[w_id][rank])
                    compute_scale = 1.0
                    runtime_us = (
                        expert_tokens * compute_scale * expert_compute_us_per_token
                    )
                else:
                    expert_tokens = 0.0
                    for e in die_to_experts.get(rank, []):
                        expert_tokens += float(
                            ws.get("expert_counts", {}).get(str(e), 0)
                        ) / max(len(cur_mapping[e]), 1)
                    compute_scale = batch_scale
                    runtime_us = (
                        expert_tokens * compute_scale * expert_compute_us_per_token
                    )
                if runtime_us > 0:
                    prev = append_comp(
                        et, node_id, f"w{w_id}_expert_compute", runtime_us, prev
                    )
                    node_id += 1

                for seg_id, seg in enumerate(comm_cache[w_id]["combine"]):
                    deps = []
                    for dst, tag, size in seg["send"][rank]:
                        deps.append(
                            append_send(et, node_id, rank, dst, tag, size, prev)
                        )
                        node_id += 1
                    for src, tag, size in seg["recv"][rank]:
                        deps.append(
                            append_recv(et, node_id, src, rank, tag, size, prev)
                        )
                        node_id += 1
                    if deps:
                        prev = append_barrier(
                            et, node_id, f"w{w_id}_combine_s{seg_id}_done", deps
                        )
                        node_id += 1
                prev = append_step_wall(et, node_id, f"w{w_id}_step_done", prev)
                node_id += 1
            if node_id == 0:
                append_comp(et, node_id, "idle_noop", 1.0, None)
    return output_dir / trace_prefix


def _build_segment_comm_cache(
    dispatch_segments: list[list[dict[int, dict[int, int]]]],
    combine_segments: list[list[dict[int, dict[int, int]]]],
    num_ranks: int,
) -> list[dict[str, list[dict[str, List[List[tuple[int, int, int]]]]]]]:
    out = []
    for w_id, dispatch_mats in enumerate(dispatch_segments):
        cache = {"dispatch": [], "combine": []}
        for seg_id, mat in enumerate(dispatch_mats):
            cache["dispatch"].append(
                _build_one_segment_cache(mat, num_ranks, w_id, seg_id, 0)
            )
        for seg_id, mat in enumerate(combine_segments[w_id]):
            cache["combine"].append(
                _build_one_segment_cache(mat, num_ranks, w_id, seg_id, 500_000_000)
            )
        out.append(cache)
    return out


def _build_one_segment_cache(
    mat: dict[int, dict[int, int]],
    num_ranks: int,
    window_id: int,
    segment_id: int,
    tag_base: int,
) -> dict[str, List[List[tuple[int, int, int]]]]:
    cache = {
        "send": [[] for _ in range(num_ranks)],
        "recv": [[] for _ in range(num_ranks)],
    }
    seg_stride = 100_000
    for src, dsts in mat.items():
        for dst, size in dsts.items():
            tag = (
                tag_base
                + window_id * 1_000_000
                + segment_id * seg_stride
                + src * 1024
                + dst
            )
            cache["send"][src].append((dst, tag, size))
            cache["recv"][dst].append((src, tag, size))
    return cache
