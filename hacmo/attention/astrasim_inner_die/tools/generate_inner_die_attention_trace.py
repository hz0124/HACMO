#!/usr/bin/env python3
import argparse
from pathlib import Path

from extern.graph_frontend.chakra.schema.protobuf.et_def_pb2 import (
    ALL_TO_ALL,
    COMM_COLL_NODE,
    COMM_RECV_NODE,
    COMM_SEND_NODE,
    COMP_NODE,
    MEM_LOAD_NODE,
    GlobalMetadata,
)
from extern.graph_frontend.chakra.schema.protobuf.et_def_pb2 import AttributeProto as ChakraAttr
from extern.graph_frontend.chakra.schema.protobuf.et_def_pb2 import Node as ChakraNode
from extern.graph_frontend.chakra.src.third_party.utils.protolib import encodeMessage as encode_message


def _append_comp(et_file, node_id: int, runtime_us: float, dep_id: int | None) -> int:
    node = ChakraNode()
    node.id = node_id
    node.name = "attention_compute"
    node.type = COMP_NODE
    node.duration_micros = max(1, int(round(runtime_us)))
    node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
    node.attr.append(ChakraAttr(name="num_ops", int64_val=max(1, int(round(runtime_us * 1_000_000)))))
    node.attr.append(ChakraAttr(name="tensor_size", uint64_val=max(1, int(round(runtime_us * 100_000)))))
    if dep_id is not None:
        node.data_deps.append(dep_id)
    encode_message(et_file, node)
    return node.id


def _append_mem_load(et_file, node_id: int, name: str, tensor_size: int, dep_id: int | None) -> int:
    node = ChakraNode()
    node.id = node_id
    node.name = name
    node.type = MEM_LOAD_NODE
    node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
    node.attr.append(ChakraAttr(name="tensor_size", uint64_val=max(1, int(tensor_size))))
    if dep_id is not None:
        node.data_deps.append(dep_id)
    encode_message(et_file, node)
    return node.id


def _append_mem_load_fragmented(
    et_file,
    node_id: int,
    name: str,
    useful_bytes: int,
    effective_payload_bytes: int,
    transfer_granularity_bytes: int,
    dep_id: int | None,
) -> int:
    payload = max(1, int(effective_payload_bytes))
    txn_count = max(1, (int(useful_bytes) + payload - 1) // payload)
    prev = dep_id
    cur = node_id
    for i in range(txn_count):
        billed = int(transfer_granularity_bytes) if int(transfer_granularity_bytes) > 0 else payload
        prev = _append_mem_load(et_file, cur, f"{name}_frag_{i}", billed, prev)
        cur += 1
    return prev


def _append_all_to_all(et_file, node_id: int, comm_size: int, dep_id: int | None) -> int:
    node = ChakraNode()
    node.id = node_id
    node.name = "remote_sram_exchange"
    node.type = COMM_COLL_NODE
    node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
    node.attr.append(ChakraAttr(name="comm_type", int64_val=ALL_TO_ALL))
    node.attr.append(ChakraAttr(name="comm_size", int64_val=max(1, int(comm_size))))
    if dep_id is not None:
        node.data_deps.append(dep_id)
    encode_message(et_file, node)
    return node.id


def _append_send_recv_pair(
    et_file,
    node_id: int,
    rank: int,
    num_ranks: int,
    comm_size: int,
    dep_id: int | None,
) -> int:
    send = ChakraNode()
    send.id = node_id
    send.name = "remote_sram_send"
    send.type = COMM_SEND_NODE
    send.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
    send.attr.append(ChakraAttr(name="comm_src", uint32_val=rank))
    send.attr.append(ChakraAttr(name="comm_dst", uint32_val=(rank + 1) % num_ranks))
    send.attr.append(ChakraAttr(name="comm_tag", uint32_val=rank))
    send.attr.append(ChakraAttr(name="comm_size", int64_val=max(1, int(comm_size))))
    if dep_id is not None:
        send.data_deps.append(dep_id)
    encode_message(et_file, send)

    recv = ChakraNode()
    recv.id = node_id + 1
    recv.name = "remote_sram_recv"
    recv.type = COMM_RECV_NODE
    recv.attr.append(ChakraAttr(name="is_cpu_op", bool_val=False))
    recv.attr.append(ChakraAttr(name="comm_src", uint32_val=(rank - 1 + num_ranks) % num_ranks))
    recv.attr.append(ChakraAttr(name="comm_dst", uint32_val=rank))
    recv.attr.append(ChakraAttr(name="comm_tag", uint32_val=(rank - 1 + num_ranks) % num_ranks))
    recv.attr.append(ChakraAttr(name="comm_size", int64_val=max(1, int(comm_size))))
    recv.data_deps.append(send.id)
    encode_message(et_file, recv)
    return recv.id


def generate(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for rank in range(args.num_ranks):
        rank_file = output_dir / f"{args.trace_prefix}.{rank}.et"
        with rank_file.open("wb") as et:
            encode_message(et, GlobalMetadata(version="0.0.4"))
            node_id = 0
            prev = None

            if rank < args.active_ranks:
                prev = _append_comp(et, node_id, args.compute_us, prev)
                node_id += 1

                if args.select_hbm_bytes > 0:
                    prev = _append_mem_load(et, node_id, "select_hbm_load", args.select_hbm_bytes, prev)
                    node_id += 1

                if args.qkv_hbm_bytes > 0:
                    if args.qkv_effective_payload_bytes > 0:
                        prev = _append_mem_load_fragmented(
                            et_file=et,
                            node_id=node_id,
                            name="qkv_hbm_load",
                            useful_bytes=args.qkv_hbm_bytes,
                            effective_payload_bytes=args.qkv_effective_payload_bytes,
                            transfer_granularity_bytes=args.qkv_transfer_granularity_bytes,
                            dep_id=prev,
                        )
                        payload = max(1, int(args.qkv_effective_payload_bytes))
                        node_id += max(1, (int(args.qkv_hbm_bytes) + payload - 1) // payload)
                    else:
                        prev = _append_mem_load(et, node_id, "qkv_hbm_load", args.qkv_hbm_bytes, prev)
                        node_id += 1

                if args.remote_sram_bytes > 0:
                    if args.remote_mode == "alltoall":
                        _append_all_to_all(et, node_id, args.remote_sram_bytes, prev)
                    else:
                        _append_send_recv_pair(
                            et_file=et,
                            node_id=node_id,
                            rank=rank,
                            num_ranks=args.num_ranks,
                            comm_size=args.remote_sram_bytes,
                            dep_id=prev,
                        )
            else:
                _append_comp(et, node_id, 1.0, prev)

    print(f"Generated {args.num_ranks} ET files at: {output_dir}")
    print(f"Trace prefix: {output_dir / args.trace_prefix}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate inner-die attention ET traces with MEM_LOAD + ALL_TO_ALL nodes.")
    parser.add_argument("--num_ranks", type=int, default=361)
    parser.add_argument("--active_ranks", type=int, default=360)
    parser.add_argument("--trace_prefix", type=str, default="inner_die_attn")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--compute_us", type=float, required=True)
    parser.add_argument("--select_hbm_bytes", type=int, required=True)
    parser.add_argument("--qkv_hbm_bytes", type=int, required=True)
    parser.add_argument("--remote_sram_bytes", type=int, required=True)
    parser.add_argument("--remote_mode", type=str, choices=["p2p", "alltoall"], default="p2p")
    parser.add_argument("--qkv_effective_payload_bytes", type=int, default=0)
    parser.add_argument("--qkv_transfer_granularity_bytes", type=int, default=0)
    args = parser.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
