"""Causal per-core residency and routed-access replay for TP1."""

from __future__ import annotations

import hashlib
import heapq
from bisect import bisect_right
from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

from .mapping import MappingPlan, consumer_schedule
from .workload import RequestLane, WorkloadWindow, selected_tokens

PrefetchHistory = Mapping[tuple[str, int], Mapping[int, int]]


@dataclass(frozen=True, order=True)
class KVBlock:
    request_id: str
    layer_id: int
    token_id: int


@dataclass(frozen=True)
class RoutedAccess:
    access_id: str
    block: KVBlock
    lane_id: int
    decode_step: int
    consumer_core: int
    source: str
    source_core: int | None
    cache_target_core: int | None
    hbm_port: int | None
    hbm_address: int | None
    address: int
    size_bytes: int

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["block"] = asdict(self.block)
        return data


@dataclass(frozen=True)
class PrefetchTransfer:
    transfer_id: str
    block: KVBlock
    target_core: int
    hbm_port: int
    hbm_address: int
    address: int
    size_bytes: int

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["block"] = asdict(self.block)
        return data


@dataclass(frozen=True)
class ReplaySummary:
    logical_accesses: int
    local_accesses: int
    remote_accesses: int
    hbm_accesses: int
    local_bytes: int
    remote_bytes: int
    hbm_requested_bytes: int
    hbm_cacheline_bytes: int
    remote_byte_hops: int
    hbm_byte_hops: int
    per_core_accesses: tuple[int, ...]

    @property
    def hit_rate(self) -> float:
        hits = self.local_accesses + self.remote_accesses
        return hits / self.logical_accesses if self.logical_accesses else 0.0

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "hit_rate": self.hit_rate}


def block_address(block: KVBlock, *, max_context: int, n_layers: int, kv_entry_bytes: int) -> int:
    if block.token_id >= max_context:
        raise ValueError("max_context does not cover the selected token")
    request_hash = int.from_bytes(hashlib.sha256(block.request_id.encode("utf-8")).digest()[:8], "little")
    request_slot = request_hash & ((1 << 24) - 1)
    request_stride = n_layers * max_context * kv_entry_bytes
    return request_slot * request_stride + (block.layer_id * max_context + block.token_id) * kv_entry_bytes


def hbm_port_address(
    address: int, kv_entry_bytes: int, num_ports: int
) -> tuple[int, int]:
    """Map a logical KV address to one port and its dense port-local address."""
    if address < 0 or kv_entry_bytes <= 0 or num_ports <= 0:
        raise ValueError("address must be non-negative and HBM dimensions positive")
    block_index, block_offset = divmod(address, kv_entry_bytes)
    port = block_index % num_ports
    local_block = block_index // num_ports
    return port, local_block * kv_entry_bytes + block_offset


def hbm_port_core(port: int, rows: int, cols: int, num_ports: int) -> int:
    if num_ports % 2 != 0:
        raise ValueError("symmetric top/bottom HBM placement requires an even port count")
    per_edge = num_ports // 2
    edge_index = port % per_edge
    col = min((2 * edge_index + 1) * cols // (2 * per_edge), cols - 1)
    row = 0 if port < per_edge else rows - 1
    return row * cols + col


def manhattan_hops(src: int, dst: int, cols: int) -> int:
    src_row, src_col = divmod(src, cols)
    dst_row, dst_col = divmod(dst, cols)
    return abs(src_row - dst_row) + abs(src_col - dst_col)


class ResidencyState:
    """Single-owner, per-core LRU state with capacity-accounted insertions."""

    def __init__(self, plan: MappingPlan) -> None:
        self.plan = plan
        self.caches = [OrderedDict() for _ in plan.cores]
        self.owner_by_block: dict[KVBlock, int] = {}
        cumulative = 0
        self.capacity_prefix: list[int] = []
        for core in plan.cores:
            cumulative += core.cache_capacity_entries
            self.capacity_prefix.append(cumulative)

    @property
    def resident_entries(self) -> int:
        return len(self.owner_by_block)

    def owner(self, block: KVBlock) -> int | None:
        return self.owner_by_block.get(block)

    def touch(self, block: KVBlock) -> None:
        owner = self.owner(block)
        if owner is not None:
            self.caches[owner].move_to_end(block)

    def clear(self) -> None:
        self.owner_by_block.clear()
        for cache in self.caches:
            cache.clear()

    def preferred_core(self, block: KVBlock) -> int | None:
        total = self.plan.total_cache_entries
        if total <= 0:
            return None
        digest = hashlib.sha256(
            f"{block.request_id}:{block.layer_id}:{block.token_id}".encode("utf-8")
        ).digest()
        slot = int.from_bytes(digest[:8], "little") % total
        core_id = bisect_right(self.capacity_prefix, slot)
        if core_id >= self.plan.num_cores:
            raise RuntimeError("cache-capacity prefix sum does not cover preferred slot")
        return core_id

    def insert(self, block: KVBlock, target_core: int | None = None, *, evict: bool = True) -> int | None:
        current = self.owner(block)
        if current is not None:
            self.touch(block)
            return current
        preferred = self.preferred_core(block) if target_core is None else target_core
        if preferred is None:
            return None
        for offset in range(self.plan.num_cores):
            core_id = (preferred + offset) % self.plan.num_cores
            capacity = self.plan.cores[core_id].cache_capacity_entries
            if capacity <= 0:
                continue
            cache = self.caches[core_id]
            if len(cache) >= capacity:
                if not evict or core_id != preferred:
                    continue
                victim, _ = cache.popitem(last=False)
                del self.owner_by_block[victim]
            cache[block] = None
            self.owner_by_block[block] = core_id
            return core_id
        return None

    def validate(self) -> None:
        total = 0
        seen: set[KVBlock] = set()
        for core_id, cache in enumerate(self.caches):
            if len(cache) > self.plan.cores[core_id].cache_capacity_entries:
                raise ValueError(f"core {core_id} exceeds its KV capacity")
            total += len(cache)
            for block in cache:
                if block in seen or self.owner_by_block.get(block) != core_id:
                    raise ValueError("KV block has inconsistent or duplicate ownership")
                seen.add(block)
        if total != len(self.owner_by_block):
            raise ValueError("owner index and per-core caches disagree")


def _hot_blocks(
    lanes: Sequence[RequestLane],
    layer_ids: Sequence[int],
    per_lane_layer_limit: int,
    history: PrefetchHistory | None = None,
) -> list[KVBlock]:
    ranked_blocks: list[KVBlock] = []
    history = history or {}
    for lane in lanes:
        for layer_id in layer_ids:
            layer = lane.profile.layers[layer_id]
            historical = history.get((lane.instance_id, layer_id), {})
            recent = sorted(
                historical,
                key=lambda token: (-historical[token], token),
            )
            lane_ranked = recent + [
                token
                for token in layer.prefill_ranked_tokens
                if token not in historical
            ]
            for token in lane_ranked[:per_lane_layer_limit]:
                ranked_blocks.append(
                    KVBlock(lane.instance_id, layer_id, token)
                )
    return ranked_blocks


def seed_static_cache(
    state: ResidencyState,
    lanes: Sequence[RequestLane],
    layer_ids: Sequence[int],
) -> None:
    if state.plan.total_cache_entries <= 0:
        return
    denominator = max(len(lanes) * len(layer_ids), 1)
    per_lane_layer = (state.plan.total_cache_entries + denominator - 1) // denominator
    # Successor links find the first non-full core at or after a preferred
    # core. This preserves the cyclic placement rule without a linear scan as
    # thousands of cores approach capacity during every offline candidate.
    num_cores = state.plan.num_cores
    successor = list(range(num_cores + 1))

    def find(core_id: int) -> int:
        while successor[core_id] != core_id:
            successor[core_id] = successor[successor[core_id]]
            core_id = successor[core_id]
        return core_id

    for core in state.plan.cores:
        if core.cache_capacity_entries == 0:
            successor[core.core_id] = core.core_id + 1
    for block in _hot_blocks(lanes, layer_ids, per_lane_layer):
        if state.resident_entries >= state.plan.total_cache_entries:
            break
        preferred = state.preferred_core(block)
        if preferred is None:
            break
        target = find(preferred)
        if target == num_cores:
            target = find(0)
        if target == num_cores:
            raise RuntimeError("static cache placement ran out of physical capacity")
        inserted = state.insert(block, target, evict=False)
        if inserted != target:
            raise RuntimeError("static cache placement did not use its selected core")
        if (
            len(state.caches[target])
            == state.plan.cores[target].cache_capacity_entries
        ):
            successor[target] = find(target + 1)
    state.validate()


def layer_prefetch_blocks(
    state: ResidencyState,
    lanes: Sequence[RequestLane],
    layer_id: int,
    history: PrefetchHistory,
) -> tuple[KVBlock, ...]:
    if state.plan.total_cache_entries <= 0:
        return ()
    per_lane = min(
        state.plan.topk,
        (state.plan.total_cache_entries + len(lanes) - 1)
        // max(len(lanes), 1),
    )
    return tuple(_hot_blocks(lanes, [layer_id], per_lane, history)[: state.plan.total_cache_entries])


def _prefetch_targets(
    state: ResidencyState,
    blocks: Sequence[KVBlock],
) -> tuple[tuple[int, KVBlock, int], ...]:
    retained_by_core = [0] * state.plan.num_cores
    for block in blocks:
        owner = state.owner(block)
        if owner is None:
            continue
        retained_by_core[owner] += 1
        # A resident member of the layer-prefetch set must survive the fills
        # for the rest of that set.
        state.touch(block)
    remaining_by_core = [
        core.cache_capacity_entries - retained_by_core[core.core_id]
        for core in state.plan.cores
    ]
    if any(remaining < 0 for remaining in remaining_by_core):
        raise RuntimeError("resident layer-prefetch set exceeds a core's KV capacity")
    num_cores = state.plan.num_cores
    successor = list(range(num_cores + 1))

    def find(core_id: int) -> int:
        while successor[core_id] != core_id:
            successor[core_id] = successor[successor[core_id]]
            core_id = successor[core_id]
        return core_id

    for core_id, remaining in enumerate(remaining_by_core):
        if remaining == 0:
            successor[core_id] = core_id + 1

    targets: list[tuple[int, KVBlock, int]] = []
    for index, block in enumerate(blocks):
        if state.owner(block) is not None:
            continue
        preferred = state.preferred_core(block)
        if preferred is None:
            continue
        target = find(preferred)
        if target == num_cores:
            target = find(0)
        if target == num_cores:
            raise RuntimeError("layer-prefetch set exceeds aggregate KV capacity")
        remaining_by_core[target] -= 1
        if remaining_by_core[target] == 0:
            successor[target] = find(target + 1)
        targets.append((index, block, target))
    return tuple(targets)


def plan_layer_prefetch(
    *,
    state: ResidencyState,
    lanes: Sequence[RequestLane],
    layer_id: int,
    decode_step: int,
    history: PrefetchHistory,
    kv_entry_bytes: int,
    cacheline_bytes: int,
    num_hbm_ports: int,
    max_context: int,
    n_layers: int,
) -> tuple[PrefetchTransfer, ...]:
    blocks = layer_prefetch_blocks(state, lanes, layer_id, history)
    transfers: list[PrefetchTransfer] = []
    for index, block, target in _prefetch_targets(state, blocks):
        address = block_address(
            block,
            max_context=max_context,
            n_layers=n_layers,
            kv_entry_bytes=kv_entry_bytes,
        )
        port, hbm_address = hbm_port_address(
            address, kv_entry_bytes, num_hbm_ports
        )
        transfers.append(
            PrefetchTransfer(
                transfer_id=f"L{layer_id}-D{decode_step}-P{index}",
                block=block,
                target_core=target,
                hbm_port=port,
                hbm_address=hbm_address,
                address=address,
                size_bytes=kv_entry_bytes,
            )
        )
    return tuple(transfers)


def commit_prefetch_fills(state: ResidencyState, transfers: Iterable[PrefetchTransfer]) -> None:
    for transfer in transfers:
        state.insert(transfer.block, transfer.target_core)
    state.validate()


def _assign_consumers(
    plan: MappingPlan,
    blocks: Sequence[KVBlock],
    *,
    kv_entry_bytes: int,
    num_hbm_ports: int,
    max_context: int,
    n_layers: int,
) -> tuple[int, ...]:
    """Assign one window before source classification or simulator replay."""
    quota_schedule = consumer_schedule(plan, len(blocks))
    if plan.policy in {"B", "C"}:
        return quota_schedule

    remaining = Counter(quota_schedule)
    candidates_by_port: dict[int, list[tuple[int, int]]] = {}
    for port in range(num_hbm_ports):
        port_core = hbm_port_core(port, plan.rows, plan.cols, num_hbm_ports)
        candidates_by_port[port] = [
            (manhattan_hops(port_core, core.core_id, plan.cols), core.core_id)
            for core in plan.cores
            if remaining[core.core_id] > 0
        ]
        heapq.heapify(candidates_by_port[port])

    consumers: list[int] = []
    for block in blocks:
        address = block_address(
            block,
            max_context=max_context,
            n_layers=n_layers,
            kv_entry_bytes=kv_entry_bytes,
        )
        port, _ = hbm_port_address(address, kv_entry_bytes, num_hbm_ports)
        candidates = candidates_by_port[port]
        while candidates and remaining[candidates[0][1]] == 0:
            heapq.heappop(candidates)
        if not candidates:
            raise RuntimeError("consumer assignment exhausted the offline quota")
        consumer = candidates[0][1]
        consumers.append(consumer)
        remaining[consumer] -= 1

    if Counter(consumers) != Counter(quota_schedule):
        raise RuntimeError("consumer assignment does not conserve per-core quota")
    return tuple(consumers)


def advance_prefetch_residency(
    state: ResidencyState,
    lanes: Sequence[RequestLane],
    layer_id: int,
    history: PrefetchHistory,
) -> None:
    """Apply an unsimulated layer prefetch directly to the physical LRU."""
    blocks = layer_prefetch_blocks(state, lanes, layer_id, history)
    for _, block, target in _prefetch_targets(state, blocks):
        state.insert(block, target)
    state.validate()


def route_window(
    *,
    window: WorkloadWindow,
    plan: MappingPlan,
    state: ResidencyState,
    kv_entry_bytes: int,
    cacheline_bytes: int,
    num_hbm_ports: int,
    max_context: int,
    n_layers: int,
) -> tuple[tuple[RoutedAccess, ...], ReplaySummary]:
    logical: list[tuple[RequestLane, int]] = []
    for lane in window.lanes:
        logical.extend((lane, token) for token in selected_tokens(window, lane))
    blocks = [
        KVBlock(lane.instance_id, window.layer_id, token)
        for lane, token in logical
    ]
    consumers = _assign_consumers(
        plan,
        blocks,
        kv_entry_bytes=kv_entry_bytes,
        num_hbm_ports=num_hbm_ports,
        max_context=max_context,
        n_layers=n_layers,
    )

    accesses: list[RoutedAccess] = []
    per_core = [0] * plan.num_cores
    for index, ((lane, token), block, consumer) in enumerate(
        zip(logical, blocks, consumers)
    ):
        address = block_address(
            block,
            max_context=max_context,
            n_layers=n_layers,
            kv_entry_bytes=kv_entry_bytes,
        )
        owner = state.owner(block)
        if owner == consumer:
            source = "local"
            source_core = owner
            target = owner
            port = None
            state.touch(block)
        elif owner is not None:
            source = "remote"
            source_core = owner
            target = owner
            port = None
            state.touch(block)
        else:
            source = "hbm"
            source_core = None
            # A demand miss is filled into the consumer's own LRU. Moving it
            # again to a hash-selected core would invent a second transfer
            # that is neither required by the mapping nor by the cache policy.
            target = consumer
            port, hbm_address = hbm_port_address(
                address, kv_entry_bytes, num_hbm_ports
            )
        if source != "hbm":
            hbm_address = None
        accesses.append(
            RoutedAccess(
                access_id=f"L{window.layer_id}-D{window.decode_step}-A{index}",
                block=block,
                lane_id=lane.lane_id,
                decode_step=window.decode_step,
                consumer_core=consumer,
                source=source,
                source_core=source_core,
                cache_target_core=target,
                hbm_port=port,
                hbm_address=hbm_address,
                address=address,
                size_bytes=kv_entry_bytes,
            )
        )
        per_core[consumer] += 1

    summary = summarize_accesses(
        accesses,
        rows=plan.rows,
        cols=plan.cols,
        cacheline_bytes=cacheline_bytes,
        num_hbm_ports=num_hbm_ports,
        per_core=per_core,
    )
    return tuple(accesses), summary


def commit_hbm_fills(state: ResidencyState, accesses: Iterable[RoutedAccess]) -> None:
    committed: set[KVBlock] = set()
    for access in accesses:
        if access.source != "hbm" or access.block in committed:
            continue
        if state.plan.total_cache_entries > 0:
            state.insert(access.block, access.cache_target_core)
        committed.add(access.block)
    state.validate()


def advance_residency(
    state: ResidencyState,
    window: WorkloadWindow,
    *,
    kv_entry_bytes: int,
    num_hbm_ports: int,
    max_context: int,
    n_layers: int,
    collect_blocks: bool = True,
) -> tuple[KVBlock, ...]:
    """Advance LRU state for an unsimulated window without inventing traffic."""
    blocks: list[KVBlock] = []
    misses: list[KVBlock] = []
    miss_assignments: list[tuple[KVBlock, int]] = []
    for lane in window.lanes:
        for token in selected_tokens(window, lane):
            block = KVBlock(lane.instance_id, window.layer_id, token)
            blocks.append(block)
    consumers = _assign_consumers(
        state.plan,
        blocks,
        kv_entry_bytes=kv_entry_bytes,
        num_hbm_ports=num_hbm_ports,
        max_context=max_context,
        n_layers=n_layers,
    )
    for block, consumer in zip(blocks, consumers):
        if state.owner(block) is None:
            misses.append(block)
            miss_assignments.append((block, consumer))
        else:
            state.touch(block)
    if state.plan.total_cache_entries > 0:
        for block, consumer in miss_assignments:
            state.insert(block, consumer)
    state.validate()
    return tuple(blocks) if collect_blocks else ()


def summarize_accesses(
    accesses: Sequence[RoutedAccess],
    *,
    rows: int,
    cols: int,
    cacheline_bytes: int,
    num_hbm_ports: int,
    per_core: Sequence[int] | None = None,
) -> ReplaySummary:
    counts = Counter(access.source for access in accesses)
    remote_hops = 0
    hbm_hops = 0
    hbm_cachelines = 0
    for access in accesses:
        if access.source == "remote":
            assert access.source_core is not None
            remote_hops += access.size_bytes * manhattan_hops(access.source_core, access.consumer_core, cols)
        elif access.source == "hbm":
            assert access.hbm_port is not None
            target = access.cache_target_core if access.cache_target_core is not None else access.consumer_core
            port_core = hbm_port_core(access.hbm_port, rows, cols, num_hbm_ports)
            hops = manhattan_hops(port_core, access.consumer_core, cols)
            if target != access.consumer_core:
                hops += manhattan_hops(access.consumer_core, target, cols)
            hbm_hops += access.size_bytes * hops
            first = access.address // cacheline_bytes
            last = (access.address + access.size_bytes - 1) // cacheline_bytes
            hbm_cachelines += last - first + 1
    size = accesses[0].size_bytes if accesses else 0
    return ReplaySummary(
        logical_accesses=len(accesses),
        local_accesses=counts["local"],
        remote_accesses=counts["remote"],
        hbm_accesses=counts["hbm"],
        local_bytes=counts["local"] * size,
        remote_bytes=counts["remote"] * size,
        hbm_requested_bytes=counts["hbm"] * size,
        hbm_cacheline_bytes=hbm_cachelines * cacheline_bytes,
        remote_byte_hops=remote_hops,
        hbm_byte_hops=hbm_hops,
        per_core_accesses=tuple(per_core or ()),
    )
