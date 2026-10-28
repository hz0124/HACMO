from __future__ import annotations

import pickle
from collections import Counter
from pathlib import Path

import numpy as np

from hacmo.attention.tp1_core.generate_audited_tp1_pack import (
    HardwareConfig,
    _parser,
)
from hacmo.attention.tp1_core.mapping import (
    build_mapping_plan,
    consumer_schedule,
)
from hacmo.attention.tp1_core.offline_solver import (
    OfflineCostModel,
    _window_cost_us,
    score_mapping_plan,
    solve_nonuniform_mapping,
)
from hacmo.attention.tp1_core.routing import (
    KVBlock,
    ResidencyState,
    RoutedAccess,
    advance_prefetch_residency,
    advance_residency,
    commit_hbm_fills,
    commit_prefetch_fills,
    hbm_port_address,
    hbm_port_core,
    manhattan_hops,
    plan_layer_prefetch,
    route_window,
)
from hacmo.attention.tp1_core.workload import (
    RequestLane,
    WorkloadWindow,
    load_profile,
    selected_tokens,
)


def _profile(path: Path) -> None:
    payload = {
        "prompt_id": "longbench_unit_" + "a" * 64,
        "prompt_len": 16,
        "gen_len": 2,
        "profiling_data": {
            0: {
                "prefill_token_counts": {0: 9, 1: 8, 2: 7},
                "decode_indices": np.asarray([[0, 1, 1, 5], [2, 3, 4, 5]], dtype=np.int32),
                "decode_positions": np.asarray([16, 17], dtype=np.int32),
            },
            1: {
                "prefill_token_counts": {0: 9, 1: 8, 2: 7},
                "decode_indices": np.asarray([[0, 1, 1, 5], [2, 3, 4, 5]], dtype=np.int32),
                "decode_positions": np.asarray([16, 17], dtype=np.int32),
            },
        },
    }
    path.write_bytes(pickle.dumps(payload))


def test_compact_profile_preserves_complete_unique_topk_row(tmp_path: Path) -> None:
    path = tmp_path / "unit_profile.pkl"
    _profile(path)
    profile = load_profile(path)
    lane = RequestLane(0, "request-0", profile)
    window = WorkloadWindow(0, 0, (lane,))

    assert profile.prompt_len == 16
    assert profile.layers[0].prefill_counts[0] == 9
    assert selected_tokens(window, lane) == (0, 1, 5)


def test_mapping_conserves_work_and_sram_and_has_expected_gradient() -> None:
    plan = build_mapping_plan(
        policy="S",
        rows=4,
        cols=2,
        batch_size=2,
        topk=8,
        kv_entry_bytes=128,
        total_sram_bytes=64 * 1024,
        cache_ratio=0.5,
        compute_gradient=1.0,
        cache_gradient=1.0,
    )

    assert plan.total_compute_entries == 16
    assert plan.total_cache_entries * 128 <= plan.cache_budget_bytes
    assert plan.cores[0].compute_quota_entries >= plan.cores[2].compute_quota_entries
    assert plan.cores[2].cache_capacity_entries >= plan.cores[0].cache_capacity_entries
    per_core_sram = plan.total_sram_bytes // plan.num_cores
    assert all(
        core.compute_memory_bytes + core.cache_capacity_entries * plan.kv_entry_bytes
        <= per_core_sram + 1
        for core in plan.cores
    )


def test_cache_gradient_changes_capacity_without_changing_compute_work() -> None:
    common = dict(
        policy="S",
        rows=4,
        cols=2,
        batch_size=2,
        topk=8,
        kv_entry_bytes=128,
        total_sram_bytes=64 * 1024,
        cache_ratio=0.25,
        compute_gradient=1.0,
    )
    uniform_cache = build_mapping_plan(cache_gradient=0.0, **common)
    center_heavy_cache = build_mapping_plan(cache_gradient=1.0, **common)

    assert [core.compute_quota_entries for core in uniform_cache.cores] == [
        core.compute_quota_entries for core in center_heavy_cache.cores
    ]
    assert [core.cache_capacity_entries for core in uniform_cache.cores] != [
        core.cache_capacity_entries for core in center_heavy_cache.cores
    ]
    assert (
        center_heavy_cache.cores[2].cache_capacity_entries
        > center_heavy_cache.cores[0].cache_capacity_entries
    )


def test_hbm_port_mapping_uses_dense_port_local_block_addresses() -> None:
    mapped = [hbm_port_address(index * 656, 656, 8) for index in range(16)]

    assert [port for port, _ in mapped[:8]] == list(range(8))
    assert [address for _, address in mapped[:8]] == [0] * 8
    assert [port for port, _ in mapped[8:]] == list(range(8))
    assert [address for _, address in mapped[8:]] == [656] * 8


def test_audited_pack_default_uses_paper_range_per_link_noc_bandwidth() -> None:
    assert HardwareConfig().noc_link_gb_per_s == 256.0
    assert _parser().parse_args([]).noc_link_gb_per_s == 256.0


def test_routing_sources_come_only_from_explicit_residency(tmp_path: Path) -> None:
    path = tmp_path / "unit_profile.pkl"
    _profile(path)
    profile = load_profile(path)
    lane = RequestLane(0, "request-0", profile)
    window = WorkloadWindow(0, 0, (lane,))
    plan = build_mapping_plan(
        policy="C",
        rows=2,
        cols=2,
        batch_size=1,
        topk=3,
        kv_entry_bytes=128,
        total_sram_bytes=4096,
        cache_ratio=0.5,
    )
    empty_state = ResidencyState(plan)
    empty_accesses, _ = route_window(
        window=window,
        plan=plan,
        state=empty_state,
        kv_entry_bytes=128,
        cacheline_bytes=128,
        num_hbm_ports=2,
        max_context=32,
        n_layers=1,
    )
    assigned = {access.block.token_id: access.consumer_core for access in empty_accesses}
    state = ResidencyState(plan)
    state.insert(KVBlock("request-0", 0, 0), target_core=assigned[0])
    remote_owner = (assigned[1] + 1) % plan.num_cores
    state.insert(KVBlock("request-0", 0, 1), target_core=remote_owner)

    accesses, summary = route_window(
        window=window,
        plan=plan,
        state=state,
        kv_entry_bytes=128,
        cacheline_bytes=128,
        num_hbm_ports=2,
        max_context=32,
        n_layers=1,
    )

    assert summary.logical_accesses == 3
    assert summary.local_accesses + summary.remote_accesses == 2
    assert summary.hbm_accesses == 1
    assert summary.hit_rate == 2 / 3
    assert {access.source for access in accesses} == {"local", "remote", "hbm"}
    miss = next(access for access in accesses if access.source == "hbm")
    assert miss.cache_target_core == miss.consumer_core
    commit_hbm_fills(state, accesses)
    assert state.owner(KVBlock("request-0", 0, 5)) == miss.consumer_core
    state.validate()


def test_frozen_mapping_owns_compute_assignment_independent_of_residency(
    tmp_path: Path,
) -> None:
    path = tmp_path / "unit_profile.pkl"
    _profile(path)
    profile = load_profile(path)
    lane = RequestLane(0, "request-0", profile)
    window = WorkloadWindow(0, 0, (lane,))
    plan = build_mapping_plan(
        policy="C",
        rows=2,
        cols=2,
        batch_size=1,
        topk=3,
        kv_entry_bytes=128,
        total_sram_bytes=4096,
        cache_ratio=0.5,
    )
    uncached_state = ResidencyState(plan)
    cached_state = ResidencyState(plan)
    cached_state.insert(KVBlock("request-0", 0, 0), target_core=2)
    cached_state.insert(KVBlock("request-0", 0, 1), target_core=1)

    uncached, _ = route_window(
        window=window,
        plan=plan,
        state=uncached_state,
        kv_entry_bytes=128,
        cacheline_bytes=128,
        num_hbm_ports=2,
        max_context=32,
        n_layers=1,
    )
    cached, _ = route_window(
        window=window,
        plan=plan,
        state=cached_state,
        kv_entry_bytes=128,
        cacheline_bytes=128,
        num_hbm_ports=2,
        max_context=32,
        n_layers=1,
    )

    assert tuple(access.consumer_core for access in uncached) == tuple(
        access.consumer_core for access in cached
    )
    assert any(access.source in {"local", "remote"} for access in cached)
    assert any(access.source == "hbm" for access in cached)


def test_routing_preserves_frozen_compute_and_reports_remote_hit(
    tmp_path: Path,
) -> None:
    path = tmp_path / "unit_profile.pkl"
    _profile(path)
    profile = load_profile(path)
    lane = RequestLane(0, "request-0", profile)
    plan = build_mapping_plan(
        policy="C",
        rows=2,
        cols=2,
        batch_size=1,
        topk=3,
        kv_entry_bytes=128,
        total_sram_bytes=4096,
        cache_ratio=0.5,
    )
    empty, _ = route_window(
        window=WorkloadWindow(0, 0, (lane,)),
        plan=plan,
        state=ResidencyState(plan),
        kv_entry_bytes=128,
        cacheline_bytes=128,
        num_hbm_ports=2,
        max_context=32,
        n_layers=1,
    )
    assigned = {access.block.token_id: access.consumer_core for access in empty}
    state = ResidencyState(plan)
    state.insert(KVBlock("request-0", 0, 0), target_core=assigned[0])
    remote_owner = (assigned[1] + 1) % plan.num_cores
    state.insert(KVBlock("request-0", 0, 1), target_core=remote_owner)

    accesses, summary = route_window(
        window=WorkloadWindow(0, 0, (lane,)),
        plan=plan,
        state=state,
        kv_entry_bytes=128,
        cacheline_bytes=128,
        num_hbm_ports=2,
        max_context=32,
        n_layers=1,
    )

    by_token = {access.block.token_id: access for access in accesses}
    assert by_token[0].consumer_core == by_token[0].source_core == assigned[0]
    assert by_token[0].source == "local"
    assert by_token[1].consumer_core == assigned[1]
    assert by_token[1].source_core == remote_owner
    assert by_token[1].source == "remote"
    assert summary.local_accesses == 1
    assert summary.remote_accesses == 1
    assert tuple(access.consumer_core for access in accesses) == tuple(
        access.consumer_core for access in empty
    )


def test_o_mapping_uses_frozen_compute_quota_before_prefetch_commit(
    tmp_path: Path,
) -> None:
    path = tmp_path / "unit_profile.pkl"
    _profile(path)
    profile = load_profile(path)
    lane = RequestLane(0, "request-0", profile)
    plan = build_mapping_plan(
        policy="O",
        rows=2,
        cols=2,
        batch_size=1,
        topk=3,
        kv_entry_bytes=128,
        total_sram_bytes=4096,
        cache_ratio=0.5,
        compute_gradient=1.0,
        cache_gradient=1.0,
    )
    state = ResidencyState(plan)
    accesses, _ = route_window(
        window=WorkloadWindow(0, 0, (lane,)),
        plan=plan,
        state=state,
        kv_entry_bytes=128,
        cacheline_bytes=128,
        num_hbm_ports=2,
        max_context=32,
        n_layers=1,
    )

    assert all(access.source == "hbm" for access in accesses)
    cached_accesses, _ = route_window(
        window=WorkloadWindow(0, 0, (lane,)),
        plan=plan,
        state=ResidencyState(plan),
        kv_entry_bytes=128,
        cacheline_bytes=128,
        num_hbm_ports=2,
        max_context=32,
        n_layers=1,
    )
    assert tuple(access.consumer_core for access in accesses) == tuple(
        access.consumer_core for access in cached_accesses
    )


def test_baseline_b_routes_every_access_to_hbm(tmp_path: Path) -> None:
    path = tmp_path / "unit_profile.pkl"
    _profile(path)
    profile = load_profile(path)
    lane = RequestLane(0, "request-0", profile)
    plan = build_mapping_plan(
        policy="B",
        rows=2,
        cols=2,
        batch_size=1,
        topk=3,
        kv_entry_bytes=128,
        total_sram_bytes=4096,
        cache_ratio=0.0,
    )
    state = ResidencyState(plan)
    accesses, summary = route_window(
        window=WorkloadWindow(0, 0, (lane,)),
        plan=plan,
        state=state,
        kv_entry_bytes=128,
        cacheline_bytes=128,
        num_hbm_ports=2,
        max_context=32,
        n_layers=1,
    )

    assert all(access.source == "hbm" for access in accesses)
    assert summary.hbm_accesses == summary.logical_accesses == 3
    repeated, _ = route_window(
        window=WorkloadWindow(0, 0, (lane,)),
        plan=plan,
        state=ResidencyState(plan),
        kv_entry_bytes=128,
        cacheline_bytes=128,
        num_hbm_ports=2,
        max_context=32,
        n_layers=1,
    )
    assert tuple(access.consumer_core for access in repeated) == tuple(
        access.consumer_core for access in accesses
    )
    commit_hbm_fills(state, accesses)
    assert state.resident_entries == 0


def test_port_aware_assignment_conserves_nonuniform_per_core_quota(tmp_path: Path) -> None:
    path = tmp_path / "unit_profile.pkl"
    _profile(path)
    profile = load_profile(path)
    lanes = tuple(
        RequestLane(index, f"request-{index}", profile)
        for index in range(4)
    )
    plan = build_mapping_plan(
        policy="S",
        rows=4,
        cols=4,
        batch_size=4,
        topk=3,
        kv_entry_bytes=128,
        total_sram_bytes=16 * 4096,
        cache_ratio=0.5,
        compute_gradient=1.0,
        cache_gradient=1.0,
    )
    state = ResidencyState(plan)

    accesses, _ = route_window(
        window=WorkloadWindow(0, 0, lanes),
        plan=plan,
        state=state,
        kv_entry_bytes=128,
        cacheline_bytes=128,
        num_hbm_ports=2,
        max_context=32,
        n_layers=1,
    )
    assert Counter(access.consumer_core for access in accesses) == Counter(
        consumer_schedule(plan, len(accesses))
    )

    baseline_plan = build_mapping_plan(
        policy="C",
        rows=4,
        cols=4,
        batch_size=4,
        topk=3,
        kv_entry_bytes=128,
        total_sram_bytes=16 * 4096,
        cache_ratio=0.5,
    )
    baseline_accesses, _ = route_window(
        window=WorkloadWindow(0, 0, lanes),
        plan=baseline_plan,
        state=ResidencyState(baseline_plan),
        kv_entry_bytes=128,
        cacheline_bytes=128,
        num_hbm_ports=2,
        max_context=32,
        n_layers=1,
    )
    baseline_hops = sum(
        manhattan_hops(
            hbm_port_core(int(access.hbm_port), 4, 4, 2),
            access.consumer_core,
            4,
        )
        for access in baseline_accesses
    )
    routed_hops = sum(
        manhattan_hops(
            hbm_port_core(int(access.hbm_port), 4, 4, 2),
            access.consumer_core,
            4,
        )
        for access in accesses
    )
    assert routed_hops < baseline_hops

    cached_state = ResidencyState(plan)
    cached_state.insert(accesses[0].block, target_core=5)
    cached_accesses, _ = route_window(
        window=WorkloadWindow(0, 0, lanes),
        plan=plan,
        state=cached_state,
        kv_entry_bytes=128,
        cacheline_bytes=128,
        num_hbm_ports=2,
        max_context=32,
        n_layers=1,
    )
    assert tuple(access.consumer_core for access in cached_accesses) == tuple(
        access.consumer_core for access in accesses
    )


def test_unsimulated_window_advances_the_same_residency_state(tmp_path: Path) -> None:
    path = tmp_path / "unit_profile.pkl"
    _profile(path)
    profile = load_profile(path)
    lane = RequestLane(0, "request-0", profile)
    plan = build_mapping_plan(
        policy="C",
        rows=2,
        cols=2,
        batch_size=1,
        topk=3,
        kv_entry_bytes=128,
        total_sram_bytes=4096,
        cache_ratio=0.5,
    )
    reference = ResidencyState(plan)
    compact = ResidencyState(plan)
    window = WorkloadWindow(0, 0, (lane,))

    accesses, _ = route_window(
        window=window,
        plan=plan,
        state=reference,
        kv_entry_bytes=128,
        cacheline_bytes=128,
        num_hbm_ports=2,
        max_context=32,
        n_layers=1,
    )
    commit_hbm_fills(reference, accesses)
    misses = advance_residency(
        compact,
        window,
        kv_entry_bytes=128,
        num_hbm_ports=2,
        max_context=32,
        n_layers=1,
    )

    assert len(misses) == 3
    assert compact.owner_by_block == reference.owner_by_block
    assert [tuple(cache) for cache in compact.caches] == [
        tuple(cache) for cache in reference.caches
    ]


def test_offline_solver_returns_frozen_candidate_without_simulator_input(tmp_path: Path) -> None:
    path = tmp_path / "unit_profile.pkl"
    _profile(path)
    profile = load_profile(path)
    lanes = (RequestLane(0, "request-0", profile),)
    model = OfflineCostModel(
        aggregate_hbm_gbps=4800.0,
        hbm_ports=2,
        hbm_internal_ns=80.0,
        hbm_concurrency_per_port=8,
        noc_link_gbps=256.0,
        gamma=1.2,
        die_compute_tflops=100.0,
        flops_per_selected_kv=1024.0,
        cacheline_bytes=128,
        kv_entry_bytes=128,
        max_context=32,
        n_layers=1,
    )

    plan, scores = solve_nonuniform_mapping(
        rows=2,
        cols=2,
        batch_size=1,
        topk=3,
        kv_entry_bytes=128,
        total_sram_bytes=4096,
        cache_ratio=0.5,
        lanes=lanes,
        layer_ids=[0],
        decode_steps=[0],
        model=model,
        compute_gradients=[0.0, 1.0],
        cache_gradients=[0.0, 1.0],
    )

    assert len(scores) == 4
    assert plan.offline_score_us is not None
    assert plan.compute_gradient == 1.0
    assert plan.cache_gradient == 1.0
    assert plan.total_compute_entries == 3


def test_offline_score_uses_full_layer_cache_population(tmp_path: Path) -> None:
    path = tmp_path / "unit_profile.pkl"
    _profile(path)
    profile = load_profile(path)
    lanes = (RequestLane(0, "request-0", profile),)
    plan = build_mapping_plan(
        policy="C",
        rows=1,
        cols=1,
        batch_size=1,
        topk=3,
        kv_entry_bytes=128,
        total_sram_bytes=256,
        cache_ratio=1.0,
    )
    model = OfflineCostModel(
        aggregate_hbm_gbps=100.0,
        hbm_ports=2,
        hbm_internal_ns=80.0,
        hbm_concurrency_per_port=1,
        noc_link_gbps=100.0,
        gamma=1.0,
        die_compute_tflops=100.0,
        flops_per_selected_kv=1024.0,
        cacheline_bytes=128,
        kv_entry_bytes=128,
        max_context=32,
        n_layers=2,
    )

    single_layer = score_mapping_plan(
        plan,
        lanes,
        layer_ids=[0],
        decode_steps=[0],
        model=model,
        residency_layer_ids=[0],
    )
    full_population = score_mapping_plan(
        plan,
        lanes,
        layer_ids=[0],
        decode_steps=[0],
        model=model,
        residency_layer_ids=[0, 1],
    )

    assert full_population.mean_hbm_us > single_layer.mean_hbm_us


def test_offline_score_adds_compute_and_equation_one_memory_cost(tmp_path: Path) -> None:
    path = tmp_path / "unit_profile.pkl"
    _profile(path)
    profile = load_profile(path)
    lanes = (RequestLane(0, "request-0", profile),)
    plan = build_mapping_plan(
        policy="B",
        rows=1,
        cols=1,
        batch_size=1,
        topk=3,
        kv_entry_bytes=128,
        total_sram_bytes=256,
        cache_ratio=0.0,
    )
    model = OfflineCostModel(
        aggregate_hbm_gbps=100.0,
        hbm_ports=2,
        hbm_internal_ns=80.0,
        hbm_concurrency_per_port=1,
        noc_link_gbps=100.0,
        gamma=1.0,
        die_compute_tflops=100.0,
        flops_per_selected_kv=1024.0,
        cacheline_bytes=128,
        kv_entry_bytes=128,
        max_context=32,
        n_layers=2,
    )

    score = score_mapping_plan(
        plan,
        lanes,
        layer_ids=[0],
        decode_steps=[0],
        model=model,
        residency_layer_ids=[0, 1],
    )

    expected = score.mean_compute_us + score.mean_hbm_us + score.mean_noc_us
    assert abs(score.mean_window_us - expected) < 1e-12


def test_offline_score_accounts_for_shared_xy_link_service() -> None:
    plan = build_mapping_plan(
        policy="B",
        rows=4,
        cols=4,
        batch_size=1,
        topk=21,
        kv_entry_bytes=128,
        total_sram_bytes=4096,
        cache_ratio=0.0,
    )
    model = OfflineCostModel(
        aggregate_hbm_gbps=1_000_000.0,
        hbm_ports=2,
        hbm_internal_ns=0.0,
        hbm_concurrency_per_port=1024,
        noc_link_gbps=100.0,
        gamma=1.0,
        die_compute_tflops=1_000_000.0,
        flops_per_selected_kv=1.0,
        cacheline_bytes=128,
        kv_entry_bytes=128,
        max_context=128,
        n_layers=1,
    )
    consumers = [6] * 10 + [10] * 10 + [14]
    accesses = tuple(
        RoutedAccess(
            access_id=f"a{index}",
            block=KVBlock("request-0", 0, index),
            lane_id=0,
            decode_step=0,
            consumer_core=consumer,
            source="hbm",
            source_core=None,
            cache_target_core=consumer,
            hbm_port=0,
            hbm_address=index * 128,
            address=index * 128,
            size_bytes=128,
        )
        for index, consumer in enumerate(consumers)
    )

    window_us, _, _, noc_us, bottleneck_us = _window_cost_us(
        plan, accesses, model
    )

    expected_bottleneck_us = 21 * 128 / (100.0 * 1e9) * 1e6
    assert abs(bottleneck_us - expected_bottleneck_us) < 1e-12
    assert noc_us == bottleneck_us
    assert window_us >= bottleneck_us


def test_offline_hbm_service_is_shared_per_physical_port() -> None:
    plan = build_mapping_plan(
        policy="B",
        rows=2,
        cols=2,
        batch_size=1,
        topk=4,
        kv_entry_bytes=128,
        total_sram_bytes=4096,
        cache_ratio=0.0,
    )
    model = OfflineCostModel(
        aggregate_hbm_gbps=100.0,
        hbm_ports=2,
        hbm_internal_ns=80.0,
        hbm_concurrency_per_port=2,
        noc_link_gbps=100.0,
        gamma=1.0,
        die_compute_tflops=100.0,
        flops_per_selected_kv=1.0,
        cacheline_bytes=128,
        kv_entry_bytes=128,
        max_context=32,
        n_layers=1,
    )

    def hbm_accesses(consumers: list[int]) -> tuple[RoutedAccess, ...]:
        return tuple(
            RoutedAccess(
                access_id=f"a{index}",
                block=KVBlock("request-0", 0, index),
                lane_id=0,
                decode_step=0,
                consumer_core=consumer,
                source="hbm",
                source_core=None,
                cache_target_core=consumer,
                hbm_port=0,
                hbm_address=index * 128,
                address=index * 128,
                size_bytes=128,
            )
            for index, consumer in enumerate(consumers)
        )

    clustered_hbm_us = _window_cost_us(
        plan, hbm_accesses([0, 0, 0, 0]), model
    )[2]
    distributed_hbm_us = _window_cost_us(
        plan, hbm_accesses([0, 1, 2, 3]), model
    )[2]

    assert clustered_hbm_us == distributed_hbm_us


def test_prefetch_becomes_resident_only_after_completion_commit(tmp_path: Path) -> None:
    path = tmp_path / "unit_profile.pkl"
    _profile(path)
    profile = load_profile(path)
    lanes = (RequestLane(0, "request-0", profile),)
    plan = build_mapping_plan(
        policy="O",
        rows=2,
        cols=2,
        batch_size=1,
        topk=3,
        kv_entry_bytes=128,
        total_sram_bytes=4096,
        cache_ratio=0.5,
        compute_gradient=1.0,
        cache_gradient=1.0,
    )
    state = ResidencyState(plan)
    transfers = plan_layer_prefetch(
        state=state,
        lanes=lanes,
        layer_id=0,
        decode_step=0,
        history={},
        kv_entry_bytes=128,
        cacheline_bytes=128,
        num_hbm_ports=2,
        max_context=32,
        n_layers=1,
    )

    assert len(transfers) == 3
    assert state.resident_entries == 0
    commit_prefetch_fills(state, transfers)
    assert state.resident_entries == 3


def test_layer_prefetch_respects_each_core_capacity(tmp_path: Path) -> None:
    path = tmp_path / "unit_profile.pkl"
    _profile(path)
    profile = load_profile(path)
    lanes = (
        RequestLane(0, "request-0", profile),
        RequestLane(1, "request-1", profile),
    )
    plan = build_mapping_plan(
        policy="O",
        rows=2,
        cols=2,
        batch_size=2,
        topk=3,
        kv_entry_bytes=128,
        total_sram_bytes=512,
        cache_ratio=1.0,
        compute_gradient=1.0,
        cache_gradient=0.0,
    )
    state = ResidencyState(plan)
    retained = KVBlock("request-0", 0, 0)
    state.insert(retained, target_core=0)

    transfers = plan_layer_prefetch(
        state=state,
        lanes=lanes,
        layer_id=0,
        decode_step=0,
        history={},
        kv_entry_bytes=128,
        cacheline_bytes=128,
        num_hbm_ports=2,
        max_context=32,
        n_layers=1,
    )

    assert all(transfer.target_core != 0 for transfer in transfers)
    assert len({transfer.target_core for transfer in transfers}) == len(transfers)
    commit_prefetch_fills(state, transfers)
    assert state.owner(retained) == 0
    assert all(state.owner(item.block) == item.target_core for item in transfers)


def test_unsimulated_prefetch_uses_the_same_targets_as_timed_prefetch(
    tmp_path: Path,
) -> None:
    path = tmp_path / "unit_profile.pkl"
    _profile(path)
    profile = load_profile(path)
    lanes = (
        RequestLane(0, "request-0", profile),
        RequestLane(1, "request-1", profile),
    )
    plan = build_mapping_plan(
        policy="O",
        rows=2,
        cols=2,
        batch_size=2,
        topk=3,
        kv_entry_bytes=128,
        total_sram_bytes=512,
        cache_ratio=1.0,
        compute_gradient=1.0,
        cache_gradient=0.0,
    )
    timed = ResidencyState(plan)
    compact = ResidencyState(plan)
    for state in (timed, compact):
        state.insert(KVBlock("request-0", 0, 0), target_core=0)

    transfers = plan_layer_prefetch(
        state=timed,
        lanes=lanes,
        layer_id=0,
        decode_step=0,
        history={},
        kv_entry_bytes=128,
        cacheline_bytes=128,
        num_hbm_ports=2,
        max_context=32,
        n_layers=1,
    )
    commit_prefetch_fills(timed, transfers)
    advance_prefetch_residency(compact, lanes, 0, {})

    assert compact.owner_by_block == timed.owner_by_block
    assert [tuple(cache) for cache in compact.caches] == [
        tuple(cache) for cache in timed.caches
    ]


def test_prefetch_lru_can_select_decode_block_absent_from_prefill(tmp_path: Path) -> None:
    path = tmp_path / "unit_profile.pkl"
    _profile(path)
    profile = load_profile(path)
    lanes = (RequestLane(0, "request-0", profile),)
    plan = build_mapping_plan(
        policy="O",
        rows=2,
        cols=2,
        batch_size=1,
        topk=3,
        kv_entry_bytes=128,
        total_sram_bytes=1024,
        cache_ratio=0.5,
        compute_gradient=1.0,
        cache_gradient=1.0,
    )
    state = ResidencyState(plan)
    recent = KVBlock("request-0", 0, 5)
    transfers = plan_layer_prefetch(
        state=state,
        lanes=lanes,
        layer_id=0,
        decode_step=1,
        history={("request-0", 0): {5: 10}},
        kv_entry_bytes=128,
        cacheline_bytes=128,
        num_hbm_ports=2,
        max_context=32,
        n_layers=1,
    )

    assert recent in {transfer.block for transfer in transfers}
