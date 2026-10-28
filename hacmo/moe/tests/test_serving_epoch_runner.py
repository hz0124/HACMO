from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
CORE = ROOT / "tp2_core"
EXPERIMENTS = ROOT / "experiments"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from run_serving_epoch_tp2 import (
    base_placement_source_mode,
    cosine,
    distribution_shift_metrics,
    dynamic_windows_path_for_batch,
    filter_usable_trace_files,
    aggregate_seed_rows,
    normalize_rows,
    ours_static_reference_scope,
    popularity_aligned_cosine,
    popularity_order,
    resolve_moentwine_attention_tp,
    runtime_sampling_contract,
    sampled_window_statistics,
    sampling_contract,
    select_moentwine_deployment,
    select_request_segments,
    source_mode_semantics,
    transfer_mapping_by_popularity,
    write_moentwine_primary_mapping,
)
from utils_topology import load_topology
from run_tp2_dynamic_headroom_diagnostic import CASE_SPECS


def test_camera_ready_high_drift_case_has_ordered_mixed_pool() -> None:
    case = CASE_SPECS["high_drift_uniform_pool_law_math_elem_human_aging"]

    assert case.initial_label == "professional_law[256:416]"
    assert case.replacement_slices == (
        ("high_school_mathematics", 0, 90),
        ("elementary_mathematics", 0, 90),
        ("human_aging", 0, 90),
    )
    assert case.replacement_label == (
        "high_school_mathematics[0:90]+"
        "elementary_mathematics[0:90]+human_aging[0:90]"
    )


def test_mapping_only_cli_stops_before_runtime_route_plans(tmp_path: Path) -> None:
    law = tmp_path / "professional_law"
    misc = tmp_path / "miscellaneous"
    law.mkdir()
    misc.mkdir()
    raw = [
        {},
        {"23": [[0]]},
        {"23": [[1]]},
    ]
    (law / "0.json").write_text(json.dumps(raw), encoding="utf-8")
    (law / "1.json").write_text(json.dumps(raw), encoding="utf-8")
    (misc / "0.json").write_text(json.dumps(raw), encoding="utf-8")
    output = tmp_path / "output"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "run_serving_epoch_tp2.py"),
            "--topologies",
            str(ROOT / "configs/network/hierarchical_4w_8x8_mesh_rail_900.yaml"),
            "--methods",
            "ours_static",
            "--library-hf-trace-roots",
            str(law),
            str(misc),
            "--runtime-hf-trace-root",
            str(law),
            "--epoch-t-start",
            "0",
            "--epoch-t-requests",
            "1",
            "--epoch-t1-requests",
            "1",
            "--library-max-requests",
            "1",
            "--decode-len",
            "2",
            "--window-size",
            "1",
            "--num-experts",
            "4",
            "--topk",
            "1",
            "--static-max-replicas",
            "1",
            "--mapping-only",
            "--output-root",
            str(output),
        ],
        cwd=ROOT.parents[1],
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    topology = output / "4w_8x8_rail_900"
    assert (topology / "mappings" / "ours_static.json").is_file()
    assert not (topology / "route_plans").exists()
    metadata = json.loads((topology / "mapping_only_metadata.json").read_text())
    assert metadata["mapping_only"] is True


def test_select_request_segments_are_adjacent_and_disjoint() -> None:
    files = [Path(f"{i}.json") for i in range(10)]

    epoch_t, epoch_t1 = select_request_segments(
        files,
        epoch_t_start=2,
        epoch_t_requests=3,
        epoch_t1_requests=4,
    )

    assert epoch_t == [Path("2.json"), Path("3.json"), Path("4.json")]
    assert epoch_t1 == [Path("5.json"), Path("6.json"), Path("7.json"), Path("8.json")]
    assert set(epoch_t).isdisjoint(epoch_t1)


def test_select_request_segments_uses_remainder_for_eval() -> None:
    files = [Path(f"{i}.json") for i in range(6)]

    epoch_t, epoch_t1 = select_request_segments(
        files,
        epoch_t_start=0,
        epoch_t_requests=2,
        epoch_t1_requests=None,
    )

    assert epoch_t == [Path("0.json"), Path("1.json")]
    assert epoch_t1 == [Path("2.json"), Path("3.json"), Path("4.json"), Path("5.json")]


def test_select_request_segments_accepts_independent_epoch_t1_domain() -> None:
    epoch_t_pool = [Path("livecodebench") / f"{i}.json" for i in range(6)]
    epoch_t1_pool = [Path("mmlu") / f"{i}.json" for i in range(8)]

    epoch_t, epoch_t1 = select_request_segments(
        epoch_t_pool,
        epoch_t_start=1,
        epoch_t_requests=3,
        epoch_t1_requests=4,
        epoch_t1_files=epoch_t1_pool,
        epoch_t1_start=2,
    )

    assert epoch_t == epoch_t_pool[1:4]
    assert epoch_t1 == epoch_t1_pool[2:6]


def test_trace_selection_filters_files_without_selected_layer_decode_data(
    tmp_path: Path,
) -> None:
    valid_0 = tmp_path / "0.json"
    invalid = tmp_path / "1.json"
    valid_2 = tmp_path / "2.json"
    valid_0.write_text('[{}, {"23": [[1, 2]]}]\n', encoding="utf-8")
    invalid.write_text('[{}, {"22": [[3, 4]]}]\n', encoding="utf-8")
    valid_2.write_text('[{}, {"23": [[5, 6]]}]\n', encoding="utf-8")

    usable, skipped = filter_usable_trace_files(
        [valid_0, invalid, valid_2], layers={"23"}
    )

    assert usable == [valid_0, valid_2]
    assert skipped == [invalid]


def test_cosine_prefers_matching_epoch_telemetry() -> None:
    telemetry = [1.0, 0.0, 1.0]
    similar = [0.5, 0.0, 0.5]
    different = [0.0, 1.0, 0.0]

    assert cosine(telemetry, similar) > cosine(telemetry, different)


def test_popularity_aligned_cosine_ignores_expert_id_permutation() -> None:
    library = [0.05, 0.30, 0.50, 0.15]
    telemetry = [0.30, 0.50, 0.15, 0.05]

    assert popularity_aligned_cosine(library, telemetry) == pytest.approx(1.0)


def test_moentwine_mapping_defaults_to_tp4_contract(tmp_path: Path) -> None:
    obj = write_moentwine_primary_mapping(
        output_json=tmp_path / "mapping.json",
        num_experts=256,
        num_dies=256,
        query_heads=128,
        attention_tp_degree=4,
        attention_tp_shape=[2, 2],
    )

    assert obj["metadata"]["attention_tp_degree"] == 4
    assert obj["metadata"]["attention_tp_shape"] == [2, 2]
    assert obj["metadata"]["attention_tp_selection_policy"] == "fixed"
    assert obj["metadata"]["attention_tp_selection_uses_runtime_trace"] is False


def test_moentwine_mapping_accepts_predeclared_tp16(tmp_path: Path) -> None:
    obj = write_moentwine_primary_mapping(
        output_json=tmp_path / "mapping.json",
        num_experts=256,
        num_dies=256,
        query_heads=128,
        attention_tp_degree=16,
        attention_tp_shape=[4, 4],
        attention_tp_selection_policy="topology_predeclared",
        attention_tp_selection_reason="16w_4x4_mesh_uses_full_wafer_tp16",
    )

    assert obj["metadata"]["attention_tp_degree"] == 16
    assert obj["metadata"]["attention_tp_shape"] == [4, 4]
    assert obj["metadata"]["attention_tp_selection_policy"] == "topology_predeclared"
    assert (
        obj["metadata"]["attention_tp_selection_reason"]
        == "16w_4x4_mesh_uses_full_wafer_tp16"
    )


def test_moentwine_topology_policy_keeps_4w_default_and_sets_16w_tp16() -> None:
    args = argparse.Namespace(
        moentwine_attention_tp_selection="topology_predeclared",
        moentwine_attention_tp_degree=4,
        moentwine_attention_tp_shape=[2, 2],
        moentwine_query_heads=128,
    )

    tp4 = resolve_moentwine_attention_tp(
        args=args,
        topo=load_topology(
            ROOT / "configs/network/hierarchical_4w_8x8_mesh_rail_900.yaml"
        ),
    )
    tp16 = resolve_moentwine_attention_tp(
        args=args,
        topo=load_topology(
            ROOT / "configs/network/hierarchical_16w_4x4_mesh_rail_500.yaml"
        ),
    )

    assert tp4["attention_tp_degree"] == 4
    assert tp4["attention_tp_shape"] == [2, 2]
    assert tp4["selection_reason"] == "topology_predeclared_fallback_to_cli_or_default"
    assert tp16["attention_tp_degree"] == 16
    assert tp16["attention_tp_shape"] == [4, 4]
    assert tp16["selection_reason"] == "16w_4x4_mesh_uses_full_wafer_tp16"
    assert tp16["uses_runtime_trace_for_selection"] is False


def test_moentwine_deployment_policy_defaults_to_full_her() -> None:
    selected = select_moentwine_deployment(policy="full_her")

    assert selected["selected_variant"] == "full_her"
    assert selected["selection_reason"] == "policy_forces_full_her"
    assert selected["full_her_total_us"] is None
    assert selected["no_her_total_us"] is None


def test_moentwine_deployment_policy_can_force_no_her() -> None:
    selected = select_moentwine_deployment(
        policy="no_her",
        full_her_total_us=10.0,
        no_her_total_us=20.0,
    )

    assert selected["selected_variant"] == "no_her"
    assert selected["selection_reason"] == "policy_forces_no_her"


def test_moentwine_cost_gated_policy_uses_analytical_totals() -> None:
    full = select_moentwine_deployment(
        policy="cost_gated_her",
        full_her_total_us=19.0,
        no_her_total_us=20.0,
    )
    no_her = select_moentwine_deployment(
        policy="cost_gated_her",
        full_her_total_us=21.0,
        no_her_total_us=20.0,
    )

    assert full["selected_variant"] == "full_her"
    assert no_her["selected_variant"] == "no_her"


def test_moentwine_cost_gated_policy_requires_candidate_totals() -> None:
    with pytest.raises(ValueError, match="requires both candidate totals"):
        select_moentwine_deployment(policy="cost_gated_her")


def test_distribution_shift_separates_shape_from_expert_identity() -> None:
    epoch_t = [0.05, 0.30, 0.50, 0.15]
    epoch_t1 = [0.30, 0.50, 0.15, 0.05]

    metrics = distribution_shift_metrics(epoch_t, epoch_t1, topks=(2,))

    assert metrics["expert_id_aligned_cosine"] < 0.8
    assert metrics["popularity_rank_aligned_cosine"] == pytest.approx(1.0)
    assert metrics["epoch_t_hottest_expert"] == 2
    assert metrics["epoch_t1_hottest_expert"] == 1
    assert metrics["top2_overlap"] == 1


def test_transfer_mapping_by_popularity_uses_runtime_expert_ranks() -> None:
    library_mapping = {0: [1], 1: [2], 2: [0], 3: [3]}
    library_h = [0.05, 0.30, 0.50, 0.15]
    telemetry_h = [0.30, 0.55, 0.10, 0.05]

    transferred, _ = transfer_mapping_by_popularity(
        library_mapping,
        library_h,
        telemetry_h,
    )

    assert transferred == {0: [2], 1: [0], 2: [3], 3: [1]}


def test_transfer_mapping_by_popularity_preserves_all_static_replicas() -> None:
    transferred, _ = transfer_mapping_by_popularity(
        {0: [4], 1: [8, 9, 10]},
        [0.1, 0.9],
        [0.8, 0.2],
    )

    assert transferred[0] == [8, 9, 10]


def test_popularity_order_breaks_frequency_ties_by_expert_id() -> None:
    assert popularity_order([0.2, 0.7, 0.7, 0.1]) == [1, 2, 0, 3]


def test_dynamic_windows_path_is_batch_specific() -> None:
    assert dynamic_windows_path_for_batch(Path("topo"), 64) == Path(
        "topo/runtime_windows/ours_dynamic_windows_bpd64.json"
    )


def test_trace_source_mode_legacy_alias_maps_to_sampling_semantics() -> None:
    assert (
        source_mode_semantics("empirical_per_die_sampling")
        == "empirical_per_die_sampling"
    )
    assert source_mode_semantics("replicate_all_dies") == "empirical_per_die_sampling"
    assert source_mode_semantics("spread_requests") == "spread_requests_to_target_dies"


def test_sampling_contract_rejects_unique_request_interpretation() -> None:
    contract = sampling_contract("empirical_per_die_sampling")

    assert "empirical request-distribution sample" in contract
    assert "does not require one unique raw trace file" in contract


def test_continuous_runtime_contract_records_fixed_slot_replacement() -> None:
    contract = runtime_sampling_contract(
        "continuous_request_slots", "empirical_per_die_sampling"
    )

    assert "fixed serving slots" in contract
    assert "global active batch is constant" in contract
    assert "all methods consume the same emitted steps" in contract


def test_rolling_runtime_contract_records_long_running_admissions() -> None:
    contract = runtime_sampling_contract(
        "rolling_continuous_request_slots", "empirical_per_die_sampling"
    )

    assert "roll over again within the serving horizon" in contract
    assert "global active batch is constant" in contract
    assert "all methods consume the same emitted steps" in contract


def test_sampled_window_statistics_separates_identity_and_rank_shift() -> None:
    statistics = sampled_window_statistics(
        [
            {"expert_counts": {"0": 5, "1": 3, "2": 1}, "tokens_per_die": {"0": 2}},
            {"expert_counts": {"0": 1, "1": 5, "2": 3}, "tokens_per_die": {"0": 2}},
        ],
        num_experts=3,
    )

    assert statistics["first_to_last"]["expert_id_aligned_cosine"] < 0.7
    assert statistics["first_to_last"]["popularity_rank_aligned_cosine"] == pytest.approx(1.0)
    assert statistics["tokens_per_window"] == [2, 2]


def test_epoch_t_base_entry_mode_is_explicit_and_causal() -> None:
    assert base_placement_source_mode(False) == "static_mapping_library_only"
    assert base_placement_source_mode(True) == "epoch_t_telemetry_plus_static_library"

    scope = ours_static_reference_scope(True)

    assert "epoch_t-derived Epoch Base Placement" in scope
    assert "epoch_t telemetry only" in scope


def test_method_subset_keeps_absolute_rows_without_vllm_reference() -> None:
    samples = [
        {
            "topology": "4w",
            "method": "ours_static",
            "seed": seed,
            "batch_per_die": 16,
            "global_batch": 4096,
            "total_us": total,
            "dispatch_us": 20.0,
            "compute_us": 10.0,
            "combine_us": 22.0,
            "serial_upper_us": 52.0,
            "cross_wafer_ratio": 0.2,
            "avg_hops": 1.5,
            "metric_source": "analytical",
        }
        for seed, total in [(17, 50.0), (29, 54.0)]
    ]

    normalized = normalize_rows(samples)
    aggregated = aggregate_seed_rows(normalized)

    assert len(normalized) == 2
    assert all(row["normalized_to_vllm"] is None for row in normalized)
    assert len(aggregated) == 1
    assert aggregated[0]["total_us"] == pytest.approx(52.0)
    assert aggregated[0]["normalized_to_vllm"] is None
