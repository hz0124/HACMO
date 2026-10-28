from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "tp2_core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from build_mapping_inputs import build_mapping_inputs_from_payloads
from build_step_inputs import (
    aggregate_steps_to_windows,
    build_continuous_step_inputs,
    build_mixed_continuous_step_inputs,
    build_rolling_continuous_step_inputs,
    build_sampled_step_inputs,
)


def _literal_empirical_per_die(req_payloads: list[dict], num_dies: int, decode_len: int, window_size: int):
    p_ue_counts = {u: Counter() for u in range(num_dies)}
    h_counts = Counter()
    s_u = Counter()
    window_stats = []
    window_src = []

    for req in req_payloads:
        for tk in req.get("decode", [])[:decode_len]:
            experts = []
            for eids in tk.get("layer_to_expert_ids", {}).values():
                experts.extend(int(e) for e in eids)
            for src_die in range(num_dies):
                s_u[src_die] += 1
                for e in experts:
                    p_ue_counts[src_die][e] += 1
                    h_counts[e] += 1

    num_windows = (decode_len + window_size - 1) // window_size
    for w in range(num_windows):
        st = w * window_size
        ed = min(decode_len, (w + 1) * window_size)
        e_counter = Counter()
        u_counter = Counter()
        src_e_counter = {u: Counter() for u in range(num_dies)}
        for req in req_payloads:
            for tk in req.get("decode", [])[st:ed]:
                experts = []
                for eids in tk.get("layer_to_expert_ids", {}).values():
                    experts.extend(int(e) for e in eids)
                for src_die in range(num_dies):
                    for e in experts:
                        e_counter[e] += 1
                        src_e_counter[src_die][e] += 1
                    u_counter[src_die] += 1
        window_stats.append(
            {
                "window_id": w,
                "start_token_idx": st + 1,
                "end_token_idx": ed,
                "expert_counts": dict(e_counter),
                "tokens_per_die": dict(u_counter),
                "active": bool(u_counter),
            }
        )
        window_src.append(
            {
                "window_id": w,
                "src_expert_counts": {str(u): dict(c) for u, c in src_e_counter.items() if c},
            }
        )
    return p_ue_counts, h_counts, s_u, window_stats, window_src


def test_empirical_per_die_fast_path_matches_literal_expansion() -> None:
    payloads = [
        {
            "decode": [
                {"layer_to_expert_ids": {"0": [1, 2], "1": [2]}},
                {"layer_to_expert_ids": {"0": [3]}},
            ]
        },
        {
            "decode": [
                {"layer_to_expert_ids": {"0": [2]}},
                {"layer_to_expert_ids": {"0": [1, 3]}},
            ]
        },
    ]

    fast = build_mapping_inputs_from_payloads(
        req_payloads=payloads,
        source_mode="empirical_per_die_sampling",
        num_dies=3,
        effective_decode_len=2,
        window_size=1,
    )
    literal = _literal_empirical_per_die(payloads, num_dies=3, decode_len=2, window_size=1)

    assert fast == literal


def test_sampled_step_inputs_are_seeded_independent_per_die_batches() -> None:
    payloads = [
        {"request_id": "a", "decode": [{"layer_to_expert_ids": {"0": [0]}}]},
        {"request_id": "b", "decode": [{"layer_to_expert_ids": {"0": [1]}}]},
        {"request_id": "c", "decode": [{"layer_to_expert_ids": {"0": [2]}}]},
    ]

    first = build_sampled_step_inputs(
        req_payloads=payloads,
        num_dies=3,
        batch_per_die=4,
        decode_len=1,
        window_size=8,
        seed=17,
    )
    second = build_sampled_step_inputs(
        req_payloads=payloads,
        num_dies=3,
        batch_per_die=4,
        decode_len=1,
        window_size=8,
        seed=17,
    )

    assert first == second
    assert first["metadata"]["sampling"] == "independent_with_replacement_per_die"
    assert first["metadata"]["batch_scale"] == 1.0
    assert first["steps"][0]["tokens_per_die"] == {"0": 4, "1": 4, "2": 4}
    assert len(first["sampled_request_indices"]) == 3
    assert all(len(indices) == 4 for indices in first["sampled_request_indices"].values())


def test_step_inputs_aggregate_to_windows_without_losing_source_counts() -> None:
    sampled = build_sampled_step_inputs(
        req_payloads=[
            {
                "decode": [
                    {"layer_to_expert_ids": {"0": [1]}},
                    {"layer_to_expert_ids": {"0": [2]}},
                ]
            }
        ],
        num_dies=2,
        batch_per_die=1,
        decode_len=2,
        window_size=8,
        seed=0,
    )

    window_stats, window_src = aggregate_steps_to_windows(sampled["steps"], window_size=8)

    assert window_stats == [
        {
            "window_id": 0,
            "start_token_idx": 1,
            "end_token_idx": 2,
            "expert_counts": {"1": 2, "2": 2},
            "tokens_per_die": {"0": 2, "1": 2},
            "active": True,
        }
    ]
    assert window_src[0]["src_expert_counts"] == {
        "0": {"1": 1, "2": 1},
        "1": {"1": 1, "2": 1},
    }


def _full_request(expert: int, decode_len: int) -> dict:
    return {
        "decode": [
            {"layer_to_expert_ids": {"0": [expert]}} for _ in range(decode_len)
        ]
    }


def test_continuous_step_inputs_preserve_concurrency_and_replace_each_slot_once() -> None:
    first = build_continuous_step_inputs(
        initial_req_payloads=[_full_request(0, 4), _full_request(9, 3)],
        replacement_req_payloads=[_full_request(1, 4)],
        num_dies=2,
        batch_per_die=2,
        decode_len=4,
        window_size=2,
        seed=17,
        initial_domain="law",
        replacement_domain="accounting",
    )
    second = build_continuous_step_inputs(
        initial_req_payloads=[_full_request(0, 4), _full_request(9, 3)],
        replacement_req_payloads=[_full_request(1, 4)],
        num_dies=2,
        batch_per_die=2,
        decode_len=4,
        window_size=2,
        seed=17,
        initial_domain="law",
        replacement_domain="accounting",
    )

    assert first == second
    assert first["metadata"]["initial_request_pool_size"] == 2
    assert first["metadata"]["initial_complete_request_pool_size"] == 1
    assert sorted(first["slot_initial_ages"]) == [0, 1, 2, 3]
    assert [event["activation_step_id"] for event in first["replacement_events"]] == [
        1,
        2,
        3,
        4,
    ]
    assert all(
        sum(step["tokens_per_die"].values()) == 4 for step in first["steps"]
    )
    assert [step["active_request_domains"] for step in first["steps"]] == [
        {"law": 4},
        {"accounting": 1, "law": 3},
        {"accounting": 2, "law": 2},
        {"accounting": 3, "law": 1},
    ]


def test_continuous_primary_and_control_share_initial_state_and_schedule() -> None:
    common = {
        "initial_req_payloads": [_full_request(0, 4), _full_request(2, 4)],
        "num_dies": 2,
        "batch_per_die": 3,
        "decode_len": 4,
        "window_size": 2,
        "seed": 29,
        "initial_domain": "law",
    }
    primary = build_continuous_step_inputs(
        **common,
        replacement_req_payloads=[_full_request(1, 4)],
        replacement_domain="accounting",
    )
    control = build_continuous_step_inputs(
        **common,
        replacement_req_payloads=[_full_request(3, 4), _full_request(4, 4)],
        replacement_domain="law_control",
    )

    assert primary["slot_initial_request_indices"] == control[
        "slot_initial_request_indices"
    ]
    assert primary["slot_initial_ages"] == control["slot_initial_ages"]
    assert primary["steps"][0] == control["steps"][0]
    assert [
        (event["slot_id"], event["activation_step_id"])
        for event in primary["replacement_events"]
    ] == [
        (event["slot_id"], event["activation_step_id"])
        for event in control["replacement_events"]
    ]


def test_mixed_continuous_slots_sample_one_ordered_uniform_replacement_pool() -> None:
    kwargs = {
        "initial_req_payloads": [_full_request(0, 4)],
        "replacement_req_payloads_by_domain": [
            ("mathematics", [_full_request(1, 4), _full_request(2, 4)]),
            ("human_aging", [_full_request(3, 4)]),
        ],
        "num_dies": 4,
        "batch_per_die": 16,
        "decode_len": 4,
        "window_size": 2,
        "seed": 17,
        "initial_domain": "professional_law",
    }

    first = build_mixed_continuous_step_inputs(**kwargs)
    second = build_mixed_continuous_step_inputs(**kwargs)

    assert first == second
    assert first["metadata"]["sampling"] == (
        "diagnostic_uniform_mixed_continuous_request_slots"
    )
    assert first["metadata"]["replacement_domain"] == "mixed_replacement_pool"
    assert first["metadata"]["replacement_domains"] == [
        "mathematics",
        "human_aging",
    ]
    assert first["metadata"]["replacement_request_domain_by_index"] == {
        "0": "mathematics",
        "1": "mathematics",
        "2": "human_aging",
    }
    assert all(
        sum(step["tokens_per_die"].values()) == 64 for step in first["steps"]
    )
    observed_domains = {
        domain
        for step in first["steps"]
        for domain in step["active_request_domains"]
    }
    assert observed_domains == {"professional_law", "mathematics", "human_aging"}
    assert {
        event["replacement_domain"] for event in first["replacement_events"]
    } == {"mixed_replacement_pool"}


def test_rolling_continuous_slots_keep_serving_after_first_replacement() -> None:
    sampled = build_rolling_continuous_step_inputs(
        initial_req_payloads=[_full_request(0, 4)],
        replacement_req_payloads=[_full_request(1, 4), _full_request(2, 4)],
        num_dies=2,
        batch_per_die=2,
        request_decode_len=4,
        serving_steps=8,
        window_size=2,
        seed=17,
        initial_domain="law",
        replacement_domain="accounting",
    )

    assert sampled["metadata"]["sampling"] == "rolling_continuous_request_slots"
    assert sampled["metadata"]["request_decode_len"] == 4
    assert sampled["metadata"]["serving_steps"] == 8
    assert all(
        sum(step["tokens_per_die"].values()) == 4 for step in sampled["steps"]
    )
    assert [step["active_request_domains"] for step in sampled["steps"][:5]] == [
        {"law": 4},
        {"accounting": 1, "law": 3},
        {"accounting": 2, "law": 2},
        {"accounting": 3, "law": 1},
        {"accounting": 4},
    ]
    assert any(event["generation"] > 1 for event in sampled["admission_events"])

    window_stats, _ = aggregate_steps_to_windows(sampled["steps"], window_size=2)
    assert [sum(row["tokens_per_die"].values()) for row in window_stats] == [
        8,
        8,
        8,
        8,
    ]
