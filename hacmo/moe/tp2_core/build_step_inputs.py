#!/usr/bin/env python3
from __future__ import annotations

import random
from collections import Counter, defaultdict


def _token_experts(token: dict) -> list[int]:
    experts: list[int] = []
    for expert_ids in token.get("layer_to_expert_ids", {}).values():
        experts.extend(int(expert) for expert in expert_ids)
    return experts


def build_sampled_step_inputs(
    *,
    req_payloads: list[dict],
    num_dies: int,
    batch_per_die: int,
    decode_len: int,
    window_size: int,
    seed: int,
) -> dict:
    if not req_payloads:
        raise ValueError("epoch t+1 request pool is empty")
    if num_dies <= 0 or batch_per_die <= 0 or decode_len <= 0 or window_size <= 0:
        raise ValueError("num_dies, batch_per_die, decode_len, and window_size must be positive")

    rng = random.Random(seed)
    sampled_request_indices = {
        str(src): [rng.randrange(len(req_payloads)) for _ in range(batch_per_die)]
        for src in range(num_dies)
    }

    steps: list[dict] = []
    for step_id in range(decode_len):
        src_expert_counts: dict[str, dict[str, int]] = {}
        tokens_per_die: dict[str, int] = {}
        expert_counts: Counter[int] = Counter()
        for src_str, request_indices in sampled_request_indices.items():
            src_counts: Counter[int] = Counter()
            tokens = 0
            for request_idx in request_indices:
                decode = req_payloads[request_idx].get("decode", [])
                if step_id >= len(decode):
                    continue
                tokens += 1
                src_counts.update(_token_experts(decode[step_id]))
            if tokens:
                tokens_per_die[src_str] = tokens
            if src_counts:
                src_expert_counts[src_str] = {
                    str(expert): int(count) for expert, count in sorted(src_counts.items())
                }
                expert_counts.update(src_counts)
        steps.append(
            {
                "step_id": step_id,
                "token_idx": step_id + 1,
                "window_id": step_id // window_size,
                "src_expert_counts": src_expert_counts,
                "tokens_per_die": tokens_per_die,
                "expert_counts": {
                    str(expert): int(count) for expert, count in sorted(expert_counts.items())
                },
                "active": bool(tokens_per_die),
            }
        )

    return {
        "metadata": {
            "sampling": "independent_with_replacement_per_die",
            "seed": seed,
            "request_pool_size": len(req_payloads),
            "num_dies": num_dies,
            "batch_per_die": batch_per_die,
            "global_batch": num_dies * batch_per_die,
            "decode_len": decode_len,
            "window_size": window_size,
            "batch_scale": 1.0,
        },
        "sampled_request_indices": sampled_request_indices,
        "steps": steps,
    }


def _complete_request_pool(
    req_payloads: list[dict], *, decode_len: int, role: str
) -> list[tuple[int, dict]]:
    complete = [
        (request_index, request)
        for request_index, request in enumerate(req_payloads)
        if len(request.get("decode", [])) >= decode_len
    ]
    if not complete:
        raise ValueError(
            f"{role} request pool has no request with at least {decode_len} decode steps"
        )
    return complete


def build_continuous_step_inputs(
    *,
    initial_req_payloads: list[dict],
    replacement_req_payloads: list[dict],
    num_dies: int,
    batch_per_die: int,
    decode_len: int,
    window_size: int,
    seed: int,
    initial_domain: str = "epoch_t",
    replacement_domain: str = "epoch_t1",
) -> dict:
    """Build a fixed-concurrency serving epoch with staggered request ages.

    Every slot starts from an epoch-t request at a seeded decode age. When that
    request finishes, an epoch-t+1 request occupies the same slot on the next
    step. Separate seeded random streams keep the initial cohort and completion
    schedule identical for paired primary/control replacement pools.
    """
    if num_dies <= 0 or batch_per_die <= 0 or decode_len <= 0 or window_size <= 0:
        raise ValueError("num_dies, batch_per_die, decode_len, and window_size must be positive")

    initial_pool = _complete_request_pool(
        initial_req_payloads, decode_len=decode_len, role="initial"
    )
    replacement_pool = _complete_request_pool(
        replacement_req_payloads, decode_len=decode_len, role="replacement"
    )
    global_batch = num_dies * batch_per_die
    initial_rng = random.Random(f"{seed}:initial_requests")
    replacement_rng = random.Random(f"{seed}:replacement_requests")
    age_rng = random.Random(f"{seed}:initial_ages")

    initial_positions = [
        initial_rng.randrange(len(initial_pool)) for _ in range(global_batch)
    ]
    replacement_positions = [
        replacement_rng.randrange(len(replacement_pool)) for _ in range(global_batch)
    ]
    initial_ages = [slot_id % decode_len for slot_id in range(global_batch)]
    age_rng.shuffle(initial_ages)
    slots = [
        {
            "initial_position": initial_positions[slot_id],
            "replacement_position": replacement_positions[slot_id],
            "age": initial_ages[slot_id],
            "replaced": False,
        }
        for slot_id in range(global_batch)
    ]

    replacement_events: list[dict] = []
    steps: list[dict] = []
    for step_id in range(decode_len):
        src_expert_counts: dict[str, dict[str, int]] = {}
        tokens_per_die: dict[str, int] = {}
        expert_counts: Counter[int] = Counter()
        active_domains: Counter[str] = Counter()
        active_ages: Counter[int] = Counter()

        for src in range(num_dies):
            src_counts: Counter[int] = Counter()
            for local_slot in range(batch_per_die):
                slot_id = src * batch_per_die + local_slot
                slot = slots[slot_id]
                pool = replacement_pool if slot["replaced"] else initial_pool
                position_key = (
                    "replacement_position" if slot["replaced"] else "initial_position"
                )
                request = pool[int(slot[position_key])][1]
                age = int(slot["age"])
                src_counts.update(_token_experts(request["decode"][age]))
                active_domains[
                    replacement_domain if slot["replaced"] else initial_domain
                ] += 1
                active_ages[age] += 1

                if age == decode_len - 1:
                    if slot["replaced"]:
                        raise AssertionError(
                            "a replacement request completed within one serving epoch"
                        )
                    replacement_request_index = replacement_pool[
                        int(slot["replacement_position"])
                    ][0]
                    replacement_events.append(
                        {
                            "slot_id": slot_id,
                            "src_die": src,
                            "after_step_id": step_id,
                            "activation_step_id": step_id + 1,
                            "activation_window_id": (step_id + 1) // window_size,
                            "initial_age": initial_ages[slot_id],
                            "replacement_domain": replacement_domain,
                            "replacement_request_index": replacement_request_index,
                        }
                    )
                    slot["age"] = 0
                    slot["replaced"] = True
                else:
                    slot["age"] = age + 1

            tokens_per_die[str(src)] = batch_per_die
            src_expert_counts[str(src)] = {
                str(expert): int(count) for expert, count in sorted(src_counts.items())
            }
            expert_counts.update(src_counts)

        steps.append(
            {
                "step_id": step_id,
                "token_idx": step_id + 1,
                "window_id": step_id // window_size,
                "src_expert_counts": src_expert_counts,
                "tokens_per_die": tokens_per_die,
                "expert_counts": {
                    str(expert): int(count)
                    for expert, count in sorted(expert_counts.items())
                },
                "active_request_domains": dict(sorted(active_domains.items())),
                "active_request_age_histogram": {
                    str(age): int(count) for age, count in sorted(active_ages.items())
                },
                "active": True,
            }
        )

    if len(replacement_events) != global_batch:
        raise AssertionError("every initial request slot must be replaced exactly once")
    if any(sum(step["tokens_per_die"].values()) != global_batch for step in steps):
        raise AssertionError("every serving step must preserve the global active batch")

    return {
        "metadata": {
            "sampling": "continuous_staggered_request_slots",
            "seed": seed,
            "initial_domain": initial_domain,
            "replacement_domain": replacement_domain,
            "initial_request_pool_size": len(initial_req_payloads),
            "initial_complete_request_pool_size": len(initial_pool),
            "replacement_request_pool_size": len(replacement_req_payloads),
            "replacement_complete_request_pool_size": len(replacement_pool),
            "num_dies": num_dies,
            "batch_per_die": batch_per_die,
            "global_batch": global_batch,
            "decode_len": decode_len,
            "window_size": window_size,
            "initial_age_assignment": (
                "balanced_over_decode_ages_then_seeded_shuffle"
            ),
            "replacement_rule": (
                "replace_after_initial_request_final_decode_step; "
                "replacement activates on the next step"
            ),
            "slots_replaced_exactly_once": global_batch,
            "random_streams": [
                "initial_requests",
                "replacement_requests",
                "initial_ages",
            ],
            "batch_scale": 1.0,
        },
        "slot_initial_request_indices": [
            initial_pool[position][0] for position in initial_positions
        ],
        "slot_replacement_request_indices": [
            replacement_pool[position][0] for position in replacement_positions
        ],
        "slot_initial_ages": initial_ages,
        "replacement_events": replacement_events,
        "steps": steps,
    }


def build_mixed_continuous_step_inputs(
    *,
    initial_req_payloads: list[dict],
    replacement_req_payloads_by_domain: list[tuple[str, list[dict]]],
    num_dies: int,
    batch_per_die: int,
    decode_len: int,
    window_size: int,
    seed: int,
    initial_domain: str = "epoch_t",
) -> dict:
    """Build fixed serving slots with one ordered, uniformly sampled mixed pool.

    Requests from every replacement domain are concatenated in the declared
    order. Sampling is uniform over the resulting complete-request pool, while
    per-step domain accounting retains the selected request's original domain.
    """
    domains = [str(domain) for domain, _ in replacement_req_payloads_by_domain]
    if not domains:
        raise ValueError("mixed replacement pool must contain at least one domain")
    if len(set(domains)) != len(domains):
        raise ValueError("mixed replacement domain names must be unique")

    combined_requests: list[dict] = []
    domain_by_index: dict[int, str] = {}
    for domain, requests in replacement_req_payloads_by_domain:
        for request in requests:
            domain_by_index[len(combined_requests)] = str(domain)
            combined_requests.append(request)

    sampled = build_continuous_step_inputs(
        initial_req_payloads=initial_req_payloads,
        replacement_req_payloads=combined_requests,
        num_dies=num_dies,
        batch_per_die=batch_per_die,
        decode_len=decode_len,
        window_size=window_size,
        seed=seed,
        initial_domain=initial_domain,
        replacement_domain="mixed_replacement_pool",
    )

    replacement_domain_for_slot = [
        domain_by_index[int(request_index)]
        for request_index in sampled["slot_replacement_request_indices"]
    ]
    for step_id, step in enumerate(sampled["steps"]):
        active_domains: Counter[str] = Counter()
        for slot_id, initial_age in enumerate(sampled["slot_initial_ages"]):
            if int(initial_age) + step_id < decode_len:
                active_domains[initial_domain] += 1
            else:
                active_domains[replacement_domain_for_slot[slot_id]] += 1
        step["active_request_domains"] = dict(sorted(active_domains.items()))

    sampled["metadata"].update(
        {
            "sampling": "diagnostic_uniform_mixed_continuous_request_slots",
            "replacement_rule": (
                "replace_after_initial_request_final_decode_step; replacement "
                "request is sampled uniformly from the combined mixed-domain pool"
            ),
            "replacement_domains": domains,
            "replacement_request_domain_by_index": {
                str(index): domain_by_index[index] for index in sorted(domain_by_index)
            },
        }
    )
    return sampled


def build_rolling_continuous_step_inputs(
    *,
    initial_req_payloads: list[dict],
    replacement_req_payloads: list[dict],
    num_dies: int,
    batch_per_die: int,
    request_decode_len: int,
    serving_steps: int,
    window_size: int,
    seed: int,
    initial_domain: str = "epoch_t",
    replacement_domain: str = "epoch_t1",
) -> dict:
    """Build a long-running fixed-concurrency serving stream.

    Slots start from epoch-t requests at staggered decode ages. Every completed
    request is replaced on the next step by a sampled replacement-domain request,
    and replacement requests may complete and roll over again within the same
    serving horizon.
    """
    if (
        num_dies <= 0
        or batch_per_die <= 0
        or request_decode_len <= 0
        or serving_steps <= 0
        or window_size <= 0
    ):
        raise ValueError(
            "num_dies, batch_per_die, request_decode_len, serving_steps, "
            "and window_size must be positive"
        )

    initial_pool = _complete_request_pool(
        initial_req_payloads, decode_len=request_decode_len, role="initial"
    )
    replacement_pool = _complete_request_pool(
        replacement_req_payloads, decode_len=request_decode_len, role="replacement"
    )
    global_batch = num_dies * batch_per_die
    initial_rng = random.Random(f"{seed}:initial_requests")
    replacement_rng = random.Random(f"{seed}:rolling_replacement_requests")
    age_rng = random.Random(f"{seed}:initial_ages")

    initial_positions = [
        initial_rng.randrange(len(initial_pool)) for _ in range(global_batch)
    ]
    initial_ages = [
        slot_id % request_decode_len for slot_id in range(global_batch)
    ]
    age_rng.shuffle(initial_ages)
    slots = [
        {
            "pool": "initial",
            "position": initial_positions[slot_id],
            "age": initial_ages[slot_id],
            "generation": 0,
        }
        for slot_id in range(global_batch)
    ]

    admission_events: list[dict] = []
    steps: list[dict] = []
    for step_id in range(serving_steps):
        src_expert_counts: dict[str, dict[str, int]] = {}
        tokens_per_die: dict[str, int] = {}
        expert_counts: Counter[int] = Counter()
        active_domains: Counter[str] = Counter()
        active_ages: Counter[int] = Counter()

        for src in range(num_dies):
            src_counts: Counter[int] = Counter()
            for local_slot in range(batch_per_die):
                slot_id = src * batch_per_die + local_slot
                slot = slots[slot_id]
                pool_name = str(slot["pool"])
                pool = initial_pool if pool_name == "initial" else replacement_pool
                request_index, request = pool[int(slot["position"])]
                age = int(slot["age"])
                src_counts.update(_token_experts(request["decode"][age]))
                active_domains[
                    initial_domain if pool_name == "initial" else replacement_domain
                ] += 1
                active_ages[age] += 1

                if age == request_decode_len - 1:
                    next_position = replacement_rng.randrange(len(replacement_pool))
                    next_request_index = replacement_pool[next_position][0]
                    generation = int(slot["generation"]) + 1
                    admission_events.append(
                        {
                            "slot_id": slot_id,
                            "src_die": src,
                            "after_step_id": step_id,
                            "activation_step_id": step_id + 1,
                            "activation_window_id": (step_id + 1) // window_size,
                            "completed_domain": (
                                initial_domain
                                if pool_name == "initial"
                                else replacement_domain
                            ),
                            "completed_request_index": request_index,
                            "next_domain": replacement_domain,
                            "next_request_index": next_request_index,
                            "generation": generation,
                        }
                    )
                    slot["pool"] = "replacement"
                    slot["position"] = next_position
                    slot["age"] = 0
                    slot["generation"] = generation
                else:
                    slot["age"] = age + 1

            tokens_per_die[str(src)] = batch_per_die
            src_expert_counts[str(src)] = {
                str(expert): int(count) for expert, count in sorted(src_counts.items())
            }
            expert_counts.update(src_counts)

        steps.append(
            {
                "step_id": step_id,
                "token_idx": step_id + 1,
                "window_id": step_id // window_size,
                "src_expert_counts": src_expert_counts,
                "tokens_per_die": tokens_per_die,
                "expert_counts": {
                    str(expert): int(count)
                    for expert, count in sorted(expert_counts.items())
                },
                "active_request_domains": dict(sorted(active_domains.items())),
                "active_request_age_histogram": {
                    str(age): int(count) for age, count in sorted(active_ages.items())
                },
                "active": True,
            }
        )

    if any(sum(step["tokens_per_die"].values()) != global_batch for step in steps):
        raise AssertionError("every serving step must preserve the global active batch")

    return {
        "metadata": {
            "sampling": "rolling_continuous_request_slots",
            "seed": seed,
            "initial_domain": initial_domain,
            "replacement_domain": replacement_domain,
            "initial_request_pool_size": len(initial_req_payloads),
            "initial_complete_request_pool_size": len(initial_pool),
            "replacement_request_pool_size": len(replacement_req_payloads),
            "replacement_complete_request_pool_size": len(replacement_pool),
            "num_dies": num_dies,
            "batch_per_die": batch_per_die,
            "global_batch": global_batch,
            "request_decode_len": request_decode_len,
            "serving_steps": serving_steps,
            "window_size": window_size,
            "initial_age_assignment": (
                "balanced_over_request_decode_ages_then_seeded_shuffle"
            ),
            "replacement_rule": (
                "every completed request is replaced by a seeded replacement-domain "
                "request on the next step; replacement requests may complete and "
                "roll over again"
            ),
            "random_streams": [
                "initial_requests",
                "rolling_replacement_requests",
                "initial_ages",
            ],
            "batch_scale": 1.0,
        },
        "slot_initial_request_indices": [
            initial_pool[position][0] for position in initial_positions
        ],
        "slot_initial_ages": initial_ages,
        "admission_events": admission_events,
        "steps": steps,
    }


def aggregate_steps_to_windows(
    steps: list[dict],
    *,
    window_size: int,
) -> tuple[list[dict], list[dict]]:
    if window_size <= 0:
        raise ValueError("window_size must be positive")

    grouped: dict[int, list[dict]] = defaultdict(list)
    for step in steps:
        window_id = int(step.get("window_id", int(step["step_id"]) // window_size))
        grouped[window_id].append(step)

    window_stats: list[dict] = []
    window_src_expert_counts: list[dict] = []
    for window_id in sorted(grouped):
        window_steps = sorted(grouped[window_id], key=lambda step: int(step["step_id"]))
        expert_counts: Counter[int] = Counter()
        tokens_per_die: Counter[int] = Counter()
        src_expert_counts: dict[int, Counter[int]] = defaultdict(Counter)
        for step in window_steps:
            expert_counts.update(
                {int(expert): int(count) for expert, count in step.get("expert_counts", {}).items()}
            )
            tokens_per_die.update(
                {int(src): int(count) for src, count in step.get("tokens_per_die", {}).items()}
            )
            for src, counts in step.get("src_expert_counts", {}).items():
                src_expert_counts[int(src)].update(
                    {int(expert): int(count) for expert, count in counts.items()}
                )

        window_stats.append(
            {
                "window_id": window_id,
                "start_token_idx": int(window_steps[0]["token_idx"]),
                "end_token_idx": int(window_steps[-1]["token_idx"]),
                "expert_counts": {
                    str(expert): int(count) for expert, count in sorted(expert_counts.items())
                },
                "tokens_per_die": {
                    str(src): int(count) for src, count in sorted(tokens_per_die.items())
                },
                "active": any(bool(step.get("active")) for step in window_steps),
            }
        )
        window_src_expert_counts.append(
            {
                "window_id": window_id,
                "src_expert_counts": {
                    str(src): {
                        str(expert): int(count)
                        for expert, count in sorted(src_expert_counts[src].items())
                    }
                    for src in sorted(src_expert_counts)
                },
            }
        )
    return window_stats, window_src_expert_counts
