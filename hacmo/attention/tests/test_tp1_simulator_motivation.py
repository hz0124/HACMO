from __future__ import annotations

import copy

import pytest

from hacmo.attention.tools.build_tp1_simulator_motivation import (
    _build_rows,
    _validate_rows,
)


def _fixtures() -> tuple[
    list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]
]:
    aggregates: list[dict[str, str]] = []
    cores: list[dict[str, str]] = []
    hybrid: list[dict[str, str]] = []
    for context_index, context in enumerate(("8k", "16k")):
        for batch in (16, 32, 64):
            total = float(batch + context_index * 4)
            aggregates.append(
                {
                    "context_group": context,
                    "array_rows": "68",
                    "batch_size": str(batch),
                    "method": "C",
                    "sampled_windows": "6",
                    "mean_exposed_us": str(total),
                    "ramulator_backend": "ramulator2_hbm3_timed_trace",
                    "noc_backend": "astra_sim_congestion_aware_mesh2d",
                }
            )
            cores.extend(
                [
                    {
                        "context_group": context,
                        "array_rows": "68",
                        "batch_size": str(batch),
                        "method": "C",
                        "core_id": "0",
                        "row": "0",
                        "col": "0",
                        "compute_busy_us": str(total * 0.20 * 6),
                        "post_phase_finish_us": str(total * 0.90 * 6),
                        "exposed_noncompute_us": str(total * 0.70 * 6),
                    },
                    {
                        "context_group": context,
                        "array_rows": "68",
                        "batch_size": str(batch),
                        "method": "C",
                        "core_id": "7",
                        "row": "0",
                        "col": "7",
                        "compute_busy_us": str(total * 0.25 * 6),
                        "post_phase_finish_us": str(total * 6),
                        "exposed_noncompute_us": str(total * 0.75 * 6),
                    },
                ]
            )
        for batch_label, global_batch, source_batch, cp_degree in (
            ("16", 16, 16, 1),
            ("32", 32, 32, 1),
            ("64", 64, 64, 1),
            ("128+CP2", 128, 64, 2),
        ):
            simulated = float(source_batch + context_index * 4)
            cp_us = 15.0 if cp_degree == 2 else 0.0
            hybrid.append(
                {
                    "context": context,
                    "array_rows": "68",
                    "method": "C",
                    "batch_label": batch_label,
                    "global_batch": str(global_batch),
                    "cp_degree": str(cp_degree),
                    "simulated_batch_per_die": str(source_batch),
                    "source_batch_size": str(source_batch),
                    "simulated_core_attention_us": str(simulated),
                    "analytical_cp_us": str(cp_us),
                    "total_us": str(simulated + cp_us),
                    "result_class": (
                        "hybrid_simulator_plus_analytical_cp"
                        if cp_degree == 2
                        else "simulator_only"
                    ),
                }
            )
    return aggregates, cores, hybrid


def test_breakdown_uses_critical_core_and_conserves_components() -> None:
    aggregate, cores, hybrid = _fixtures()
    rows = _build_rows(aggregate, cores, hybrid, method="C", array_side=68)
    validation = _validate_rows(rows)

    assert validation["status"] == "ok"
    assert len(rows) == 8
    assert all(row["critical_core_id"] == 7 for row in rows)
    assert all(
        abs(
            float(row["simulated_compute_busy_us"])
            + float(row["simulated_exposed_wait_tail_us"])
            + float(row["analytical_cp_us"])
            - float(row["total_us"])
        )
        < 1e-9
        for row in rows
    )
    cp2 = [row for row in rows if row["batch_label"] == "128+CP2"]
    assert len(cp2) == 2
    assert all(row["source_batch_size"] == 64 for row in cp2)
    assert all(float(row["analytical_cp_us"]) == 15.0 for row in cp2)


def test_breakdown_rejects_nonconserved_critical_core() -> None:
    aggregate, cores, hybrid = _fixtures()
    broken = copy.deepcopy(cores)
    target = next(
        row
        for row in broken
        if row["context_group"] == "8k"
        and row["batch_size"] == "16"
        and row["core_id"] == "7"
    )
    target["exposed_noncompute_us"] = "1.0"

    with pytest.raises(ValueError, match="do not conserve"):
        _build_rows(aggregate, broken, hybrid, method="C", array_side=68)
