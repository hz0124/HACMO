from __future__ import annotations

from hacmo.attention.tools.build_tp1_hybrid_cp_pack import (
    METHODS,
    _build_rows,
    _validate_rows,
    analytical_cp_components,
)


def _source_rows() -> dict[tuple[str, int, int, str], dict[str, str]]:
    rows: dict[tuple[str, int, int, str], dict[str, str]] = {}
    for context in ("8k", "16k"):
        for side in (20, 68):
            for batch in (16, 32, 64):
                for index, method in enumerate(METHODS):
                    rows[(context, side, batch, method)] = {
                        "mean_exposed_us": str(batch + index),
                        "ramulator_backend": "ramulator2_hbm3_timed_trace",
                        "noc_backend": "astra_sim_congestion_aware_mesh2d",
                    }
    return rows


def test_cp2_formula_matches_preserved_experiment_definition() -> None:
    cp_bytes, cp_us = analytical_cp_components(
        simulated_batch_per_die=64,
        cp_degree=2,
        topk=2048,
        vector_dim=576,
        element_bytes=2,
        bandwidth_tbps=5.0,
        hops=1.0,
    )

    assert cp_bytes == 75_497_472
    assert abs(cp_us - 15.0994944) < 1e-9


def test_hybrid_pack_reuses_batch64_and_keeps_cp_separate() -> None:
    long_rows, summary_rows = _build_rows(
        _source_rows(),
        topk=2048,
        vector_dim=576,
        element_bytes=2,
        cp_bandwidth_tbps=5.0,
        cp2_hops=1.0,
    )
    validation = _validate_rows(long_rows, summary_rows)

    assert validation["status"] == "ok"
    assert len(long_rows) == 64
    assert len(summary_rows) == 16
    cp2_rows = [row for row in long_rows if row["batch_label"] == "128+CP2"]
    assert len(cp2_rows) == 16
    assert all(row["source_batch_size"] == 64 for row in cp2_rows)
    assert all(float(row["analytical_cp_us"]) > 0.0 for row in cp2_rows)
    assert all(
        abs(
            float(row["total_us"])
            - float(row["simulated_core_attention_us"])
            - float(row["analytical_cp_us"])
        )
        < 1e-9
        for row in cp2_rows
    )
    assert {str(row["method"]) for row in long_rows} == set(METHODS)
