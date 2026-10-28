from __future__ import annotations

import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import json

import pytest

from build_camera_ready_tp2_pack import (
    CASE_NAME,
    build_pack,
    scrub_absolute_paths,
    scrub_route_plan_provenance,
)


def _write_astra_tail(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "case",
                "replay",
                "method",
                "total_us",
                "compute_us",
                "communication_us",
                "speedup_over_vllm",
                "dynamic_speedup_over_static",
                "route_plan_consumed",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _main_rows(moentwine_total: float) -> list[dict[str, object]]:
    totals = {
        "vllm": 12.0,
        "moentwine": moentwine_total,
        "ours_static": 9.0,
        "ours_dynamic": 8.0,
    }
    return [
        {
            "case": CASE_NAME,
            "replay": "tail16",
            "method": method,
            "total_us": total,
            "compute_us": total / 2,
            "communication_us": total / 2,
            "speedup_over_vllm": 12.0 / total,
            "dynamic_speedup_over_static": 9.0 / 8.0 if method == "ours_dynamic" else "",
            "route_plan_consumed": "True",
        }
        for method, total in totals.items()
    ]


def _stage_run(work: Path, topology: str, batch_size: int, *, seed: int = 17) -> Path:
    """Lay out one diagnostic-runner cell exactly as the runner leaves it."""
    main_run = work / "runs" / topology / f"bpd{batch_size}"
    no_her_run = work / "variants" / topology / f"bpd{batch_size}" / "no_her"
    _write_astra_tail(main_run / CASE_NAME / "astra_tail16.csv", _main_rows(11.0))
    _write_astra_tail(
        no_her_run / CASE_NAME / "astra_tail16.csv",
        [row for row in _main_rows(7.0) if row["method"] == "moentwine"],
    )
    artifact = main_run / CASE_NAME / topology
    for path in (
        artifact / "mapping_inputs" / "runtime" / "window_stats.json",
        artifact / "mapping_inputs" / "runtime" / "window_src_expert_counts.json",
        artifact / "runtime_windows" / f"ours_dynamic_windows_bpd{batch_size}.json",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")

    # Solver output records where its inputs lived on the machine that ran it.
    for name in ("ours_static.json", "uniform.json"):
        path = artifact / "mappings" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "expert_to_dies": {"0": [0]},
                    "metadata": {
                        "popularity_rank_transfer": {
                            "selected_library_mapping_json": "/solve-host/work/x.json"
                        },
                        "alpha": 1.0,
                    },
                }
            ),
            encoding="utf-8",
        )

    # The diagnostic runner emits route plans but never a traffic matrix.
    route_plan = (
        artifact / "route_plans" / f"seed{seed}" / f"ours_dynamic_bpd{batch_size}.json"
    )
    route_plan.parent.mkdir(parents=True, exist_ok=True)
    route_plan.write_text(
        json.dumps(
            {
                "metadata": {
                    "batch_per_die": batch_size,
                    "global_batch": batch_size * 256,
                    "seed": seed,
                },
                "steps": [],
            }
        ),
        encoding="utf-8",
    )
    return artifact


def test_pack_builder_derives_the_tp3_traffic_matrix_from_the_route_plan(
    tmp_path: Path,
) -> None:
    topology = "16w_4x4_rail_500"
    batch_size = 16
    work = tmp_path / "work"
    artifact = _stage_run(work, topology, batch_size)
    calls: list[dict] = []

    def fake_builder(route_plan: Path, destination: Path, **kwargs) -> None:
        calls.append({"route_plan": route_plan, **kwargs})
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(
                {
                    "steps": [],
                    "summary": {"avg_step_dispatch_us": 1.0, "avg_step_combine_us": 2.0},
                    "params": {"route_plan_json": str(route_plan)},
                }
            ),
            encoding="utf-8",
        )

    output = tmp_path / "pack"
    build_pack(
        work_root=work,
        output_root=output,
        topologies=[topology],
        batch_sizes=[batch_size],
        tp3_mapping_input_basis_bpd=batch_size,
        traffic_matrix_builder=fake_builder,
    )

    assert len(calls) == 1, "one traffic matrix per topology-batch cell"
    call = calls[0]
    assert call["route_plan"] == (
        artifact / "route_plans" / "seed17" / f"ours_dynamic_bpd{batch_size}.json"
    )
    assert call["batch_per_die"] == batch_size
    assert call["num_dies"] == 256, "num_dies comes from global_batch / batch_per_die"

    emitted = (
        output / topology / "traffic_matrices" / f"ours_dynamic_bpd{batch_size}.json"
    )
    assert emitted.is_file()


def test_emitted_traffic_matrix_carries_no_build_host_route_plan_path(
    tmp_path: Path,
) -> None:
    matrix = {
        "summary": {"avg_step_dispatch_us": 1.0},
        "params": {"route_plan_json": "/tmp/build-host/route_plans/seed17/x.json"},
    }

    scrubbed = scrub_route_plan_provenance(matrix)

    assert scrubbed["params"]["route_plan_json"] is None
    assert scrubbed["params"]["route_plan_storage"] == (
        "consumed_at_build_time; embedded traffic matrix is the TP3 input"
    )
    assert "/tmp/build-host" not in json.dumps(scrubbed)


def test_pack_builder_names_the_missing_route_plan_when_it_cannot_derive(
    tmp_path: Path,
) -> None:
    topology = "16w_4x4_rail_500"
    batch_size = 16
    work = tmp_path / "work"
    artifact = _stage_run(work, topology, batch_size)
    (artifact / "route_plans" / "seed17" / f"ours_dynamic_bpd{batch_size}.json").unlink()

    with pytest.raises(FileNotFoundError, match=r"route_plans.*ours_dynamic_bpd16"):
        build_pack(
            work_root=work,
            output_root=tmp_path / "pack",
            topologies=[topology],
            batch_sizes=[batch_size],
            tp3_mapping_input_basis_bpd=batch_size,
            traffic_matrix_builder=lambda *a, **k: None,
        )


def test_absolute_paths_are_dropped_but_solver_parameters_are_kept() -> None:
    scrubbed = scrub_absolute_paths(
        {
            "expert_to_dies": {"0": [0]},
            "metadata": {
                "popularity_rank_transfer": {
                    "selected_library_mapping_json": "/solve-host/work/x.json"
                },
                "alpha": 1.0,
                "note": "relative/path/is/fine.json",
            },
        }
    )

    transfer = scrubbed["metadata"]["popularity_rank_transfer"]
    assert transfer["selected_library_mapping_json"] is None
    assert scrubbed["metadata"]["alpha"] == 1.0
    assert scrubbed["metadata"]["note"] == "relative/path/is/fine.json"
    assert scrubbed["expert_to_dies"] == {"0": [0]}
    assert "/solve-host" not in json.dumps(scrubbed)


def test_the_handoff_carries_the_mappings_the_overall_stage_replays(
    tmp_path: Path,
) -> None:
    # The Overall GPU replay reads uniform.json out of the TP2 pack, so a pack
    # that only carries ours_static.json cannot drive Figure 9.
    topology = "16w_4x4_rail_500"
    batch_size = 16
    work = tmp_path / "work"
    _stage_run(work, topology, batch_size)
    output = tmp_path / "pack"

    build_pack(
        work_root=work,
        output_root=output,
        topologies=[topology],
        batch_sizes=[batch_size],
        tp3_mapping_input_basis_bpd=batch_size,
        traffic_matrix_builder=lambda route_plan, destination, **kwargs: (
            destination.parent.mkdir(parents=True, exist_ok=True),
            destination.write_text("{}", encoding="utf-8"),
        ),
    )

    for name in ("ours_static.json", "uniform.json"):
        emitted = output / topology / "mappings" / name
        assert emitted.is_file(), name
        assert "/solve-host" not in emitted.read_text(encoding="utf-8"), name


def test_pack_builder_cost_gates_16w_and_emits_tp3_handoff(tmp_path: Path) -> None:
    topology = "16w_4x4_rail_500"
    batch_size = 16
    work = tmp_path / "work"
    _stage_run(work, topology, batch_size)

    def fake_builder(route_plan: Path, destination: Path, **kwargs) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps({"steps": [], "summary": {}, "params": {}}), encoding="utf-8"
        )

    output = tmp_path / "pack"
    build_pack(
        work_root=work,
        output_root=output,
        topologies=[topology],
        batch_sizes=[batch_size],
        tp3_mapping_input_basis_bpd=batch_size,
        traffic_matrix_builder=fake_builder,
    )

    with (output / "paper_figures" / "paper_figure_summary.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    moentwine = next(row for row in rows if row["method"] == "moentwine")
    assert float(moentwine["total_us"]) == 7.0
    assert moentwine["moentwine_deployment_selected_variant"] == "no_her"
    assert (output / topology / "mappings" / "ours_static.json").is_file()
    assert (
        output
        / topology
        / "runtime_windows"
        / f"ours_dynamic_windows_bpd{batch_size}.json"
    ).is_file()
    assert (
        output
        / topology
        / "traffic_matrices"
        / f"ours_dynamic_bpd{batch_size}.json"
    ).is_file()
