from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_camera_ready_tp2.py"
if str(SCRIPT.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT.parent))

from run_camera_ready_tp2 import (
    CASE_NAME,
    verify_prepared_input,
    verify_static_mapping,
)


def test_dry_run_connects_generated_mapping_to_mixed_pool_prepare(
    tmp_path: Path,
) -> None:
    trace_root = tmp_path / "mmlu"
    work_root = tmp_path / "work"
    topology = ROOT / "configs/network/hierarchical_4w_8x8_mesh_rail_900.yaml"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--trace-root",
            str(trace_root),
            "--work-root",
            str(work_root),
            "--topologies",
            str(topology),
            "--batch-sizes",
            "16",
            "--stages",
            "mapping",
            "prepare",
            "--dry-run",
        ],
        cwd=ROOT.parents[1],
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    plan = json.loads(completed.stdout)
    mapping_command = plan["commands"][0]
    prepare_command = plan["commands"][1]
    generated_mapping = (
        work_root
        / "static_mapping"
        / "4w_cap256"
        / "4w_8x8_rail_900"
        / "mappings"
        / "ours_static.json"
    )

    assert "--mapping-only" in mapping_command
    assert str(trace_root / "professional_law") in mapping_command
    assert str(trace_root / "miscellaneous") in mapping_command
    assert "high_drift_uniform_pool_law_math_elem_human_aging" in prepare_command
    assert str(generated_mapping) in prepare_command
    assert plan["protocol"]["seed"] == 17
    assert plan["protocol"]["batch_sizes"] == [16]
    assert plan["protocol"]["expected_input_hashes"]["16"]["step_inputs"] == (
        "a9668e2c831ae052d02013bdea73ab2f76c94cedf5eeadaab78d66d3b656feb3"
    )
    assert plan["trace_root"] == str(trace_root)
    assert plan["work_root"] == str(work_root)


def test_the_plan_never_points_outside_the_roots_it_was_given(tmp_path: Path) -> None:
    """No command may reference a directory the caller did not select.

    An earlier plan carried the author machine's data root, which is how a
    released workflow ends up depending on a path only the authors have.
    """
    trace_root = tmp_path / "mmlu"
    work_root = tmp_path / "work"
    plan = _plan_for(tmp_path, [TOPO_4W], [16], ["mapping", "prepare", "pack"])
    allowed = (
        str(trace_root),
        str(work_root),
        str(ROOT.parents[1]),
        sys.executable,
        str(Path(sys.executable).resolve()),
    )

    escapes = [
        token
        for command in plan["commands"]
        for token in command
        if token.startswith("/") and not token.startswith(allowed)
    ]

    assert escapes == [], f"plan reaches outside the selected roots: {escapes}"


def test_prepare_verification_rejects_a_golden_hash_mismatch(tmp_path: Path) -> None:
    summary = tmp_path / CASE_NAME / "input_summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(
        json.dumps(
            {
                "input_hashes": {
                    "step_inputs": "wrong",
                    "window_stats": "expected-window",
                    "window_src_expert_counts": "expected-source",
                }
            }
        ),
        encoding="utf-8",
    )

    expected = {
        "step_inputs": "expected-step",
        "window_stats": "expected-window",
        "window_src_expert_counts": "expected-source",
    }
    with pytest.raises(RuntimeError, match="bpd16.*step_inputs"):
        verify_prepared_input(tmp_path, batch_size=16, expected=expected)


def test_static_mapping_verification_ignores_provenance_but_checks_placement(
    tmp_path: Path,
) -> None:
    mapping = tmp_path / "ours_static.json"
    mapping.write_text(
        json.dumps(
            {
                "expert_to_dies": {"0": [0], "1": [1]},
                "metadata": {"source_path": "/machine-specific/path"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="Static Mapping.*semantic hash mismatch"):
        verify_static_mapping(mapping, expected_sha256="wrong")


def _plan_for(tmp_path: Path, topologies: list[Path], batch_sizes: list[int], stages: list[str]) -> dict:
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--trace-root",
            str(tmp_path / "mmlu"),
            "--work-root",
            str(tmp_path / "work"),
            "--topologies",
            *[str(path) for path in topologies],
            "--batch-sizes",
            *[str(size) for size in batch_sizes],
            "--stages",
            *stages,
            "--dry-run",
        ],
        cwd=ROOT.parents[1],
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


TOPO_4W = ROOT / "configs/network/hierarchical_4w_8x8_mesh_rail_900.yaml"
TOPO_16W = ROOT / "configs/network/hierarchical_16w_4x4_mesh_rail_500.yaml"


def test_mapping_stage_schedules_one_solve_per_distinct_provenance(
    tmp_path: Path,
) -> None:
    plan = _plan_for(tmp_path, [TOPO_4W, TOPO_16W], [16, 32, 64, 128], ["mapping"])

    solves = plan["protocol"]["static_mapping_solves"]
    assert set(solves) == {"4w_cap256", "4w_cap96", "16w_cap96"}

    caps_by_topology: dict[str, set[str]] = {}
    for command in plan["commands"]:
        assert "--mapping-only" in command
        cap = command[command.index("--library-max-requests") + 1]
        topology = command[command.index("--topologies") + 1]
        caps_by_topology.setdefault(Path(topology).stem, set()).add(cap)

    assert caps_by_topology == {
        "hierarchical_4w_8x8_mesh_rail_900": {"256", "96"},
        "hierarchical_16w_4x4_mesh_rail_500": {"96"},
    }

    output_roots = [
        command[command.index("--output-root") + 1] for command in plan["commands"]
    ]
    assert len(set(output_roots)) == 3, "each solve needs its own output root"


def test_each_cell_consumes_the_mapping_its_provenance_names(tmp_path: Path) -> None:
    plan = _plan_for(tmp_path, [TOPO_4W, TOPO_16W], [16, 32, 64, 128], ["prepare"])
    work_root = tmp_path / "work"

    def mapping_for(solve_key: str, topology_label: str) -> str:
        return str(
            work_root
            / "static_mapping"
            / solve_key
            / topology_label
            / "mappings"
            / "ours_static.json"
        )

    consumed: dict[tuple[str, str], str] = {}
    for command in plan["commands"]:
        topology = Path(command[command.index("--network-config") + 1]).stem
        batch = command[command.index("--batch-per-die") + 1]
        consumed[(topology, batch)] = command[
            command.index("--static-mapping-json") + 1
        ]

    four_w = "hierarchical_4w_8x8_mesh_rail_900"
    sixteen_w = "hierarchical_16w_4x4_mesh_rail_500"

    assert consumed[(four_w, "16")] == mapping_for("4w_cap256", "4w_8x8_rail_900")
    assert consumed[(four_w, "32")] == mapping_for("4w_cap96", "4w_8x8_rail_900")
    assert consumed[(four_w, "64")] == mapping_for("4w_cap256", "4w_8x8_rail_900")
    assert consumed[(four_w, "128")] == mapping_for("4w_cap256", "4w_8x8_rail_900")
    assert consumed[(sixteen_w, "16")] == mapping_for("4w_cap256", "4w_8x8_rail_900")
    assert consumed[(sixteen_w, "32")] == mapping_for("16w_cap96", "16w_4x4_rail_500")
    assert consumed[(sixteen_w, "64")] == mapping_for("16w_cap96", "16w_4x4_rail_500")
    assert consumed[(sixteen_w, "128")] == mapping_for("4w_cap256", "4w_8x8_rail_900")


def test_plan_records_the_expected_hash_of_every_scheduled_solve(
    tmp_path: Path,
) -> None:
    plan = _plan_for(tmp_path, [TOPO_4W, TOPO_16W], [16, 32, 64, 128], ["mapping"])
    solves = plan["protocol"]["static_mapping_solves"]

    assert solves["4w_cap256"]["expert_to_dies_sha256"] == (
        "8d78a5554fef87f61085f588ffaf17d320b595a6d87914dc90d1bc8ee5e51f70"
    )
    assert solves["4w_cap96"]["expert_to_dies_sha256"] == (
        "b6b05fd5b66b48aa6d038e9f2052b06708f555e7dea3ccaf942cf06626d4f475"
    )
    assert solves["16w_cap96"]["expert_to_dies_sha256"] == (
        "a43a211e7a2a378b81dd96713809f492beb8207aca03aab396cba2013f8cfc2e"
    )


def test_mapping_stage_omits_solves_no_selected_cell_consumes(tmp_path: Path) -> None:
    plan = _plan_for(tmp_path, [TOPO_4W], [16], ["mapping"])

    assert set(plan["protocol"]["static_mapping_solves"]) == {"4w_cap256"}
    assert len(plan["commands"]) == 1


def test_pack_stage_emits_one_pack_per_downstream_consumer(tmp_path: Path) -> None:
    """TP3 and Overall need different batches behind the runtime inputs.

    They read the same path inside a pack, so a single pack cannot serve both:
    TP3 replays the batch-64 window statistics and Overall runs its GPU replay
    at 32 requests per worker.
    """
    plan = _plan_for(tmp_path, [TOPO_4W, TOPO_16W], [16, 32, 64, 128], ["pack"])
    work_root = tmp_path / "work"

    assert len(plan["commands"]) == 2
    by_root = {
        command[command.index("--output-root") + 1]: command[
            command.index("--tp3-mapping-input-basis-bpd") + 1
        ]
        for command in plan["commands"]
    }
    assert by_root == {
        str(work_root / "camera_ready_pack"): "64",
        str(work_root / "camera_ready_pack_overall"): "32",
    }
    assert plan["pack_root"] == str(work_root / "camera_ready_pack")
    assert plan["overall_pack_root"] == str(work_root / "camera_ready_pack_overall")
    assert plan["mapping_input_basis_bpd"] == {"tp3": 64, "overall": 32}


def test_a_basis_outside_the_selected_batches_falls_back(tmp_path: Path) -> None:
    plan = _plan_for(tmp_path, [TOPO_4W], [16], ["pack"])

    bases = {
        command[command.index("--tp3-mapping-input-basis-bpd") + 1]
        for command in plan["commands"]
    }
    assert bases == {"16"}


def test_dry_run_schedules_16w_no_her_variant_before_pack(tmp_path: Path) -> None:
    topology = ROOT / "configs/network/hierarchical_16w_4x4_mesh_rail_500.yaml"
    work_root = tmp_path / "work"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--trace-root",
            str(tmp_path / "mmlu"),
            "--work-root",
            str(work_root),
            "--topologies",
            str(topology),
            "--batch-sizes",
            "16",
            "--stages",
            "variants",
            "pack",
            "--dry-run",
        ],
        cwd=ROOT.parents[1],
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    commands = json.loads(completed.stdout)["commands"]
    assert len(commands) == 3
    variant, pack, overall_pack = commands
    assert variant[variant.index("--methods") + 1] == "moentwine"
    assert variant[variant.index("--moentwine-deployment-policy") + 1] == "no_her"
    assert str(work_root / "variants" / "16w_4x4_rail_500" / "bpd16" / "no_her") in variant
    assert pack[1].endswith("build_camera_ready_tp2_pack.py")
    assert str(work_root / "camera_ready_pack") in pack
    assert str(work_root / "camera_ready_pack_overall") in overall_pack
