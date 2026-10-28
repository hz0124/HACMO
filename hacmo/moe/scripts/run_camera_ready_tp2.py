#!/usr/bin/env python3
"""Run the camera-ready TP2 protocol from pinned trace inputs.

This orchestrator deliberately separates Static Mapping Library construction
from the per-topology/per-batch runtime and ASTRA stages. Large route plans stay
under the caller-selected work root and are never treated as source inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
SERVING_EPOCH_RUNNER = PROJECT_DIR / "scripts" / "run_serving_epoch_tp2.py"
DYNAMIC_RUNNER = PROJECT_DIR / "experiments" / "run_tp2_dynamic_headroom_diagnostic.py"
PACK_BUILDER = PROJECT_DIR / "scripts" / "build_camera_ready_tp2_pack.py"
MAPPING_TOPOLOGY = (
    PROJECT_DIR / "configs/network/hierarchical_4w_8x8_mesh_rail_900.yaml"
)
DEFAULT_TOPOLOGIES = [
    MAPPING_TOPOLOGY,
    PROJECT_DIR / "configs/network/hierarchical_16w_4x4_mesh_rail_500.yaml",
]
CASE_NAME = "high_drift_uniform_pool_law_math_elem_human_aging"

# TP3 and Overall read the topology-level runtime inputs from the same path
# but need different batches behind them: TP3 replays the batch-64 window
# statistics, while the Overall GPU replay runs at 32 requests per worker.
# One pack cannot serve both, so the stage emits one per consumer.
TP3_MAPPING_INPUT_BASIS_BPD = 64
OVERALL_MAPPING_INPUT_BASIS_BPD = 32

DIAGNOSTIC_STAGES = ("prepare", "dynamic", "route", "astra", "summarize")
STAGE_ORDER = ("mapping", *DIAGNOSTIC_STAGES, "variants", "pack")
MANIFEST_PATH = PROJECT_DIR / "configs/camera_ready_tp2_manifest.json"


TOPOLOGY_CONFIGS = {
    "4w_8x8_rail_900": PROJECT_DIR
    / "configs/network/hierarchical_4w_8x8_mesh_rail_900.yaml",
    "16w_4x4_rail_500": PROJECT_DIR
    / "configs/network/hierarchical_16w_4x4_mesh_rail_500.yaml",
}


def topology_label(path: Path) -> str:
    return path.stem.replace("hierarchical_", "").replace("_mesh_rail_", "_rail_")


def solve_mapping_path(mapping_root: Path, solve_key: str, solve: dict) -> Path:
    """Where a named solve deposits its placement.

    Two solves can share a wafer topology and differ only in library cap, so the
    solve key has to sit above the topology directory to keep them apart.
    """
    return (
        mapping_root
        / solve_key
        / str(solve["solver_topology"])
        / "mappings"
        / "ours_static.json"
    )


def expand_stages(stages: list[str]) -> list[str]:
    return list(STAGE_ORDER) if "all" in stages else [s for s in STAGE_ORDER if s in stages]


def build_plan(args: argparse.Namespace) -> dict[str, object]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    trace_root = Path(args.trace_root).resolve()
    work_root = Path(args.work_root).resolve()
    pack_root = (
        Path(args.pack_root).resolve()
        if args.pack_root is not None
        else work_root / "camera_ready_pack"
    )
    overall_pack_root = (
        Path(args.overall_pack_root).resolve()
        if args.overall_pack_root is not None
        else pack_root.with_name(pack_root.name + "_overall")
    )
    topologies = [Path(path).resolve() for path in args.topologies]
    stages = expand_stages(args.stages)
    mapping_root = work_root / "static_mapping"
    all_solves = manifest["static_mapping_solves"]
    provenance = manifest["cell_mapping_provenance"]

    # The published grid is not uniform in solve coordinates, so resolve each
    # cell against the manifest instead of imposing one placement everywhere.
    def solve_key_for(topology: Path, batch_size: int) -> str:
        label = topology_label(topology)
        try:
            return provenance[label][str(batch_size)]
        except KeyError as error:
            raise RuntimeError(
                f"No mapping provenance recorded for {label} bpd{batch_size}"
            ) from error

    selected_cells = [
        (topology, batch_size)
        for topology in topologies
        for batch_size in args.batch_sizes
    ]
    required_solves: dict[str, dict] = {}
    for topology, batch_size in selected_cells:
        key = solve_key_for(topology, batch_size)
        required_solves[key] = all_solves[key]

    def mapping_for(topology: Path, batch_size: int) -> Path:
        key = solve_key_for(topology, batch_size)
        return solve_mapping_path(mapping_root, key, all_solves[key])

    commands: list[list[str]] = []

    if "mapping" in stages:
        for solve_key, solve in sorted(required_solves.items()):
            solver_topology = TOPOLOGY_CONFIGS[str(solve["solver_topology"])]
            cap = str(solve["library_max_requests"])
            commands.append(
                [
                    sys.executable,
                    str(SERVING_EPOCH_RUNNER),
                    "--topologies",
                    str(solver_topology),
                    "--methods",
                    "ours_static",
                    "--library-hf-trace-roots",
                    str(trace_root / "professional_law"),
                    str(trace_root / "miscellaneous"),
                    "--runtime-hf-trace-root",
                    str(trace_root / "professional_law"),
                    "--epoch-t1-hf-trace-root",
                    str(trace_root / "professional_law"),
                    "--epoch-t-start",
                    "256",
                    "--epoch-t1-start",
                    "416",
                    "--epoch-t-requests",
                    "160",
                    "--epoch-t1-requests",
                    "1",
                    "--library-max-requests",
                    cap,
                    "--decode-len",
                    "128",
                    "--window-size",
                    "8",
                    "--mapping-only",
                    "--output-root",
                    str(mapping_root / solve_key),
                ]
            )

    runtime_stages = [stage for stage in DIAGNOSTIC_STAGES if stage in stages]
    if runtime_stages:
        for topology in topologies:
            label = topology_label(topology)
            for batch_size in args.batch_sizes:
                commands.append(
                    [
                        sys.executable,
                        str(DYNAMIC_RUNNER),
                        "--cases",
                        CASE_NAME,
                        "--stages",
                        *runtime_stages,
                        "--network-config",
                        str(topology),
                        "--batch-per-die",
                        str(batch_size),
                        "--seed",
                        str(args.seed),
                        "--trace-root",
                        str(trace_root),
                        "--static-mapping-json",
                        str(mapping_for(topology, batch_size)),
                        "--moentwine-deployment-policy",
                        "full_her",
                        "--output-root",
                        str(work_root / "runs" / label / f"bpd{batch_size}"),
                    ]
                )

    if "variants" in stages:
        for topology in topologies:
            label = topology_label(topology)
            if not label.startswith("16w_"):
                continue
            for batch_size in args.batch_sizes:
                commands.append(
                    [
                        sys.executable,
                        str(DYNAMIC_RUNNER),
                        "--cases",
                        CASE_NAME,
                        "--methods",
                        "moentwine",
                        "--stages",
                        "prepare",
                        "route",
                        "astra",
                        "summarize",
                        "--network-config",
                        str(topology),
                        "--batch-per-die",
                        str(batch_size),
                        "--seed",
                        str(args.seed),
                        "--trace-root",
                        str(trace_root),
                        "--static-mapping-json",
                        str(mapping_for(topology, batch_size)),
                        "--moentwine-deployment-policy",
                        "no_her",
                        "--output-root",
                        str(
                            work_root
                            / "variants"
                            / label
                            / f"bpd{batch_size}"
                            / "no_her"
                        ),
                    ]
                )

    def pack_command(output_root: Path, basis: int) -> list[str]:
        return [
            sys.executable,
            str(PACK_BUILDER),
            "--work-root",
            str(work_root),
            "--output-root",
            str(output_root),
            "--topologies",
            *[topology_label(path) for path in topologies],
            "--batch-sizes",
            *[str(batch_size) for batch_size in args.batch_sizes],
            "--tp3-mapping-input-basis-bpd",
            str(basis if basis in args.batch_sizes else args.batch_sizes[0]),
        ]

    if "pack" in stages:
        commands.append(pack_command(pack_root, TP3_MAPPING_INPUT_BASIS_BPD))
        commands.append(
            pack_command(overall_pack_root, OVERALL_MAPPING_INPUT_BASIS_BPD)
        )

    return {
        "version": "camera-ready-v1",
        "protocol": {
            "case": CASE_NAME,
            "seed": args.seed,
            "decode_len": 128,
            "window_size": 8,
            "tail_steps": 16,
            "batch_scope": "per_die",
            "batch_sizes": list(args.batch_sizes),
            "topologies": [topology_label(path) for path in topologies],
            "epoch_t_telemetry": "professional_law[256:416]",
            "replacement_pool": [
                "high_school_mathematics[0:90]",
                "elementary_mathematics[0:90]",
                "human_aging[0:90]",
            ],
            "expected_input_hashes": {
                str(batch_size): manifest["expected_input_hashes"][str(batch_size)]
                for batch_size in args.batch_sizes
            },
            "static_mapping_solves": required_solves,
            "cell_mapping_provenance": {
                topology_label(topology): {
                    str(batch_size): solve_key_for(topology, batch_size)
                    for batch_size in args.batch_sizes
                }
                for topology in topologies
            },
        },
        "manifest": str(MANIFEST_PATH),
        "trace_root": str(trace_root),
        "work_root": str(work_root),
        "pack_root": str(pack_root),
        "overall_pack_root": str(overall_pack_root),
        "mapping_input_basis_bpd": {
            "tp3": TP3_MAPPING_INPUT_BASIS_BPD,
            "overall": OVERALL_MAPPING_INPUT_BASIS_BPD,
        },
        "generated_static_mappings": {
            solve_key: str(solve_mapping_path(mapping_root, solve_key, solve))
            for solve_key, solve in sorted(required_solves.items())
        },
        "stages": stages,
        "commands": commands,
    }


def verify_prepared_input(
    output_root: Path, *, batch_size: int, expected: dict[str, str]
) -> dict[str, str]:
    summary_path = output_root / CASE_NAME / "input_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing TP2 prepare summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    actual = summary.get("input_hashes", {})
    mismatches = [
        key
        for key, expected_hash in expected.items()
        if actual.get(key) != expected_hash
    ]
    if mismatches:
        details = "; ".join(
            f"{key} expected={expected[key]} actual={actual.get(key)}"
            for key in mismatches
        )
        raise RuntimeError(f"bpd{batch_size} input hash mismatch: {details}")
    return {key: str(actual[key]) for key in expected}


def verify_static_mapping(mapping_path: Path, *, expected_sha256: str) -> str:
    if not mapping_path.is_file():
        raise FileNotFoundError(f"Generated Static Mapping is missing: {mapping_path}")
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    if "expert_to_dies" not in mapping:
        raise RuntimeError(f"Static Mapping has no expert_to_dies: {mapping_path}")
    canonical = (
        json.dumps(
            mapping["expert_to_dies"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")
    actual_sha256 = hashlib.sha256(canonical).hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "Static Mapping expert_to_dies semantic hash mismatch: "
            f"expected={expected_sha256} actual={actual_sha256}"
        )
    return actual_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace-root",
        default=(
            PROJECT_DIR
            / "data/raw_traces/hf_moe_expert_selection_trace/cognitivecomputations"
            / "DeepSeek-R1-AWQ/mmlu"
        ),
    )
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--pack-root", default=None)
    parser.add_argument("--overall-pack-root", default=None)
    parser.add_argument("--topologies", nargs="+", default=DEFAULT_TOPOLOGIES)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[16, 32, 64, 128])
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["all", *STAGE_ORDER],
        default=["all"],
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = build_plan(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return

    work_root = Path(str(plan["work_root"]))
    work_root.mkdir(parents=True, exist_ok=True)
    (work_root / "orchestrator_plan.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    generated_mappings = {
        key: Path(str(path))
        for key, path in dict(plan["generated_static_mappings"]).items()
    }
    expected_hashes = {
        key: str(solve["expert_to_dies_sha256"])
        for key, solve in dict(plan["protocol"]["static_mapping_solves"]).items()
    }
    path_to_solve = {str(path): key for key, path in generated_mappings.items()}
    verified_solves: set[str] = set()

    def verify_solve(solve_key: str) -> None:
        if solve_key in verified_solves:
            return
        verified_hash = verify_static_mapping(
            generated_mappings[solve_key],
            expected_sha256=expected_hashes[solve_key],
        )
        verified_solves.add(solve_key)
        print(
            f"[camera-ready-tp2] verified Static Mapping {solve_key} "
            f"semantic hash: {verified_hash}",
            flush=True,
        )

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(REPO_ROOT)
    for command in plan["commands"]:
        if command[1] == str(DYNAMIC_RUNNER):
            consumed = command[command.index("--static-mapping-json") + 1]
            verify_solve(path_to_solve[consumed])
        print("[camera-ready-tp2] " + " ".join(command), flush=True)
        subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)
        if command[1] == str(SERVING_EPOCH_RUNNER):
            produced = Path(command[command.index("--output-root") + 1]).name
            verify_solve(produced)
        if command[1] == str(DYNAMIC_RUNNER) and "prepare" in command:
            batch_size = int(command[command.index("--batch-per-die") + 1])
            output_root = Path(command[command.index("--output-root") + 1])
            expected = plan["protocol"]["expected_input_hashes"][str(batch_size)]
            verified = verify_prepared_input(
                output_root,
                batch_size=batch_size,
                expected=expected,
            )
            print(
                f"[camera-ready-tp2] verified bpd{batch_size} input hashes: "
                + json.dumps(verified, sort_keys=True),
                flush=True,
            )


if __name__ == "__main__":
    main()
