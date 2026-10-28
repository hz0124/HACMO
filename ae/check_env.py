#!/usr/bin/env python3
"""Check the software and source prerequisites for HACMO AE workflows."""

from __future__ import annotations

import argparse
import importlib
import platform
import resource
import shutil
import subprocess
import sys
from pathlib import Path


MODULES = [
    "numpy",
    "matplotlib",
    "yaml",
    "huggingface_hub",
    "google.protobuf",
    "grpc_tools",
    "pytest",
]

# ASTRA-Sim opens one Chakra trace per NPU, and the largest TP1 array is
# 68x68 = 4624 NPUs. At the usual shell default of 1024 the simulator exits 1
# with an empty log, so refuse to start rather than fail silently later.
REQUIRED_NOFILE_SOFT_LIMIT = 8192

LOADER_FAILURE_MARKERS = (
    "error while loading shared libraries",
    "cannot open shared object file",
)


def binary_starts(path: Path) -> tuple[bool, str]:
    """Report whether a prebuilt simulator can actually be loaded and run.

    Checking is_file() is not enough: a binary carrying an RPATH from the
    packaging host exists but cannot resolve its own libraries. Both
    simulators exit non-zero when invoked without a workload, so only a
    dynamic-loader failure counts as broken.
    """
    if not path.is_file():
        return False, f"missing binary: {path}"
    try:
        completed = subprocess.run(
            [str(path)],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except OSError as exc:
        return False, f"{path} could not be executed: {exc}"
    except subprocess.TimeoutExpired:
        return True, f"{path} started (timed out waiting for it to exit)"

    output = f"{completed.stdout}\n{completed.stderr}"
    if any(marker in output for marker in LOADER_FAILURE_MARKERS):
        detail = next(
            (line.strip() for line in output.splitlines() if line.strip()),
            "dynamic loader failure",
        )
        return False, f"{path} cannot load its libraries: {detail}"
    return True, f"{path} started (exit={completed.returncode})"


def file_descriptor_limit_error(*, soft_limit: int) -> str | None:
    if soft_limit >= REQUIRED_NOFILE_SOFT_LIMIT:
        return None
    return (
        f"open file limit is {soft_limit}, but ASTRA-Sim needs at least "
        f"{REQUIRED_NOFILE_SOFT_LIMIT} for the 68x68 TP1 cases; "
        f"raise it with `ulimit -n {REQUIRED_NOFILE_SOFT_LIMIT}`"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    errors: list[str] = []
    print(f"python={platform.python_version()} platform={platform.platform()}")
    if not ((3, 10) <= sys.version_info[:2] <= (3, 12)):
        errors.append("Python 3.10-3.12 is required for the tested environment")
    for name in MODULES:
        try:
            module = importlib.import_module(name)
            print(f"module {name}: {getattr(module, '__version__', 'installed')}")
        except Exception as exc:  # pragma: no cover - diagnostic path
            errors.append(f"missing Python module {name}: {exc}")

    root = Path(__file__).resolve().parents[1]
    if args.full:
        if platform.system() != "Linux":
            errors.append("the full ASTRA-Sim workflow is supported on Linux")
        astra = root / "astra-sim"
        astra_binary = (
            astra
            / "build/astra_analytical/build/bin/"
            / "AstraSim_Analytical_Congestion_Aware"
        )
        ramulator = root / "third_party" / "ramulator2"
        ramulator_binary = root / "build/third_party/ramulator2-build/ramulator2"
        if not astra.exists() or not any(astra.iterdir()):
            errors.append("missing bundled astra-sim source tree")
        soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
        print(f"open file limit (soft): {soft_limit}")
        limit_error = file_descriptor_limit_error(soft_limit=soft_limit)
        if limit_error is not None:
            errors.append(limit_error)

        if astra_binary.is_file():
            ok, detail = binary_starts(astra_binary)
            print(f"prebuilt ASTRA-Sim: {detail}")
            if not ok:
                errors.append(detail)
        else:
            errors.append(f"missing required ASTRA-Sim binary: {astra_binary}; run bash scripts/build_simulators.sh")
            print("prebuilt ASTRA-Sim: missing; source build tools are required")
            for command in ["cmake", "g++", "make"]:
                path = shutil.which(command)
                print(f"command {command}: {path or 'missing'}")
                if path is None:
                    errors.append(f"missing command: {command}")
        if not ramulator.exists() or not any(ramulator.iterdir()):
            errors.append("missing bundled Ramulator2 source tree")
        if ramulator_binary.is_file():
            ok, detail = binary_starts(ramulator_binary)
            print(f"prebuilt Ramulator2: {detail}")
            if not ok:
                errors.append(detail)
        else:
            errors.append(f"missing required Ramulator2 binary: {ramulator_binary}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
    print("PASS: environment checks")


if __name__ == "__main__":
    main()
