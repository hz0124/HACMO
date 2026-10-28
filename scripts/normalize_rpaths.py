#!/usr/bin/env python3
"""Make the prebuilt simulators relocatable before they are packaged.

Both binaries are linked inside the packaging directory, so they come out
carrying an absolute RPATH that only resolves on the machine that built them.
Rewriting each RPATH relative to $ORIGIN lets the archive work wherever a
reviewer unpacks it.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


# Binary, relative to the artifact root, and the library directories it needs.
RELOCATABLE_BINARIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "build/third_party/ramulator2-build/ramulator2",
        ("build/third_party/ramulator2-src", "ae-env/lib"),
    ),
    (
        "astra-sim/build/astra_analytical/build/bin/"
        "AstraSim_Analytical_Congestion_Aware",
        ("ae-env/lib",),
    ),
)


def origin_relative_rpath(binary: str, library_dirs: tuple[str, ...] | list[str]) -> str:
    """Express each library directory relative to the binary's own directory."""
    binary_dir = os.path.dirname(binary)
    return ":".join(
        "$ORIGIN/" + os.path.relpath(library_dir, binary_dir).replace(os.sep, "/")
        for library_dir in library_dirs
    )


def absolute_entries(rpath: str) -> list[str]:
    """Return the RPATH entries that are still tied to a build host."""
    return [
        entry
        for entry in rpath.split(":")
        if entry and not entry.startswith("$ORIGIN")
    ]


def read_rpath(binary: Path) -> str:
    completed = subprocess.run(
        ["patchelf", "--print-rpath", str(binary)],
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def normalize(root: Path, *, strict: bool = True) -> list[tuple[str, str, str]]:
    if shutil.which("patchelf") is None:
        raise RuntimeError("patchelf is required to make the binaries relocatable")

    changes: list[tuple[str, str, str]] = []
    for relative_binary, library_dirs in RELOCATABLE_BINARIES:
        binary = root / relative_binary
        if not binary.is_file():
            if strict:
                raise FileNotFoundError(f"Missing binary to normalize: {binary}")
            continue
        before = read_rpath(binary)
        desired = origin_relative_rpath(relative_binary, library_dirs)
        subprocess.run(
            ["patchelf", "--set-rpath", desired, str(binary)],
            check=True,
        )
        after = read_rpath(binary)
        leftovers = absolute_entries(after)
        if leftovers:
            raise RuntimeError(
                f"{binary}: RPATH still contains build-host paths: {leftovers}"
            )
        changes.append((relative_binary, before, after))
    return changes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="artifact root containing ae/")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="skip binaries that are not present instead of failing",
    )
    args = parser.parse_args()

    changes = normalize(args.root.resolve(), strict=not args.allow_missing)
    if not changes:
        print("no prebuilt binaries found to normalize", file=sys.stderr)
        raise SystemExit(1 if not args.allow_missing else 0)
    for binary, before, after in changes:
        print(f"{binary}\n  before: {before or '(none)'}\n  after:  {after}")
    print(f"PASS: {len(changes)} binaries are relocatable")


if __name__ == "__main__":
    main()
