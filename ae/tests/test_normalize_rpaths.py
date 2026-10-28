from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from normalize_rpaths import (
    RELOCATABLE_BINARIES,
    absolute_entries,
    origin_relative_rpath,
)


def test_ramulator_finds_its_library_next_door() -> None:
    # libramulator.so ships in ramulator2-src, one directory up from the
    # binary. The AE v2 archive pointed at the packaging host instead.
    rpath = origin_relative_rpath(
        "build/third_party/ramulator2-build/ramulator2",
        ["build/third_party/ramulator2-src"],
    )

    assert rpath == "$ORIGIN/../ramulator2-src"


def test_astra_reaches_the_bundled_environment_from_five_levels_down() -> None:
    rpath = origin_relative_rpath(
        "astra-sim/build/astra_analytical/build/bin/"
        "AstraSim_Analytical_Congestion_Aware",
        ["ae-env/lib"],
    )

    assert rpath == "$ORIGIN/../../../../../ae-env/lib"


def test_multiple_library_directories_are_joined_in_order() -> None:
    rpath = origin_relative_rpath(
        "build/third_party/ramulator2-build/ramulator2",
        ["build/third_party/ramulator2-src", "ae-env/lib"],
    )

    assert rpath == "$ORIGIN/../ramulator2-src:$ORIGIN/../../../ae-env/lib"


def test_a_build_host_path_is_flagged_but_origin_entries_are_not() -> None:
    packaging_path = "/opt/packaging-host/runtime/HACMO-AE/build"

    leftovers = absolute_entries(f"$ORIGIN/../ramulator2-src:{packaging_path}")

    assert leftovers == [packaging_path]


def test_a_fully_relative_rpath_has_nothing_to_flag() -> None:
    assert absolute_entries("$ORIGIN/../ramulator2-src:$ORIGIN/../../../ae-env/lib") == []
    assert absolute_entries("") == []


def test_every_configured_binary_normalizes_to_origin_relative_entries() -> None:
    for binary, library_dirs in RELOCATABLE_BINARIES:
        rpath = origin_relative_rpath(binary, library_dirs)
        assert absolute_entries(rpath) == [], binary
        assert rpath.startswith("$ORIGIN/"), binary
