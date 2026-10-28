from __future__ import annotations

import os
import stat
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from check_env import (
    REQUIRED_NOFILE_SOFT_LIMIT,
    binary_starts,
    file_descriptor_limit_error,
)


def _executable(path: Path, script: str) -> Path:
    path.write_text(script, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


def test_a_binary_that_exits_nonzero_still_counts_as_startable(tmp_path: Path) -> None:
    # Both simulators print usage and exit non-zero when invoked with no
    # workload; that is a healthy binary, not a broken one.
    binary = _executable(tmp_path / "usage", "#!/bin/sh\necho 'usage: ...' >&2\nexit 1\n")

    ok, detail = binary_starts(binary)

    assert ok, detail


def test_a_binary_whose_loader_cannot_find_its_library_is_reported(
    tmp_path: Path,
) -> None:
    # This is AE defect (1): an RPATH baked to the packaging host, which
    # check_env previously passed because it only tested is_file().
    binary = _executable(
        tmp_path / "broken",
        "#!/bin/sh\n"
        "echo 'broken: error while loading shared libraries: "
        "libramulator.so: cannot open shared object file' >&2\n"
        "exit 127\n",
    )

    ok, detail = binary_starts(binary)

    assert not ok
    assert "libramulator.so" in detail


def test_a_missing_binary_is_reported_rather_than_raising(tmp_path: Path) -> None:
    ok, detail = binary_starts(tmp_path / "absent")

    assert not ok
    assert "absent" in detail


def test_the_default_1024_descriptor_limit_is_rejected() -> None:
    # AE defect (2): ASTRA-Sim opens one trace file per NPU, so 4624 NPUs
    # silently fail at the shell default of 1024.
    error = file_descriptor_limit_error(soft_limit=1024)

    assert error is not None
    assert "1024" in error
    assert str(REQUIRED_NOFILE_SOFT_LIMIT) in error


def test_a_sufficient_descriptor_limit_passes() -> None:
    assert file_descriptor_limit_error(soft_limit=REQUIRED_NOFILE_SOFT_LIMIT) is None
    assert file_descriptor_limit_error(soft_limit=1048576) is None


def test_the_required_limit_covers_the_largest_npu_count() -> None:
    # 68x68 is the largest TP1 array; anything at or below its NPU count
    # would reintroduce the silent failure.
    assert REQUIRED_NOFILE_SOFT_LIMIT > 68 * 68
