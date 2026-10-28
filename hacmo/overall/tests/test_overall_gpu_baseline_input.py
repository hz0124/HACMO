from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_overall_rebuttal import resolve_tp1_gpu_baseline


def test_without_an_override_the_baseline_is_read_from_the_tp1_pack(
    tmp_path: Path,
) -> None:
    pack = tmp_path / "tp1-final"

    resolved = resolve_tp1_gpu_baseline(pack, override=None)

    assert resolved == pack / "summary/tp1_gpu_measured_baseline.csv"


def test_an_override_lets_the_cpu_only_workflow_use_the_archived_table(
    tmp_path: Path,
) -> None:
    # AE defect (4): the TP1 stage never writes this table on a CPU-only host,
    # so the Overall stage has to be pointed at the shipped copy.
    archived = tmp_path / "expected-results" / "tp1_gpu_measured_baseline.csv"
    archived.parent.mkdir(parents=True)
    archived.write_text("method,value\nG,1.0\n", encoding="utf-8")

    resolved = resolve_tp1_gpu_baseline(tmp_path / "tp1-final", override=str(archived))

    assert resolved == archived.resolve()


def test_an_override_pointing_nowhere_fails_by_name(tmp_path: Path) -> None:
    missing = tmp_path / "absent.csv"

    with pytest.raises(FileNotFoundError, match="absent.csv"):
        resolve_tp1_gpu_baseline(tmp_path / "tp1-final", override=str(missing))
