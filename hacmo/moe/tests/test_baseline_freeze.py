from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from check_tp2_baseline_freeze import check_baseline_freeze


def test_baseline_freeze_detects_metric_drift() -> None:
    reference = [
        {"method": "vllm", "topology": "4w", "batch_size": "64", "total_us": "100.0"},
        {"method": "moentwine", "topology": "4w", "batch_size": "64", "total_us": "90.0"},
    ]
    candidate = [
        {"method": "vllm", "topology": "4w", "batch_size": "64", "total_us": "101.0"},
        {"method": "ours_static", "topology": "4w", "batch_size": "64", "total_us": "80.0"},
    ]

    errors = check_baseline_freeze(reference, candidate, metric_columns=("total_us",))

    assert any("metric drift" in e for e in errors)


def test_baseline_freeze_can_allow_ours_only_candidates() -> None:
    reference = [
        {"method": "vllm", "topology": "4w", "batch_size": "64", "total_us": "100.0"},
    ]
    candidate = [
        {"method": "ours_static", "topology": "4w", "batch_size": "64", "total_us": "80.0"},
    ]

    errors = check_baseline_freeze(reference, candidate, metric_columns=("total_us",))
    filtered = [e for e in errors if not e.startswith("no candidate baseline rows")]

    assert filtered == []
