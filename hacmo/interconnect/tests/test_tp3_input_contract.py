from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_tp3_experiment import validate_tp2_handoff


def test_tp3_handoff_rejects_missing_batch_runtime_and_traffic(tmp_path: Path) -> None:
    topology = tmp_path / "4w_8x8_rail_900"
    (topology / "mappings").mkdir(parents=True)
    (topology / "mappings" / "ours_static.json").write_text("{}\n")
    runtime_inputs = topology / "mapping_inputs" / "runtime"
    runtime_inputs.mkdir(parents=True)
    (runtime_inputs / "window_stats.json").write_text("[]\n")
    (runtime_inputs / "window_src_expert_counts.json").write_text("[]\n")

    with pytest.raises(FileNotFoundError) as error:
        validate_tp2_handoff(
            tmp_path,
            topology_labels=["4w_8x8_rail_900"],
            batch_sizes=[16],
        )

    message = str(error.value)
    assert "runtime_windows/ours_dynamic_windows_bpd16.json" in message
    assert "traffic_matrices/ours_dynamic_bpd16.json" in message
