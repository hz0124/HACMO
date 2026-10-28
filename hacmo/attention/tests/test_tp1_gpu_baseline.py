from __future__ import annotations

import csv
from pathlib import Path

import pytest

from hacmo.attention.tools.plot_tp1_audited_pack import _load_gpu_baseline


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _gpu_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    profile = tmp_path / "vllm_kernel_times.csv"
    _write_csv(
        profile,
        [
            {
                "module": "sparse_attention_mla",
                "batch_size": batch,
                "s_q": 1,
                "s_kv": 32768,
                "topk": 2048,
                "e2e_time_us": latency,
            }
            for batch, latency in ((16, 71.0), (32, 91.2), (64, 120.7))
        ],
    )
    context_sweep = tmp_path / "flashmla_detailed_results.csv"
    _write_csv(
        context_sweep,
        [
            {
                "batch_size": 32,
                "s_q": 1,
                "s_kv": context,
                "topk": 2048,
                "e2e_time_us": latency,
            }
            for context, latency in ((8192, 93.2), (16384, 93.7), (32768, 94.9))
        ],
    )
    env_info = tmp_path / "env_info.txt"
    env_info.write_text("gpu: NVIDIA H200\n", encoding="utf-8")
    return profile, context_sweep, env_info


def test_gpu_baseline_uses_direct_batch_measurements_for_both_contexts(
    tmp_path: Path,
) -> None:
    by_group, rows, validation = _load_gpu_baseline(*_gpu_inputs(tmp_path))

    assert by_group[("8k", 16)] == 71.0
    assert by_group[("16k", 16)] == 71.0
    assert by_group[("8k", 64)] == 120.7
    assert len(rows) == 6
    assert validation["max_abs_8k_16k_delta_vs_32k_pct"] < 5.0


def test_gpu_baseline_rejects_unsupported_context_reuse(tmp_path: Path) -> None:
    profile, context_sweep, env_info = _gpu_inputs(tmp_path)
    with context_sweep.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["e2e_time_us"] = "70.0"
    _write_csv(context_sweep, rows)

    with pytest.raises(ValueError, match="context reuse is not supported"):
        _load_gpu_baseline(profile, context_sweep, env_info)
