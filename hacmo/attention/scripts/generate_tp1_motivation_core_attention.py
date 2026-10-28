#!/usr/bin/env python3
"""Generate the TP1 motivation figure for indexer-excluded core attention.

The figure focuses on the C baseline at 68x68 under 32K context and shows
absolute latency broken into:
  - ideal compute
  - HBM service
  - wafer-core NoC + tail exposure

This reuses the current TP1 final-pack modeling helpers so the figure stays
aligned with the rebuttal-era TP1 pipeline.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hacmo.attention.tp1_core.generate_tp1_final_pack import (  # noqa: E402
    estimate_latency,
    get_hit_rate,
    load_hbm_service_curve,
    load_hit_rate_overrides,
    load_mapping_profile,
    parse_profile_file,
    mapping_for_shape,
)


DEFAULT_FINAL_PACK = (
    REPO_ROOT
    / "hacmo"
    / "attention"
    / "outputs"
    / "technical_point1_final_pack_static_traffic_shared_20260612"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "hacmo"
    / "attention"
    / "motivation"
    / "tp1_core_attention_motivation_20260617"
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows to write: {path}")
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, object]], out_dir: Path) -> None:
    batch_order = [16, 32, 64, 128]
    x = list(range(len(batch_order)))
    width = 0.62

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ideal = [float(r["ideal_compute_us"]) for r in rows]
    hbm = [float(r["hbm_service_us"]) for r in rows]
    noc_tail = [float(r["noc_tail_us"]) for r in rows]

    ax.bar(x, ideal, width=width, color="#9aa0a6", label="Ideal compute")
    ax.bar(x, hbm, width=width, bottom=ideal, color="#d97706", label="HBM Service")
    ax.bar(
        x,
        noc_tail,
        width=width,
        bottom=[i + j for i, j in zip(ideal, hbm)],
        color="#2563eb",
        label="Wafer-Core NoC + Tail Exposure",
    )

    totals = [float(r["total_us"]) for r in rows]
    for i, total in enumerate(totals):
        ax.text(i, total + 2.0, f"{total:.1f} us", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x, [str(b) for b in batch_order])
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Absolute latency (us)")
    ax.set_title("Indexer-excluded core attention (68x68, ctx=32k, C)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "tp1_core_attention_motivation.png", dpi=240)
    fig.savefig(out_dir / "tp1_core_attention_motivation.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-pack", type=Path, default=DEFAULT_FINAL_PACK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    pack = args.final_pack
    csv_path = pack / "csv" / "ablation1_tp1.csv"
    meta_path = pack / "raw" / "run_meta.json"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)

    import json

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    mapping_profile = load_mapping_profile(str(meta.get("mapping_profile_json", "")))
    hit_rate_overrides = load_hit_rate_overrides(str(meta.get("hit_rate_override_csv", "")))
    hbm_config = dict(meta.get("hbm_model", {}))
    cache_ratio = float(meta.get("ablation_cache_ratio", 0.4))
    hbm_config["ablation_cache_ratio"] = cache_ratio
    hbm_config["service_curve"] = load_hbm_service_curve(str(hbm_config.get("hbm_service_curve_json", "")))

    rows = read_rows(csv_path)
    batch_order = [16, 32, 64, 128]
    context_len = 32768
    array_rows = 68
    array_cols = 68
    memo: dict[tuple, dict] = {}
    selected = []
    mapping_derived = mapping_for_shape(mapping_profile, array_rows, array_cols)
    profile_files = [Path(p) for p in meta.get("profile_files", [])]
    layer_ids = list(range(0, int(meta.get("n_layers", 61)), 6))
    layer_streams_all = [
        parse_profile_file(profile_file, decode_steps_limit=int(meta.get("decode_steps", 50)), layer_ids=layer_ids)
        for profile_file in profile_files
        if profile_file.exists()
    ]
    if not layer_streams_all:
        raise FileNotFoundError("No profile files available in run_meta.json for hit-rate reconstruction.")

    for batch in batch_order:
        # Fallback to the already-generated CSV if the override lookup above
        # cannot resolve the cache hit rate from the local profile bundle.
        match = [
            r
            for r in rows
            if r["context_len"] == str(context_len)
            and r["array_rows"] == str(array_rows)
            and r["array_cols"] == str(array_cols)
            and r.get("batch_label") == str(batch)
            and r.get("cp_degree", "1") == "1"
            and r["method"] == "C"
        ]
        if match:
            hit_rate = float(match[0]["hit_rate"])
        else:
            hit_rate = get_hit_rate(
                overrides=hit_rate_overrides,
                memo=memo,
                layer_streams_all=layer_streams_all,
                batch_size=batch,
                cache_ratio=cache_ratio,
                context_len=context_len,
                policy="C",
                kv_entry_bytes=int(hbm_config.get("kv_entry_bytes", 656)),
                total_sram_bytes=440 * 1024 * 1024,
                n_layers=int(meta.get("n_layers", 61)),
            )
        lat = estimate_latency(
            batch_size=batch,
            context_len=context_len,
            baseline="C",
            hit_rate=hit_rate,
            rows=array_rows,
            cols=array_cols,
            gamma=float(meta.get("gamma", 1.0)),
            mapping_derived=mapping_derived,
            hbm_config=hbm_config,
        )
        selected.append(
            {
                "batch_size": batch,
                "hit_rate": float(hit_rate),
                "ideal_compute_us": float(lat["T_comp_us"]),
                "hbm_service_us": float(lat["T_hbm_service_us"]),
                "noc_tail_us": float(lat["T_noc_hbm_us"] + lat["T_noc_near_us"] + lat["T_noc_remote_us"] + (lat["T_qkv_mem_us"] - lat["T_qkv_pre_tail_us"])),
                "total_us": float(lat["T_comp_us"] + lat["T_hbm_service_us"] + lat["T_noc_hbm_us"] + lat["T_noc_near_us"] + lat["T_noc_remote_us"] + (lat["T_qkv_mem_us"] - lat["T_qkv_pre_tail_us"])),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_dir / "tp1_core_attention_motivation.csv", selected)
    plot(selected, args.output_dir)

    (args.output_dir / "README.md").write_text(
        "\n".join(
            [
                "# TP1 Core-Attention Motivation",
                "",
                "Generated from the current TP1 final pack.",
                "",
                "- Scope: indexer-excluded core attention.",
                "- Setting: context 32k, array 68x68, C baseline, batch 16/32/64/128.",
                "- Provenance: 16/32/64 reuse current final-pack rows; 128 is recomputed from the current profile bundle because the ablation CSV only carries 128+CP4.",
                "- Decomposition: ideal compute, HBM Service, Wafer-Core NoC + Tail Exposure.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
