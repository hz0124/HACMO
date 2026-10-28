#!/usr/bin/env python3
"""Download a bounded multi-step subset from core12345/MoE_expert_selection_trace."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError


REPO_ID = "core12345/MoE_expert_selection_trace"

DATASETS = [
    {
        "model": "deepseek",
        "root": "cognitivecomputations/DeepSeek-R1-AWQ",
        "datasets": {
            "Chinese-SimpleQA": [
                "中华文化",
                "人文与社会科学",
                "工程、技术与应用科学",
                "生活、艺术与文化",
                "社会",
                "自然与自然科学",
            ],
            "HuggingFaceH4": ["aime_2024"],
            "livecodebench": ["execution"],
            "mmlu": ["abstract_algebra", "computer_security", "high_school_mathematics", "machine_learning"],
            "mmlu_ZH_CN": ["abstract_algebra", "computer_security", "high_school_mathematics", "machine_learning"],
        },
    },
    {
        "model": "qwen",
        "root": "Qwen/Qwen3-235B-A22B-FP8",
        "datasets": {
            "Chinese-SimpleQA": [
                "中华文化",
                "人文与社会科学",
                "工程、技术与应用科学",
                "生活、艺术与文化",
                "社会",
                "自然与自然科学",
            ],
            "HuggingFaceH4": ["aime_2024"],
            "livecodebench": ["execution"],
            "mmlu": ["abstract_algebra", "computer_security", "high_school_mathematics", "machine_learning"],
            "mmlu_ZH_CN": ["abstract_algebra", "computer_security", "high_school_mathematics", "machine_learning"],
        },
    },
    {
        "model": "llama",
        "root": "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
        "datasets": {
            "HuggingFaceH4": ["aime_2024"],
            "hellaswag": ["test"],
            "livecodebench": ["execution"],
            "mmlu": ["abstract_algebra", "computer_security", "high_school_mathematics", "machine_learning"],
            "mmlu_ZH_CN": ["abstract_algebra", "computer_security", "high_school_mathematics", "machine_learning"],
        },
    },
]


def _natural_key(path: str) -> tuple:
    return tuple(int(x) if x.isdigit() else x for x in re.split(r"(\d+)", path))


def _decode_steps_and_layer(path: Path, layer: str) -> tuple[int, bool]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return 0, False
    decode = payload[1:]
    has_layer = any(isinstance(tok, dict) and layer in tok for tok in decode)
    return len(decode), has_layer


def _list_json_files(api: HfApi, prefix: str) -> list[str]:
    try:
        items = list(api.list_repo_tree(REPO_ID, path_in_repo=prefix, repo_type="dataset", recursive=False))
    except EntryNotFoundError:
        print(f"skip missing leaf: {prefix}", flush=True)
        return []
    return sorted((item.path for item in items if item.path.endswith(".json")), key=_natural_key)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-dir", type=Path, default=Path("hacmo/moe/data/raw_traces/hf_moe_expert_selection_trace"))
    parser.add_argument("--summary-json", type=Path, default=Path("hacmo/moe/data/raw_traces/hf_moe_subset_summary.json"))
    parser.add_argument("--samples-per-dataset", type=int, default=12)
    parser.add_argument("--min-decode-steps", type=int, default=16)
    parser.add_argument("--layer", default="23")
    parser.add_argument("--max-candidates-per-leaf", type=int, default=80)
    parser.add_argument("--models", nargs="+", default=None, help="Optional model keys to download, e.g. deepseek qwen llama.")
    args = parser.parse_args()

    endpoint = os.environ.get("HF_ENDPOINT")
    api = HfApi(endpoint=endpoint)
    summary: list[dict] = []
    enabled_models = set(args.models) if args.models is not None else None

    for model_spec in DATASETS:
        if enabled_models is not None and model_spec["model"] not in enabled_models:
            continue
        for dataset, leaves in model_spec["datasets"].items():
            selected: list[dict] = []
            candidates: list[str] = []
            for leaf in leaves:
                prefix = f"{model_spec['root']}/{dataset}/{leaf}"
                candidates.extend(_list_json_files(api, prefix)[: args.max_candidates_per_leaf])

            for remote_path in sorted(candidates, key=_natural_key):
                if len(selected) >= args.samples_per_dataset:
                    break
                local_path = Path(hf_hub_download(
                    repo_id=REPO_ID,
                    filename=remote_path,
                    repo_type="dataset",
                    local_dir=args.local_dir,
                ))
                decode_steps, has_layer = _decode_steps_and_layer(local_path, args.layer)
                if decode_steps < args.min_decode_steps or not has_layer:
                    continue
                selected.append({
                    "remote_path": remote_path,
                    "local_path": str(local_path),
                    "decode_steps": decode_steps,
                    "has_layer": has_layer,
                })

            summary.append({
                "model": model_spec["model"],
                "model_root": model_spec["root"],
                "dataset": dataset,
                "requested_samples": args.samples_per_dataset,
                "selected_samples": len(selected),
                "min_decode_steps": args.min_decode_steps,
                "layer": args.layer,
                "selected": selected,
            })
            print(f"{model_spec['model']}/{dataset}: selected {len(selected)}", flush=True)

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.summary_json}")


if __name__ == "__main__":
    main()
