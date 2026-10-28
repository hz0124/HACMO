#!/usr/bin/env python3
"""Download only the third-party MoE requests consumed by HACMO AE."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath

from huggingface_hub import HfApi, hf_hub_download


REPO_ID = "core12345/MoE_expert_selection_trace"
REVISION = "27febb7b2d24169560a9f6f83d38eb53f94916cf"
TARGET_PROFILES = {
    "camera-ready-v1": [
        ("cognitivecomputations/DeepSeek-R1-AWQ/mmlu/professional_law", 417),
        ("cognitivecomputations/DeepSeek-R1-AWQ/mmlu/miscellaneous", 256),
        (
            "cognitivecomputations/DeepSeek-R1-AWQ/mmlu/high_school_mathematics",
            90,
        ),
        (
            "cognitivecomputations/DeepSeek-R1-AWQ/mmlu/elementary_mathematics",
            90,
        ),
        ("cognitivecomputations/DeepSeek-R1-AWQ/mmlu/human_aging", 90),
    ],
    "ae-v2": [
        ("cognitivecomputations/DeepSeek-R1-AWQ/mmlu/professional_law", 96),
        ("cognitivecomputations/DeepSeek-R1-AWQ/mmlu/miscellaneous", 96),
        ("cognitivecomputations/DeepSeek-R1-AWQ/livecodebench/execution", 440),
    ],
}


def natural_key(path: str) -> tuple:
    parent = str(PurePosixPath(path).parent)
    stem = PurePosixPath(path).stem
    parts = re.split(r"(\d+)", stem)
    return (parent, tuple(int(part) if part.isdigit() else part for part in parts))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("ae/data/moe"))
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument(
        "--profile",
        choices=sorted(TARGET_PROFILES),
        default="camera-ready-v1",
        help="Pinned request selection to download.",
    )
    args = parser.parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit(
            "HF_TOKEN is required. Accept the dataset access conditions at "
            f"https://huggingface.co/datasets/{REPO_ID} first."
        )

    api = HfApi(token=token)
    records: list[dict[str, object]] = []
    targets = TARGET_PROFILES[args.profile]
    for prefix, limit in targets:
        items = api.list_repo_tree(
            REPO_ID,
            path_in_repo=prefix,
            repo_type="dataset",
            recursive=True,
            revision=args.revision,
        )
        files = sorted(
            (item.path for item in items if item.path.endswith(".json")),
            key=natural_key,
        )[:limit]
        if len(files) < limit:
            raise RuntimeError(f"{prefix}: found {len(files)} JSON files, need {limit}")
        for index, filename in enumerate(files, 1):
            local = Path(
                hf_hub_download(
                    repo_id=REPO_ID,
                    filename=filename,
                    repo_type="dataset",
                    revision=args.revision,
                    local_dir=args.output,
                    token=token,
                )
            )
            records.append(
                {
                    "remote_path": filename,
                    "local_path": str(local.relative_to(args.output)),
                    "size": local.stat().st_size,
                    "sha256": sha256(local),
                }
            )
            if index % 50 == 0 or index == len(files):
                print(f"{prefix}: {index}/{len(files)}", flush=True)

    manifest = {
        "repo_id": REPO_ID,
        "revision": args.revision,
        "profile": args.profile,
        "redistributed_by_hacmo": False,
        "selection": (
            f"{sum(limit for _, limit in targets)} DeepSeek requests consumed by "
            f"the {args.profile} TP2 workflow"
        ),
        "expected_file_count": sum(limit for _, limit in targets),
        "targets": [
            {"prefix": prefix, "limit": limit} for prefix, limit in targets
        ],
        "files": records,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"PASS: downloaded and checksummed {len(records)} files")


if __name__ == "__main__":
    main()
