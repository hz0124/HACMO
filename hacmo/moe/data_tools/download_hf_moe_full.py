#!/usr/bin/env python3
"""Download a complete model subtree from core12345/MoE_expert_selection_trace.

This avoids wildcard `hf download --include`, which asks the Hub to enumerate
the whole dataset. Instead it walks exactly one model subtree, downloads JSON
files one by one, and records a resumable manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError


DEFAULT_REPO_ID = "core12345/MoE_expert_selection_trace"


def _list_children(api: HfApi, repo_id: str, path: str):
    try:
        return list(api.list_repo_tree(repo_id, path_in_repo=path, repo_type="dataset", recursive=False))
    except EntryNotFoundError:
        return []


def _discover_json_files(api: HfApi, repo_id: str, root: str) -> list[str]:
    root = root.strip("/")
    try:
        items = api.list_repo_tree(repo_id, path_in_repo=root, repo_type="dataset", recursive=True)
        return sorted(item.path for item in items if item.path.endswith(".json"))
    except EntryNotFoundError:
        return []


def _read_manifest(path: Path, repo_id: str, root: str) -> dict:
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest.setdefault("repo_id", repo_id)
        manifest.setdefault("root", root)
        manifest.setdefault("files", {})
        return manifest
    return {"repo_id": repo_id, "root": root, "files": {}}


def _write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _download_one(repo_id: str, filename: str, local_dir: Path, retries: int) -> Path:
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            return Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    repo_type="dataset",
                    local_dir=local_dir,
                )
            )
        except HfHubHTTPError as exc:
            last_exc = exc
            wait = min(2 ** attempt, 60)
            print(f"retry {attempt + 1}/{retries} after {wait}s: {filename}: {exc}", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"failed to download {filename}") from last_exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--repo-root", required=True, help="Model subtree, e.g. Qwen/Qwen3-235B-A22B-FP8.")
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=Path("hacmo/moe/data/raw_traces/hf_moe_expert_selection_trace"),
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Optional debug limit.")
    parser.add_argument("--discover-only", action="store_true", help="Only update the manifest with discovered files.")
    parser.add_argument("--rediscover", action="store_true", help="Ignore discovered_file_paths in the manifest.")
    args = parser.parse_args()

    endpoint = os.environ.get("HF_ENDPOINT")
    api = HfApi(endpoint=endpoint)
    repo_root = args.repo_root.strip("/")
    manifest = _read_manifest(args.manifest, args.repo_id, repo_root)

    if manifest.get("repo_id") != args.repo_id or manifest.get("root") != repo_root:
        raise ValueError(
            f"Manifest {args.manifest} is for repo={manifest.get('repo_id')} "
            f"root={manifest.get('root')}, not repo={args.repo_id} root={repo_root}"
        )

    print(f"endpoint={endpoint or 'default'}", flush=True)
    cached_files = manifest.get("discovered_file_paths")
    if cached_files and not args.rediscover and args.limit is None:
        files = [str(path) for path in cached_files]
        print(f"using cached discovery for {repo_root}", flush=True)
    else:
        print(f"discovering {repo_root}", flush=True)
        files = _discover_json_files(api, args.repo_id, repo_root)
    if args.limit is not None:
        files = files[: args.limit]
    manifest["discovered_files"] = len(files)
    manifest["discovered_file_paths"] = files
    _write_manifest(args.manifest, manifest)
    print(f"discovered_json={len(files)}", flush=True)
    if args.discover_only:
        print("discover_only=true", flush=True)
        return

    pending: list[str] = []
    skipped = 0
    for filename in files:
        entry = manifest["files"].get(filename)
        local_path = Path(entry["local_path"]) if entry and entry.get("local_path") else None
        if local_path is not None and local_path.exists():
            skipped += 1
            continue
        pending.append(filename)

    print(f"pending={len(pending)} skipped={skipped} workers={args.workers}", flush=True)
    downloaded = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_download_one, args.repo_id, filename, args.local_dir, args.retries): filename
            for filename in pending
        }
        for future in as_completed(futures):
            filename = futures[future]
            local_path = future.result()
            manifest["files"][filename] = {
                "local_path": str(local_path),
                "size": local_path.stat().st_size,
            }
            downloaded += 1
            if downloaded % 50 == 0 or downloaded == len(pending):
                _write_manifest(args.manifest, manifest)
                print(
                    f"progress downloaded_now={downloaded}/{len(pending)} "
                    f"manifest={len(manifest['files'])}/{len(files)} skipped={skipped}",
                    flush=True,
                )

    _write_manifest(args.manifest, manifest)
    print(f"done discovered={len(files)} downloaded={downloaded} skipped={skipped}", flush=True)


if __name__ == "__main__":
    main()
