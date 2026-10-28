#!/usr/bin/env python3
"""Convert HuggingFace-style MoE expert-selection traces to TP2 inputs.

Expected input layout:
  <trace-root>/<dataset>/<subject>/<request_id>.json

Each JSON file is a list of generation steps.  Step 0 is usually prefill and
contains many token rows per layer; later steps contain one generated-token row
per layer.  For TP2 decode experiments this tool uses decode steps only by
default and writes the normalized request format consumed by
build_mapping_inputs.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List


def _request_sort_key(path: Path) -> tuple:
    try:
        return (str(path.parent), int(path.stem))
    except ValueError:
        return (str(path.parent), path.stem)


def _iter_json_files(root: Path) -> List[Path]:
    return sorted(root.glob("**/*.json"), key=_request_sort_key)


def _selected_layers(values: List[str], all_layers: bool) -> set[str] | None:
    return None if all_layers else {str(v) for v in values}


def _step_to_layer_ids(step_obj: dict, layers: set[str] | None) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for layer, rows in sorted(step_obj.items(), key=lambda x: int(x[0]) if str(x[0]).isdigit() else str(x[0])):
        layer_key = str(layer)
        if layers is not None and layer_key not in layers:
            continue
        if rows is None:
            continue
        if not isinstance(rows, list) or not rows:
            continue
        # Decode steps normally contain one row.  If a row contains more than
        # one token, flatten it so the selected layer still contributes its
        # observed expert activations to this normalized token.
        expert_ids: list[int] = []
        for row in rows:
            if isinstance(row, list):
                expert_ids.extend(int(e) for e in row)
        if expert_ids:
            out[layer_key] = expert_ids
    return out


def _write_normalized(
    out_dir: Path,
    files: Iterable[Path],
    *,
    layers: set[str] | None,
    include_prefill_as_decode: bool,
    max_decode_steps: int | None,
    metadata: dict,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for req_idx, path in enumerate(files):
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            continue
        decode = []
        step_iter = enumerate(raw if include_prefill_as_decode else raw[1:], start=0 if include_prefill_as_decode else 1)
        for step_idx, step_obj in step_iter:
            if max_decode_steps is not None and len(decode) >= max_decode_steps:
                break
            if not isinstance(step_obj, dict):
                continue
            layer_to_ids = _step_to_layer_ids(step_obj, layers)
            if not layer_to_ids:
                continue
            decode.append({
                "token_idx": int(step_idx),
                "layer_to_expert_ids": layer_to_ids,
            })
        if not decode:
            continue
        obj = {
            "request_id": f"hf_{path.parent.name}_{path.stem}",
            "source_file": str(path),
            "prefill": {"layer_to_expert_ids": {}},
            "decode": decode,
        }
        (out_dir / f"hf_req{count:05d}.json").write_text(
            json.dumps(obj, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        count += 1
    meta = dict(metadata)
    meta["num_requests"] = count
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return count


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare HF MoE expert-selection traces for TP2.")
    p.add_argument("--trace-root", required=True)
    p.add_argument("--runtime-trace-root", default=None,
                   help="Optional separate trace root for cross-dataset heldout runtime evaluation.")
    p.add_argument("--layers", nargs="+", default=["23"])
    p.add_argument("--all-layers", action="store_true")
    p.add_argument("--offline-fraction", type=float, default=0.5)
    p.add_argument("--max-requests", type=int, default=None)
    p.add_argument("--runtime-max-requests", type=int, default=None)
    p.add_argument("--max-decode-steps", type=int, default=None)
    p.add_argument("--runtime-max-decode-steps", type=int, default=None)
    p.add_argument("--include-prefill-as-decode", action="store_true",
                   help="Use step 0 as a decode-like token. Default skips prefill.")
    p.add_argument("--output-root", required=True)
    args = p.parse_args()

    root = Path(args.trace_root)
    files = _iter_json_files(root)
    if args.max_requests is not None:
        files = files[:args.max_requests]
    if not files:
        raise ValueError(f"Need at least one JSON request trace under {root}")

    runtime_root = Path(args.runtime_trace_root) if args.runtime_trace_root else None
    if runtime_root is not None:
        offline_files = files
        runtime_files = _iter_json_files(runtime_root)
        runtime_max_requests = args.runtime_max_requests if args.runtime_max_requests is not None else args.max_requests
        if runtime_max_requests is not None:
            runtime_files = runtime_files[:runtime_max_requests]
        if not runtime_files:
            raise ValueError(f"Need at least one JSON request trace under {runtime_root}")
        split_mode = "cross_dataset_heldout"
    else:
        if len(files) < 2:
            raise ValueError(f"Need at least two JSON request traces under {root}")
        cut = int(len(files) * args.offline_fraction)
        cut = min(max(cut, 1), len(files) - 1)
        offline_files = files[:cut]
        runtime_files = files[cut:]
        split_mode = "request_heldout"
    layers = _selected_layers(args.layers, args.all_layers)
    out_root = Path(args.output_root)
    runtime_max_decode_steps = (
        args.runtime_max_decode_steps
        if args.runtime_max_decode_steps is not None
        else args.max_decode_steps
    )

    common_meta = {
        "trace_source": "hf_moe_expert_selection_trace",
        "trace_root": str(root),
        "runtime_trace_root": str(runtime_root) if runtime_root is not None else str(root),
        "selected_layers": sorted(layers) if layers is not None else "all",
        "total_input_files": len(files),
        "offline_files": len(offline_files),
        "runtime_files": len(runtime_files),
        "split_mode": split_mode,
        "include_prefill_as_decode": args.include_prefill_as_decode,
        "max_decode_steps": args.max_decode_steps,
        "runtime_max_decode_steps": runtime_max_decode_steps,
    }
    offline_count = _write_normalized(
        out_root / "offline_normalized_traces",
        offline_files,
        layers=layers,
        include_prefill_as_decode=args.include_prefill_as_decode,
        max_decode_steps=args.max_decode_steps,
        metadata={**common_meta, "split": "offline"},
    )
    runtime_count = _write_normalized(
        out_root / "runtime_normalized_traces",
        runtime_files,
        layers=layers,
        include_prefill_as_decode=args.include_prefill_as_decode,
        max_decode_steps=runtime_max_decode_steps,
        metadata={**common_meta, "split": "runtime"},
    )
    print(
        f"Wrote HF normalized traces to {out_root}: "
        f"offline_requests={offline_count}, runtime_requests={runtime_count}, "
        f"layers={common_meta['selected_layers']}"
    )


if __name__ == "__main__":
    main()
