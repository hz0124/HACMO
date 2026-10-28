#!/usr/bin/env python3
"""Analyze HF MoE expert-selection trace stability for TP2 rebuttal.

The input format is the HF expert-selection trace format used by
prepare_hf_moe_selection_traces.py: each request JSON is a list of generation
steps, step 0 is prefill, and later steps are decode tokens.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Iterable


TOP_KS = (8, 16, 32)


def _request_sort_key(path: Path) -> tuple[str, int | str]:
    try:
        return (str(path.parent), int(path.stem))
    except ValueError:
        return (str(path.parent), path.stem)


def _iter_trace_roots(base: Path) -> list[Path]:
    roots: list[Path] = []
    for child in sorted(base.rglob("*")):
        if not child.is_dir():
            continue
        if any(part.startswith(".") for part in child.relative_to(base).parts):
            continue
        if list(child.glob("*.json")):
            roots.append(child)
    return roots


def _iter_json_files(root: Path, max_files: int | None) -> list[Path]:
    files = sorted(root.glob("*.json"), key=_request_sort_key)
    if max_files is not None:
        files = files[:max_files]
    return files


def _trace_label(base: Path, root: Path) -> tuple[str, str, str]:
    parts = root.relative_to(base).parts
    if len(parts) < 3:
        return (parts[0] if parts else "unknown", "unknown", "/".join(parts[1:]) or root.name)
    owner = parts[0]
    model = parts[1]
    workload = "/".join(parts[2:])
    return owner, model, workload


def _count_rows(counter: Counter[int], rows: object) -> int:
    token_rows = 0
    if not isinstance(rows, list):
        return token_rows
    for row in rows:
        if not isinstance(row, list):
            continue
        token_rows += 1
        for expert in row:
            counter[int(expert)] += 1
    return token_rows


def _add_step_counts(
    layer_counts: dict[str, Counter[int]],
    layer_tokens: Counter[str],
    step_obj: object,
) -> None:
    if not isinstance(step_obj, dict):
        return
    for layer, rows in step_obj.items():
        counter = layer_counts[str(layer)]
        token_rows = _count_rows(counter, rows)
        if token_rows:
            layer_tokens[str(layer)] += token_rows


def _add_decode_split_counts(
    early_counts: dict[int, dict[str, Counter[int]]],
    early_tokens: dict[int, Counter[str]],
    later_counts: dict[int, dict[str, Counter[int]]],
    later_tokens: dict[int, Counter[str]],
    steps: list[object],
    *,
    window_size: int,
    early_windows: Iterable[int],
) -> None:
    decode_steps = steps[1:]
    for early_window_count in early_windows:
        cutoff = early_window_count * window_size
        for idx, step_obj in enumerate(decode_steps):
            if idx < cutoff:
                _add_step_counts(early_counts[early_window_count], early_tokens[early_window_count], step_obj)
            else:
                _add_step_counts(later_counts[early_window_count], later_tokens[early_window_count], step_obj)


def _dot(a: Counter[int], b: Counter[int]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return float(sum(v * b.get(k, 0) for k, v in a.items()))


def _norm(a: Counter[int]) -> float:
    return math.sqrt(sum(v * v for v in a.values()))


def _cosine(a: Counter[int], b: Counter[int]) -> float:
    denom = _norm(a) * _norm(b)
    if denom == 0:
        return math.nan
    return _dot(a, b) / denom


def _dense_values(a: Counter[int], b: Counter[int]) -> tuple[list[float], list[float]]:
    max_id = max(a.keys() | b.keys()) if a or b else -1
    return ([float(a.get(i, 0)) for i in range(max_id + 1)], [float(b.get(i, 0)) for i in range(max_id + 1)])


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for pos in range(i, j):
            ranks[order[pos]] = avg_rank
        i = j
    return ranks


def _pearson(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or not x:
        return math.nan
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    dx = [v - mx for v in x]
    dy = [v - my for v in y]
    denom = math.sqrt(sum(v * v for v in dx) * sum(v * v for v in dy))
    if denom == 0:
        return math.nan
    return sum(a * b for a, b in zip(dx, dy)) / denom


def _spearman(a: Counter[int], b: Counter[int]) -> float:
    av, bv = _dense_values(a, b)
    if not av:
        return math.nan
    return _pearson(_average_ranks(av), _average_ranks(bv))


def _topk(counter: Counter[int], k: int) -> set[int]:
    return {expert for expert, _ in counter.most_common(k)}


def _metrics(a: Counter[int], b: Counter[int]) -> dict[str, float]:
    out = {
        "cosine": _cosine(a, b),
        "spearman": _spearman(a, b),
    }
    for k in TOP_KS:
        ak = _topk(a, k)
        bk = _topk(b, k)
        inter = len(ak & bk)
        union = len(ak | bk)
        out[f"top{k}_overlap_frac"] = inter / k if k else math.nan
        out[f"top{k}_jaccard"] = inter / union if union else math.nan
    return out


def _fmt(value: float | int | str) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.6f}"
    return str(value)


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(row.get(k, "")) for k in fieldnames})


def _quantile(values: list[float], q: float) -> float:
    clean = sorted(v for v in values if not math.isnan(v))
    if not clean:
        return math.nan
    if len(clean) == 1:
        return clean[0]
    pos = (len(clean) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return clean[lo]
    frac = pos - lo
    return clean[lo] * (1.0 - frac) + clean[hi] * frac


def _summarize(rows: list[dict[str, object]], group_keys: list[str]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in group_keys)].append(row)

    out: list[dict[str, object]] = []
    for key, items in sorted(grouped.items(), key=lambda x: tuple(str(v) for v in x[0])):
        cosines = [float(r["cosine"]) for r in items]
        top16 = [float(r["top16_overlap_frac"]) for r in items]
        spearman = [float(r["spearman"]) for r in items]
        layer23 = next((r for r in items if str(r["layer"]) == "23"), None)
        row = {k: v for k, v in zip(group_keys, key)}
        row.update({
            "files": max(int(r["files"]) for r in items),
            "layers": len(items),
            "median_cosine": median(v for v in cosines if not math.isnan(v)) if any(not math.isnan(v) for v in cosines) else math.nan,
            "p10_cosine": _quantile(cosines, 0.10),
            "min_cosine": min((v for v in cosines if not math.isnan(v)), default=math.nan),
            "median_spearman": median(v for v in spearman if not math.isnan(v)) if any(not math.isnan(v) for v in spearman) else math.nan,
            "median_top16_overlap": median(v for v in top16 if not math.isnan(v)) if any(not math.isnan(v) for v in top16) else math.nan,
            "p10_top16_overlap": _quantile(top16, 0.10),
            "layer23_cosine": float(layer23["cosine"]) if layer23 else math.nan,
            "layer23_top16_overlap": float(layer23["top16_overlap_frac"]) if layer23 else math.nan,
        })
        out.append(row)
    return out


def analyze_root(base: Path, root: Path, *, max_files: int | None, window_size: int, early_windows: list[int]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    owner, model, workload = _trace_label(base, root)
    files = _iter_json_files(root, max_files)
    prefill_counts: dict[str, Counter[int]] = defaultdict(Counter)
    decode_counts: dict[str, Counter[int]] = defaultdict(Counter)
    prefill_tokens: Counter[str] = Counter()
    decode_tokens: Counter[str] = Counter()
    early_counts: dict[int, dict[str, Counter[int]]] = {n: defaultdict(Counter) for n in early_windows}
    later_counts: dict[int, dict[str, Counter[int]]] = {n: defaultdict(Counter) for n in early_windows}
    early_tokens: dict[int, Counter[str]] = {n: Counter() for n in early_windows}
    later_tokens: dict[int, Counter[str]] = {n: Counter() for n in early_windows}
    valid_files = 0

    for path in files:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list) or len(raw) < 2:
            continue
        valid_files += 1
        _add_step_counts(prefill_counts, prefill_tokens, raw[0])
        for step_obj in raw[1:]:
            _add_step_counts(decode_counts, decode_tokens, step_obj)
        _add_decode_split_counts(
            early_counts,
            early_tokens,
            later_counts,
            later_tokens,
            raw,
            window_size=window_size,
            early_windows=early_windows,
        )

    prefill_rows: list[dict[str, object]] = []
    for layer in sorted(set(prefill_counts) | set(decode_counts), key=lambda x: int(x) if x.isdigit() else x):
        a = prefill_counts[layer]
        b = decode_counts[layer]
        if not a or not b:
            continue
        row: dict[str, object] = {
            "owner": owner,
            "model": model,
            "workload": workload,
            "trace_root": str(root),
            "files": valid_files,
            "layer": layer,
            "prefill_token_rows": prefill_tokens[layer],
            "decode_token_rows": decode_tokens[layer],
            "num_experts_seen": len(set(a) | set(b)),
        }
        row.update(_metrics(a, b))
        prefill_rows.append(row)

    early_rows: list[dict[str, object]] = []
    for early_window_count in early_windows:
        for layer in sorted(set(early_counts[early_window_count]) | set(later_counts[early_window_count]), key=lambda x: int(x) if x.isdigit() else x):
            a = early_counts[early_window_count][layer]
            b = later_counts[early_window_count][layer]
            if not a or not b:
                continue
            row = {
                "owner": owner,
                "model": model,
                "workload": workload,
                "trace_root": str(root),
                "files": valid_files,
                "early_windows": early_window_count,
                "window_size": window_size,
                "layer": layer,
                "early_token_rows": early_tokens[early_window_count][layer],
                "later_token_rows": later_tokens[early_window_count][layer],
                "num_experts_seen": len(set(a) | set(b)),
            }
            row.update(_metrics(a, b))
            early_rows.append(row)
    return prefill_rows, early_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze prefill/decode and early/later decode expert stability.")
    parser.add_argument("--base-root", default="hacmo/moe/data/raw_traces/hf_moe_expert_selection_trace")
    parser.add_argument("--trace-roots", nargs="*", default=None,
                        help="Optional explicit leaf trace directories. Defaults to all leaf directories under base-root.")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--early-windows", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    base = Path(args.base_root)
    roots = [Path(p) for p in args.trace_roots] if args.trace_roots else _iter_trace_roots(base)
    if not roots:
        raise ValueError(f"No trace roots found under {base}")

    prefill_rows: list[dict[str, object]] = []
    early_rows: list[dict[str, object]] = []
    for root in roots:
        p_rows, e_rows = analyze_root(
            base,
            root,
            max_files=args.max_files,
            window_size=args.window_size,
            early_windows=args.early_windows,
        )
        prefill_rows.extend(p_rows)
        early_rows.extend(e_rows)

    out_dir = Path(args.output_dir)
    prefill_fields = [
        "owner", "model", "workload", "trace_root", "files", "layer",
        "prefill_token_rows", "decode_token_rows", "num_experts_seen",
        "cosine", "spearman",
        "top8_overlap_frac", "top8_jaccard",
        "top16_overlap_frac", "top16_jaccard",
        "top32_overlap_frac", "top32_jaccard",
    ]
    early_fields = [
        "owner", "model", "workload", "trace_root", "files", "early_windows",
        "window_size", "layer", "early_token_rows", "later_token_rows",
        "num_experts_seen", "cosine", "spearman",
        "top8_overlap_frac", "top8_jaccard",
        "top16_overlap_frac", "top16_jaccard",
        "top32_overlap_frac", "top32_jaccard",
    ]
    summary_fields = [
        "owner", "model", "workload", "files", "layers", "median_cosine",
        "p10_cosine", "min_cosine", "median_spearman",
        "median_top16_overlap", "p10_top16_overlap",
        "layer23_cosine", "layer23_top16_overlap",
    ]
    early_summary_fields = [
        "owner", "model", "workload", "early_windows", "files", "layers",
        "median_cosine", "p10_cosine", "min_cosine", "median_spearman",
        "median_top16_overlap", "p10_top16_overlap",
        "layer23_cosine", "layer23_top16_overlap",
    ]

    _write_csv(out_dir / "q1_prefill_decode_by_layer.csv", prefill_rows, prefill_fields)
    _write_csv(out_dir / "q1_prefill_decode_summary.csv", _summarize(prefill_rows, ["owner", "model", "workload"]), summary_fields)
    _write_csv(out_dir / "q1_early_decode_stability_by_layer.csv", early_rows, early_fields)
    _write_csv(out_dir / "q1_early_decode_stability_summary.csv", _summarize(early_rows, ["owner", "model", "workload", "early_windows"]), early_summary_fields)
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
