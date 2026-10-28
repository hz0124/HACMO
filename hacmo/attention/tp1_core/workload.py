"""Trace-qualified TP1 workload loading and deterministic request ensembles."""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class LayerProfile:
    prefill_counts: Mapping[int, int]
    prefill_ranked_tokens: tuple[int, ...]
    decode_indices: np.ndarray
    decode_positions: np.ndarray


@dataclass(frozen=True)
class RequestProfile:
    prompt_id: str
    task: str
    prompt_len: int
    gen_len: int
    layers: Mapping[int, LayerProfile]
    source_path: str
    sha256: str


@dataclass(frozen=True)
class RequestLane:
    lane_id: int
    instance_id: str
    profile: RequestProfile


@dataclass(frozen=True)
class WorkloadWindow:
    layer_id: int
    decode_step: int
    lanes: tuple[RequestLane, ...]

    @property
    def logical_accesses(self) -> int:
        return sum(len(selected_tokens(self, lane)) for lane in self.lanes)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _task_name(data: Mapping[str, object], prompt_id: str) -> str:
    source = data.get("source")
    if isinstance(source, Mapping):
        nested = source.get("source")
        if isinstance(nested, Mapping) and nested.get("dataset"):
            return str(nested["dataset"])
    if prompt_id.startswith("longbench_"):
        body = prompt_id[len("longbench_") :]
        for suffix_len in (64, 40):
            suffix = body[-suffix_len:]
            if len(suffix) == suffix_len and all(c in "0123456789abcdef" for c in suffix):
                return body[: -(suffix_len + 1)]
    return str(data.get("category", "unknown"))


def _counts_from_array(values: object, prompt_len: int) -> dict[int, int]:
    arr = np.asarray(values)
    if arr.size == 0:
        return {}
    flat = arr.reshape(-1)
    valid = flat[(flat >= 0) & (flat < prompt_len)].astype(np.int64, copy=False)
    if valid.size == 0:
        return {}
    tokens, counts = np.unique(valid, return_counts=True)
    return {int(token): int(count) for token, count in zip(tokens, counts)}


def _layer_profile(raw: Mapping[str, object], prompt_len: int) -> LayerProfile:
    if "prefill_token_counts" in raw:
        prefill_counts = {
            int(token): int(count)
            for token, count in dict(raw["prefill_token_counts"]).items()
            if 0 <= int(token) < prompt_len and int(count) > 0
        }
    elif "prefill_indices" in raw:
        value = raw.get("prefill_indices", [])
        if isinstance(value, list) and value:
            value = value[0]
        prefill_counts = _counts_from_array(value, prompt_len)
    elif "topk_indices" in raw:
        arr = np.asarray(raw["topk_indices"])
        seq = arr[0] if arr.ndim == 3 else arr
        prefill_counts = _counts_from_array(seq[:prompt_len], prompt_len)
    else:
        prefill_counts = {}

    if "decode_indices" in raw:
        decode = np.asarray(raw["decode_indices"], dtype=np.int64)
    elif "topk_indices" in raw:
        arr = np.asarray(raw["topk_indices"])
        seq = arr[0] if arr.ndim == 3 else arr
        decode = np.asarray(seq[prompt_len:], dtype=np.int64)
    else:
        decode = np.empty((0, 0), dtype=np.int64)
    if decode.ndim == 1:
        decode = decode.reshape(1, -1)
    if decode.ndim != 2:
        raise ValueError(f"decode_indices must be 2-D, got shape {decode.shape}")

    positions = np.asarray(raw.get("decode_positions", []), dtype=np.int64)
    if positions.size == 0:
        positions = np.arange(prompt_len, prompt_len + len(decode), dtype=np.int64)
    if positions.ndim != 1 or len(positions) != len(decode):
        raise ValueError("decode_positions must have one entry per decode row")
    return LayerProfile(
        prefill_counts=prefill_counts,
        prefill_ranked_tokens=tuple(
            token
            for token, _ in sorted(
                prefill_counts.items(), key=lambda item: (-item[1], item[0])
            )
        ),
        decode_indices=decode,
        decode_positions=positions,
    )


def load_profile(path: str | Path) -> RequestProfile:
    profile_path = Path(path).resolve()
    data = pickle.loads(profile_path.read_bytes())
    if not isinstance(data, Mapping):
        raise ValueError(f"Profile must contain a dictionary: {profile_path}")
    prompt_id = str(data.get("prompt_id", profile_path.stem))
    prompt_len = int(data["prompt_len"])
    raw_layers = data.get("profiling_data")
    if not isinstance(raw_layers, Mapping):
        raise ValueError(f"profiling_data must be keyed by layer: {profile_path}")
    layers = {
        int(layer_id): _layer_profile(raw, prompt_len)
        for layer_id, raw in raw_layers.items()
        if isinstance(raw, Mapping)
    }
    if not layers:
        raise ValueError(f"Profile has no layer data: {profile_path}")
    return RequestProfile(
        prompt_id=prompt_id,
        task=_task_name(data, prompt_id),
        prompt_len=prompt_len,
        gen_len=int(data.get("gen_len", min(len(x.decode_indices) for x in layers.values()))),
        layers=layers,
        source_path=str(profile_path),
        sha256=_sha256(profile_path),
    )


def load_profiles(directory: str | Path) -> list[RequestProfile]:
    root = Path(directory).resolve()
    paths = sorted(root.glob("*_profile.pkl"))
    if not paths:
        raise FileNotFoundError(f"No *_profile.pkl files found under {root}")
    return [load_profile(path) for path in paths]


def split_middle_train(profiles: Sequence[RequestProfile]) -> tuple[list[RequestProfile], list[RequestProfile]]:
    """Use the middle-length request of each task for mapping, others for evaluation."""
    by_task: dict[str, list[RequestProfile]] = {}
    for profile in profiles:
        by_task.setdefault(profile.task, []).append(profile)
    train: list[RequestProfile] = []
    evaluation: list[RequestProfile] = []
    for task in sorted(by_task):
        ordered = sorted(by_task[task], key=lambda item: (item.prompt_len, item.prompt_id))
        middle = len(ordered) // 2
        train.append(ordered[middle])
        evaluation.extend(item for index, item in enumerate(ordered) if index != middle)
    if not evaluation:
        raise ValueError("Heldout evaluation requires at least two profiles per task")
    return train, evaluation


def evaluation_length_groups(profiles: Sequence[RequestProfile]) -> dict[str, list[RequestProfile]]:
    """Group heldout profiles by the measured prompt lengths, never by a nominal label."""
    groups: dict[str, list[RequestProfile]] = {"8k": [], "16k": []}
    for profile in profiles:
        if 7_500 <= profile.prompt_len < 12_000:
            groups["8k"].append(profile)
        elif 12_000 <= profile.prompt_len < 18_000:
            groups["16k"].append(profile)
    return {name: sorted(items, key=lambda item: item.prompt_id) for name, items in groups.items() if items}


def build_lanes(profiles: Sequence[RequestProfile], batch_size: int, seed: int) -> tuple[RequestLane, ...]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not profiles:
        raise ValueError("profiles must not be empty")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(profiles), size=batch_size)
    return tuple(
        RequestLane(
            lane_id=lane_id,
            instance_id=f"req-{seed}-{lane_id:04d}",
            profile=profiles[int(profile_index)],
        )
        for lane_id, profile_index in enumerate(indices)
    )


def common_layer_ids(profiles: Iterable[RequestProfile], requested: Sequence[int]) -> tuple[int, ...]:
    profiles = list(profiles)
    return tuple(
        layer_id
        for layer_id in requested
        if all(layer_id in profile.layers for profile in profiles)
    )


def valid_decode_steps(lanes: Sequence[RequestLane], layer_ids: Sequence[int], requested: Sequence[int]) -> tuple[int, ...]:
    return tuple(
        step
        for step in requested
        if all(step < len(lane.profile.layers[layer].decode_indices) for lane in lanes for layer in layer_ids)
    )


def selected_tokens(window: WorkloadWindow, lane: RequestLane) -> tuple[int, ...]:
    layer = lane.profile.layers[window.layer_id]
    row = layer.decode_indices[window.decode_step]
    valid = (int(token) for token in row if 0 <= int(token) < lane.profile.prompt_len)
    return tuple(dict.fromkeys(valid))
