#!/usr/bin/env python3
"""Config loading helpers for SupernodeSim."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

from .types import (
    CachePolicy,
    HardwareConfig,
    MappingPolicy,  # backward compat alias
    MoEConfig,
    ParallelConfig,
    SimulationConfig,
    SupernodeConfig,
    WorkloadConfig,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _construct(cls: type[T], data: dict[str, Any]) -> T:
    valid = {f.name for f in fields(cls)}
    unknown = {k for k in data if k not in valid}
    if unknown:
        logger.warning("Config has unknown keys for %s: %s (ignored)", cls.__name__, unknown)
    kwargs = {k: v for k, v in data.items() if k in valid}
    obj = cls(**kwargs)
    for f in fields(cls):
        value = getattr(obj, f.name)
        if is_dataclass(value) and isinstance(data.get(f.name), dict):
            setattr(obj, f.name, _construct(type(value), data[f.name]))
    return obj


def load_config(path: str | Path) -> SimulationConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    # Accept both "mapping" and "cache_policy" as keys for backward compat.
    mapping_data = data.get("mapping", data.get("cache_policy", {}))
    return SimulationConfig(
        hardware=_construct(HardwareConfig, data.get("hardware", {})),
        workload=_construct(WorkloadConfig, data.get("workload", {})),
        parallel=_construct(ParallelConfig, data.get("parallel", {})),
        mapping=_construct(CachePolicy, mapping_data),
        moe=_construct(MoEConfig, data.get("moe", {})),
        supernode=_construct(SupernodeConfig, data.get("supernode", {})),
        backend=data.get("backend", "analytic"),
    )


def dump_config(config: SimulationConfig) -> dict[str, Any]:
    return {
        "hardware": asdict(config.hardware),
        "workload": asdict(config.workload),
        "parallel": asdict(config.parallel),
        "mapping": asdict(config.mapping),
        "moe": asdict(config.moe),
        "supernode": asdict(config.supernode),
        "backend": config.backend,
    }
