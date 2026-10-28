#!/usr/bin/env python3
"""Public SupernodeSim API."""

from __future__ import annotations

from .backends import analytic, astra, ramulator
from .types import SimulationConfig, SimulationResult


def simulate_layer(config: SimulationConfig) -> SimulationResult:
    if config.backend == "analytic":
        return analytic.simulate(config)
    if config.backend == "astra":
        return astra.simulate(config)
    if config.backend == "ramulator":
        return ramulator.simulate(config)
    if config.backend == "astra_ramulator":
        raise NotImplementedError(f"Backend {config.backend!r} is reserved but not implemented yet.")
    raise ValueError(f"Unsupported backend: {config.backend}")
