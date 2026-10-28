#!/usr/bin/env python3
"""Trace event primitives for future Astra/Ramulator backends."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CoreEvent:
    core_row: int
    core_col: int
    kind: str
    bytes: int = 0
    flops: float = 0.0
    depends_on: tuple[int, ...] = ()

