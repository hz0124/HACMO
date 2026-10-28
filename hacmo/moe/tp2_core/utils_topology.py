#!/usr/bin/env python3
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


@dataclass(frozen=True)
class TopologyConfig:
    num_wafers: int
    dies_per_wafer: int
    mesh_rows: int
    mesh_cols: int
    switch_bw_gbps: float
    mesh_bw_gbps: float
    switch_latency_ns: float
    mesh_latency_ns: float
    rail_ports: List[int]
    rail_bw_scales: List[float]
    rail_optimized: bool


@dataclass(frozen=True)
class PathEstimate:
    src_die: int
    dst_die: int
    path_edges: List[Tuple[str, int, int]]
    hop_latency_ns: float
    bottleneck_bw_gbps: float
    serialization_ns: float
    total_ns: float


class HierarchicalTopology:
    def __init__(self, cfg: TopologyConfig) -> None:
        self.cfg = cfg
        self.total_dies = cfg.num_wafers * cfg.dies_per_wafer

    def get_all_dies(self) -> List[int]:
        return list(range(self.total_dies))

    def die_to_wafer_local(self, die: int) -> Tuple[int, int]:
        return die // self.cfg.dies_per_wafer, die % self.cfg.dies_per_wafer

    def local_to_rc(self, local_die: int) -> Tuple[int, int]:
        return local_die // self.cfg.mesh_cols, local_die % self.cfg.mesh_cols

    def rc_to_local(self, row: int, col: int) -> int:
        return row * self.cfg.mesh_cols + col

    def mesh_hops(self, local_a: int, local_b: int) -> int:
        ra, ca = self.local_to_rc(local_a)
        rb, cb = self.local_to_rc(local_b)
        return abs(ra - rb) + abs(ca - cb)

    def get_edge_distance(self, die: int) -> int:
        _, local = self.die_to_wafer_local(die)
        r, c = self.local_to_rc(local)
        return min(r, c, self.cfg.mesh_rows - 1 - r, self.cfg.mesh_cols - 1 - c)

    def get_outer_to_inner_weights(self) -> Dict[int, float]:
        distances = [self.get_edge_distance(d) for d in self.get_all_dies()]
        max_dist = max(distances) if distances else 1
        raw = {d: float(max_dist - self.get_edge_distance(d) + 1) for d in self.get_all_dies()}
        mean = (sum(raw.values()) / max(len(raw), 1)) or 1.0
        return {d: v / mean for d, v in raw.items()}

    def get_link_bandwidth(self, edge: Tuple[str, int, int]) -> float:
        kind, _, _ = edge
        if kind == "mesh":
            return self.cfg.mesh_bw_gbps
        if kind == "switch":
            port_die = edge[2] if edge[1] == self.total_dies else edge[1]
            _, local = self.die_to_wafer_local(port_die)
            return self.cfg.switch_bw_gbps * self._rail_scale(local)
        return self.cfg.switch_bw_gbps

    def _rail_scale(self, local_port: int) -> float:
        for port, scale in zip(self.cfg.rail_ports, self.cfg.rail_bw_scales):
            if port == local_port:
                return float(scale)
        return 1.0

    def _mesh_path_edges(self, wafer: int, local_src: int, local_dst: int) -> List[Tuple[str, int, int]]:
        """Return deterministic Manhattan mesh edges between local dies."""
        if local_src == local_dst:
            return []
        r, c = self.local_to_rc(local_src)
        tr, tc = self.local_to_rc(local_dst)
        edges: List[Tuple[str, int, int]] = []
        while r != tr:
            nr = r + (1 if tr > r else -1)
            a = wafer * self.cfg.dies_per_wafer + self.rc_to_local(r, c)
            b = wafer * self.cfg.dies_per_wafer + self.rc_to_local(nr, c)
            edges.append(("mesh", a, b))
            r = nr
        while c != tc:
            nc = c + (1 if tc > c else -1)
            a = wafer * self.cfg.dies_per_wafer + self.rc_to_local(r, c)
            b = wafer * self.cfg.dies_per_wafer + self.rc_to_local(r, nc)
            edges.append(("mesh", a, b))
            c = nc
        return edges

    def _nearest_rail_port(self, local_die: int) -> int:
        best_port = self.cfg.rail_ports[0]
        best_hops = self.mesh_hops(local_die, best_port)
        best_scale = self._rail_scale(best_port)
        for port in self.cfg.rail_ports[1:]:
            hops = self.mesh_hops(local_die, port)
            scale = self._rail_scale(port)
            if (
                hops < best_hops
                or (hops == best_hops and scale > best_scale)
                or (hops == best_hops and scale == best_scale and port < best_port)
            ):
                best_port = port
                best_hops = hops
                best_scale = scale
        return best_port

    def get_path(self, src_die: int, dst_die: int) -> List[Tuple[str, int, int]]:
        src_w, src_local = self.die_to_wafer_local(src_die)
        dst_w, dst_local = self.die_to_wafer_local(dst_die)
        if src_w == dst_w:
            return self._mesh_path_edges(src_w, src_local, dst_local)
        if not self.cfg.rail_optimized:
            return [("switch", src_die, self.total_dies), ("switch", self.total_dies, dst_die)]

        src_port = self._nearest_rail_port(src_local)
        dst_port = self._nearest_rail_port(dst_local)
        src_port_die = src_w * self.cfg.dies_per_wafer + src_port
        dst_port_die = dst_w * self.cfg.dies_per_wafer + dst_port
        path: List[Tuple[str, int, int]] = []
        path.extend(self._mesh_path_edges(src_w, src_local, src_port))
        path.append(("switch", src_port_die, self.total_dies))
        path.append(("switch", self.total_dies, dst_port_die))
        path.extend(self._mesh_path_edges(dst_w, dst_port, dst_local))
        return path

    def get_path_cost(
        self,
        src_die: int,
        dst_die: int,
        link_loads: Dict[Tuple[str, int, int], float],
        lambda_: float,
    ) -> float:
        cost = 0.0
        for edge in self.get_path(src_die, dst_die):
            kind = edge[0]
            bw = self.get_link_bandwidth(edge)
            load = link_loads.get(edge, 0.0)
            if kind == "mesh":
                z_l = 1.0
            else:
                z_l = (self.cfg.mesh_bw_gbps / max(self.cfg.switch_bw_gbps, 1e-9)
                       + self.cfg.switch_latency_ns / max(self.cfg.mesh_latency_ns, 1e-9))
            cost += z_l + lambda_ * (load / max(bw, 1e-6))
        return cost


class TopologyCost:
    """Shared path/bandwidth estimator for mapping, routing and sanity checks.

    Bandwidth values are stored as GB/s. Numerically, 1 GB/s is 1 byte/ns, so
    byte serialization time in ns is ``nbytes / bw_gbps``.
    """

    def __init__(self, topo: HierarchicalTopology) -> None:
        self.topo = topo

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "TopologyCost":
        return cls(load_topology(config_path))

    @property
    def rail_port_count(self) -> int:
        # Duplicate corner entries in the YAML represent distinct physical
        # rail/I/O ports. load_topology folds them into one local die with a
        # larger bandwidth scale, so the physical port count is the scale sum.
        return int(round(sum(self.topo.cfg.rail_bw_scales)))

    def estimate(self, src_die: int, dst_die: int, nbytes: int) -> PathEstimate:
        if src_die == dst_die or nbytes <= 0:
            return PathEstimate(
                src_die=src_die,
                dst_die=dst_die,
                path_edges=[],
                hop_latency_ns=0.0,
                bottleneck_bw_gbps=float("inf"),
                serialization_ns=0.0,
                total_ns=0.0,
            )

        edges = self.topo.get_path(src_die, dst_die)
        hop_latency = 0.0
        bottleneck = float("inf")
        serialization_sum = 0.0
        for edge in edges:
            kind = edge[0]
            if kind == "mesh":
                bw = max(self.topo.cfg.mesh_bw_gbps, 1e-9)
                hop_latency += self.topo.cfg.mesh_latency_ns
            elif kind == "switch":
                bw = max(self.topo.cfg.switch_bw_gbps, 1e-9)
                hop_latency += self.topo.cfg.switch_latency_ns
            else:
                bw = max(self.topo.get_link_bandwidth(edge), 1e-9)
            bottleneck = min(bottleneck, bw)
            serialization_sum += nbytes / bw
        if bottleneck == float("inf"):
            bottleneck = 0.0
        # This sanity estimator reports the pipelined path serialization at the
        # bottleneck link. Phase-wall modeling in build_window_traffic_matrix.py
        # still accounts for per-message routed phases separately.
        serialization = nbytes / bottleneck if bottleneck > 0 else serialization_sum
        return PathEstimate(
            src_die=src_die,
            dst_die=dst_die,
            path_edges=edges,
            hop_latency_ns=hop_latency,
            bottleneck_bw_gbps=bottleneck,
            serialization_ns=serialization,
            total_ns=hop_latency + serialization,
        )


def load_topology(config_path: str | Path) -> HierarchicalTopology:
    p = Path(config_path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    npus_count = raw.get("npus_count", [16, 16])
    bandwidth = raw.get("bandwidth", [500.0, 2000.0])
    latency = raw.get("latency", [2000.0, 500.0])
    dies_per_wafer = int(npus_count[1])
    rows = int(math.sqrt(dies_per_wafer))
    cols = dies_per_wafer // max(rows, 1)
    if rows * cols != dies_per_wafer:
        raise ValueError(
            f"dies_per_wafer={dies_per_wafer} is not a perfect-square mesh "
            f"(rows={rows}, cols={cols}, rows*cols={rows*cols})"
        )
    # Default rail ports: only edge dies (distance 0), not all dies.
    # If the YAML specifies rail-port-nodes explicitly, use those.
    if "rail-port-nodes" in raw:
        rail_ports_raw = list(raw["rail-port-nodes"])
    else:
        # Auto-detect edge dies: dies on mesh boundary (row=0, row=rows-1,
        # col=0, col=cols-1).
        edge_dies = []
        for r in range(rows):
            for c in range(cols):
                if r == 0 or r == rows - 1 or c == 0 or c == cols - 1:
                    edge_dies.append(r * cols + c)
        rail_ports_raw = edge_dies
    rail_scales_raw = list(raw.get("rail-port-bw-scales", [1.0] * len(rail_ports_raw)))
    scale_by_port: Dict[int, float] = {}
    for port, scale in zip(rail_ports_raw, rail_scales_raw):
        scale_by_port[int(port)] = scale_by_port.get(int(port), 0.0) + float(scale)
    rail_ports = sorted(scale_by_port)
    rail_scales = [scale_by_port[p] for p in rail_ports]
    cfg = TopologyConfig(
        num_wafers=int(npus_count[0]),
        dies_per_wafer=dies_per_wafer,
        mesh_rows=rows,
        mesh_cols=cols,
        switch_bw_gbps=float(bandwidth[0]),
        mesh_bw_gbps=float(bandwidth[1]),
        switch_latency_ns=float(latency[0]),
        mesh_latency_ns=float(latency[1]),
        rail_ports=rail_ports,
        rail_bw_scales=rail_scales,
        rail_optimized=bool(raw.get("rail-optimized", False)),
    )
    return HierarchicalTopology(cfg)
