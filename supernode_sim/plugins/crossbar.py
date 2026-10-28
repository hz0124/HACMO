#!/usr/bin/env python3
"""Edge-local crossbar interconnect model for supernode_sim.

TP3 models four independent edge-local crossbars per wafer.  Each wafer edge
owns ``side`` external logical I/O ports and may reassign those ports only among
the ``side`` die-side groups on that same edge.  The model keeps external I/O
bandwidth fixed; it improves utilization by changing local port grants.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import yaml


def load_crossbar_config(network_yaml: str) -> dict:
    """Load crossbar parameters from network YAML."""
    raw = yaml.safe_load(open(network_yaml).read())
    npus = raw.get("npus_count", [4, 64])
    num_wafers = int(npus[0])
    dies_per_wafer = int(npus[1])
    side = int(math.isqrt(dies_per_wafer))
    if side * side != dies_per_wafer:
        raise ValueError(f"Crossbar requires square mesh, got {dies_per_wafer} dies")
    num_io_ports = 4 * side
    per_port_bw = float(raw.get("bandwidth", [900.0, 2000.0])[0])
    return {
        "num_wafers": num_wafers,
        "dies_per_wafer": dies_per_wafer,
        "side": side,
        "num_io_ports": num_io_ports,
        "external_ports_per_edge": side,
        "switch_bw_total_gbps": num_io_ports * per_port_bw,
        "per_port_bw_gbps": per_port_bw,
        "mesh_bw_gbps": float(raw.get("bandwidth", [900.0, 2000.0])[1]),
        "switch_latency_ns": float(raw.get("latency", [2000.0, 500.0])[0]),
        "mesh_latency_ns": float(raw.get("latency", [2000.0, 500.0])[1]),
        "crossbar_latency_ns": float(raw.get("crossbar-latency-ns", 20.0)),
        "crossbar_mode": raw.get("crossbar-mode", False),
    }


EDGE_TOP = 0
EDGE_RIGHT = 1
EDGE_BOTTOM = 2
EDGE_LEFT = 3
EDGE_NAMES = {
    EDGE_TOP: "top",
    EDGE_RIGHT: "right",
    EDGE_BOTTOM: "bottom",
    EDGE_LEFT: "left",
}


def edge_die_ports(side: int) -> Dict[int, List[int]]:
    """Map each edge die to the I/O port indices it connects to.

    Port numbering: 0..side-1 = top edge (L->R), side..2*side-1 = right edge (T->B),
    2*side..3*side-1 = bottom edge (R->L), 3*side..4*side-1 = left edge (B->T).
    Corner dies get 2 ports.
    """
    ports: Dict[int, List[int]] = defaultdict(list)
    for c in range(side):
        ports[c].append(c)  # top row
    for r in range(side):
        die = r * side + (side - 1)
        ports[die].append(side + r)  # right col
    for c in range(side):
        die = (side - 1) * side + c
        ports[die].append(2 * side + (side - 1 - c))  # bottom row
    for r in range(side):
        die = r * side
        ports[die].append(3 * side + (side - 1 - r))  # left col
    return dict(ports)


def edge_groups(side: int) -> Dict[int, List[int]]:
    """Return edge-local die groups indexed by side id."""
    return {
        EDGE_TOP: [c for c in range(side)],
        EDGE_RIGHT: [r * side + (side - 1) for r in range(side)],
        EDGE_BOTTOM: [(side - 1) * side + c for c in range(side)],
        EDGE_LEFT: [r * side for r in range(side)],
    }


def _edge_local_index(edge: int, die_local: int, side: int) -> int:
    r, c = _die_to_rc(die_local, side)
    if edge == EDGE_TOP:
        return c
    if edge == EDGE_RIGHT:
        return r
    if edge == EDGE_BOTTOM:
        return c
    if edge == EDGE_LEFT:
        return r
    raise ValueError(f"Unknown edge id: {edge}")


def _edge_die(edge: int, local_index: int, side: int) -> int:
    if edge == EDGE_TOP:
        return local_index
    if edge == EDGE_RIGHT:
        return local_index * side + (side - 1)
    if edge == EDGE_BOTTOM:
        return (side - 1) * side + local_index
    if edge == EDGE_LEFT:
        return local_index * side
    raise ValueError(f"Unknown edge id: {edge}")


def _die_to_rc(die_local: int, side: int) -> Tuple[int, int]:
    return divmod(die_local, side)


def _mesh_hops(a: int, b: int, side: int) -> int:
    ar, ac = _die_to_rc(a, side)
    br, bc = _die_to_rc(b, side)
    return abs(ar - br) + abs(ac - bc)


def _edge_distance(die_local: int, side: int) -> int:
    r, c = _die_to_rc(die_local, side)
    return min(r, c, side - 1 - r, side - 1 - c)


def _nearest_edges(die_local: int, side: int) -> List[Tuple[int, int]]:
    """Return all shortest edge exits as ``(edge_id, edge_die_local)``."""
    r, c = _die_to_rc(die_local, side)
    candidates = [
        (EDGE_TOP, r, c),
        (EDGE_RIGHT, side - 1 - c, r * side + (side - 1)),
        (EDGE_BOTTOM, side - 1 - r, (side - 1) * side + c),
        (EDGE_LEFT, c, r * side),
    ]
    best = min(dist for _edge, dist, _die in candidates)
    return [(edge, edge_die_local) for edge, dist, edge_die_local in candidates if dist == best]


def _split_bytes(nbytes: float, n: int) -> Iterable[float]:
    if n <= 0:
        return []
    return [float(nbytes) / n for _ in range(n)]


def _empty_edge_traffic(side: int) -> Dict[int, Dict[int, Dict[str, float]]]:
    return {
        edge: {d: {"exit": 0.0, "transit": 0.0} for d in dies}
        for edge, dies in edge_groups(side).items()
    }


def _empty_wafer_edge_traffic(
    num_wafers: int,
    side: int,
) -> Dict[int, Dict[int, Dict[int, Dict[str, float]]]]:
    return {w: _empty_edge_traffic(side) for w in range(num_wafers)}


def _add_edge_traffic_from_phase(
    traffic: Dict[int, Dict[int, Dict[int, Dict[str, float]]]],
    matrix: Dict[int, Dict[int, int]],
    dies_per_wafer: int,
    side: int,
) -> None:
    """Project cross-wafer phase traffic to nearest source/destination edges."""
    for src, dsts in matrix.items():
        sw, sl = divmod(src, dies_per_wafer)
        for dst, nbytes in dsts.items():
            dw, dl = divmod(dst, dies_per_wafer)
            if sw == dw or nbytes <= 0:
                continue

            src_edges = _nearest_edges(sl, side)
            for (edge, edge_die_local), share in zip(src_edges, _split_bytes(nbytes, len(src_edges))):
                hops = _mesh_hops(sl, edge_die_local, side)
                traffic[sw][edge][edge_die_local]["exit"] += share
                traffic[sw][edge][edge_die_local]["transit"] += share * hops

            dst_edges = _nearest_edges(dl, side)
            for edge, edge_die_local in dst_edges:
                share = float(nbytes) / len(dst_edges)
                hops = _mesh_hops(dl, edge_die_local, side)
                traffic[dw][edge][edge_die_local]["exit"] += share
                traffic[dw][edge][edge_die_local]["transit"] += share * hops


def build_projected_edge_traffic(
    dispatch_matrix: Dict[int, Dict[int, int]],
    combine_matrix: Dict[int, Dict[int, int]],
    dies_per_wafer: int,
    side: int,
    num_wafers: int | None = None,
) -> Dict[int, Dict[int, Dict[int, Dict[str, float]]]]:
    """Compute projected edge-local exit/transit traffic for an epoch."""
    if num_wafers is None:
        max_rank = -1
        for matrix in (dispatch_matrix, combine_matrix):
            for src, dsts in matrix.items():
                max_rank = max(max_rank, int(src), *(int(d) for d in dsts))
        num_wafers = max_rank // dies_per_wafer + 1 if max_rank >= 0 else 1
    traffic = _empty_wafer_edge_traffic(int(num_wafers), side)
    _add_edge_traffic_from_phase(traffic, dispatch_matrix, dies_per_wafer, side)
    _add_edge_traffic_from_phase(traffic, combine_matrix, dies_per_wafer, side)
    return traffic


def build_edge_traffic(
    dispatch_matrix: Dict[int, Dict[int, int]],
    combine_matrix: Dict[int, Dict[int, int]],
    dies_per_wafer: int,
    side: int,
) -> Dict[int, Dict[str, int]]:
    """Backward-compatible aggregate view of projected edge traffic."""
    projected = build_projected_edge_traffic(dispatch_matrix, combine_matrix, dies_per_wafer, side)
    traffic: Dict[int, Dict[str, float]] = defaultdict(lambda: {"exit": 0.0, "transit": 0.0})
    for wafer_data in projected.values():
        for edge_data in wafer_data.values():
            for die, vals in edge_data.items():
                traffic[die]["exit"] += vals["exit"]
                traffic[die]["transit"] += vals["transit"]
    return {d: {"exit": int(v["exit"]), "transit": int(v["transit"])} for d, v in traffic.items()}


def _round_to_total(alloc: Dict[int, int], active: List[int], total_ports: int) -> Dict[int, int]:
    diff = total_ports - sum(alloc.values())
    if not active:
        return alloc
    for i in range(abs(diff)):
        d = active[i % len(active)]
        if diff > 0:
            alloc[d] += 1
        elif alloc[d] > 1:
            alloc[d] -= 1
    return alloc


def allocate_crossbar_ports(
    method: str,
    edge_traffic: Dict[int, Dict[str, int]],
    side: int,
    num_io_ports: int,
    per_port_bw_gbps: float,
    beta: float = 0.5,
    rho: float = 0.4,
) -> Dict[int, Dict]:
    """Allocate a finite set of logical ports within one crossbar scope."""
    total_ports = num_io_ports
    edge_dies = sorted(edge_traffic.keys())

    exit_demand = {d: float(edge_traffic.get(d, {}).get("exit", 0)) for d in edge_dies}
    transit_demand = {d: float(edge_traffic.get(d, {}).get("transit", 0)) for d in edge_dies}
    active = [d for d in edge_dies if exit_demand[d] > 0]

    if not active:
        return {
            d: {"ports": 0, "exit_bw_gbps": 0.0, "service_ns": 0.0}
            for d in edge_dies
        }

    if method == "naive_crossbar":
        port_alloc = {d: 0 for d in edge_dies}
        for d in active:
            port_alloc[d] = 1
        for _ in range(max(total_ports - len(active), 0)):
            best = max(active, key=lambda d: exit_demand[d] / (port_alloc[d] + 1.0))
            port_alloc[best] += 1

    elif method == "dynamic_bw":
        demand = {d: exit_demand[d] + beta * transit_demand[d] for d in edge_dies}
        port_alloc = {d: 0 for d in edge_dies}
        for d in active:
            port_alloc[d] = 1
        for _ in range(max(total_ports - len(active), 0)):
            best = max(active, key=lambda d: demand[d] / (port_alloc[d] + 1.0))
            port_alloc[best] += 1
    else:
        raise ValueError(f"Unknown crossbar method: {method}")

    result = {}
    for d in edge_dies:
        n_ports = port_alloc.get(d, 0)
        exit_bytes = exit_demand[d]
        eff_bw = n_ports * per_port_bw_gbps
        if n_ports > 0 and exit_bytes > 0:
            service_ns = exit_bytes / max(eff_bw, 1e-9)
        else:
            service_ns = 0.0
        result[d] = {
            "ports": n_ports,
            "exit_bw_gbps": eff_bw,
            "service_ns": service_ns,
        }

    if method == "dynamic_bw":
        avg_load_per_port = sum(exit_demand.values()) / max(total_ports, 1)
        reusable_pool = 0.0
        overload = {}
        for d in edge_dies:
            cap = port_alloc.get(d, 0) * avg_load_per_port
            if exit_demand[d] < cap:
                reusable_pool += cap - exit_demand[d]
            elif exit_demand[d] > cap:
                overload[d] = exit_demand[d] - cap
        sum_overload = sum(overload.values())
        if reusable_pool > 0 and sum_overload > 0:
            for d in edge_dies:
                base_ns = result[d]["service_ns"]
                if base_ns <= 0:
                    continue
                if d in overload:
                    share = reusable_pool * overload[d] / sum_overload
                    bonus_ns = rho * share / max(per_port_bw_gbps, 1e-9)
                    result[d]["service_ns"] = max(0.85 * base_ns, base_ns - bonus_ns)

    return result


def allocate_edge_local_ports(
    method: str,
    projected_traffic: Dict[int, Dict[int, Dict[int, Dict[str, float]]]],
    side: int,
    per_port_bw_gbps: float,
    beta: float = 0.5,
    rho: float = 0.4,
) -> Dict[int, Dict[int, Dict[int, Dict]]]:
    """Allocate ports independently for each edge-local crossbar."""
    return {
        wafer: {
            edge: allocate_crossbar_ports(
                method,
                edge_traffic,
                side,
                side,
                per_port_bw_gbps,
                beta=beta,
                rho=rho,
            )
            for edge, edge_traffic in wafer_traffic.items()
        }
        for wafer, wafer_traffic in projected_traffic.items()
    }


def _phase_projected_traffic(
    matrix: Dict[int, Dict[int, int]],
    dies_per_wafer: int,
    side: int,
    num_wafers: int | None = None,
) -> Dict[int, Dict[int, Dict[int, float]]]:
    if num_wafers is None:
        max_rank = -1
        for src, dsts in matrix.items():
            max_rank = max(max_rank, int(src), *(int(d) for d in dsts))
        num_wafers = max_rank // dies_per_wafer + 1 if max_rank >= 0 else 1
    traffic: Dict[int, Dict[int, Dict[int, float]]] = {
        wafer: {edge: {d: 0.0 for d in dies} for edge, dies in edge_groups(side).items()}
        for wafer in range(int(num_wafers))
    }
    for src, dsts in matrix.items():
        sw, sl = divmod(src, dies_per_wafer)
        for dst, nbytes in dsts.items():
            dw, dl = divmod(dst, dies_per_wafer)
            if sw == dw or nbytes <= 0:
                continue
            src_edges = _nearest_edges(sl, side)
            for (edge, edge_die_local), share in zip(src_edges, _split_bytes(nbytes, len(src_edges))):
                traffic[sw][edge][edge_die_local] += share
            dst_edges = _nearest_edges(dl, side)
            for (edge, edge_die_local), share in zip(dst_edges, _split_bytes(nbytes, len(dst_edges))):
                traffic[dw][edge][edge_die_local] += share
    return traffic


def _apply_phase_service_times(
    phase_exit: Dict[int, Dict[int, Dict[int, float]]],
    port_alloc: Dict[int, Dict[int, Dict[int, Dict]]],
    per_port_bw_gbps: float,
    method: str,
    rho: float,
) -> Dict[int, Dict[int, Dict[int, Dict]]]:
    out: Dict[int, Dict[int, Dict[int, Dict]]] = {}
    for wafer, wafer_exit in phase_exit.items():
        out[wafer] = {}
        for edge, exits in wafer_exit.items():
            edge_alloc = port_alloc.get(wafer, {}).get(edge, {})
            total_ports = sum(int(v.get("ports", 0)) for v in edge_alloc.values())
            total_exit = sum(exits.values())
            avg_load_per_port = total_exit / max(total_ports, 1)
            reusable_pool = 0.0
            overload: Dict[int, float] = {}
            for die, exit_bytes in exits.items():
                ports = int(edge_alloc.get(die, {}).get("ports", 0))
                cap = ports * avg_load_per_port
                if exit_bytes < cap:
                    reusable_pool += cap - exit_bytes
                elif exit_bytes > cap:
                    overload[die] = exit_bytes - cap
            sum_overload = sum(overload.values())

            out[wafer][edge] = {}
            for die, exit_bytes in exits.items():
                ports = int(edge_alloc.get(die, {}).get("ports", 0))
                service_ns = exit_bytes / max(ports * per_port_bw_gbps, 1e-9) if ports > 0 and exit_bytes > 0 else 0.0
                if method == "dynamic_bw" and service_ns > 0 and die in overload and sum_overload > 0 and reusable_pool > 0:
                    share = reusable_pool * overload[die] / sum_overload
                    bonus_ns = rho * share / max(per_port_bw_gbps, 1e-9)
                    service_ns = max(0.85 * service_ns, service_ns - bonus_ns)
                out[wafer][edge][die] = {
                    "ports": ports,
                    "exit_bytes": exit_bytes,
                    "exit_bw_gbps": ports * per_port_bw_gbps,
                    "service_ns": service_ns,
                }
    return out


def _allocation_summary(
    projected_traffic: Dict[int, Dict[int, Dict[int, Dict[str, float]]]],
    port_alloc: Dict[int, Dict[int, Dict[int, Dict]]],
    topk: int = 16,
) -> Dict[str, object]:
    edge_rows = []
    allocation_rows = []
    for wafer, wafer_traffic in sorted(projected_traffic.items()):
        for edge, edge_traffic in sorted(wafer_traffic.items()):
            alloc = port_alloc.get(wafer, {}).get(edge, {})
            total_exit = sum(vals["exit"] for vals in edge_traffic.values())
            total_transit = sum(vals["transit"] for vals in edge_traffic.values())
            total_ports = sum(int(vals.get("ports", 0)) for vals in alloc.values())
            service_ns = max((float(vals.get("service_ns", 0.0)) for vals in alloc.values()), default=0.0)
            edge_rows.append({
                "wafer": wafer,
                "edge": EDGE_NAMES.get(edge, str(edge)),
                "edge_id": edge,
                "exit_bytes": round(total_exit, 3),
                "transit_byte_hops": round(total_transit, 3),
                "ports_allocated": total_ports,
                "service_ns": round(service_ns, 6),
            })
            for die, vals in sorted(edge_traffic.items()):
                a = alloc.get(die, {})
                allocation_rows.append({
                    "wafer": wafer,
                    "edge": EDGE_NAMES.get(edge, str(edge)),
                    "edge_id": edge,
                    "die_local": die,
                    "exit_bytes": round(vals["exit"], 3),
                    "transit_byte_hops": round(vals["transit"], 3),
                    "ports": int(a.get("ports", 0)),
                    "service_ns": round(float(a.get("service_ns", 0.0)), 6),
                })
    allocation_rows.sort(key=lambda r: (r["exit_bytes"], r["transit_byte_hops"]), reverse=True)
    return {
        "edge_crossbars": edge_rows,
        "top_allocations": allocation_rows[:topk],
        "num_edge_crossbars": len(edge_rows),
        "max_edge_service_ns": max((r["service_ns"] for r in edge_rows), default=0.0),
        "max_edge_exit_bytes": max((r["exit_bytes"] for r in edge_rows), default=0.0),
    }


def _phase_rail_analytical_ns(
    matrix: Dict[int, Dict[int, int]],
    cfg: dict,
    edge_congestion_factor: float = 1.0,
) -> float:
    """Compute analytical latency for one phase using the rail-only model.

    Mirrors build_window_traffic_matrix.py: per-message path latency accumulated
    per destination rank, phase time = max across ranks. Includes edge-port
    congestion model: all cross-wafer traffic from a wafer must serialize
    through limited edge ports, creating queuing delay.

    Cross-wafer path: mesh-to-edge + 2 switch edges + mesh-from-edge.
    Same-wafer path: mesh hops only.
    Edge congestion: wafer_exit_bytes / (num_edge_ports * switch_bw) per wafer,
    applied once per source wafer to its busiest receiving rank.
    """
    dpw = cfg["dies_per_wafer"]
    side = cfg["side"]
    num_io_ports = cfg["num_io_ports"]
    mesh_bw = cfg["mesh_bw_gbps"]
    switch_bw = cfg["per_port_bw_gbps"]
    switch_lat_ns = cfg["switch_latency_ns"]
    mesh_lat_ns = cfg["mesh_latency_ns"]

    rank_recv_ns = defaultdict(float)
    # Per-wafer cross-wafer exit bytes (for edge congestion)
    wafer_exit_bytes = defaultdict(float)
    # Per-wafer busiest destination rank (for applying congestion)
    wafer_busiest_dst = {}
    wafer_busiest_bytes = defaultdict(float)
    # Track which source dies have cross-wafer traffic per wafer
    wafer_cross_wafer_srcs = defaultdict(set)

    # First pass: compute per-message latencies and accumulate wafer exit bytes
    for src, dsts in matrix.items():
        sw, sl = divmod(src, dpw)
        for dst, nbytes in dsts.items():
            dw, dl = divmod(dst, dpw)
            if sw == dw:
                hops = _mesh_hops(sl, dl, side)
                lat = hops * (mesh_lat_ns + nbytes / max(mesh_bw, 1e-9))
                rank_recv_ns[dst] += lat
            else:
                wafer_exit_bytes[sw] += nbytes
                wafer_cross_wafer_srcs[sw].add(sl)
                # Track busiest dst per wafer
                if nbytes > wafer_busiest_bytes[sw]:
                    wafer_busiest_bytes[sw] = nbytes
                    wafer_busiest_dst[sw] = dst
                # Cross-wafer: mesh-to-edge + switch edges + mesh-from-edge
                src_hops = _edge_distance(sl, side)
                dst_hops = _edge_distance(dl, side)
                mesh_to_edge = max(src_hops, 1) * (mesh_lat_ns + nbytes / max(mesh_bw, 1e-9))
                switch_lat = 2 * (switch_lat_ns + nbytes / max(switch_bw, 1e-9))
                mesh_from_edge = max(dst_hops, 1) * (mesh_lat_ns + nbytes / max(mesh_bw, 1e-9))
                rank_recv_ns[dst] += mesh_to_edge + switch_lat + mesh_from_edge

    # Apply edge-port congestion: once per source wafer
    # wafer_exit_bytes / (num_edge_ports * switch_bw) * congestion_factor
    if edge_congestion_factor > 0:
        for sw, load_bytes in wafer_exit_bytes.items():
            congestion_ns = (load_bytes
                             / (num_io_ports * max(switch_bw, 1e-9))
                             * edge_congestion_factor)
            # Apply to busiest destination rank for this wafer
            busiest = wafer_busiest_dst.get(sw)
            if busiest is not None:
                rank_recv_ns[busiest] += congestion_ns

    return max(rank_recv_ns.values(), default=0.0)


def _phase_crossbar_ns(
    matrix: Dict[int, Dict[int, int]],
    phase_service: Dict[int, Dict[int, Dict[int, Dict]]],
    cfg: dict,
) -> float:
    """Compute analytical latency for one phase using the crossbar model.

    This keeps the TP2 traffic-matrix timing basis: message latencies are
    accumulated at receiving ranks.  For cross-wafer messages, mesh movement
    and fixed switch/crossbar latency are accumulated per message, while the
    external I/O serialization is replaced by one edge-local finite-port
    service bottleneck.
    """
    dpw = cfg["dies_per_wafer"]
    side = cfg["side"]
    mesh_bw = cfg["mesh_bw_gbps"]
    switch_lat_ns = cfg["switch_latency_ns"]
    crossbar_lat_ns = cfg.get("crossbar_latency_ns", 20.0)
    mesh_lat_ns = cfg["mesh_latency_ns"]

    same_wafer_rank_ns = defaultdict(float)
    cross_wafer_rank_base_ns = defaultdict(float)
    has_cross_wafer = False

    for src, dsts in matrix.items():
        sw, sl = divmod(src, dpw)
        for dst, nbytes in dsts.items():
            dw, dl = divmod(dst, dpw)
            if sw == dw:
                hops = _mesh_hops(sl, dl, side)
                lat = hops * (mesh_lat_ns + nbytes / max(mesh_bw, 1e-9))
                same_wafer_rank_ns[dst] += lat
            else:
                has_cross_wafer = True
                src_hops = _edge_distance(sl, side)
                dst_hops = _edge_distance(dl, side)
                mesh_to_edge = src_hops * (mesh_lat_ns + nbytes / max(mesh_bw, 1e-9))
                mesh_from_edge = dst_hops * (mesh_lat_ns + nbytes / max(mesh_bw, 1e-9))
                external_fixed = 2.0 * switch_lat_ns + crossbar_lat_ns
                cross_wafer_rank_base_ns[dst] += mesh_to_edge + external_fixed + mesh_from_edge

    same_wafer_ns = max(same_wafer_rank_ns.values(), default=0.0)

    if not has_cross_wafer:
        return same_wafer_ns

    external_service_ns = max(
        (
            edge_service["service_ns"]
            for wafer_service in phase_service.values()
            for edge_services in wafer_service.values()
            for edge_service in edge_services.values()
            if edge_service["service_ns"] > 0
        ),
        default=0.0,
    )
    cross_wafer_ns = max(cross_wafer_rank_base_ns.values(), default=0.0) + external_service_ns

    return max(same_wafer_ns, cross_wafer_ns)


def crossbar_comm_us_per_step(
    dispatch_matrices: List[Dict[int, Dict[int, int]]],
    combine_matrices: List[Dict[int, Dict[int, int]]],
    cfg: dict,
    method: str,
    batch_scale: float,
    window_size: int,
    beta: float = 0.5,
    rho: float = 0.4,
    crossbar_latency_ns: float | None = None,
    collect_debug: bool = False,
) -> Dict[str, float]:
    """Compute per-step communication latency using crossbar model.

    The crossbar model replaces the per-message switch-edge serialization with
    the crossbar's demand-allocated per-source-die serialization bottleneck.

    Also computes the rail-only analytical latency for calibration: the ratio
    rail_analytical / astra_sim gives the congestion scaling factor that accounts
    for effects the analytical model misses (pipelining, queuing, collectives).

    Returns {
        "dispatch_us": float, "combine_us": float, "comm_us": float,
        "rail_analytical_dispatch_us": float, "rail_analytical_combine_us": float,
        "rail_analytical_comm_us": float,
    }
    """
    per_port_bw = cfg["per_port_bw_gbps"]
    dpw = cfg["dies_per_wafer"]
    side = cfg["side"]
    num_wafers = int(cfg.get("num_wafers", 1))
    if crossbar_latency_ns is not None:
        cfg = dict(cfg)
        cfg["crossbar_latency_ns"] = float(crossbar_latency_ns)

    dispatch_us_list = []
    combine_us_list = []
    rail_dispatch_us_list = []
    rail_combine_us_list = []
    debug_windows = []

    for w_idx, (dispatch_mat, combine_mat) in enumerate(zip(dispatch_matrices, combine_matrices)):
        projected_traffic = build_projected_edge_traffic(dispatch_mat, combine_mat, dpw, side, num_wafers=num_wafers)
        port_alloc = allocate_edge_local_ports(
            method, projected_traffic, side, per_port_bw, beta=beta, rho=rho,
        )
        dispatch_phase_exit = _phase_projected_traffic(dispatch_mat, dpw, side, num_wafers=num_wafers)
        combine_phase_exit = _phase_projected_traffic(combine_mat, dpw, side, num_wafers=num_wafers)
        dispatch_service = _apply_phase_service_times(dispatch_phase_exit, port_alloc, per_port_bw, method, rho)
        combine_service = _apply_phase_service_times(combine_phase_exit, port_alloc, per_port_bw, method, rho)

        # Crossbar analytical latency
        dispatch_ns = _phase_crossbar_ns(dispatch_mat, dispatch_service, cfg)
        combine_ns = _phase_crossbar_ns(combine_mat, combine_service, cfg)
        dispatch_us_list.append(dispatch_ns / 1000.0)
        combine_us_list.append(combine_ns / 1000.0)

        # Rail analytical latency (for calibration)
        rail_dispatch_ns = _phase_rail_analytical_ns(dispatch_mat, cfg)
        rail_combine_ns = _phase_rail_analytical_ns(combine_mat, cfg)
        rail_dispatch_us_list.append(rail_dispatch_ns / 1000.0)
        rail_combine_us_list.append(rail_combine_ns / 1000.0)

        if collect_debug:
            summary = _allocation_summary(projected_traffic, port_alloc)
            summary.update({
                "window_index": w_idx,
                "dispatch_ns": round(dispatch_ns, 6),
                "combine_ns": round(combine_ns, 6),
                "rail_dispatch_ns": round(rail_dispatch_ns, 6),
                "rail_combine_ns": round(rail_combine_ns, 6),
                "dispatch_max_phase_service_ns": max(
                    (
                        vals["service_ns"]
                        for wafer_service in dispatch_service.values()
                        for edge_service in wafer_service.values()
                        for vals in edge_service.values()
                    ),
                    default=0.0,
                ),
                "combine_max_phase_service_ns": max(
                    (
                        vals["service_ns"]
                        for wafer_service in combine_service.values()
                        for edge_service in wafer_service.values()
                        for vals in edge_service.values()
                    ),
                    default=0.0,
                ),
            })
            debug_windows.append(summary)

    num_windows = len(dispatch_us_list) or 1
    avg_dispatch_us = sum(dispatch_us_list) / num_windows / max(window_size, 1)
    avg_combine_us = sum(combine_us_list) / num_windows / max(window_size, 1)
    rail_avg_dispatch_us = sum(rail_dispatch_us_list) / num_windows / max(window_size, 1)
    rail_avg_combine_us = sum(rail_combine_us_list) / num_windows / max(window_size, 1)

    result = {
        "dispatch_us": round(avg_dispatch_us, 6),
        "combine_us": round(avg_combine_us, 6),
        "comm_us": round(avg_dispatch_us + avg_combine_us, 6),
        "rail_analytical_dispatch_us": round(rail_avg_dispatch_us, 6),
        "rail_analytical_combine_us": round(rail_avg_combine_us, 6),
        "rail_analytical_comm_us": round(rail_avg_dispatch_us + rail_avg_combine_us, 6),
        "crossbar_latency_ns": round(float(cfg.get("crossbar_latency_ns", 20.0)), 6),
        "external_ports_per_edge": side,
        "per_port_bw_gbps": round(per_port_bw, 6),
    }
    if collect_debug:
        result["debug_windows"] = debug_windows
    return result
