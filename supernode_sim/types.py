#!/usr/bin/env python3
"""Typed public data model for SupernodeSim.

The simulator models inference on a layered supernode (rack + wafer + die/core).
Its public contract is SimulationConfig -> SimulationResult.

Design principles:
- Generic MoE parameters stay in MoEConfig (num_experts, topk, hidden_size, etc.)
- Experiment-specific knobs (mapping_method, locality_factor) stay in MoEConfig
  for convenience but are clearly documented as policy-level, not simulator-level.
- Expert placement is accepted as a pre-computed ExpertMapping input, allowing
  external policy code to control mapping without modifying the simulator.
- No hardcoded paths to experiment directories in defaults.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class HardwareConfig:
    """Die-level hardware parameters for the analytic backend."""

    array_rows: int = 20
    array_cols: int = 20
    hbm_edges: tuple[str, ...] = ("top", "bottom")
    hbm_peak_tbps: float = 4.8
    noc_bw_tbps: float = 5.0
    compute_peak_tflops: float = 2250.0
    total_sram_mb: float = 440.0
    near_sram_hops: float = 0.0


@dataclass
class WorkloadConfig:
    """LLM workload description."""

    model_name: str = "DeepSeek-V3.2"
    layer_type: str = "attention"  # "attention" or "moe"
    batch_size: int = 32
    context_len: int = 32768
    topk: int = (
        2048  # For attention: context_len/seq_per_head. For MoE: see MoEConfig.topk.
    )
    selected_vector_bytes: int = 576
    hidden_size: int = 7168
    n_heads: int = 128
    head_dim: int = 576
    n_layers: int = 61
    decode_len: int = 16


@dataclass
class ParallelConfig:
    """Parallelism strategy."""

    cp_degree: int = 1
    horizontal_strategy: str = "dp"  # "dp" or "tp"
    tp_degree_horizontal: int = 1
    batch_coalesce: bool = False
    owner_compute: bool = False


@dataclass
class CachePolicy:
    """KV-cache and prefetching policy for the analytic attention backend.

    This controls SRAM/HBM cache behaviour, NOT expert-to-die mapping.
    Expert mapping is controlled via MoEConfig.mapping_method or by
    passing an explicit ExpertMapping to the astra backend.
    """

    name: str = "baseline"
    cache_policy: str = "none"
    static_mapping: bool = False
    dynamic_prefetch: bool = False
    row_only_static_mapping: bool = True
    prefetch_overlap_us: float = 0.0


# Backward compatibility alias.
MappingPolicy = CachePolicy


@dataclass
class ExpertMapping:
    """A pre-computed expert-to-die placement.

    When provided to the astra backend (via MoEConfig.expert_mapping or
    SimulationConfig.metadata), the simulator uses this mapping directly
    instead of computing one from mapping_method.  This allows external
    policy code (e.g., the ILP solver in hacmo/moe/) to control
    expert placement without modifying the simulator.

    Format: expert_id -> list of die IDs (primary + replicas).
    """

    expert_to_dies: Dict[int, List[int]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {str(k): v for k, v in self.expert_to_dies.items()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExpertMapping":
        return cls(expert_to_dies={int(k): list(v) for k, v in d.items()})


@dataclass
class MoEConfig:
    """MoE layer parameters.

    Generic simulator parameters:
        num_experts, topk, hidden_size, bytes_per_hidden_elem,
        expert_compute_us_per_token, attention_us_per_die, seed

    Trace generation parameters:
        num_requests, window_size

    Policy/experiment parameters (external to simulator core):
        mapping_method, real_freq_json, locality_factor, locality_wafer_shift

    External mapping override:
        expert_mapping -- if provided, supersedes mapping_method
    """

    # Generic MoE parameters.
    num_experts: int = 256
    topk: int = 8
    num_requests: int = 256
    window_size: int = 16
    hidden_size: int = 7168
    bytes_per_hidden_elem: int = 2
    dispatch_bytes_per_hidden_elem: int = 0  # 0 means use bytes_per_hidden_elem.
    combine_bytes_per_hidden_elem: int = 0  # 0 means use bytes_per_hidden_elem.
    expert_compute_us_per_token: float = 0.1
    compute_backend: str = (
        "constant_per_token"  # "constant_per_token" or "roofline_h200".
    )
    expert_intermediate_size: int = 2048
    h200_effective_tflops: float = 1979.0
    h200_hbm_bw_tbps: float = 4.8
    weight_bytes_per_elem: int = 1
    activation_bytes_per_elem: int = 2
    attention_us_per_die: float = 0.0
    seed: int = 42
    # Policy parameters (can be overridden by external code).
    mapping_method: str = "uniform"
    real_freq_json: str = ""
    locality_factor: float = 0.0
    locality_wafer_shift: int = 4
    # External mapping override: if set, use this instead of mapping_method.
    expert_mapping: Dict[str, List[int]] = field(default_factory=dict)
    expert_mapping_json: str = ""
    route_plan_json: str = ""
    window_stats_json: str = ""
    window_src_expert_json: str = ""
    window_mappings_json: str = ""
    exclude_leading_windows: int = 0
    route_via_edge_ports: bool = False
    astra_trace_mode: str = (
        "full"  # "full", "communication_only", "dispatch_only", or "combine_only".
    )


@dataclass
class SupernodeConfig:
    """Multi-wafer supernode topology and ASTRA-Sim backend configuration.

    All path fields default to empty strings and must be provided by the
    caller.  No hardcoded experiment-directory paths.
    """

    num_wafers: int = 16
    dies_per_wafer: int = 16
    network_config: str = ""
    system_config: str = ""
    remote_memory_config: str = ""
    astra_sim_dir: str = ""
    astra_network_backend: str = (
        "congestion_aware"  # "congestion_aware" or "congestion_unaware".
    )
    output_root: str = "supernode_sim/outputs/smoke"
    run_astra: bool = True
    keep_traces: bool = True


@dataclass
class SimulationConfig:
    """Top-level configuration for a single simulation run."""

    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    workload: WorkloadConfig = field(default_factory=WorkloadConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    mapping: CachePolicy = field(default_factory=CachePolicy)
    moe: MoEConfig = field(default_factory=MoEConfig)
    supernode: SupernodeConfig = field(default_factory=SupernodeConfig)
    backend: str = "analytic"  # "analytic", "astra", "astra_tracegen"


@dataclass
class SimulationResult:
    """Output of a single simulation run.

    Universal fields are at the top level.  Backend-specific details go
    into metadata.
    """

    backend: str
    total_latency_us: float
    compute_us: float
    hbm_us: float
    sram_noc_us: float
    communication_us: float
    prefetch_exposed_us: float
    hbm_bytes: float
    sram_local_bytes: float
    sram_near_bytes: float
    sram_remote_bytes: float
    horizontal_link_bytes: float
    vertical_link_bytes: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
