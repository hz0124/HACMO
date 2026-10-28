# Architecture and source map

The public simulator interface is `supernode_sim.api.simulate_layer(config)`.
JSON configs are loaded by `supernode_sim.config`; typed objects are defined in
`supernode_sim.types`. The CLI is `python -m supernode_sim simulate`.

| Layer | Entry points | Responsibility |
| --- | --- | --- |
| Simulator | `supernode_sim/api.py`, `backends/`, `tracegen/` | Execute configured analytic, ASTRA, and Ramulator workloads |
| Attention | `hacmo/attention/tp1_core/` | Profile loading, placement, routing, and trace-driven attention evaluation |
| MoE | `hacmo/moe/tp2_core/` | Mapping inputs, static placement, causal runtime scheduling, traffic matrices |
| Interconnect | `hacmo/interconnect/scripts/`, `supernode_sim/plugins/crossbar.py` | Rail/crossbar topology and lane allocation |
| Overall | `hacmo/overall/scripts/run_overall_rebuttal.py` | Consume method outputs and support curves to calculate model performance |
| Reproduction | `ae/run_full.sh`, `ae/build_paper_results.py` | Orchestrate experiments, compare CSVs, audit claims, render plots |

`ae/run_full.sh` is the supported conference command interface. Its producer
order is TP1 and TP2, followed by TP3, Overall, and result aggregation. Overall
consumes the method evidence; it does not define the method implementations.

The modified ASTRA-Sim tree is pinned under `astra-sim/`; Ramulator2 is under
`third_party/ramulator2/`. `scripts/build_simulators.sh` compiles the generic
ASTRA executables, applies the conference Mesh2D build overlay, and builds an
isolated Ramulator2 tree with the supplied adapters. See
[third-party notices](THIRD_PARTY.md) for base revisions and modifications.

## Inputs and outputs

- `ae/expected-results/`: immutable author-generated tables and metadata.
- `ae/data/`: downloaded profiles, support tables, and separately obtained MoE
  traces; ignored by Git.
- `ae/work/`: generated simulation traces, logs, CSVs, plots, and audits;
  ignored by Git.
- `hacmo/DSE/`: optional architecture exploration. Its MoE path is distinct
  from the camera-ready TP2 serving-epoch protocol.
- `hacmo/hardware_analysis/`: optional hardware-analysis scripts and tables.

Some optional exploratory helpers retain historical default input pack names.
Supply their explicit input arguments when using them with new experiments.
The supported `ae/` workflows supply their dependencies explicitly.
