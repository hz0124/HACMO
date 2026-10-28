# HACMO

**Heterogeneity-Aware Cross-Layer Mapping Optimization for Sparse LLMs on Layered Supernodes** · MICRO 2026

Hanbin Zhang, Jun Liu, Rongquan Ni, Yuxuan Zhang, and Guohao Dai.

HACMO models sparse LLM inference across rack, wafer, and core levels. It
combines spatially aware sparse-attention mapping, global cost-aware MoE expert
placement, and traffic-aware wafer-edge interconnect allocation. The simulator
integrates modified ASTRA-Sim and Ramulator2 backends with Python mapping and
experiment workflows.

This repository contains the conference implementation derived from the
immutable [camera-ready-v1 artifact](https://doi.org/10.5281/zenodo.22658850).
The required CPU workflow covers Figures 9–12. See
[reproduction scope and version correspondence](docs/RELEASE.md) for the
distinction between simulated, analytical, measured, and archived inputs.

## Quick start

Use Linux x86-64, Python 3.10–3.12, CMake 3.x, and a C++17/20 toolchain. Ubuntu 22.04/24.04
is the target platform. Full reproduction was designed for 32 CPU cores,
64 GB RAM, and 100 GB free disk; actual run requirements are recorded in
[validation](docs/VALIDATION.md). No GPU or model weights are needed for the
required workflow.

```bash
git clone https://github.com/hz0124/HACMO.git
cd HACMO
sudo apt-get update
sudo apt-get install -y build-essential cmake git ninja-build patch \
  pkg-config protobuf-compiler libprotobuf-dev libyaml-cpp-dev \
  python3-dev python3-venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Commands below run from this checkout. The editable installation uses the
bundled configs and simulator source trees. C++ dependency sources are bundled;
simulator compilation does not need GitHub access. Installing Python packages
and downloading experiment data require network access.

Run a small analytical simulation:

```bash
python -m supernode_sim simulate \
  --config supernode_sim/configs/deepseek_v32_attention_smoke.json \
  --output ae/work/quick/attention_smoke.json
```

Build the simulator stack and check the installation:

```bash
JOBS=8 bash scripts/build_simulators.sh
bash ae/run_quick.sh
```

The quick check runs tests, a smoke workload, and a twelve-claim audit using
archived reference tables. Full reproduction requires the input data below.

## Reproduce the paper

Download `hacmo-ae-data-camera-ready-v1.tar.gz` from the
[fixed Zenodo record](https://zenodo.org/records/22658850). Verify its SHA256:

```text
6a2cdbfbcde64720fe10c903350e6afc1cd359fed0f616bf6e3b4118212e0d55
```

Install the author-provided profiles and support tables, then fetch the pinned
third-party MoE trace subset after accepting its original access conditions:

```bash
bash ae/setup_data.sh /path/to/hacmo-ae-data-camera-ready-v1.tar.gz
# Set HF_TOKEN in your shell after accepting the dataset's access conditions.
python ae/download_moe_traces.py --profile camera-ready-v1 --output ae/data/moe
ulimit -n 8192
python ae/check_env.py --full
bash ae/run_full.sh all
```

The downloader selects **943 requests** at a fixed dataset revision. Raw
third-party traces remain with their original distributor. See
[data provenance](docs/DATA.md).

| Figure | Workflow | Output under `ae/work/` |
| --- | --- | --- |
| 9 | Overall performance | `overall/data/simulated_layers.csv` |
| 10 | Sparse-attention mapping (TP1) | `tp1/final/csv/tp1_hybrid_cp2_summary.csv` |
| 11 | Expert mapping (TP2) | `tp2/camera_ready_pack/paper_figures/paper_figure_summary.csv` |
| 12 | Interconnect allocation (TP3) | `tp3/final/tp3_metrics.csv` |

Plots and the twelve headline-value checks appear in `ae/work/paper-results/`.
The workflow compares generated tables with `ae/expected-results/`, preserves
the reference tables, and exits nonzero on failure. Read the
[reproduction guide](docs/REPRODUCING.md) for stage dependencies, expected
differences, and the source of each input.

## Repository layout

```text
supernode_sim/     Shared simulation API, backends, trace generation, crossbar model
hacmo/
  attention/      Spatially aware sparse-attention mapping (TP1)
  moe/            Static and dynamic expert mapping (TP2)
  interconnect/   Traffic-aware interconnect experiments (TP3)
  overall/        Aggregation of model and topology performance
  DSE/            Optional architecture exploration
  hardware_analysis/  Optional hardware analysis and source tables
ae/               Reproduction entry points, reference tables, result auditing
scripts/          Simulator builds and protobuf generation
astra-sim/        Pinned modified ASTRA-Sim with expanded dependencies
third_party/      Pinned Ramulator2 source
docs/             Architecture, reproduction, data, and release provenance
```

Tests stay beside the modules they verify. Third-party source is expanded in
the checkout; no submodule initialization is required. The ASTRA-Sim location
is retained to preserve its build interfaces.

## Extend or contribute

Start with the [architecture guide](docs/repository-map.md) and module READMEs.
Please include the configuration, command, environment, and relevant error log
in an issue. Changes should keep reference data immutable and pass
`bash ae/run_quick.sh`; simulator or numerical changes also need the affected
full stages and result comparisons. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Citation and licenses

Please cite the MICRO 2026 paper and the fixed artifact version used in your
experiments. [CITATION.cff](CITATION.cff) records the artifact DOI and authors;
the proceedings DOI will be added when verified.

Author-created code: [MIT](LICENSE). Author-generated profiles, support tables,
and reference results: [CC BY 4.0](DATA_LICENSE). Bundled dependencies retain
their own licenses; see [third-party notices](docs/THIRD_PARTY.md).
