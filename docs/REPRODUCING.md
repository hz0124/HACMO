# Reproducing Figures 9--12

Run commands from the repository root on Linux. Generated files stay
under `ae/work/`; archived tables under `ae/expected-results/` are read-only.

## 1. Prepare the environment and data

For a source checkout, install the requirements and build the simulators as
shown in the root README. Then:

```bash
source .venv/bin/activate
bash ae/setup_data.sh /path/to/hacmo-ae-data-camera-ready-v1.tar.gz
# Set HF_TOKEN after accepting the original dataset conditions.
python ae/download_moe_traces.py --profile camera-ready-v1 --output ae/data/moe
ulimit -n 8192
python ae/check_env.py --full
```

The optional Zenodo Linux runtime contains prebuilt binaries and shared
libraries. It still requires a Python environment with `requirements.txt`
installed; it does not contain a complete Conda environment. Run its commands
in its own extracted tree. Do not overlay its older source layout onto this
checkout. See [release correspondence](RELEASE.md).

`check_env.py --full` runs each prebuilt simulator once rather than only
checking that the file exists, and it refuses to pass when the open-file limit
is below 8192. ASTRA-Sim opens one Chakra trace per NPU and the largest
Technical Point 1 array is 68x68 = 4624 NPUs, so the usual shell default of
1024 makes the simulator exit with an empty log. `run_full.sh` and
`run_quick.sh` raise the limit themselves; raise it manually with
`ulimit -n 8192` if you invoke the stages directly.

## 2. Quick integrity and functional check

```bash
bash ae/run_quick.sh
```

This runs the test suites and a simulator smoke test, then reconstructs four
audit plots and twelve headline checks from the archived expected tables under
`ae/work/reference-paper-results/`. It validates installation and result
interpretation but is not the independent paper reproduction.

## 3. Full required reproduction

```bash
bash ae/run_full.sh all
```

This executes TP1, TP2, TP3, Overall, table comparison, plot generation, and
headline-value aggregation. TP1 rebuilds its direct ASTRA-Sim/Ramulator2
source pack before deriving the hybrid `128+CP2` points. It may take several
hours and can create large temporary Chakra traces; reserve approximately
100 GB.

Every stage runs even if an earlier one fails a table comparison. The run ends
with a stage summary and a table-comparison summary so that a single
discrepancy does not hide the later stages, and exits non-zero if anything
failed.

Stages can be rerun in dependency order:

```bash
bash ae/run_full.sh tp1
bash ae/run_full.sh tp2
bash ae/run_full.sh tp3
bash ae/run_full.sh overall
bash ae/run_full.sh results
```

TP2 solves each offline expert placement its result grid needs, replays the
protocol for both topologies at four batch sizes, and writes two compact packs.
They differ only in which batch's runtime window statistics sit at the
topology level: TP3 replays the batch-64 statistics, while the Overall GPU
replay runs at 32 requests per worker. Feeding either consumer the other's pack
is silently wrong rather than an error, so the stage emits both and the later
stages take the one they need. TP3 requires the batch-64 pack; Overall requires
the batch-32 pack. `results` consumes all four final CSV tables.

## 4. What Overall regenerates and what it consumes

The Overall stage recomputes the DeepSeek columns from the Technical Point 1,
2 and 3 output of this workflow. The Qwen3 and Llama4 columns come from
evidence packs bundled in the data archive under
`ae/data/overall-support/`, and the Technical Point 1 service curves come from
the same place. Regenerating those columns would require redistributing two
further third-party trace sets and running two more full simulation campaigns,
so they ship as inputs; `ae/data/overall-support/PROVENANCE.md` states exactly
which paths are and are not regenerated.

This differs from the artifact evaluated for MICRO 2026, where every Qwen,
Llama and Technical Point 2/3 field was frozen from an earlier audited Overall
table. Here the DeepSeek Technical Point 2 and 3 fields are regenerated end to
end, and the bundled Qwen and Llama packs were recomputed for the camera-ready
paper.

The measured H200 baseline table is an input, not an output: the CPU-only
workflow cannot produce it. It is passed explicitly with
`--tp1-gpu-baseline-csv` from `ae/expected-results/tp1/source/`.

## 5. Required outputs

| Figure | Generated final CSV |
| --- | --- |
| 9 | `ae/work/overall/data/simulated_layers.csv` |
| 10 | `ae/work/tp1/final/csv/tp1_hybrid_cp2_summary.csv` |
| 11 | `ae/work/tp2/camera_ready_pack/paper_figures/paper_figure_summary.csv` |
| 12 | `ae/work/tp3/final/tp3_metrics.csv` |

Comparable paper plots and the value audit are written to
`ae/work/paper-results/`. A successful run prints PASS for each table and each
of the twelve claims, then ends with `PASS: full workflow stage=all`.

The Figure 10 headline checks are a 63.5% maximum full-executor reduction,
18.9% mean static reduction, 53.9% mean full-executor reduction, and 2.21x
arithmetic mean of the per-group speedups `C/O`. All four averages/maxima
cover the same 16 groups.

Figure 10 contains 16 groups: contexts 8K/16K, arrays 20x20/68x68, and batch
configurations 16/32/64/`128+CP2`. The first three batch configurations are
direct simulator results. `128+CP2` reuses per-die batch 64 and adds 15.099494
us of analytical CP communication equally to B/C/S/O.
