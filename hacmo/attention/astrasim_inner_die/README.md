# ASTRA-Sim Inner-Die Memory Pipeline

This folder runs a memory-centric inner-die attention experiment where:

- compute time uses the analytical formula output (`T_comp_us`) from `profile_driven_lru_step`
- HBM-like accesses are modeled by ASTRA-Sim `MEM_LOAD_NODE` with remote-memory backend
- remote-SRAM-like accesses are modeled by ASTRA-Sim point-to-point communication on a mesh topology

## Why 19x19 mesh

Current ASTRA-Sim Mesh implementation requires a perfect-square NPU count.  
To approximate a `20x18=360` core array, this pipeline uses a `19x19=361` mesh and runs all 361 ranks.

## Run

```bash
cd hacmo/attention
bash astrasim_inner_die/scripts/run_inner_die_astrasim_pipeline.sh
```

## Output

Main output directory:

- `outputs/astrasim_inner_die_lru_step_full_361/`
  - `csv/inner_die_astrasim_summary.csv`
  - `raw_summary.json`
  - `logs/` (astra-sim logs per case)
  - `parsed/` (parsed wall/comm/remote-mem per case)

## Notes

- This is a first bridge to ASTRA-Sim memory modeling, not a cycle-accurate microarchitecture model.
- `MEM_LOAD_NODE` latency comes from remote-memory backend config (`per_npu_hbm_like.json`).
- In this ASTRA-Sim branch, mixing `REMOTE_MEM` and `COMM` in one run can crash stats post-processing.
  Therefore each case is decomposed into two runs:
  1) HBM-only (`MEM_LOAD`) and 2) remote-SRAM-only (`COMM`), then summed in the CSV as `astra_wall_us`.

## Fragmentation-aware HBM option

The ASTRA-Sim analytical remote-memory backend now supports optional fragmentation-aware timing:

- `remote-mem-effective-payload-bytes`
- `remote-mem-transfer-granularity-bytes`

Runtime model:

- `txn_count = ceil(useful_bytes / effective_payload_bytes)`
- `runtime = txn_count * remote_mem_latency + transferred_bytes / remote_mem_bw`
- if `transfer_granularity_bytes > 0`, then `transferred_bytes = txn_count * transfer_granularity_bytes`

Example config:

- `configs/remote_memory/per_npu_hbm_fragmented_4k_25p.json`
