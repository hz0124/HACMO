#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HACMO_DIR="$(cd "${ROOT_DIR}/../../.." && pwd)"
ASTRA_SIM_DIR="${HACMO_DIR}/astra-sim"

python3 "${ROOT_DIR}/tools/run_inner_die_astrasim_pipeline.py" \
  --astra_sim_dir "${ASTRA_SIM_DIR}" \
  --ablation_csv "${HACMO_DIR}/hacmo/attention/outputs/profile_driven_lru_step/csv/ablation1_attn_latency_profile_driven.csv" \
  --motivation_csv "${HACMO_DIR}/hacmo/attention/outputs/profile_driven_lru_step/csv/motivation_bw_util_profile_driven.csv" \
  --output_root "${HACMO_DIR}/hacmo/attention/outputs/astrasim_inner_die_lru_step_full_361" \
  --network_config "${ROOT_DIR}/configs/network/inner_die_mesh_19x19.yaml" \
  --system_config "${ROOT_DIR}/configs/system/inner_die_attention_system.json" \
  --remote_memory_config "${ROOT_DIR}/configs/remote_memory/per_npu_hbm_like.json" \
  --num_ranks 361 \
  --active_ranks 361

echo "Done: outputs/astrasim_inner_die_lru_step_full_361"
