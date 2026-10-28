#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$(cd "${ROOT_DIR}/../.." && pwd)"
TP1_DIR="${ROOT_DIR}/tp1_core"
GAMMA_JSON="${ROOT_DIR}/support/gamma_calibration_neighbor.json"

usage() {
  cat <<'EOF'
Usage: run_tp1_rebuttal_pipeline.sh [options]

Options:
  --profile-dir PATH          Sparse-attention profile directory (repeatable)
  --support-output-root PATH  HBM traces, service curve, and mapping output
  --final-output-root PATH    TP1 paper-pack output
  -h, --help                  Show this help
EOF
}

PROFILE_DIRS=()
OUT_ROOT="${ROOT_DIR}/outputs/technical_point1_rebuttal_pipeline"
FINAL_ROOT="${ROOT_DIR}/outputs/technical_point1_final_pack_static_traffic_shared_20260612"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile-dir)
      PROFILE_DIRS+=("$2")
      shift 2
      ;;
    --support-output-root)
      OUT_ROOT="$2"
      shift 2
      ;;
    --final-output-root)
      FINAL_ROOT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[TP1-R][ERROR] Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "${GAMMA_JSON}" ]]; then
  echo "[TP1-R][ERROR] Missing gamma calibration file: ${GAMMA_JSON}" >&2
  exit 1
fi

# The release ships the compact processed profiles. Raw H200 measurement
# directories only exist on a machine that ran the profiling, so pass them
# explicitly or through HACMO_TP1_PROFILE_DIRS when starting from raw data.
if [[ ${#PROFILE_DIRS[@]} -eq 0 ]]; then
  if [[ -n "${HACMO_TP1_PROFILE_DIRS:-}" ]]; then
    IFS=: read -r -a PROFILE_DIRS <<< "${HACMO_TP1_PROFILE_DIRS}"
  else
    PROFILE_DIRS=("${REPO_DIR}/ae/data/tp1/profiles")
  fi
fi

mkdir -p "${OUT_ROOT}" "${FINAL_ROOT}"

echo "[TP1-R] Stage 1: core-level static mapping"
python3 "${TP1_DIR}/stage2_core_static_mapping.py" \
  --array_shapes 20x20 32x32 68x68 \
  --output_json "${OUT_ROOT}/core_static_mapping_profile.json"

echo "[TP1-R] Stage 2: HBM trace generation"
python3 "${ROOT_DIR}/tools/generate_hbm_attention_trace.py" \
  --profile_dirs "${PROFILE_DIRS[@]}" \
  --output_root "${OUT_ROOT}/hbm_traces" \
  --batch_sizes 8 16 32 64 128 \
  --context_lens 8192 32768 65536 \
  --patterns profile page_grouped random dense \
  --topk 2048 \
  --selected_vector_bytes 656 \
  --kv_entry_bytes 656 \
  --decode_steps 50 \
  --layer_sample_stride 6 \
  --max_profile_files 3

echo "[TP1-R] Stage 3: HBM service curve"
python3 "${ROOT_DIR}/tools/run_hbm_service_curve.py" \
  --trace_summary "${OUT_ROOT}/hbm_traces/trace_summary.json" \
  --output_root "${OUT_ROOT}/hbm_service_curve" \
  --hbm_peak_tbps 4.8 \
  --dense_util 0.47

echo "[TP1-R] Stage 4: final rebuttal pack"
MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-tp1" python3 "${TP1_DIR}/generate_tp1_final_pack.py" \
  --profile_dirs "${PROFILE_DIRS[@]}" \
  --output_root "${FINAL_ROOT}" \
  --gamma_json "${GAMMA_JSON}" \
  --decode_steps 50 \
  --layer_sample_stride 6 \
  --max_profile_files 3 \
  --total_sram_mb 440 \
  --kv_entry_bytes 656 \
  --selected_kv_bytes_per_token 656 \
  --n_layers 61 \
  --ablation_cache_ratio 0.40 \
  --gpu_vllm_profile neutral \
  --hbm_model service_curve \
  --hbm_service_curve_json "${OUT_ROOT}/hbm_service_curve/hbm_service_curve.json" \
  --hbm_peak_tbps 4.8 \
  --dense_hbm_util 0.47 \
  --sparse_pattern profile \
  --sparse_compute_tokens 2048 \
  --indexer_model deepseek_v32 \
  --indexer_k_bytes_per_token 128 \
  --indexer_scale_bytes_per_token 4 \
  --indexer_n_heads 64 \
  --indexer_head_dim 128 \
  --indexer_compute_tflops 2250 \
  --noc_model "${TP1_NOC_MODEL:-shared}" \
  --noc_bw_scale_mode cols \
  --mapping_profile_json "${OUT_ROOT}/core_static_mapping_profile.json"

echo "[TP1-R] Stage 5: visual inspection report"
MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-tp1" python3 "${ROOT_DIR}/tools/plot_tp1_rebuttal_report.py" \
  --pipeline_root "${OUT_ROOT}" \
  --final_root "${FINAL_ROOT}" \
  --output_dir "${FINAL_ROOT}/visual_report"

echo "[TP1-R] Done."
echo "[TP1-R] Pipeline outputs: ${OUT_ROOT}"
echo "[TP1-R] Final pack: ${FINAL_ROOT}"
echo "[TP1-R] Visual report: ${FINAL_ROOT}/visual_report/README.md"
