#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}"
export MPLCONFIGDIR="${ROOT}/ae/work/matplotlib"

# The prebuilt simulators load libraries from two different places: protobuf
# from the relocatable environment, and libramulator.so from the Ramulator2
# source tree it was built in. Missing the second one is what stopped the
# artifact-evaluation runs after fourteen minutes.
for lib_dir in "${ROOT}/ae-env/lib" "${ROOT}/build/third_party/ramulator2-src"; do
  if [[ -d "${lib_dir}" ]]; then
    export LD_LIBRARY_PATH="${lib_dir}:${LD_LIBRARY_PATH:-}"
  fi
done

# ASTRA-Sim opens one Chakra trace per NPU and the largest TP1 array is
# 68x68 = 4624 NPUs. Under the usual 1024 default it exits 1 with an empty
# log, so raise the limit here rather than let it fail silently.
REQUIRED_NOFILE=8192
if [[ "$(ulimit -Sn)" -lt "${REQUIRED_NOFILE}" ]]; then
  ulimit -n "${REQUIRED_NOFILE}" 2>/dev/null || true
fi

mkdir -p ae/work "${MPLCONFIGDIR}"
STAGE="${1:-all}"

# TP3 replays the batch-64 window statistics while the Overall GPU replay runs
# at 32 requests per worker. Both read the same path inside a pack, so the TP2
# stage emits one pack per consumer.
TP2_PACK="ae/work/tp2/camera_ready_pack"
TP2_PACK_OVERALL="ae/work/tp2/camera_ready_pack_overall"
MOE_TRACE_ROOT="ae/data/moe/cognitivecomputations/DeepSeek-R1-AWQ/mmlu"

STAGE_NAMES=()
STAGE_STATUS=()
CHECK_NAMES=()
CHECK_STATUS=()

# Every comparison runs and is reported at the end. Stopping at the first
# mismatch hid TP3, Overall and the claim audit from the AE reviewers.
compare() {
  local label="$1"
  shift
  CHECK_NAMES+=("${label}")
  if python ae/compare_csv.py "$@"; then
    CHECK_STATUS+=("PASS")
  else
    CHECK_STATUS+=("FAIL")
  fi
}

run_stage() {
  local name="$1"
  shift
  echo "=== stage ${name} ==="
  STAGE_NAMES+=("${name}")
  if "$@"; then
    STAGE_STATUS+=("OK")
  else
    STAGE_STATUS+=("ERROR")
    echo "ERROR: stage ${name} did not complete; later stages may compare stale data" >&2
  fi
}

run_tp1() {
  bash hacmo/attention/scripts/run_tp1_hybrid_cp_pack.sh \
    --source-pack ae/work/tp1/source \
    --output-root ae/work/tp1/final \
    --profile-dir ae/data/tp1/profiles \
    --preceding-phase-csv ae/data/tp1/tp2_serving_epoch_astra_comm_only_summary.csv \
    --rebuild-source --skip-build || return 1
  compare tp1-source \
    --expected ae/expected-results/tp1/source/tp1_audited_metrics.csv \
    --actual ae/work/tp1/source/csv/tp1_audited_metrics.csv \
    --keys context_group array_rows array_cols batch_size method \
    --metrics mean_exposed_us compute_busy_us_sum hbm_requested_bytes astra_communication_bytes
  compare tp1-hybrid \
    --expected ae/expected-results/tp1/hybrid/tp1_hybrid_cp2_summary.csv \
    --actual ae/work/tp1/final/csv/tp1_hybrid_cp2_summary.csv \
    --keys context array batch_label \
    --metrics analytical_cp_us B_us C_us S_us O_us
}

run_tp2() {
  python hacmo/moe/scripts/run_camera_ready_tp2.py \
    --trace-root "${MOE_TRACE_ROOT}" \
    --work-root ae/work/tp2 \
    --pack-root "${TP2_PACK}" \
    --overall-pack-root "${TP2_PACK_OVERALL}" \
    --stages all || return 1
  compare tp2 \
    --expected ae/expected-results/tp2/paper_figure_summary.csv \
    --actual "${TP2_PACK}/paper_figures/paper_figure_summary.csv" \
    --keys topology batch_size method --metrics total_us compute_us communication_us
}

run_tp3() {
  python hacmo/interconnect/scripts/run_tp3_experiment.py \
    --tp2-results-dir "${TP2_PACK}" --batch-sizes 16 32 64 128 \
    --output-root ae/work/tp3/whole || return 1
  python hacmo/interconnect/scripts/probe_lane_allocation.py \
    --tp2-results-dir "${TP2_PACK}" --batch-sizes 16 32 64 128 \
    --random-samples 2 --output-root ae/work/tp3/probe || return 1
  python hacmo/interconnect/scripts/build_lane_level_final_pack.py \
    --whole-port-pack ae/work/tp3/whole --lane-probe-dir ae/work/tp3/probe \
    --output-root ae/work/tp3/final || return 1
  compare tp3 \
    --expected ae/expected-results/tp3/tp3_metrics.csv \
    --actual ae/work/tp3/final/tp3_metrics.csv \
    --keys topology batch_size bar_key --metrics total_us compute_us communication_us
}

run_overall() {
  # Qwen and Llama TP2/TP3 evidence ships as bundled support packs; the
  # required workflow regenerates the DeepSeek columns only.
  python hacmo/overall/scripts/run_overall_rebuttal.py \
    --output-root ae/work/overall --skip-root-copy \
    --tp1-final-pack ae/work/tp1/source \
    --tp1-support-pack ae/data/overall-support/tp1-support \
    --tp1-context-16k-pack ae/data/overall-support/tp1-context-16k \
    --tp1-profile-dir ae/data/tp1/profiles \
    --tp1-gpu-baseline-csv ae/expected-results/tp1/source/tp1_gpu_measured_baseline.csv \
    --deepseek-tp2-pack "${TP2_PACK_OVERALL}" \
    --deepseek-tp3-pack ae/work/tp3/final \
    --qwen-tp2-pack ae/data/overall-support/qwen/tp2 \
    --qwen-tp3-pack ae/data/overall-support/qwen/tp3 \
    --llama-tp2-pack ae/data/overall-support/llama/tp2 \
    --llama-tp3-pack ae/data/overall-support/llama/tp3 || return 1
  # validate_overall_outputs.py is a development-time provenance check: it
  # asserts that the consumed packs carry the authors' own directory names, so
  # it cannot pass for a run that regenerates them under ae/work. The release
  # gate is the table comparison below plus the twelve paper-claim checks.
  compare overall \
    --expected ae/expected-results/overall/simulated_layers.csv \
    --actual ae/work/overall/data/simulated_layers.csv \
    --keys scenario model_key baseline_key topology batch_per_worker context_len \
    --metrics tpot_ms throughput_tps
}

build_paper_results() {
  python ae/build_paper_results.py \
    --tp1-csv ae/work/tp1/final/csv/tp1_hybrid_cp2_summary.csv \
    --tp2-csv "${TP2_PACK}/paper_figures/paper_figure_summary.csv" \
    --tp3-csv ae/work/tp3/final/tp3_metrics.csv \
    --overall-csv ae/work/overall/data/simulated_layers.csv \
    --output-dir ae/work/paper-results
}

report() {
  local failures=0
  echo
  echo "=== stage summary ==="
  for index in "${!STAGE_NAMES[@]}"; do
    printf '%-10s %s\n' "${STAGE_NAMES[$index]}" "${STAGE_STATUS[$index]}"
    [[ "${STAGE_STATUS[$index]}" == "OK" ]] || failures=$((failures + 1))
  done
  echo
  echo "=== table comparisons ==="
  if [[ ${#CHECK_NAMES[@]} -eq 0 ]]; then
    echo "(none run)"
  fi
  for index in "${!CHECK_NAMES[@]}"; do
    printf '%-14s %s\n' "${CHECK_NAMES[$index]}" "${CHECK_STATUS[$index]}"
    [[ "${CHECK_STATUS[$index]}" == "PASS" ]] || failures=$((failures + 1))
  done
  echo
  if [[ ${failures} -gt 0 ]]; then
    echo "FAIL: full workflow stage=${STAGE} (${failures} failing)" >&2
    return 1
  fi
  echo "PASS: full workflow stage=${STAGE}"
}

python ae/check_env.py --full || {
  echo "ERROR: environment checks failed; fix them before running the workflow" >&2
  exit 1
}

case "${STAGE}" in
  tp1) run_stage tp1 run_tp1 ;;
  tp2) run_stage tp2 run_tp2 ;;
  tp3) run_stage tp3 run_tp3 ;;
  overall) run_stage overall run_overall ;;
  results) run_stage results build_paper_results ;;
  all)
    run_stage tp1 run_tp1
    run_stage tp2 run_tp2
    run_stage tp3 run_tp3
    run_stage overall run_overall
    run_stage results build_paper_results
    ;;
  *) echo "Usage: bash ae/run_full.sh [all|tp1|tp2|tp3|overall|results]" >&2; exit 2 ;;
esac

report
