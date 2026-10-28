#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_DIR}"

SOURCE_PACK="hacmo/attention/outputs/technical_point1_final_pack_astra_ramulator_trace_driven_20260724"
OUTPUT_ROOT="hacmo/attention/outputs/technical_point1_final_pack_astra_ramulator_hybrid_cp2_20260725"
CP_BANDWIDTH_TBPS="5.0"
CP2_HOPS="1.0"
REBUILD_SOURCE=0
SKIP_BUILD=0
PROFILE_DIR=""
PRECEDING_PHASE_CSV=""

usage() {
  cat <<'EOF'
Usage: run_tp1_hybrid_cp_pack.sh [options]

Options:
  --source-pack PATH         Validated non-CP simulator pack to reuse.
  --output-root PATH         Derived final-pack output directory.
  --cp-bandwidth-tbps VALUE  Analytical CP bandwidth in TB/s (default: 5.0).
  --cp2-hops VALUE           Analytical CP2 hop multiplier (default: 1.0).
  --rebuild-source           Rebuild batches 16/32/64 before deriving the pack.
  --skip-build               Reuse prebuilt ASTRA-Sim and Ramulator2 binaries.
  --profile-dir PATH         Compact H200 profile directory for source rebuild.
  --preceding-phase-csv PATH Frozen TP2 communication input for source rebuild.
  -h, --help                 Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-pack)
      SOURCE_PACK="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --cp-bandwidth-tbps)
      CP_BANDWIDTH_TBPS="$2"
      shift 2
      ;;
    --cp2-hops)
      CP2_HOPS="$2"
      shift 2
      ;;
    --rebuild-source)
      REBUILD_SOURCE=1
      shift
      ;;
    --skip-build)
      SKIP_BUILD=1
      shift
      ;;
    --profile-dir)
      PROFILE_DIR="$2"
      shift 2
      ;;
    --preceding-phase-csv)
      PRECEDING_PHASE_CSV="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "${REBUILD_SOURCE}" -eq 1 ]]; then
  SOURCE_ARGS=(
    --output-root "${SOURCE_PACK}"
    --batch-sizes 16 32 64
  )
  if [[ "${SKIP_BUILD}" -eq 1 ]]; then
    SOURCE_ARGS+=(--skip-build)
  fi
  if [[ -n "${PROFILE_DIR}" ]]; then
    SOURCE_ARGS+=(--profile-dir "${PROFILE_DIR}")
  fi
  if [[ -n "${PRECEDING_PHASE_CSV}" ]]; then
    SOURCE_ARGS+=(--preceding-phase-csv "${PRECEDING_PHASE_CSV}")
  fi
  bash hacmo/attention/scripts/run_tp1_audited_pipeline.sh \
    "${SOURCE_ARGS[@]}"
fi

PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-tp1-hybrid-cp" \
python3 hacmo/attention/tools/build_tp1_hybrid_cp_pack.py \
  --source-pack "${SOURCE_PACK}" \
  --output-root "${OUTPUT_ROOT}" \
  --cp-bandwidth-tbps "${CP_BANDWIDTH_TBPS}" \
  --cp2-hops "${CP2_HOPS}"
