#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_DIR}"

SOURCE_PACK="hacmo/attention/outputs/technical_point1_final_pack_astra_ramulator_trace_driven_20260724"
HYBRID_PACK="hacmo/attention/outputs/technical_point1_final_pack_astra_ramulator_hybrid_cp2_20260725"
OUTPUT_DIR="hacmo/attention/motivation/tp1_core_attention_motivation_astra_ramulator_hybrid_cp2_20260725"
REBUILD_HYBRID=0
REBUILD_SOURCE=0

usage() {
  cat <<'EOF'
Usage: run_tp1_simulator_motivation.sh [options]

Options:
  --source-pack PATH   Audited ASTRA-Sim + Ramulator source pack.
  --hybrid-pack PATH   Hybrid 16/32/64/128+CP2 final support pack.
  --output-dir PATH    Motivation artifact output directory.
  --rebuild-hybrid     Regenerate the hybrid pack before plotting.
  --rebuild-source     Rerun the source simulator pack and hybrid pack first.
  -h, --help           Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-pack)
      SOURCE_PACK="$2"
      shift 2
      ;;
    --hybrid-pack)
      HYBRID_PACK="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --rebuild-hybrid)
      REBUILD_HYBRID=1
      shift
      ;;
    --rebuild-source)
      REBUILD_SOURCE=1
      REBUILD_HYBRID=1
      shift
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

if [[ "${REBUILD_HYBRID}" -eq 1 ]]; then
  REBUILD_ARGS=(
    --source-pack "${SOURCE_PACK}"
    --output-root "${HYBRID_PACK}"
  )
  if [[ "${REBUILD_SOURCE}" -eq 1 ]]; then
    REBUILD_ARGS+=(--rebuild-source)
  fi
  bash hacmo/attention/scripts/run_tp1_hybrid_cp_pack.sh "${REBUILD_ARGS[@]}"
fi

PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-tp1-motivation" \
python3 hacmo/attention/tools/build_tp1_simulator_motivation.py \
  --source-pack "${SOURCE_PACK}" \
  --hybrid-pack "${HYBRID_PACK}" \
  --output-dir "${OUTPUT_DIR}"
