#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_DIR}"

OUTPUT_ROOT="hacmo/attention/outputs/tp1_audited_astra_ramulator_candidate_20260723"
SKIP_BUILD=0
GENERATOR_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-build)
      SKIP_BUILD=1
      shift
      ;;
    --output-root)
      [[ $# -ge 2 ]] || { echo "--output-root requires a value" >&2; exit 2; }
      OUTPUT_ROOT="$2"
      GENERATOR_ARGS+=("$1" "$2")
      shift 2
      ;;
    --output-root=*)
      OUTPUT_ROOT="${1#--output-root=}"
      GENERATOR_ARGS+=("$1")
      shift
      ;;
    *)
      GENERATOR_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "${SKIP_BUILD}" -eq 0 ]]; then
  bash scripts/build_ramulator2.sh
  bash scripts/build_astra_analytical.sh
fi
PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
  python3 -m hacmo.attention.tp1_core.generate_audited_tp1_pack "${GENERATOR_ARGS[@]}"
PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
  MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-tp1-audited" \
  python3 hacmo/attention/tools/plot_tp1_audited_pack.py \
    --pack-root "${OUTPUT_ROOT}"
