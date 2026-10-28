#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}"
export MPLCONFIGDIR="${ROOT}/ae/work/matplotlib"
mkdir -p "${MPLCONFIGDIR}"

python ae/build_paper_results.py \
  --tp1-csv ae/expected-results/tp1/hybrid/tp1_hybrid_cp2_summary.csv \
  --tp2-csv ae/expected-results/tp2/paper_figure_summary.csv \
  --tp3-csv ae/expected-results/tp3/tp3_metrics.csv \
  --overall-csv ae/expected-results/overall/simulated_layers.csv \
  --output-dir ae/work/reference-paper-results
