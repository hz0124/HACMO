#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}"
export MPLCONFIGDIR="${ROOT}/ae/work/matplotlib"
# Keep this in step with run_full.sh: the prebuilt Ramulator2 resolves
# libramulator.so out of the source tree it was built in.
for lib_dir in "${ROOT}/ae-env/lib" "${ROOT}/build/third_party/ramulator2-src"; do
  if [[ -d "${lib_dir}" ]]; then
    export LD_LIBRARY_PATH="${lib_dir}:${LD_LIBRARY_PATH:-}"
  fi
done
if [[ "$(ulimit -Sn)" -lt 8192 ]]; then
  ulimit -n 8192 2>/dev/null || true
fi
mkdir -p ae/work/quick "${MPLCONFIGDIR}"

python ae/check_env.py
python scripts/generate_chakra_python.py
python -m pytest -q \
  supernode_sim/tests \
  ae/tests \
  hacmo/attention/tests \
  hacmo/moe/tests \
  hacmo/interconnect/tests \
  hacmo/overall/tests
python -m supernode_sim simulate \
  --config supernode_sim/configs/deepseek_v32_attention_smoke.json \
  --output ae/work/quick/attention_smoke.json
bash ae/run_reference.sh

for path in ae/expected-results/tp1/hybrid/tp1_hybrid_cp2_summary.csv \
            ae/expected-results/tp2/paper_figure_summary.csv \
            ae/expected-results/tp3/tp3_metrics.csv \
            ae/expected-results/overall/simulated_layers.csv; do
  test -s "${path}"
done
echo "PASS: quick functional workflow"
