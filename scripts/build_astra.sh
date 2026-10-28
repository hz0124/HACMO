#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASTRA_DIR="${ROOT}/astra-sim"
ASTRA_BIN="${ASTRA_DIR}/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware"

if [[ ! -f "${ASTRA_DIR}/HACMO_CHANGES.md" ]]; then
  git -C "${ROOT}" submodule update --init --recursive astra-sim
fi

python "${ROOT}/scripts/generate_chakra_python.py"
# Recreate C++ bindings with the installed Protobuf compiler. The archived
# generated files were produced on a different host and may not match its ABI.
SCHEMA_DIR="${ASTRA_DIR}/extern/graph_frontend/chakra/schema/protobuf"
protoc --proto_path="${SCHEMA_DIR}" --cpp_out="${SCHEMA_DIR}" "${SCHEMA_DIR}/et_def.proto"
bash "${ASTRA_DIR}/build/astra_analytical/build.sh" all
if [[ ! -x "${ASTRA_BIN}" ]]; then
  echo "[astra-build][ERROR] Missing expected executable: ${ASTRA_BIN}" >&2
  exit 1
fi
echo "[astra-build] Built: ${ASTRA_BIN}"
