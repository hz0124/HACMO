#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASTRA_DIR="${REPO_DIR}/astra-sim"
NETWORK_DIR="${ASTRA_DIR}/extern/network_backend/analytical"
HELPER="${NETWORK_DIR}/congestion_aware/topology/Helper.cpp"
MESH_HEADER="${NETWORK_DIR}/include/astra-network-analytical/congestion_aware/Mesh2D.h"
MESH_SOURCE="${NETWORK_DIR}/congestion_aware/topology/Mesh2D.cpp"
OVERLAY_HEADER="${REPO_DIR}/supernode_sim/tools/astra_mesh2d.h"
OVERLAY_SOURCE="${REPO_DIR}/supernode_sim/tools/astra_mesh2d.cpp"
STATISTICS_SOURCE="${ASTRA_DIR}/astra-sim/workload/Statistics.cc"
STATISTICS_PATCH="${REPO_DIR}/supernode_sim/tools/astra_statistics_cpu_timer.patch"

python3 "${REPO_DIR}/scripts/generate_chakra_python.py"

if [[ ! -f "${HELPER}" ]]; then
  echo "[astra-build][ERROR] Missing analytical backend at ${NETWORK_DIR}" >&2
  exit 1
fi

BACKUP_DIR="$(mktemp -d)"
cp "${HELPER}" "${BACKUP_DIR}/Helper.cpp"
cp "${STATISTICS_SOURCE}" "${BACKUP_DIR}/Statistics.cc"
cleanup() {
  cp "${BACKUP_DIR}/Helper.cpp" "${HELPER}"
  cp "${BACKUP_DIR}/Statistics.cc" "${STATISTICS_SOURCE}"
  rm -f "${MESH_HEADER}" "${MESH_SOURCE}"
  rm -rf "${BACKUP_DIR}"
}
trap cleanup EXIT

install -D -m 0644 "${OVERLAY_HEADER}" "${MESH_HEADER}"
install -D -m 0644 "${OVERLAY_SOURCE}" "${MESH_SOURCE}"

if ! grep -q 'congestion_aware/Mesh2D.h' "${HELPER}"; then
  sed -i '/#include "congestion_aware\/FullyConnected.h"/a#include "congestion_aware/Mesh2D.h"' "${HELPER}"
fi
if ! grep -q 'case TopologyBuildingBlock::Mesh:' "${HELPER}"; then
  sed -i '/case TopologyBuildingBlock::FullyConnected:/i\    case TopologyBuildingBlock::Mesh:\n        return std::make_shared<Mesh2D>(npus_count, bandwidth, latency);' "${HELPER}"
fi
if ! grep -q 'std::make_shared<Mesh2D>' "${HELPER}"; then
  echo "[astra-build][ERROR] Failed to register congestion-aware Mesh2D" >&2
  exit 1
fi

patch --quiet --forward --directory "${ASTRA_DIR}" --strip 1 < "${STATISTICS_PATCH}"

bash "${ASTRA_DIR}/build/astra_analytical/build.sh" congestion_aware

ASTRA_BIN="${ASTRA_DIR}/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware"
if [[ ! -x "${ASTRA_BIN}" ]]; then
  echo "[astra-build][ERROR] Expected binary was not produced: ${ASTRA_BIN}" >&2
  exit 1
fi
echo "[astra-build] Built: ${ASTRA_BIN}"
