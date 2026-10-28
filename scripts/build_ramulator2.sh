#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUBMODULE_DIR="${REPO_DIR}/third_party/ramulator2"
NO_REFRESH_SOURCE="${REPO_DIR}/supernode_sim/tools/ramulator2_no_refresh.cpp"
TIMED_TRACE_SOURCE="${REPO_DIR}/supernode_sim/tools/ramulator2_timed_trace.cpp"
WORK_ROOT="${REPO_DIR}/build/third_party"
WORK_SRC="${WORK_ROOT}/ramulator2-src"
BUILD_DIR="${WORK_ROOT}/ramulator2-build"

# The development checkout carries Ramulator2 as a git submodule, but the
# released artifact bundles it as plain pinned source. Support both: the
# release must not need git to build the simulator it documents.
if [[ -e "${SUBMODULE_DIR}/.git" ]]; then
  SOURCE_MODE="submodule"
elif [[ -f "${SUBMODULE_DIR}/CMakeLists.txt" ]]; then
  SOURCE_MODE="bundled"
elif command -v git >/dev/null 2>&1 && [[ -e "${REPO_DIR}/.git" ]]; then
  git -C "${REPO_DIR}" submodule update --init --recursive third_party/ramulator2
  SOURCE_MODE="submodule"
else
  echo "[ramulator2-build][ERROR] Missing Ramulator2 source at ${SUBMODULE_DIR}" >&2
  exit 1
fi

if [[ ! -f "${SUBMODULE_DIR}/CMakeLists.txt" ]]; then
  echo "[ramulator2-build][ERROR] Missing Ramulator2 source at ${SUBMODULE_DIR}" >&2
  exit 1
fi

# Fingerprint the pinned source so a working copy left over from a different
# revision is reported instead of silently rebuilt.
source_fingerprint() {
  if [[ "${SOURCE_MODE}" == "submodule" ]]; then
    git -C "${SUBMODULE_DIR}" rev-parse HEAD
  else
    find "${SUBMODULE_DIR}" -type f -name '*.h' -o -type f -name '*.cpp' \
      -o -type f -name 'CMakeLists.txt' | sort | xargs sha256sum \
      | sha256sum | cut -d' ' -f1
  fi
}

mkdir -p "${WORK_ROOT}"
PINNED_ID="$(source_fingerprint)"
STAMP_FILE="${WORK_SRC}/.hacmo-source-id"

if [[ ! -d "${WORK_SRC}" ]]; then
  if [[ "${SOURCE_MODE}" == "submodule" ]]; then
    git clone --quiet "${SUBMODULE_DIR}" "${WORK_SRC}"
  else
    cp -a "${SUBMODULE_DIR}" "${WORK_SRC}"
  fi
  printf '%s\n' "${PINNED_ID}" > "${STAMP_FILE}"
fi

WORK_ID="$(cat "${STAMP_FILE}" 2>/dev/null || \
  (git -C "${WORK_SRC}" rev-parse HEAD 2>/dev/null || echo unknown))"
if [[ "${WORK_ID}" != "${PINNED_ID}" ]]; then
  echo "[ramulator2-build][ERROR] Build source is at ${WORK_ID}, but the pinned source is at ${PINNED_ID}." >&2
  echo "[ramulator2-build][ERROR] Remove ${WORK_SRC} and rerun this script to rebuild from the pinned source." >&2
  exit 1
fi
printf '%s\n' "${PINNED_ID}" > "${STAMP_FILE}"

CONTROLLER_CMAKE="${WORK_SRC}/src/dram_controller/CMakeLists.txt"
NO_REFRESH_TARGET="${WORK_SRC}/src/dram_controller/impl/refresh/no_refresh.cpp"
if ! grep -q "impl/refresh/no_refresh.cpp" "${CONTROLLER_CMAKE}"; then
  sed -i 's/\r$//' "${CONTROLLER_CMAKE}"
  sed -i '/^[[:space:]]*impl\/refresh\/all_bank_refresh\.cpp[[:space:]]*$/a\  impl/refresh/no_refresh.cpp' "${CONTROLLER_CMAKE}"
fi
if ! grep -q "impl/refresh/no_refresh.cpp" "${CONTROLLER_CMAKE}"; then
  echo "[ramulator2-build][ERROR] Failed to register NoRefresh in ${CONTROLLER_CMAKE}" >&2
  exit 1
fi
install -D -m 0644 "${NO_REFRESH_SOURCE}" "${NO_REFRESH_TARGET}"

FRONTEND_CMAKE="${WORK_SRC}/src/frontend/CMakeLists.txt"
TIMED_TRACE_TARGET="${WORK_SRC}/src/frontend/impl/memory_trace/timed_trace.cpp"
if ! grep -q "impl/memory_trace/timed_trace.cpp" "${FRONTEND_CMAKE}"; then
  sed -i '/impl\/memory_trace\/loadstore_trace.cpp/a\  impl/memory_trace/timed_trace.cpp' "${FRONTEND_CMAKE}"
fi
if ! grep -q "impl/memory_trace/timed_trace.cpp" "${FRONTEND_CMAKE}"; then
  echo "[ramulator2-build][ERROR] Failed to register TimedTrace in ${FRONTEND_CMAKE}" >&2
  exit 1
fi
install -D -m 0644 "${TIMED_TRACE_SOURCE}" "${TIMED_TRACE_TARGET}"

# A CMakeCache carries the absolute source and build paths of whatever host
# generated it. The released archive ships one produced under the packaging
# directory, and cmake refuses to reuse a cache whose paths have moved, so
# discard a relocated cache instead of failing on it.
CACHE_FILE="${BUILD_DIR}/CMakeCache.txt"
if [[ -f "${CACHE_FILE}" ]]; then
  CACHED_SRC="$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "${CACHE_FILE}" | head -1)"
  CACHED_BUILD="$(sed -n 's/^Ramulator_BINARY_DIR:STATIC=//p' "${CACHE_FILE}" | head -1)"
  if [[ "${CACHED_SRC}" != "${WORK_SRC}" || ( -n "${CACHED_BUILD}" && "${CACHED_BUILD}" != "${BUILD_DIR}" ) ]]; then
    echo "[ramulator2-build] Discarding stale CMake cache generated at ${CACHED_SRC:-unknown}"
    rm -rf "${BUILD_DIR}"
  fi
fi

# The camera-ready runtime archived these exact dependency sources. Use them
# directly so source builds do not depend on GitHub access or mutable tags.
for dependency in yaml-cpp spdlog argparse; do
  if [[ ! -f "${REPO_DIR}/third_party/ramulator2-deps/${dependency}/CMakeLists.txt" ]]; then
    echo "[ramulator2-build][ERROR] Missing bundled dependency: ${dependency}" >&2
    exit 1
  fi
done
cmake -S "${WORK_SRC}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}" \
  -DFETCHCONTENT_SOURCE_DIR_YAML-CPP="${REPO_DIR}/third_party/ramulator2-deps/yaml-cpp" \
  -DFETCHCONTENT_SOURCE_DIR_SPDLOG="${REPO_DIR}/third_party/ramulator2-deps/spdlog" \
  -DFETCHCONTENT_SOURCE_DIR_ARGPARSE="${REPO_DIR}/third_party/ramulator2-deps/argparse"
cmake --build "${BUILD_DIR}" -j "${JOBS:-$(nproc)}"

echo "[ramulator2-build] Built: ${BUILD_DIR}/ramulator2"
