#!/bin/bash
set -e

SCRIPT_DIR=$(dirname "$(realpath "$0")")
CHAKRA_ET_DIR="${SCRIPT_DIR:?}/../../extern/graph_frontend/chakra/schema/protobuf"

# compile Chakra ET
compile_chakra_et() {
  if [[ ! -f "${CHAKRA_ET_DIR:?}"/et_def.pb.h || ! -f "${CHAKRA_ET_DIR:?}"/et_def.pb.cc ]]; then
    echo "Compiling Chakra protobuf..."
    cd "${CHAKRA_ET_DIR:?}"
    protoc et_def.proto \
      --proto_path="${CHAKRA_ET_DIR:?}" \
      --cpp_out="${CHAKRA_ET_DIR:?}"
  fi
}

# setup
setup() {
  if [[ ! -d "${SCRIPT_DIR:?}"/build ]]; then
    mkdir -p "${SCRIPT_DIR:?}"/build
  fi
  NUM_THREADS=$(nproc)
  if [[ ${NUM_THREADS} -ge 16 ]]; then
    NUM_THREADS=16
  fi
}

# compile
compile() {
  TARGET="${1:-all}"
  cd "${SCRIPT_DIR:?}"/build
  cmake "${SCRIPT_DIR:?}" -DBUILDTARGET="${TARGET}"
  cmake --build . -j "${NUM_THREADS:?}"
}

# cleanup
cleanup() {
  rm -rf "${SCRIPT_DIR:?}"/build
}

compile_chakra_et
setup
compile "$1"
