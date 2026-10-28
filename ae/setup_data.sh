#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash ae/setup_data.sh /path/to/hacmo-ae-data-camera-ready-v1.tar.gz" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCHIVE="$(realpath "$1")"
mkdir -p "${ROOT}/ae/data"
tar -xzf "${ARCHIVE}" -C "${ROOT}/ae/data"
python3 "${ROOT}/ae/verify_sha256.py" "${ROOT}/ae/data/SHA256SUMS"
echo "PASS: AE data installed under ${ROOT}/ae/data"
