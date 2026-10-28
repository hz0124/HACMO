#!/usr/bin/env bash
# Build the conference simulator stack from the bundled source snapshots.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
bash scripts/build_astra.sh
bash scripts/build_astra_analytical.sh
bash scripts/build_ramulator2.sh
