# Third-Party Software

HACMO-owned code is licensed under the top-level MIT `LICENSE`. The licenses
described below apply to their respective bundled third-party trees and are not
replaced by the HACMO license.

## ASTRA-Sim

HACMO uses a modified ASTRA-Sim version for communication-only MoE replay and
wafer/core network modeling. The complete modified tree is bundled under
`astra-sim/` in the fixed Zenodo source archive. It is based directly on
upstream commit `6d9c0d2093ef7dd84003753f5c28710c22e56e7b`; the included
`HACMO_CHANGES.md` documents the analytical-path changes.

The archive also includes the modified analytical network backend at commit
`ef35968d8f214b72009f5e236fc9bf4d06085b8d`. That commit adds the rail-optimized
switch-mesh topology required by HACMO and is based on upstream analytical
backend commit `599a047`. Both forks retain their upstream MIT licenses and
copyright notices.

## Ramulator2

Ramulator2 is pinned at upstream commit
`be93be78055d922aa1d4d33e15bcc8f2b0c61a9d` under
`third_party/ramulator2/`. HACMO's `NoRefresh` registration adapter and timed
trace frontend are kept under `supernode_sim/tools/`; the bundled upstream
source is not modified. The build script applies these additions only to an
isolated build tree. Ramulator2 is required for the Figure 10 HBM replay.

Its build dependencies are expanded under `third_party/ramulator2-deps/`:
yaml-cpp 0.7.0, spdlog 1.11.0, and argparse 2.9. These exact sources were
already included in the verified camera-ready Linux runtime archive under
`build/third_party/ramulator2-src/ext/`. Their license files and source bytes
are preserved. The build script passes them to CMake FetchContent explicitly,
so rebuilding does not require an external Git clone.

The archive omitted four `.cmake` helper files. They were restored from the
original upstream versions below. Every dependency file already present in the
archive matches the respective upstream commit byte-for-byte. The recovered
file names and hashes are recorded in `RELEASE_MANIFEST.json`.

| Dependency | Immutable source commit |
| --- | --- |
| yaml-cpp 0.7.0 | [0579ae3d976091d7d664aa9d2527e0d0cff25763](https://github.com/jbeder/yaml-cpp/tree/0579ae3d976091d7d664aa9d2527e0d0cff25763) |
| spdlog 1.11.0 | [ad0e89cbfb4d0c1ce4d097e134eb7be67baebb36](https://github.com/gabime/spdlog/tree/ad0e89cbfb4d0c1ce4d097e134eb7be67baebb36) |
| argparse 2.9 | [997da9255618311d1fcb0135ce86022729d1f1cb](https://github.com/p-ranav/argparse/tree/997da9255618311d1fcb0135ce86022729d1f1cb) |

## MoE trace dataset

The MoE trace dataset is not software bundled with HACMO and is not
redistributed. See `docs/DATA.md` for its citation, revision, and access terms.
