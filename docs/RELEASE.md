# Conference release provenance

## Immutable source

The primary source baseline is `HACMO-AE-camera-ready-v1-source.tar.gz` from Zenodo
record **22658850**, version `camera-ready-v1`, published 2026-09-08.
Its SHA256 is
`db9e3320cd81ec1b4cb72e8fa3d06f8395f0b8266dcf04d74c65299d59fc499a`.
The local release files were checked against the sizes and MD5 checksums
returned by the public Zenodo API, and the archive SHA256 manifest was verified.

| Identifier | Meaning |
| --- | --- |
| `10.5281/zenodo.22658850` | Fixed camera-ready-v1 version used as this source baseline |
| `10.5281/zenodo.21543359` | Earlier evaluated ae-v2 artifact |
| `10.5281/zenodo.21534258` | Concept DOI for the artifact version series |

Ramulator2 build dependencies additionally come from the same record's
Linux runtime archive (SHA256
`b4623e2a41ee83d3600fc81ae44175edceb7fcfbc78b3a044e190450dbad9f0b`).
Their source bytes and licenses are preserved under
`third_party/ramulator2-deps/`.
Four missing CMake helpers were recovered from the original fixed upstream
commits; all other dependency files match those commits exactly. See
[third-party provenance](THIRD_PARTY.md) and `RELEASE_MANIFEST.json`.

The GitHub concept link and paper's fixed-version link refer to the same
artifact series. Reproduction instructions use the fixed camera-ready DOI.

## Correspondence with the paper

The reference tables and twelve headline values agree with the supplied
`MICRO26-HACMO-v3.pdf`, Figures 9–12. The paper appendix retains some ae-v2
workflow details that were superseded in the published camera-ready package:

| Topic | Earlier description | Executable camera-ready workflow |
| --- | --- | --- |
| MoE subset | 632 requests | 943 requests: 417 law, 256 miscellaneous, 90 each in three replacement domains |
| Python environment | Activate `ae-env` and run `conda-unpack` | Install Python requirements; runtime archive contains binaries and shared libraries, no complete Conda environment |
| Overall | Substitute TP1 into an older frozen table | Recompute DeepSeek using TP1/TP2/TP3 output and bundled service curves; consume bundled Qwen/Llama support packs |
| TP2 output | `tp2/deepseek/...` | `tp2/camera_ready_pack/...` |
| Internal manifest | `artifact_version: ae-v2` | Source package itself belongs to camera-ready-v1; corrected in this repository |

These corrections document the code already published on Zenodo. They do not
change the conference algorithm, parameters, reference values, or archived
records.

## Reproduction boundary

- **Figure 10:** direct ASTRA-Sim and Ramulator2 execution for batches 16, 32,
  and 64. The `128+CP2` points reuse per-die batch 64 and add the archived
  analytical communication term. H200 measurements are supplied inputs.
- **Figure 11:** regenerate the distinct expert placements and runtime
  replays specified by `hacmo/moe/configs/camera_ready_tp2_manifest.json`.
  The published eight-cell grid uses different static-library caps in some
  cells. The [provenance decision](adr/0001-tp2-per-cell-mapping-provenance.md)
  explains this and preserves the per-cell configuration and hash checks.
- **Figure 12:** derive topology and lane-allocation results from the new
  batch-64 TP2 pack.
- **Figure 9:** recompute DeepSeek with the new batch-32 TP2 pack, TP1 source
  pack, and TP3 final pack; also consume archived service curves and H200
  baselines. Qwen/Llama use bundled support packs. Contexts 32K/64K include
  extrapolation beyond the compact measured profile range.
- **Quick check:** regenerate figures from archived reference tables. This
  verifies installation and aggregation; full execution is a separate step.
- **Optional source:** DSE, hardware-analysis, sensitivity, and profile-processing
  helpers are retained from the same source archive. Their extra input
  requirements are outside the Figures 9–12 acceptance claim. The archived
  DSE MoE model is a diagnostic path with different semantics from TP2.

## Repository preparation

The three method namespaces changed from `Ablation1`, `Ablation2_new`, and
`Ablation3` to `attention`, `moe`, and `interconnect`. Internal imports and path
strings changed consistently. The reference tables and their metadata remain
byte-for-byte archived, including historical path labels inside metadata.

Documentation, build entry points, editable-install metadata, and short CI
checks were added or repaired. The environment preflight now rejects a missing
ASTRA executable; quick checks raise the documented file-descriptor limit.
Trace-generation-only dry runs no longer require an ASTRA executable. Source
builds regenerate C++ protobuf bindings with the installed compiler. These
changes affect setup and dry runs, leaving executed simulation calculations intact.
The source provenance audit distinguishes these maintenance changes from
path-only migrations. Source control history was not imported from the
experimental workspace. See `RELEASE_MANIFEST.json` and [validation](VALIDATION.md).
