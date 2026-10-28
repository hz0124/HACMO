# HACMO conference repository

This checkout derives from the immutable camera-ready-v1 artifact. Read
`README.md`, `docs/RELEASE.md`, and `RELEASE_MANIFEST.json` before changing scope.

- Keep `ae/expected-results/` byte-identical to the archived reference tables.
- Keep implementation, experimental configuration, and measured inputs
  distinguishable. Never tune reference values to make validation pass.
- The supported entry points are `ae/run_quick.sh` and `ae/run_full.sh`.
- Use the same fixed data revision and per-cell mapping provenance for the
  conference workflow. Document deviations in new experiments explicitly.
- Preserve third-party licenses and pinned source identities.
- Keep generated data, credentials, environment files, and build products out
  of source control. Do not import development-workspace Git history.
- Record changes in `docs/change-log.md`; keep public documentation concise.
- Run quick checks after changes. Changes affecting simulator behavior or
  numerical results require the relevant full stages and CSV comparisons.
