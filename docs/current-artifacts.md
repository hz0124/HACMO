# Conference inputs and results

The fixed source of record is Zenodo `camera-ready-v1`, DOI
`10.5281/zenodo.22658850`. See [release provenance](RELEASE.md).

Reference tables are bundled in `ae/expected-results/` and verified with
`python ae/verify_sha256.py ae/expected-results/SHA256SUMS`.

Author-provided input data is installed into `ae/data/` from the fixed data
archive. Third-party traces are fetched separately by the pinned downloader.
Generated conference results are written to `ae/work/`; the complete table
mapping appears in [REPRODUCING.md](REPRODUCING.md).

Historical directory names inside reference metadata identify the original
source runs. Those work directories are not prerequisites for this checkout.
