# Validation record

The conference candidate passed the complete required reproduction in an
isolated Linux x86-64 checkout derived from the fixed camera-ready source
archive. Validation finished on 2026-10-28 at 04:08:19 UTC (12:08:19 UTC+8).
The source-build, quick-check, and full-workflow session took 11 h 35 min 19 s.
The wrapper exited with code 0.

## Completed checks

- All five original release files match the sizes and MD5 checksums in Zenodo
  record 22658850; the archived SHA256 checks also pass.
- The supplied paper's Figures 9–12 and appendix were inspected. The twelve
  headline values recomputed from archived tables pass their original checks.
- All reference CSVs and metadata remain byte-identical to the source archive.
- The completed execution checkout matches the local candidate for all 1,908 checked
  execution source/config files.
- Every candidate file is traced to the source archive, runtime dependency
  sources, four upstream CMake repairs, or a reviewed maintenance change.
- All 621 archived Ramulator dependency files match their original upstream
  version commits. Four missing CMake helpers were restored without upgrading
  those dependencies.
- Both repaired preflight/dry-run regression checks pass.
- ASTRA and Ramulator2 build from source, with regenerated Protobuf C++
  bindings, the conference Mesh2D overlay, and fixed offline dependencies.
- The final quick workflow passes: 174 tests, analytical smoke execution, and
  all twelve reference-table headline checks. The tests took 60.81 seconds.

## Full numerical reproduction

`bash ae/run_full.sh all` completed TP1, TP2, TP3, Overall, and result auditing.
Every stage returned OK. All five keyed CSV comparisons and all twelve paper
headline checks passed. Generated outputs started in an empty experiment-output
directory; the workflow did not substitute expected tables for generated results.
Reference-table SHA256 verification passed again after completion.

| Comparison | Rows | Largest relative difference in checked fields | Result |
| --- | ---: | ---: | --- |
| TP1 direct source | 48 | 0% | PASS |
| TP1 hybrid CP2 | 16 | 0% | PASS |
| TP2 expert mapping | 32 | 0% | PASS |
| TP3 interconnect | 32 | 0% | PASS |
| Overall | 108 | 1.230859% | PASS |

The reproduced tables contain exactly the expected row keys. TP1/TP2/TP3
checked values match the archived decimal values exactly. Overall differs only
in four numeric fields belonging to two DeepSeek HACMO `context_tps` rows:
context 32768 has TPOT +1.230859% and throughput -1.215890%; context 65536 has
TPOT +0.868627% and throughput -0.861145%. All other checked Overall values
match exactly. These are the documented compact-profile/extrapolation differences,
within the original 5% relative tolerance. Neither references nor parameters
were adjusted to obtain a pass.

The headline checks yield 8.901087x over DGX, 2.025575x over the layered-supernode
baseline, attention latency reductions of 63.525736% maximum and 53.856186% mean,
MoE mapping reductions of 44.861569% maximum and 33.394222% mean, and interconnect
full-method reductions of 49.326405% maximum and 41.067549% mean. All twelve
formulas, tolerances, and results are emitted in `ae/work/paper-results/`.

## Scope and remaining external checks

This is the required conference CPU workflow described in [RELEASE.md](RELEASE.md).
It uses the supplied measured H200 baselines and support packs, including
archived Qwen/Llama evidence; it does not remeasure GPU hardware. Optional DSE,
hardware analysis, and extra sensitivity campaigns are outside this validation.
The original per-cell TP2 solve protocol and analytical CP2 terms are preserved.
GitHub Actions is configured but has not yet been executed on GitHub.
No GitHub upload or Zenodo modification was performed during preparation.

## Environment

- Ubuntu 24.04 userspace, Linux x86-64
- Python 3.12.3, GCC 13.3.0, CMake 3.31.4, Protobuf compiler 3.21.12
- Python packages recorded in `requirements-tested-py312.txt`
- Fresh virtual environment; no existing experiment outputs used as generated
  results. The author data archive supplies measured and support inputs.
- Third-party MoE inputs are copied from the existing conference rehearsal;
  input inventories are recorded and the workflow checks the archived prepared
  workload and mapping hashes.

The public reference tables remain unchanged even if a validation check fails.
