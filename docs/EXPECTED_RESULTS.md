# Expected Results for Figures 9--12

Reproduced tables are compared with archived tables at 5% relative tolerance.
Missing rows or required metrics are failures. `ae/build_paper_results.py`
then computes and prints the following paper-facing checks.

| Figure | Formula | Paper value |
| --- | --- | ---: |
| 9 | `max(HACMO_TPS / DGX_TPS)` over models | `8.90x` |
| 9 | `max(HACMO_TPS / vLLM+LS_TPS)` over models | `2.03x` |
| 10 | `max((1 - O/C) * 100)` over all 16 groups | `63.5%` |
| 10 | `(1 - mean(S/C)) * 100` over all 16 groups | `18.9%` |
| 10 | `(1 - mean(O/C)) * 100` over all 16 groups | `53.9%` |
| 10 | `mean(C/O)` over all 16 groups | `2.21x` |
| 11 | `max((1 - Full / vLLM) * 100)` | `44.9%` |
| 11 | `mean((1 - Full / vLLM) * 100)` | `33.4%` |
| 12 | `max((1 - Full / Rail+Naive) * 100)` | `49.3%` |
| 12 | `mean((1 - Full / Rail+Naive) * 100)` | `41.1%` |
| 12 | `max((1 - Full / Rail+Mapping) * 100)` | `19.0%` |
| 12 | `mean((1 - Full / Rail+Mapping) * 100)` | `11.1%` |

Both Figure 9 checks are read off the same model-comparison configuration:
three models at context 8192 and 32 requests per worker. The Overall table also
contains a batch sweep whose best software speedup is higher, and quoting the
sweep for one claim and the model comparison for the other would make the two
numbers incomparable, so the audit uses the model-comparison rows for both.

Figure 10's average speedup is the arithmetic mean of the 16
per-configuration speedups `C/O`, matching the paper's aggregation. It is
not the reciprocal of the mean normalized latency. The aggregation includes
the four `128+CP2` hybrid points. The four Figure 10 checks are unchanged from
the artifact evaluated for MICRO 2026; Technical Point 1 evidence did not move.

The exact computed numbers, sample counts, formulas, tolerances, and statuses
are saved in `ae/work/paper-results/paper_claims.csv` and `.json`. The four PNG
and PDF plots in the same directory are generated directly from the final CSVs
and should preserve the groups, method ordering, and trends in Figures 9--12.

## Known deviation in the Overall table

In the archived camera-ready rehearsal, Figure 9 rows reproduced exactly except
the `context_tps` rows at contexts
32768 and 65536, which land within 1.3% and 0.9% of the archived table. Those
two context groups are the ones the sparse-attention model extrapolates beyond
the measured range, and the archived values were produced from the raw H200
measurement directories. The artifact ships the compact processed profiles
instead, because the raw measurement trees are not distributable. Every other
scenario — `model_compare`, `throughput_batch`, `tpot`, `topology`,
`breakdown`, and `context_tps` at 8192 and 16384 — matches to the digit, and
both Figure 9 claims are read off `model_compare`, which matched exactly in that
rehearsal. The current candidate run is recorded separately in
[VALIDATION.md](VALIDATION.md).

## Technical Point 2 mapping provenance

The Figure 11 grid has eight topology-by-batch cells, and they were not all
produced from the same offline expert-placement solve. Five cells used the
solve at wafer topology `4w_8x8_rail_900` with a static-mapping library capped
at 256 requests; `4w_8x8_rail_900` at batch 32 used the same topology capped at
96; and `16w_4x4_rail_500` at batches 32 and 64 used the `16w_4x4_rail_500`
solve capped at 96.

`hacmo/moe/configs/camera_ready_tp2_manifest.json` records this per
cell, and the workflow solves each distinct placement and checks it against a
pinned `expert_to_dies` hash before consuming it. The solve is deterministic:
fixing the topology and the library cap reproduces each placement exactly, so
nothing here is a frozen binary the workflow cannot re-derive.

Driving all eight cells from one solve reproduces five of them to six decimal
places and misses three, which moves the Figure 11 average outside its own
tolerance. `docs/adr/0001-tp2-per-cell-mapping-provenance.md` records why the
released workflow reproduces the published grid instead.
