# TP2 per-cell mapping provenance

Figure 11 contains eight topology-by-batch cells. The original runs consumed
different offline placement solves in three of those cells. The conference
workflow records the actual solve used by each cell in
`hacmo/moe/configs/camera_ready_tp2_manifest.json` and regenerates each solve
before replaying its workload.

| Evaluation cells | Solver topology | Library request cap |
| --- | --- | ---: |
| 4w batches 16, 64, 128; 16w batches 16, 128 | `4w_8x8_rail_900` | 256 |
| 4w batch 32 | `4w_8x8_rail_900` | 96 |
| 16w batches 32, 64 | `16w_4x4_rail_500` | 96 |

The request cap and, in some cells, the solve topology are therefore
non-uniform across the published grid. This is a limitation of the historical
experiment protocol. Replacing it with one uniform solve changes the results;
the archived investigation found an average reduction of about 30.4% rather
than the published 33.4% for that alternative. Such a run must be labeled as a
new experiment.

Each placement is derived from trace inputs by the released deterministic
solver. The workflow verifies the canonical `expert_to_dies` SHA256 before
using a placement:

| Solve | Canonical SHA256 |
| --- | --- |
| 4w, cap 256 | `8d78a5554fef87f61085f588ffaf17d320b595a6d87914dc90d1bc8ee5e51f70` |
| 4w, cap 96 | `b6b05fd5b66b48aa6d038e9f2052b06708f555e7dea3ccaf942cf06626d4f475` |
| 16w, cap 96 | `a43a211e7a2a378b81dd96713809f492beb8207aca03aab396cba2013f8cfc2e` |

These are hashes of regenerated mappings, not substitutes for solving them.
The protocol also checks prepared workload hashes for all four batch sizes.
MoEnTwine uses its own placement; the manifest's per-cell label keeps result
provenance aligned across methods.

Future methodological comparisons should choose and report a consistent
placement protocol. The conference reproduction retains the published cell
settings, reference tables, and headline values.
