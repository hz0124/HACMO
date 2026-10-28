# Interconnect allocation (TP3)

The scripts evaluate whole-port rail/crossbar configurations, probe lane
allocation, and build the final Figure 12 table. They consume the newly
generated batch-64 TP2 pack; the crossbar model lives in
`supernode_sim/plugins/crossbar.py`.

Run from the repository root after the setup in the main README:

```bash
bash ae/run_full.sh tp3
```

See [reproduction](../../docs/REPRODUCING.md),
[expected results](../../docs/EXPECTED_RESULTS.md), and
[source map](../../docs/repository-map.md) for dependencies and outputs.
Optional exploratory helpers retain historical input defaults; the `ae/`
workflow supplies explicit paths for the supported conference experiment.
