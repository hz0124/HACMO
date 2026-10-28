# Sparse-attention mapping (TP1)

The `tp1_core/` package loads measured profiles, constructs placements and
routes, and evaluates attention using ASTRA-Sim and Ramulator2. Direct batches
16/32/64 feed the hybrid `128+CP2` analytical derivation.

Run from the repository root after the setup in the main README:

```bash
bash ae/run_full.sh tp1
```

See [reproduction](../../docs/REPRODUCING.md),
[expected results](../../docs/EXPECTED_RESULTS.md), and
[source map](../../docs/repository-map.md) for dependencies and outputs.
Optional exploratory helpers retain historical input defaults; the `ae/`
workflow supplies explicit paths for the supported conference experiment.
