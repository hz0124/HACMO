# Expert mapping (TP2)

`tp2_core/` owns mapping input construction, the static solver, causal runtime
scheduling, and dispatch/combine traffic. `scripts/run_camera_ready_tp2.py`
executes the frozen per-cell protocol in `configs/camera_ready_tp2_manifest.json`.
It emits separate batch-64 and batch-32 packs for TP3 and Overall.

Run from the repository root after the setup in the main README:

```bash
bash ae/run_full.sh tp2
```

See [reproduction](../../docs/REPRODUCING.md),
[expected results](../../docs/EXPECTED_RESULTS.md), and
[source map](../../docs/repository-map.md) for dependencies and outputs.
Optional exploratory helpers retain historical input defaults; the `ae/`
workflow supplies explicit paths for the supported conference experiment.
