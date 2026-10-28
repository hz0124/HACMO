# Overall performance

`scripts/run_overall_rebuttal.py` aggregates method outputs and support
curves. The required workflow regenerates DeepSeek while consuming archived
Qwen/Llama support packs. TP1, TP2 and TP3 must finish first.

Run from the repository root after the setup in the main README:

```bash
bash ae/run_full.sh overall
```

See [reproduction](../../docs/REPRODUCING.md),
[expected results](../../docs/EXPECTED_RESULTS.md), and
[source map](../../docs/repository-map.md) for dependencies and outputs.
Optional exploratory helpers retain historical input defaults; the `ae/`
workflow supplies explicit paths for the supported conference experiment.
