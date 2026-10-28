# SupernodeSim

Shared simulation interfaces for HACMO. `api.simulate_layer(config)` dispatches
typed configurations to the analytic, ASTRA-Sim, or Ramulator wrappers.
`tracegen/` produces traffic and Chakra inputs; `plugins/crossbar.py` models
edge-local crossbars and lane allocation. The method modules under `hacmo/`
construct placements and experiment policies.

From the checkout root:

```bash
python -m supernode_sim simulate \
  --config supernode_sim/configs/deepseek_v32_attention_smoke.json \
  --output ae/work/quick/attention_smoke.json
python -m pytest -q supernode_sim/tests
```

The minimal example uses the analytical backend. The complete conference
workflow invokes the external simulator backends through `ae/run_full.sh`.
See [architecture](../docs/repository-map.md) and
[reproduction](../docs/REPRODUCING.md) for those paths and their data contracts.
