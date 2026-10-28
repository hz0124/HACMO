# HACMO ASTRA-Sim Fork

This fork supports the ASTRA-Sim analytical workflows used by the HACMO
artifact. It is based on upstream ASTRA-Sim commit
`6d9c0d2093ef7dd84003753f5c28710c22e56e7b`.

The HACMO branch:

- pins the bundled analytical network backend to
  `ef35968d8f214b72009f5e236fc9bf4d06085b8d`;
- restores and pins Chakra to
  `0e3cd40c569f0a4cacb6d961bb56be53407abd2f`;
- supports Chakra FeederV3 process-group metadata and operation-specific
  collective implementations;
- preserves workload node identifiers through collective generation; and
- includes the event-scheduling and congestion-aware analytical fixes used by
  HACMO's communication-only MoE replay.

HACMO uses only the analytical backend. The NS-3 and Garnet backends are not
modified by this branch.

From the expanded Zenodo source archive, build both analytical executables with:

```bash
bash build/astra_analytical/build.sh all
```

The congestion-aware executable is written to
`build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware`.
