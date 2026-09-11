# HACMO

**Heterogeneity-Aware Cross-Layer Mapping Optimization for Sparse LLMs on Layered Supernodes**

MICRO 2026 · Hanbin Zhang, Jun Liu, Rongquan Ni, Yuxuan Zhang, Guohao Dai

---

## 🚧 Code coming soon

We are preparing this repository for public release. The artifact that
accompanies the paper is already archived on Zenodo; this repository will
become the maintained home of the framework, with continued fixes and updates.

⭐ Star or watch the repository to be notified when the code lands.

## What is HACMO?

Layered supernodes — rack-scale switch networks connecting wafer-scale chips
whose dies are multi-core NPUs — offer a promising substrate for serving sparse
LLMs. Existing mapping frameworks, however, flatten this three-tier hierarchy
into a single-tier abstraction and therefore miss the architectural
heterogeneity that actually governs performance: the bandwidth and capacity gap
between HBM and SRAM inside a wafer, the mismatch between symmetric external
switch fabrics and asymmetric on-wafer meshes, and the uneven traffic demand at
wafer boundaries.

HACMO models inference behaviour on the full rack / wafer / core abstraction and
introduces three cross-layer optimizations:

- **Spatially-aware dual-granularity attention mapping** — abstracts the
  two-tier on-wafer mesh into an equivalent representation and partitions
  KV-cache and compute non-uniformly across cores.
- **Bottleneck-aware MoE mapping** — places experts by global communication
  cost rather than 2D mesh distance alone, and rebalances them at runtime over
  underutilized links.
- **Traffic-aware decoupled edge interconnect** — adds lightweight crossbars at
  wafer boundaries so external bandwidth can be allocated asymmetrically to
  match the traffic the software mapping actually produces.

Evaluated with an extended ASTRA-sim 2.0 and Ramulator2 on DeepSeek-V3.2,
Qwen3-MoE and Llama-4-Maverick, HACMO reduces sparse-attention latency by up to
**63.5%** and MoE latency by up to **49.3%**, delivering up to **8.90×** higher
generation throughput than vLLM on a GPU cluster baseline.

## Artifact

The evaluated research artifact is archived on Zenodo:

> https://doi.org/10.5281/zenodo.21534258

It contains the framework source, prebuilt simulators, compact measured
profiles, and an automated CPU-only workflow that regenerates the paper's
figures and audits its headline values.

## Citation

A BibTeX entry will be added here once the proceedings are published.

## License

Author-created software will be released under the MIT License. Bundled
third-party components retain their upstream licenses, and third-party datasets
are not redistributed.

## Contact

For questions about the paper or the artifact, please open an issue.
