# Data Dependencies

## TP1 sparse-attention profiles

The TP1 profiles were measured by the HACMO authors on NVIDIA H200 GPUs. The
Zenodo data archive contains only the compact, processed fields consumed by the
artifact: prefill access histograms and the first 50 decode steps for sampled
layers. The required workflow consumes these archived measurements.
`ae/extract_compact_tp1_profiles.py` converts existing raw profiles to the compact
format; this checkout does not include a validated raw H200 model-profiling
pipeline. Regenerating the GPU measurements is outside this release's tested
reproduction scope.

Expected location after `ae/setup_data.sh`:

```text
ae/data/tp1/profiles/
  profile_index.json
  longbench_*_profile.pkl  # 15 files
ae/data/tp1/tp2_serving_epoch_astra_comm_only_summary.csv
```

The profiles cover five LongBench workloads with three samples per workload
and use `hacmo_compact_profile_v1`. They are derived from author-owned
measurements and are distributed under CC BY 4.0 with SHA256 checksums in the
Zenodo archive. The small CSV is the frozen preceding-phase communication input
consumed by TP1. The top-level `DATA_LICENSE` defines the scope and terms.

## TP2 MoE traces

The raw expert-selection traces are third-party data from:

> Zhongkai Yu, Yue Guan, Zihao Yu, Chenyang Zhou, Zhengding Hu, Shuyi Pei,
> Yangwook Kang, Yufei Ding, and Po-An Tsai. *Patterns behind Chaos:
> Forecasting Data Movement for Efficient Large-Scale MoE LLM Inference*,
> 2026. https://arxiv.org/abs/2510.05497

- Dataset: `core12345/MoE_expert_selection_trace`
- Pinned revision: `27febb7b2d24169560a9f6f83d38eb53f94916cf`
- Access: public but automatically gated; a Hugging Face account and acceptance
  of the dataset conditions may be required.
- License: no redistribution license was stated in the dataset metadata when
  the AE release was prepared.

HACMO therefore does **not** mirror the raw MoE traces in the Zenodo artifact.
`ae/download_moe_traces.py` downloads the 943 DeepSeek requests consumed by
the required TP2 workflow, preserves the original directory layout, pins the
dataset revision, and writes file sizes and SHA256 hashes to
`ae/data/moe/manifest.json`. The selected domains contain 417 professional-law requests, 256 miscellaneous
requests, and 90 requests each from high-school mathematics, elementary
mathematics, and human aging. Overall consumes new DeepSeek results and bundled
Qwen/Llama support packs; additional Qwen/Llama trace downloads are not required.

## Generated data

Chakra traces, ASTRA-Sim logs, intermediate traffic matrices, figures, and
temporary solver output are generated under `ae/work/`. They can be deleted
after comparing the final CSV files. The Overall stage consumes new DeepSeek TP1/TP2/TP3 outputs, measured H200
baselines, and bundled support curves and Qwen/Llama packs. See
`ae/data/overall-support/PROVENANCE.md` and `docs/RELEASE.md`. The archived `ae/expected-results/` contains only author-generated
result tables and metadata. These materials are covered by `DATA_LICENSE`.
