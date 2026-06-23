# TileMaxSim

GPU Triton kernels for MaxSim scoring in multi-vector retrieval (ColBERT-style late interaction).

Naive MaxSim implementations materialize the full `N_q x N_d` query-by-document
similarity matrix in HBM and then reduce it, which wastes most of the available
memory bandwidth. TileMaxSim avoids materializing that matrix. It streams document
embeddings through SRAM, keeps the per-query-token running maxima in registers, and
reads each embedding from HBM once. On an H100 this reaches about 80% of peak HBM
bandwidth while producing the same rankings as a reference MaxSim implementation.

The kernels also handle two cases that a basic fused kernel does not:

* Embedding dimensions above 128, via tiling over the embedding dimension.
* Product-quantized documents, scored directly from PQ codes through a per-query
  distance table (asymmetric distance computation) without reconstructing vectors.

This repository contains the kernels, baselines, and benchmark scripts. It does not
contain the paper.

Developed concurrently with and independently of Pony et al., "Flash-MaxSim"
(arXiv:2605.29517). The system was named TileMaxSim to avoid the name collision.

## Layout

```
src/
  flash_maxsim_kernel.py    V1 per-query-token kernel and PyTorch baselines
  flash_maxsim_v2.py        V2 (per-document) and V2-MQ (multi-query) kernels
  flash_pqsim_kernel.py     PQ scoring kernel (fused distance-table lookup)
  benchmark.py              throughput and bandwidth benchmarks
  beir_evaluation.py        BEIR quality check against reference MaxSim
  e2e_retrieval_quality.py  MS MARCO re-ranking quality check
baselines/                  WARP / PLAID comparison harnesses
experiment_results/         measured JSON outputs behind the reported numbers
```

The recommended entry point is `flash_maxsim_v2_multiquery` in
`src/flash_maxsim_v2.py`. Set the query-tile size `BQ` equal to the query length
`N_q` so each document embedding is read from HBM exactly once.

## Requirements

An NVIDIA GPU with CUDA. Results were measured on an H100 80GB with CUDA 12.4,
Python 3.11, PyTorch 2.6, and Triton 3.2.

```
pip install -r requirements.txt
```

## Quick start

```
# Throughput and bandwidth across kernel variants
python src/flash_maxsim_v2.py --device cuda:0

# Exact-quality check on BEIR
python src/beir_evaluation.py
```

## Notes

The kernel computes the same arithmetic as reference MaxSim and only reorders memory
accesses, so rankings match to floating-point rounding. Default precision is FP16
embeddings with FP32 accumulation.

The WARP and PLAID baseline harnesses under `baselines/` need a local checkout of the
respective upstream repositories. Set the location through the `WARP_DIR` environment
variable.
