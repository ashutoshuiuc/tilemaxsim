#!/bin/bash
# TileMaxSim: Reproduce all paper results
#
# Requirements:
#   - 1x NVIDIA H100 80GB (or A100 80GB with reduced scale)
#   - CUDA 12.x with Triton 2.1+
#   - Python 3.10+ with: torch, triton, transformers, numpy
#   - ~80GB GPU memory for full-scale benchmarks
#   - Optional: MS MARCO passage collection for real-data evaluation
#
# Environment variables:
#   DATA_ROOT: Path to datasets directory (default: ../datasets)
#
# Outputs results to experiment_results/*.json

set -e

source ~/.bashrc
conda activate retrieval_research

cd "$(dirname "$(dirname "$(readlink -f "$0")")")"

echo "=========================================="
echo "TileMaxSim: Reproducing All Paper Results"
echo "=========================================="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'No GPU')"
echo "Date: $(date)"
echo "Working dir: $(pwd)"
echo ""

mkdir -p experiment_results tracker

# ============================================================
# Phase 1: Core kernel benchmarks (Table 1 in paper)
# ============================================================
echo "=== Phase 1: Core Kernel Benchmarks ==="
echo "Running: cycle2_benchmark.py (primary throughput results)"
python src/cycle2_benchmark.py
echo ""

echo "Running: cycle6_rerun_scaling.py (scaling experiments)"
python src/cycle6_rerun_scaling.py
echo ""

# ============================================================
# Phase 2: Large-scale benchmark (500K passages)
# ============================================================
echo "=== Phase 2: Large-Scale Benchmark ==="
echo "Running: cycle2_largescale.py"
python src/cycle2_largescale.py
echo ""

# ============================================================
# Phase 3: Real MS MARCO evaluation (if data available)
# ============================================================
echo "=== Phase 3: Real MS MARCO Evaluation ==="
echo "Running: encode_real_msmarco.py (encode + benchmark)"
python src/encode_real_msmarco.py
echo ""

# ============================================================
# Phase 4: End-to-end retrieval quality
# ============================================================
echo "=== Phase 4: E2E Retrieval Quality ==="
echo "Running: e2e_retrieval_quality.py (MRR@10 verification)"
python src/e2e_retrieval_quality.py
echo ""

# ============================================================
# Phase 5: Detailed profiling (bandwidth utilization)
# ============================================================
echo "=== Phase 5: Profiling ==="
echo "Running: ncu_profile.py (memory throughput analysis)"
python src/ncu_profile.py
echo ""

# ============================================================
# Summary
# ============================================================
echo ""
echo "=========================================="
echo "All experiments complete."
echo "Results:"
echo "=========================================="
for f in experiment_results/*.json; do
    echo "  $f ($(wc -c < "$f") bytes)"
done
echo ""
echo "Key result files for paper tables:"
echo "  cycle2_benchmark.json     -> Table 1 (throughput, bandwidth)"
echo "  real_msmarco_benchmark.json -> Table 2 (real-data throughput)"
echo "  e2e_retrieval_quality.json  -> Table 3 (MRR@10, quality)"
echo "  cycle6_scaling.json        -> Figure 2 (scaling curves)"
