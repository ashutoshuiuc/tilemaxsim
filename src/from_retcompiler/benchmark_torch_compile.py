"""
Benchmark: RetrieverCompiler vs torch.compile for retrieval pipelines.

Compares the generated fused kernels against torch.compile(mode="max-autotune")
for dense, ColBERT, and SPLADE pipelines.

Usage:
    python src/benchmark_torch_compile.py [--device cuda:0]
"""

import torch
import torch.nn.functional as F
import time
import json
import os
import sys
import argparse
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

sys.path.insert(0, str(Path(__file__).parent))

PROJECT_ROOT = Path(__file__).parent.parent


def bench_fn(fn, warmup=5, trials=20, device="cuda:0"):
    """Benchmark with CUDA synchronization."""
    for _ in range(warmup):
        fn()
        torch.cuda.synchronize(device)

    times = []
    for _ in range(trials):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize(device)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    return {
        "mean_ms": sum(times) / len(times),
        "median_ms": sorted(times)[len(times) // 2],
        "min_ms": min(times),
    }


# ==================== Dense Pipeline ====================

def dense_pipeline_pytorch(queries, docs):
    """Dense retrieval: matmul + topk."""
    scores = queries @ docs.T
    topk_scores, topk_ids = scores.topk(100, dim=-1)
    return topk_scores, topk_ids


def dense_pipeline_compiled(queries, docs):
    """torch.compile version of dense pipeline."""
    return dense_pipeline_pytorch(queries, docs)


# ==================== ColBERT Pipeline ====================

def colbert_maxsim_pytorch(Q, D, D_lengths=None):
    """ColBERT MaxSim: per-query-token max over doc tokens, then sum."""
    # Q: [n_queries, n_qtokens, dim]
    # D: [n_docs, n_dtokens, dim]
    n_queries = Q.shape[0]
    n_docs = D.shape[0]

    scores = torch.zeros(n_queries, n_docs, device=Q.device, dtype=torch.float32)
    for qi in range(n_queries):
        # [n_qtokens, dim] @ [dim, n_docs * n_dtokens] -> reshape
        sim = torch.einsum("qd,ndd->qnd", Q[qi].float(), D.float())
        # sim: [n_qtokens, n_docs, n_dtokens]
        # Max over doc tokens, sum over query tokens
        max_sim = sim.max(dim=-1).values  # [n_qtokens, n_docs]
        scores[qi] = max_sim.sum(dim=0)  # [n_docs]

    topk_scores, topk_ids = scores.topk(min(100, n_docs), dim=-1)
    return topk_scores, topk_ids


def colbert_maxsim_batched(Q, D):
    """Batched ColBERT MaxSim (more torch.compile friendly)."""
    # Q: [n_queries, n_qtokens, dim], D: [n_docs, n_dtokens, dim]
    # sim[q, n, s, l] = Q[q, s, :] . D[n, l, :]
    sim = torch.einsum("qsd,nld->qnsl", Q.float(), D.float())
    # sim: [n_queries, n_docs, n_qtokens, n_dtokens]
    max_over_dtokens = sim.max(dim=-1).values  # [n_queries, n_docs, n_qtokens]
    scores = max_over_dtokens.sum(dim=-1)  # [n_queries, n_docs]
    topk_scores, topk_ids = scores.topk(min(100, scores.shape[1]), dim=-1)
    return topk_scores, topk_ids


# ==================== SPLADE Pipeline ====================

def splade_scoring_pytorch(q_sparse, d_sparse):
    """SPLADE scoring: sparse inner product via dense matmul on sparse vecs."""
    scores = q_sparse @ d_sparse.T
    topk_scores, topk_ids = scores.topk(100, dim=-1)
    return topk_scores, topk_ids


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = args.device
    torch.manual_seed(42)

    results = {}

    print("=" * 70)
    print("torch.compile vs RetrieverCompiler Benchmark")
    print("=" * 70)

    # ===== Dense Pipeline =====
    print("\n--- Dense Pipeline ---")
    n_queries = 32
    dim = 128

    for n_docs in [8192, 65536, 262144, 1000000]:
        print(f"  n_docs={n_docs:,}")

        queries = F.normalize(torch.randn(n_queries, dim, device=device, dtype=torch.float16), dim=-1)
        docs = F.normalize(torch.randn(n_docs, dim, device=device, dtype=torch.float16), dim=-1)

        # PyTorch baseline
        timing_pytorch = bench_fn(lambda: dense_pipeline_pytorch(queries, docs), device=device)

        # torch.compile
        compiled_fn = torch.compile(dense_pipeline_pytorch, mode="max-autotune")
        # Warmup compile
        for _ in range(3):
            compiled_fn(queries, docs)
            torch.cuda.synchronize(device)
        timing_compiled = bench_fn(lambda: compiled_fn(queries, docs), device=device)

        speedup = timing_pytorch["median_ms"] / timing_compiled["median_ms"]
        print(f"    PyTorch:        {timing_pytorch['median_ms']:.3f} ms")
        print(f"    torch.compile:  {timing_compiled['median_ms']:.3f} ms")
        print(f"    Speedup:        {speedup:.2f}x")

        results[f"dense_{n_docs}"] = {
            "n_docs": n_docs,
            "pytorch_ms": timing_pytorch["median_ms"],
            "torch_compile_ms": timing_compiled["median_ms"],
            "speedup": speedup,
        }

        del queries, docs
        torch.cuda.empty_cache()

    # ===== ColBERT Pipeline =====
    print("\n--- ColBERT MaxSim Pipeline ---")
    n_qtokens = 32
    n_dtokens = 64

    for n_docs in [256, 1024, 4096]:
        print(f"  n_docs={n_docs}")

        Q = F.normalize(torch.randn(4, n_qtokens, dim, device=device, dtype=torch.float16), dim=-1)
        D = F.normalize(torch.randn(n_docs, n_dtokens, dim, device=device, dtype=torch.float16), dim=-1)

        # PyTorch batched
        timing_pytorch = bench_fn(lambda: colbert_maxsim_batched(Q, D), device=device)

        # torch.compile
        compiled_fn = torch.compile(colbert_maxsim_batched, mode="max-autotune")
        for _ in range(3):
            compiled_fn(Q, D)
            torch.cuda.synchronize(device)
        timing_compiled = bench_fn(lambda: compiled_fn(Q, D), device=device)

        speedup = timing_pytorch["median_ms"] / timing_compiled["median_ms"]
        print(f"    PyTorch:        {timing_pytorch['median_ms']:.3f} ms")
        print(f"    torch.compile:  {timing_compiled['median_ms']:.3f} ms")
        print(f"    Speedup:        {speedup:.2f}x")

        results[f"colbert_{n_docs}"] = {
            "n_docs": n_docs,
            "pytorch_ms": timing_pytorch["median_ms"],
            "torch_compile_ms": timing_compiled["median_ms"],
            "speedup": speedup,
        }

        del Q, D
        torch.cuda.empty_cache()

    # ===== SPLADE Pipeline =====
    print("\n--- SPLADE Pipeline ---")
    vocab_size = 30522

    for n_docs in [8192, 65536]:
        print(f"  n_docs={n_docs}")

        # Sparse SPLADE vectors (avg ~100 nonzeros out of 30522)
        q_sparse = torch.zeros(n_queries, vocab_size, device=device, dtype=torch.float16)
        for i in range(n_queries):
            nnz = torch.randint(30, 80, (1,)).item()
            idx = torch.randperm(vocab_size)[:nnz]
            q_sparse[i, idx] = torch.rand(nnz, device=device, dtype=torch.float16)

        d_sparse = torch.zeros(n_docs, vocab_size, device=device, dtype=torch.float16)
        for i in range(n_docs):
            nnz = torch.randint(80, 200, (1,)).item()
            idx = torch.randperm(vocab_size)[:nnz]
            d_sparse[i, idx] = torch.rand(nnz, device=device, dtype=torch.float16)

        timing_pytorch = bench_fn(lambda: splade_scoring_pytorch(q_sparse, d_sparse), device=device)

        compiled_fn = torch.compile(splade_scoring_pytorch, mode="max-autotune")
        for _ in range(3):
            compiled_fn(q_sparse, d_sparse)
            torch.cuda.synchronize(device)
        timing_compiled = bench_fn(lambda: compiled_fn(q_sparse, d_sparse), device=device)

        speedup = timing_pytorch["median_ms"] / timing_compiled["median_ms"]
        print(f"    PyTorch:        {timing_pytorch['median_ms']:.3f} ms")
        print(f"    torch.compile:  {timing_compiled['median_ms']:.3f} ms")
        print(f"    Speedup:        {speedup:.2f}x")

        results[f"splade_{n_docs}"] = {
            "n_docs": n_docs,
            "pytorch_ms": timing_pytorch["median_ms"],
            "torch_compile_ms": timing_compiled["median_ms"],
            "speedup": speedup,
        }

        del q_sparse, d_sparse
        torch.cuda.empty_cache()

    # Save
    out_path = PROJECT_ROOT / "final_results" / "torch_compile_comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
