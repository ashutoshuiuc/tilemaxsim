"""
Cycle 3 experiments:
1. Profiling V2-MQ kernel
2. Dimension tiling benchmark (d=128, 256, 384, 768)
3. End-to-end ColBERT retrieval demo
"""

import torch
import torch.nn.functional as F
import time
import json
import os
import sys
import numpy as np
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

sys.path.insert(0, str(Path(__file__).parent))

PROJECT_ROOT = Path(__file__).parent.parent
from flash_maxsim_v2 import flash_maxsim_v2_multiquery
from flash_maxsim_kernel import pytorch_maxsim_naive, pytorch_maxsim_loop

device = "cuda:0"


def precise_timing(fn, warmup=10, repeat=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return np.array(times)


def benchmark_dimension_tiling():
    """Test V2-MQ across d=64..768 with the new dimension-tiled kernel."""
    print("=" * 80)
    print("DIMENSION TILING BENCHMARK")
    print("=" * 80)

    Nq = 32
    Nd = 128
    B = 10000
    results = {}

    for d in [64, 128, 256, 384, 768]:
        Q = torch.randn(Nq, d, dtype=torch.float16, device=device)
        D = torch.randn(B, Nd, d, dtype=torch.float16, device=device)

        # V2-MQ
        try:
            times_mq = precise_timing(lambda: flash_maxsim_v2_multiquery(Q, D), warmup=5, repeat=30)
            mean_ms = float(np.mean(times_mq))
            io_bytes = Nq * d * 2 + B * Nd * d * 2 + B * 4
            bw_gbs = io_bytes / (mean_ms / 1000) / 1e9
            throughput = B / (mean_ms / 1000)

            results[f"d{d}"] = {
                "d": d,
                "method": "V2-MQ",
                "mean_ms": mean_ms,
                "std_ms": float(np.std(times_mq)),
                "throughput": throughput,
                "bw_gbs": bw_gbs,
                "bw_pct": bw_gbs / 3350 * 100,
            }
            print(f"  d={d:>3}: V2-MQ = {mean_ms:.3f}ms | {throughput:.2e} docs/s | BW={bw_gbs:.0f} GB/s ({bw_gbs/3350*100:.1f}%)")
        except Exception as e:
            print(f"  d={d:>3}: V2-MQ ERROR: {e}")
            results[f"d{d}"] = {"d": d, "method": "V2-MQ", "error": str(e)}

        # PyTorch Naive baseline
        try:
            times_naive = precise_timing(lambda: pytorch_maxsim_naive(Q, D), warmup=3, repeat=20)
            mean_naive = float(np.mean(times_naive))
            tp_naive = B / (mean_naive / 1000)
            results[f"d{d}"]["naive_ms"] = mean_naive
            results[f"d{d}"]["naive_throughput"] = tp_naive
            if "mean_ms" in results[f"d{d}"]:
                speedup = mean_naive / results[f"d{d}"]["mean_ms"]
                results[f"d{d}"]["speedup_vs_naive"] = speedup
                print(f"         PT Naive = {mean_naive:.3f}ms | {tp_naive:.2e} docs/s | {speedup:.1f}x speedup")
        except torch.cuda.OutOfMemoryError:
            print(f"         PT Naive: OOM")
            torch.cuda.empty_cache()

        # Correctness check
        if B <= 1000 or d <= 256:
            try:
                Q_small = Q[:, :d].contiguous() if d <= 768 else Q
                D_small = D[:100]
                s_mq = flash_maxsim_v2_multiquery(Q_small[:Nq], D_small)
                s_naive = pytorch_maxsim_naive(Q_small[:Nq], D_small)
                max_diff = (s_mq - s_naive).abs().max().item()
                results[f"d{d}"]["correctness_max_diff"] = max_diff
                print(f"         Correctness (100 docs): max_diff = {max_diff:.6f}")
            except Exception as e:
                print(f"         Correctness check failed: {e}")

        torch.cuda.empty_cache()

    return results


def benchmark_profiling():
    """Detailed profiling of V2-MQ kernel."""
    print("\n" + "=" * 80)
    print("DETAILED PROFILING (torch.profiler)")
    print("=" * 80)

    Nq, d, Nd = 32, 128, 128
    results = {}

    for B in [1000, 10000, 100000]:
        Q = torch.randn(Nq, d, dtype=torch.float16, device=device)
        D = torch.randn(B, Nd, d, dtype=torch.float16, device=device)

        # Warmup
        for _ in range(10):
            flash_maxsim_v2_multiquery(Q, D)
        torch.cuda.synchronize()

        # Profile
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
        ) as prof:
            for _ in range(5):
                flash_maxsim_v2_multiquery(Q, D)
                torch.cuda.synchronize()

        # Print kernel-level stats
        print(f"\n--- B={B} ---")
        table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)
        print(table)

        # Extract top kernel info
        kernels = []
        for evt in prof.key_averages():
            cuda_t = getattr(evt, 'cuda_time_total', 0) or getattr(evt, 'device_time_total', 0) or 0
            if cuda_t > 0 or evt.cpu_time_total > 0:
                kernels.append({
                    "name": evt.key[:60],
                    "cuda_time_us": cuda_t / max(evt.count, 1),
                    "cpu_time_us": evt.cpu_time_total / max(evt.count, 1),
                    "count": evt.count,
                })
        results[f"B{B}"] = {"B": B, "kernels": kernels[:5]}

        # Detailed timing
        times = precise_timing(lambda: flash_maxsim_v2_multiquery(Q, D))
        io_bytes = Nq * d * 2 + B * Nd * d * 2 + B * 4
        bw_gbs = io_bytes / (float(np.mean(times)) / 1000) / 1e9

        results[f"B{B}"]["mean_ms"] = float(np.mean(times))
        results[f"B{B}"]["std_ms"] = float(np.std(times))
        results[f"B{B}"]["bw_gbs"] = bw_gbs
        results[f"B{B}"]["bw_pct"] = bw_gbs / 3350 * 100

        torch.cuda.empty_cache()

    return results


def colbert_retrieval_demo():
    """
    End-to-end ColBERT retrieval demo:
    1. Create a "corpus" of document embeddings
    2. Score all documents against a query using TileMaxSim
    3. Retrieve top-k and measure end-to-end latency
    """
    print("\n" + "=" * 80)
    print("END-TO-END ColBERT RETRIEVAL DEMO")
    print("=" * 80)

    Nq, d, Nd = 32, 128, 128
    results = {}

    for corpus_size in [10000, 50000, 100000, 500000]:
        print(f"\n--- Corpus: {corpus_size} documents ---")

        # Create corpus (GPU-resident)
        corpus = torch.randn(corpus_size, Nd, d, dtype=torch.float16, device=device)
        corpus = F.normalize(corpus.view(-1, d), dim=-1).view(corpus_size, Nd, d)

        # Create query
        query = torch.randn(Nq, d, dtype=torch.float16, device=device)
        query = F.normalize(query, dim=-1)

        # Warmup
        for _ in range(3):
            flash_maxsim_v2_multiquery(query, corpus[:1000])
        torch.cuda.synchronize()

        # End-to-end: score all docs + top-k
        k = 10
        times_e2e = []
        for _ in range(20):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            # Score all documents
            scores = flash_maxsim_v2_multiquery(query, corpus)
            # Retrieve top-k
            topk_scores, topk_indices = torch.topk(scores, k=k)

            end.record()
            torch.cuda.synchronize()
            times_e2e.append(start.elapsed_time(end))

        times_e2e = np.array(times_e2e)
        mean_ms = float(np.mean(times_e2e))

        results[f"corpus_{corpus_size}"] = {
            "corpus_size": corpus_size,
            "mean_ms": mean_ms,
            "std_ms": float(np.std(times_e2e)),
            "throughput": corpus_size / (mean_ms / 1000),
            "top_k": k,
            "memory_gb": corpus_size * Nd * d * 2 / 1e9,
        }

        print(f"  E2E latency: {mean_ms:.2f} +/- {float(np.std(times_e2e)):.2f} ms")
        print(f"  Throughput: {corpus_size/(mean_ms/1000):.2e} docs/s")
        print(f"  Corpus memory: {corpus_size * Nd * d * 2 / 1e9:.1f} GB")
        print(f"  Top-{k} scores: {topk_scores[:5].tolist()}")

        del corpus
        torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    all_results = {}

    # 1. Dimension tiling
    all_results["dim_tiling"] = benchmark_dimension_tiling()

    # 2. Profiling
    all_results["profiling"] = benchmark_profiling()

    # 3. E2E demo
    all_results["e2e_demo"] = colbert_retrieval_demo()

    # Save
    out_path = PROJECT_ROOT / "experiment_results" / "cycle3_experiments.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")
