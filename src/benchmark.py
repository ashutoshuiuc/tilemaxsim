"""
Comprehensive benchmarks for TileMaxSim and TileMaxSimPQ kernels.

Measures:
- Throughput (documents/second)
- Latency (ms)
- Memory bandwidth utilization (GB/s)
- Comparison against PyTorch baselines
- Scaling across document counts (100 to 1M)
"""

import torch
import time
import json
import os
import sys
import argparse
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flash_maxsim_kernel import (
    flash_maxsim_single, flash_maxsim_batch,
    pytorch_maxsim_naive, pytorch_maxsim_loop,
)
from flash_pqsim_kernel import TileMaxSimPQ, pytorch_pqsim_baseline


@dataclass
class BenchmarkResult:
    method: str
    num_docs: int
    num_query_tokens: int
    num_doc_tokens: int
    embedding_dim: int
    batch_size: int
    latency_ms: float
    throughput_docs_per_sec: float
    bandwidth_gb_per_sec: float
    flops_achieved: float
    arithmetic_intensity: float
    notes: str = ""


def benchmark_fn(fn, warmup=10, repeat=50):
    """Benchmark a function with proper CUDA synchronization."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Timed runs
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return elapsed / repeat  # seconds per call


def compute_maxsim_io_bytes(Nq, Nd, d, dtype_bytes=2):
    """
    Compute HBM I/O for different MaxSim implementations.

    Naive: Read Q (Nq*d*2), Read D (Nd*d*2), Write S (Nq*Nd*4), Read S (Nq*Nd*4), Write max (Nq*4)
    Flash: Read Q (Nq*d*2), Read D (Nd*d*2), Write max (Nq*4)
    """
    naive_bytes = (Nq * d * dtype_bytes +  # read Q
                   Nd * d * dtype_bytes +  # read D
                   2 * Nq * Nd * 4)  # write + read similarity matrix (fp32)
    flash_bytes = (Nq * d * dtype_bytes +  # read Q
                   Nd * d * dtype_bytes +  # read D
                   Nq * 4)  # write output
    return naive_bytes, flash_bytes


def compute_maxsim_flops(Nq, Nd, d):
    """FLOPs for MaxSim: Nq * Nd * (2*d) for matmul + Nq * Nd for max reduction."""
    return Nq * Nd * (2 * d + 1)


def run_maxsim_benchmarks(device="cuda:0", results_dir=None):
    """Run comprehensive MaxSim benchmarks."""
    results = []

    # Parameters
    d = 128
    Nq = 32
    Nd_values = [64, 128, 256]  # tokens per document
    # Scale: number of documents (we batch documents)
    num_docs_values = [100, 1000, 10000, 100000]

    print("=" * 80)
    print("TileMaxSim Benchmarks")
    print("=" * 80)
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Query tokens: {Nq}, Embedding dim: {d}")
    print()

    for Nd in Nd_values:
        print(f"\n--- Document token length: {Nd} ---")
        for num_docs in num_docs_values:
            # Determine batch size based on memory
            # Each doc: Nd * d * 2 bytes (fp16)
            doc_mem = Nd * d * 2
            max_batch = min(num_docs, int(4e9 / doc_mem))  # 4GB limit per batch
            batch_size = min(max_batch, num_docs)

            if batch_size < 1:
                print(f"  num_docs={num_docs}: SKIP (too large for GPU memory)")
                continue

            print(f"\n  num_docs={num_docs} (batch_size={batch_size}):")

            Q = torch.randn(Nq, d, dtype=torch.float16, device=device)
            D = torch.randn(batch_size, Nd, d, dtype=torch.float16, device=device)

            total_flops = compute_maxsim_flops(Nq, Nd, d) * batch_size
            naive_bytes, flash_bytes = compute_maxsim_io_bytes(Nq, Nd, d)
            naive_bytes *= batch_size
            flash_bytes *= batch_size

            methods = {}

            # 1. PyTorch Naive (only for smaller sizes)
            if batch_size <= 50000:
                try:
                    t = benchmark_fn(lambda: pytorch_maxsim_naive(Q, D), warmup=5, repeat=20)
                    methods["pytorch_naive"] = t
                    bw = naive_bytes / t / 1e9
                    tput = batch_size / t
                    ai = total_flops / naive_bytes
                    print(f"    PyTorch Naive:   {t*1000:8.3f} ms | {tput:12.0f} docs/s | {bw:8.2f} GB/s | AI={ai:.1f}")
                    results.append(BenchmarkResult(
                        "pytorch_naive", num_docs, Nq, Nd, d, batch_size,
                        t * 1000, tput, bw, total_flops / t, ai
                    ))
                except torch.cuda.OutOfMemoryError:
                    print(f"    PyTorch Naive:   OOM")
                    torch.cuda.empty_cache()

            # 2. PyTorch Loop
            try:
                t = benchmark_fn(lambda: pytorch_maxsim_loop(Q, D), warmup=5, repeat=20)
                methods["pytorch_loop"] = t
                bw = flash_bytes / t / 1e9  # loop approach has similar IO to flash
                tput = batch_size / t
                ai = total_flops / flash_bytes
                print(f"    PyTorch Loop:    {t*1000:8.3f} ms | {tput:12.0f} docs/s | {bw:8.2f} GB/s | AI={ai:.1f}")
                results.append(BenchmarkResult(
                    "pytorch_loop", num_docs, Nq, Nd, d, batch_size,
                    t * 1000, tput, bw, total_flops / t, ai
                ))
            except torch.cuda.OutOfMemoryError:
                print(f"    PyTorch Loop:    OOM")
                torch.cuda.empty_cache()

            # 3. TileMaxSim
            try:
                t = benchmark_fn(lambda: flash_maxsim_batch(Q, D), warmup=5, repeat=20)
                methods["flash_maxsim"] = t
                bw = flash_bytes / t / 1e9
                tput = batch_size / t
                ai = total_flops / flash_bytes
                print(f"    TileMaxSim:     {t*1000:8.3f} ms | {tput:12.0f} docs/s | {bw:8.2f} GB/s | AI={ai:.1f}")
                results.append(BenchmarkResult(
                    "flash_maxsim", num_docs, Nq, Nd, d, batch_size,
                    t * 1000, tput, bw, total_flops / t, ai
                ))
            except torch.cuda.OutOfMemoryError:
                print(f"    TileMaxSim:     OOM")
                torch.cuda.empty_cache()

            # Print speedups
            if "flash_maxsim" in methods:
                for baseline_name, baseline_t in methods.items():
                    if baseline_name != "flash_maxsim":
                        speedup = baseline_t / methods["flash_maxsim"]
                        print(f"    -> TileMaxSim vs {baseline_name}: {speedup:.2f}x speedup")

            torch.cuda.empty_cache()

    return results


def run_pqsim_benchmarks(device="cuda:0"):
    """Run PQ scoring benchmarks."""
    results = []

    M = 16       # sub-quantizers
    K = 256      # centroids
    dsub = 8     # sub-vector dim
    d = M * dsub  # 128
    Nq = 32

    print("\n" + "=" * 80)
    print("TileMaxSimPQ Benchmarks")
    print("=" * 80)

    Nd_values = [64, 128, 256]
    num_docs_values = [100, 1000, 10000, 100000]

    codebook = torch.randn(M, K, dsub, dtype=torch.float16, device=device)
    Q = torch.randn(Nq, d, dtype=torch.float16, device=device)
    pqsim = TileMaxSimPQ(codebook)

    for Nd in Nd_values:
        print(f"\n--- Document token length: {Nd} ---")
        for num_docs in num_docs_values:
            # PQ codes are much smaller: Nd * M bytes per doc
            doc_mem = Nd * M  # 1 byte per code
            batch_size = min(num_docs, int(4e9 / max(doc_mem, 1)))
            batch_size = min(batch_size, num_docs)

            if batch_size < 1:
                continue

            print(f"\n  num_docs={num_docs} (batch_size={batch_size}):")

            codes = torch.randint(0, K, (batch_size, Nd, M), dtype=torch.uint8, device=device)

            # PQ IO: read table (Nq*M*K*4) + read codes (B*Nd*M) + write output (B*Nq*4)
            pq_io_bytes = (Nq * M * K * 4 + batch_size * Nd * M + batch_size * Nq * 4)
            # Full IO: read Q (Nq*d*2) + decompress (B*Nd*d*4) + score (B*Nq*Nd*4*2)
            full_io_bytes = (Nq * d * 2 + batch_size * Nd * d * 4 + 2 * batch_size * Nq * Nd * 4)
            pq_flops = batch_size * Nq * Nd * M  # M lookups + additions per score

            # TileMaxSimPQ
            try:
                t = benchmark_fn(lambda: pqsim.score_batch(Q, codes), warmup=5, repeat=20)
                bw = pq_io_bytes / t / 1e9
                tput = batch_size / t
                ai = pq_flops / pq_io_bytes
                print(f"    TileMaxSimPQ:      {t*1000:8.3f} ms | {tput:12.0f} docs/s | {bw:8.2f} GB/s | AI={ai:.1f}")
                results.append(BenchmarkResult(
                    "flash_pqsim", num_docs, Nq, Nd, d, batch_size,
                    t * 1000, tput, bw, pq_flops / t, ai
                ))
            except torch.cuda.OutOfMemoryError:
                print(f"    TileMaxSimPQ:      OOM")
                torch.cuda.empty_cache()

            # PyTorch PQ baseline (decompress + score)
            if batch_size <= 10000:
                try:
                    t = benchmark_fn(
                        lambda: pytorch_pqsim_baseline(Q, codebook, codes),
                        warmup=3, repeat=10
                    )
                    bw = full_io_bytes / t / 1e9
                    tput = batch_size / t
                    ai = compute_maxsim_flops(Nq, Nd, d) * batch_size / full_io_bytes
                    print(f"    PyTorch PQ:      {t*1000:8.3f} ms | {tput:12.0f} docs/s | {bw:8.2f} GB/s | AI={ai:.1f}")
                    results.append(BenchmarkResult(
                        "pytorch_pq", num_docs, Nq, Nd, d, batch_size,
                        t * 1000, tput, bw,
                        compute_maxsim_flops(Nq, Nd, d) * batch_size / t, ai
                    ))
                except torch.cuda.OutOfMemoryError:
                    print(f"    PyTorch PQ:      OOM")
                    torch.cuda.empty_cache()

            torch.cuda.empty_cache()

    return results


def run_roofline_analysis(device="cuda:0"):
    """Compute roofline model parameters for H100."""
    print("\n" + "=" * 80)
    print("Roofline Analysis (H100 SXM)")
    print("=" * 80)

    # H100 specs
    peak_fp16_tflops = 1979  # fp16 Tensor Core
    peak_fp32_tflops = 989.5  # fp32 Tensor Core
    peak_bw_tb_s = 3.35  # HBM3 bandwidth TB/s
    sram_per_sm_kb = 228  # shared memory per SM
    num_sms = 132

    peak_fp16_flops = peak_fp16_tflops * 1e12
    peak_bw_bytes = peak_bw_tb_s * 1e12

    # Operational intensity crossover point
    crossover = peak_fp16_flops / peak_bw_bytes
    print(f"\nH100 SXM Specifications:")
    print(f"  Peak FP16 Tensor Core: {peak_fp16_tflops} TFLOP/s")
    print(f"  Peak HBM3 Bandwidth:   {peak_bw_tb_s} TB/s")
    print(f"  SRAM per SM:           {sram_per_sm_kb} KB")
    print(f"  Number of SMs:         {num_sms}")
    print(f"  Crossover AI:          {crossover:.1f} FLOP/byte")

    print(f"\nMaxSim Operational Intensity Analysis:")

    configs = [
        ("Nq=32, Nd=128, d=128", 32, 128, 128),
        ("Nq=32, Nd=256, d=128", 32, 256, 128),
        ("Nq=32, Nd=64, d=128", 32, 64, 128),
        ("Nq=64, Nd=128, d=128", 64, 128, 128),
    ]

    for name, Nq, Nd, d in configs:
        flops = compute_maxsim_flops(Nq, Nd, d)
        naive_bytes, flash_bytes = compute_maxsim_io_bytes(Nq, Nd, d)

        naive_ai = flops / naive_bytes
        flash_ai = flops / flash_bytes

        # Theoretical min time
        naive_time_compute = flops / peak_fp16_flops
        naive_time_memory = naive_bytes / peak_bw_bytes
        flash_time_compute = flops / peak_fp16_flops
        flash_time_memory = flash_bytes / peak_bw_bytes

        print(f"\n  {name}:")
        print(f"    FLOPs: {flops:,}")
        print(f"    Naive IO: {naive_bytes:,} bytes (AI = {naive_ai:.1f} FLOP/byte) -> {'compute' if naive_ai > crossover else 'MEMORY'}-bound")
        print(f"    Flash IO: {flash_bytes:,} bytes (AI = {flash_ai:.1f} FLOP/byte) -> {'compute' if flash_ai > crossover else 'MEMORY'}-bound")
        print(f"    Naive min time:  {max(naive_time_compute, naive_time_memory)*1e6:.2f} us (bottleneck: {'compute' if naive_time_compute > naive_time_memory else 'memory'})")
        print(f"    Flash min time:  {max(flash_time_compute, flash_time_memory)*1e6:.2f} us (bottleneck: {'compute' if flash_time_compute > flash_time_memory else 'memory'})")

    return crossover


def save_results(results: List[BenchmarkResult], output_path: str):
    """Save benchmark results to JSON."""
    data = [asdict(r) for r in results]
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TileMaxSim Benchmarks")
    parser.add_argument("--device", default="cuda:0", help="CUDA device")
    parser.add_argument("--output", default=None, help="Output JSON file")
    parser.add_argument("--maxsim-only", action="store_true")
    parser.add_argument("--pqsim-only", action="store_true")
    parser.add_argument("--roofline-only", action="store_true")
    args = parser.parse_args()

    all_results = []

    if not args.pqsim_only and not args.roofline_only:
        all_results.extend(run_maxsim_benchmarks(args.device))

    if not args.maxsim_only and not args.roofline_only:
        all_results.extend(run_pqsim_benchmarks(args.device))

    run_roofline_analysis(args.device)

    if args.output:
        save_results(all_results, args.output)
    else:
        output_path = os.path.join(os.path.dirname(__file__), "..", "benchmark_results.json")
        save_results(all_results, output_path)
