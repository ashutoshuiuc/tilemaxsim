"""
Profiling utilities for TileMaxSim kernels.
Uses torch.profiler to measure memory bandwidth, kernel execution time, and occupancy.
"""

import torch
import torch.profiler
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flash_maxsim_kernel import (
    flash_maxsim_single, flash_maxsim_batch,
    pytorch_maxsim_naive, pytorch_maxsim_loop,
)
from flash_pqsim_kernel import TileMaxSimPQ


def profile_kernel(fn, name, warmup=5, active=3, output_dir=None):
    """Profile a kernel with torch.profiler."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "..", "profiling_results")
    os.makedirs(output_dir, exist_ok=True)

    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range(active):
            fn()
            torch.cuda.synchronize()

    # Print summary
    print(f"\n{'='*60}")
    print(f"Profile: {name}")
    print(f"{'='*60}")

    table = prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=15,
    )
    print(table)

    # Export chrome trace
    trace_path = os.path.join(output_dir, f"{name}_trace.json")
    prof.export_chrome_trace(trace_path)
    print(f"Chrome trace saved to: {trace_path}")

    return prof


def measure_bandwidth_utilization(device="cuda:0"):
    """Measure actual vs theoretical memory bandwidth for MaxSim operations."""
    print("\n" + "=" * 60)
    print("Memory Bandwidth Utilization Analysis")
    print("=" * 60)

    # H100 theoretical peak
    peak_bw_gb_s = 3350  # GB/s HBM3

    configs = [
        (32, 128, 128, 1000, "Nq=32,Nd=128,B=1K"),
        (32, 128, 128, 10000, "Nq=32,Nd=128,B=10K"),
        (32, 128, 128, 100000, "Nq=32,Nd=128,B=100K"),
        (32, 256, 128, 1000, "Nq=32,Nd=256,B=1K"),
        (32, 256, 128, 10000, "Nq=32,Nd=256,B=10K"),
    ]

    results = {}

    for Nq, Nd, d, B, label in configs:
        doc_mem = B * Nd * d * 2  # fp16
        if doc_mem > 40e9:  # 40 GB limit
            print(f"\n  {label}: SKIP (would use {doc_mem/1e9:.1f} GB)")
            continue

        Q = torch.randn(Nq, d, dtype=torch.float16, device=device)
        D = torch.randn(B, Nd, d, dtype=torch.float16, device=device)

        # Data moved (minimum): read Q + read D + write output
        min_bytes = Nq * d * 2 + B * Nd * d * 2 + B * Nq * 4
        # Naive also writes/reads sim matrix
        naive_bytes = min_bytes + 2 * B * Nq * Nd * 4

        import time

        methods = [
            ("TileMaxSim", lambda: flash_maxsim_batch(Q, D), min_bytes),
            ("PyTorch Loop", lambda: pytorch_maxsim_loop(Q, D), min_bytes),
        ]

        if B <= 50000:
            methods.append(("PyTorch Naive", lambda: pytorch_maxsim_naive(Q, D), naive_bytes))

        print(f"\n  {label}:")
        for method_name, fn, io_bytes in methods:
            try:
                # Warmup
                for _ in range(5):
                    fn()
                torch.cuda.synchronize()

                # Time
                start = time.perf_counter()
                for _ in range(20):
                    fn()
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start) / 20

                achieved_bw = io_bytes / elapsed / 1e9
                utilization = achieved_bw / peak_bw_gb_s * 100

                print(f"    {method_name:20s}: {elapsed*1000:8.3f} ms | "
                      f"BW: {achieved_bw:8.1f} GB/s ({utilization:5.1f}% of peak)")

                results[f"{label}_{method_name}"] = {
                    "latency_ms": elapsed * 1000,
                    "bandwidth_gb_s": achieved_bw,
                    "utilization_pct": utilization,
                }
            except torch.cuda.OutOfMemoryError:
                print(f"    {method_name:20s}: OOM")
                torch.cuda.empty_cache()

        torch.cuda.empty_cache()

    return results


def profile_all(device="cuda:0"):
    """Run all profiling."""
    Nq, Nd, d = 32, 128, 128
    B = 1000

    Q = torch.randn(Nq, d, dtype=torch.float16, device=device)
    D = torch.randn(B, Nd, d, dtype=torch.float16, device=device)

    # Profile each method
    profile_kernel(
        lambda: flash_maxsim_batch(Q, D),
        "flash_maxsim_batch"
    )

    profile_kernel(
        lambda: pytorch_maxsim_loop(Q, D),
        "pytorch_loop"
    )

    if B <= 10000:
        profile_kernel(
            lambda: pytorch_maxsim_naive(Q, D),
            "pytorch_naive"
        )

    # PQ profiling
    M, K, dsub = 16, 256, 8
    codebook = torch.randn(M, K, dsub, dtype=torch.float16, device=device)
    Q_pq = torch.randn(Nq, M * dsub, dtype=torch.float16, device=device)
    codes = torch.randint(0, K, (B, Nd, M), dtype=torch.uint8, device=device)
    pqsim = TileMaxSimPQ(codebook)

    profile_kernel(
        lambda: pqsim.score_batch(Q_pq, codes),
        "flash_pqsim_batch"
    )

    # Bandwidth analysis
    bw_results = measure_bandwidth_utilization(device)

    # Save results
    output_dir = os.path.join(os.path.dirname(__file__), "..", "profiling_results")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "bandwidth_results.json"), 'w') as f:
        json.dump(bw_results, f, indent=2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bandwidth-only", action="store_true")
    args = parser.parse_args()

    if args.bandwidth_only:
        measure_bandwidth_utilization(args.device)
    else:
        profile_all(args.device)
