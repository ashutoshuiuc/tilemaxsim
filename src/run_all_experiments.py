"""
Comprehensive experiments for TileMaxSim paper submission.

Covers:
1. Embedding dimension scaling (64, 128, 256, 384, 768)
2. Query/doc token count scaling
3. Memory bandwidth utilization (achieved vs peak)
4. Precision comparison (FP32, FP16, BF16)
5. Multi-GPU scaling test (2x H100)
6. Precise timing with torch.cuda.Event
7. Ablation: tile size sensitivity
"""

import torch
import time
import json
import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flash_maxsim_kernel import (
    flash_maxsim_single, flash_maxsim_batch,
    pytorch_maxsim_naive, pytorch_maxsim_loop,
)
from flash_maxsim_v2 import flash_maxsim_v2, flash_maxsim_v2_multiquery
from flash_pqsim_kernel import TileMaxSimPQ, pytorch_pqsim_baseline

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "experiment_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def cuda_event_timer(fn, warmup=10, repeat=50):
    """Precise timing with CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(repeat)]

    for i in range(repeat):
        start_events[i].record()
        fn()
        end_events[i].record()

    torch.cuda.synchronize()
    times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    return {
        "mean_ms": sum(times_ms) / len(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "median_ms": sorted(times_ms)[len(times_ms) // 2],
        "std_ms": (sum((t - sum(times_ms)/len(times_ms))**2 for t in times_ms) / len(times_ms)) ** 0.5,
    }


# =============================================================================
# Experiment 1: Embedding Dimension Scaling
# =============================================================================
def exp_embedding_dim_scaling(device="cuda:0"):
    """Measure throughput as embedding dimension varies."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: Embedding Dimension Scaling")
    print("=" * 80)

    dims = [64, 128, 256, 384, 768]
    Nq = 32
    Nd = 128
    B = 10000
    results = []

    for d in dims:
        print(f"\n  d={d}:")
        Q = torch.randn(Nq, d, dtype=torch.float16, device=device)
        D = torch.randn(B, Nd, d, dtype=torch.float16, device=device)

        # TileMaxSim V1 (batched)
        try:
            timing = cuda_event_timer(lambda: flash_maxsim_batch(Q, D), warmup=5, repeat=30)
            tput = B / (timing["mean_ms"] / 1000)
            io_bytes = (Nq * d * 2 + B * Nd * d * 2 + B * Nq * 4)
            bw = io_bytes / (timing["mean_ms"] / 1000) / 1e9
            print(f"    V1 batch:  {timing['mean_ms']:8.3f} ms | {tput/1e6:.2f}M docs/s | {bw:.1f} GB/s")
            results.append({"method": "flash_v1", "d": d, "mean_ms": timing["mean_ms"],
                           "throughput": tput, "bw_gb_s": bw, **timing})
        except Exception as e:
            print(f"    V1 batch:  ERROR {e}")

        # TileMaxSim V2-MQ
        try:
            timing = cuda_event_timer(lambda: flash_maxsim_v2_multiquery(Q, D), warmup=5, repeat=30)
            tput = B / (timing["mean_ms"] / 1000)
            io_bytes = (Nq * d * 2 + B * Nd * d * 2 + B * 4)
            bw = io_bytes / (timing["mean_ms"] / 1000) / 1e9
            print(f"    V2-MQ:     {timing['mean_ms']:8.3f} ms | {tput/1e6:.2f}M docs/s | {bw:.1f} GB/s")
            results.append({"method": "flash_v2mq", "d": d, "mean_ms": timing["mean_ms"],
                           "throughput": tput, "bw_gb_s": bw, **timing})
        except Exception as e:
            print(f"    V2-MQ:     ERROR {e}")

        # PyTorch naive
        try:
            timing = cuda_event_timer(lambda: pytorch_maxsim_naive(Q, D), warmup=5, repeat=30)
            tput = B / (timing["mean_ms"] / 1000)
            print(f"    PT Naive:  {timing['mean_ms']:8.3f} ms | {tput/1e6:.2f}M docs/s")
            results.append({"method": "pytorch_naive", "d": d, "mean_ms": timing["mean_ms"],
                           "throughput": tput, **timing})
        except (torch.cuda.OutOfMemoryError, Exception) as e:
            print(f"    PT Naive:  ERROR/OOM")
            torch.cuda.empty_cache()

        torch.cuda.empty_cache()

    with open(os.path.join(OUTPUT_DIR, "exp1_dim_scaling.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


# =============================================================================
# Experiment 2: Query Token Count Scaling
# =============================================================================
def exp_query_token_scaling(device="cuda:0"):
    """Measure throughput as Nq varies."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: Query Token Count Scaling")
    print("=" * 80)

    Nq_values = [8, 16, 32, 64]
    d = 128
    Nd = 128
    B = 10000
    results = []

    for Nq in Nq_values:
        print(f"\n  Nq={Nq}:")
        Q = torch.randn(Nq, d, dtype=torch.float16, device=device)
        D = torch.randn(B, Nd, d, dtype=torch.float16, device=device)

        for method_name, fn in [
            ("flash_v2mq", lambda: flash_maxsim_v2_multiquery(Q, D)),
            ("flash_v1", lambda: flash_maxsim_batch(Q, D)),
            ("pytorch_naive", lambda: pytorch_maxsim_naive(Q, D)),
        ]:
            try:
                timing = cuda_event_timer(fn, warmup=5, repeat=30)
                tput = B / (timing["mean_ms"] / 1000)
                io_flash = Nq * d * 2 + B * Nd * d * 2 + B * 4
                bw = io_flash / (timing["mean_ms"] / 1000) / 1e9
                print(f"    {method_name:15s}: {timing['mean_ms']:8.3f} ms | {tput/1e6:.2f}M docs/s | {bw:.1f} GB/s")
                results.append({"method": method_name, "Nq": Nq, "mean_ms": timing["mean_ms"],
                               "throughput": tput, "bw_gb_s": bw, **timing})
            except (torch.cuda.OutOfMemoryError, Exception) as e:
                print(f"    {method_name:15s}: ERROR/OOM")
                torch.cuda.empty_cache()

        torch.cuda.empty_cache()

    with open(os.path.join(OUTPUT_DIR, "exp2_query_token_scaling.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


# =============================================================================
# Experiment 3: Document Token Count Scaling
# =============================================================================
def exp_doc_token_scaling(device="cuda:0"):
    """Measure throughput as Nd varies."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: Document Token Count Scaling")
    print("=" * 80)

    Nd_values = [32, 64, 128, 256, 512]
    d = 128
    Nq = 32
    B = 10000
    results = []

    for Nd in Nd_values:
        print(f"\n  Nd={Nd}:")
        # Check memory
        mem_needed = B * Nd * d * 2  # bytes
        if mem_needed > 30e9:
            B_use = int(30e9 / (Nd * d * 2))
            print(f"    Reducing B to {B_use} due to memory")
        else:
            B_use = B

        Q = torch.randn(Nq, d, dtype=torch.float16, device=device)
        D = torch.randn(B_use, Nd, d, dtype=torch.float16, device=device)

        for method_name, fn in [
            ("flash_v2mq", lambda: flash_maxsim_v2_multiquery(Q, D)),
            ("flash_v1", lambda: flash_maxsim_batch(Q, D)),
        ]:
            try:
                timing = cuda_event_timer(fn, warmup=5, repeat=30)
                tput = B_use / (timing["mean_ms"] / 1000)
                io_bytes = Nq * d * 2 + B_use * Nd * d * 2 + B_use * 4
                bw = io_bytes / (timing["mean_ms"] / 1000) / 1e9
                flops = B_use * Nq * Nd * (2 * d + 1)
                gflops = flops / (timing["mean_ms"] / 1000) / 1e9
                print(f"    {method_name:15s}: {timing['mean_ms']:8.3f} ms | {tput/1e6:.2f}M docs/s | {bw:.1f} GB/s | {gflops:.1f} GFLOP/s")
                results.append({"method": method_name, "Nd": Nd, "B": B_use,
                               "mean_ms": timing["mean_ms"], "throughput": tput,
                               "bw_gb_s": bw, "gflops": gflops, **timing})
            except Exception as e:
                print(f"    {method_name:15s}: ERROR {e}")

        # PyTorch naive (only smaller sizes)
        if B_use * Nd * Nq * 4 < 10e9:
            try:
                timing = cuda_event_timer(lambda: pytorch_maxsim_naive(Q, D), warmup=5, repeat=20)
                tput = B_use / (timing["mean_ms"] / 1000)
                print(f"    {'pytorch_naive':15s}: {timing['mean_ms']:8.3f} ms | {tput/1e6:.2f}M docs/s")
                results.append({"method": "pytorch_naive", "Nd": Nd, "B": B_use,
                               "mean_ms": timing["mean_ms"], "throughput": tput, **timing})
            except (torch.cuda.OutOfMemoryError, Exception):
                print(f"    {'pytorch_naive':15s}: OOM")
                torch.cuda.empty_cache()

        torch.cuda.empty_cache()

    with open(os.path.join(OUTPUT_DIR, "exp3_doc_token_scaling.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


# =============================================================================
# Experiment 4: Memory Bandwidth Utilization
# =============================================================================
def exp_bandwidth_utilization(device="cuda:0"):
    """Detailed bandwidth measurement with CUDA events."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 4: Memory Bandwidth Utilization")
    print("=" * 80)

    PEAK_BW_GB_S = 3350  # H100 HBM3 peak
    d = 128
    Nq = 32
    Nd = 128
    results = []

    for B in [100, 1000, 10000, 50000, 100000]:
        print(f"\n  B={B}:")
        Q = torch.randn(Nq, d, dtype=torch.float16, device=device)
        D = torch.randn(B, Nd, d, dtype=torch.float16, device=device)

        # Minimum IO for flash approach
        flash_io = Nq * d * 2 + B * Nd * d * 2 + B * 4
        # Naive IO includes sim matrix
        naive_io = Nq * d * 2 + B * Nd * d * 2 + 2 * B * Nq * Nd * 4

        methods = [
            ("V2-MQ", lambda: flash_maxsim_v2_multiquery(Q, D), flash_io),
            ("V1-batch", lambda: flash_maxsim_batch(Q, D), flash_io),
            ("V2-seq", lambda: flash_maxsim_v2(Q, D), flash_io),
        ]
        if B <= 50000:
            methods.append(("PT-Naive", lambda: pytorch_maxsim_naive(Q, D), naive_io))
        methods.append(("PT-Loop", lambda: pytorch_maxsim_loop(Q, D), flash_io))

        for name, fn, io_bytes in methods:
            try:
                timing = cuda_event_timer(fn, warmup=5, repeat=30)
                achieved_bw = io_bytes / (timing["mean_ms"] / 1000) / 1e9
                util_pct = achieved_bw / PEAK_BW_GB_S * 100
                print(f"    {name:12s}: {timing['mean_ms']:8.3f} ms | {achieved_bw:8.1f} GB/s ({util_pct:5.1f}% peak)")
                results.append({"method": name, "B": B, "mean_ms": timing["mean_ms"],
                               "bw_gb_s": achieved_bw, "util_pct": util_pct,
                               "io_bytes": io_bytes, **timing})
            except (torch.cuda.OutOfMemoryError, Exception) as e:
                print(f"    {name:12s}: ERROR/OOM")
                torch.cuda.empty_cache()

        torch.cuda.empty_cache()

    with open(os.path.join(OUTPUT_DIR, "exp4_bandwidth.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


# =============================================================================
# Experiment 5: Precision Comparison
# =============================================================================
def exp_precision_comparison(device="cuda:0"):
    """Compare FP32, FP16, BF16 performance."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 5: Precision Comparison")
    print("=" * 80)

    d = 128
    Nq = 32
    Nd = 128
    B = 10000
    results = []

    precisions = [
        ("FP32", torch.float32, 4),
        ("FP16", torch.float16, 2),
        ("BF16", torch.bfloat16, 2),
    ]

    for prec_name, dtype, bytes_per_elem in precisions:
        print(f"\n  {prec_name}:")
        Q = torch.randn(Nq, d, dtype=dtype, device=device)
        D = torch.randn(B, Nd, d, dtype=dtype, device=device)

        io_bytes = Nq * d * bytes_per_elem + B * Nd * d * bytes_per_elem + B * 4

        # V2-MQ
        try:
            timing = cuda_event_timer(lambda: flash_maxsim_v2_multiquery(Q, D), warmup=5, repeat=30)
            tput = B / (timing["mean_ms"] / 1000)
            bw = io_bytes / (timing["mean_ms"] / 1000) / 1e9
            print(f"    V2-MQ:      {timing['mean_ms']:8.3f} ms | {tput/1e6:.2f}M docs/s | {bw:.1f} GB/s")
            results.append({"method": "flash_v2mq", "precision": prec_name,
                           "mean_ms": timing["mean_ms"], "throughput": tput, "bw_gb_s": bw, **timing})
        except Exception as e:
            print(f"    V2-MQ:      ERROR {e}")

        # V1 batch
        try:
            timing = cuda_event_timer(lambda: flash_maxsim_batch(Q, D), warmup=5, repeat=30)
            tput = B / (timing["mean_ms"] / 1000)
            bw = io_bytes / (timing["mean_ms"] / 1000) / 1e9
            print(f"    V1 batch:   {timing['mean_ms']:8.3f} ms | {tput/1e6:.2f}M docs/s | {bw:.1f} GB/s")
            results.append({"method": "flash_v1", "precision": prec_name,
                           "mean_ms": timing["mean_ms"], "throughput": tput, "bw_gb_s": bw, **timing})
        except Exception as e:
            print(f"    V1 batch:   ERROR {e}")

        # PyTorch naive
        try:
            timing = cuda_event_timer(lambda: pytorch_maxsim_naive(Q, D), warmup=5, repeat=20)
            tput = B / (timing["mean_ms"] / 1000)
            print(f"    PT Naive:   {timing['mean_ms']:8.3f} ms | {tput/1e6:.2f}M docs/s")
            results.append({"method": "pytorch_naive", "precision": prec_name,
                           "mean_ms": timing["mean_ms"], "throughput": tput, **timing})
        except (torch.cuda.OutOfMemoryError, Exception) as e:
            print(f"    PT Naive:   ERROR/OOM")
            torch.cuda.empty_cache()

        torch.cuda.empty_cache()

    with open(os.path.join(OUTPUT_DIR, "exp5_precision.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


# =============================================================================
# Experiment 6: Multi-GPU Scaling
# =============================================================================
def exp_multi_gpu_scaling():
    """Test 1 vs 2 GPU throughput scaling."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 6: Multi-GPU Scaling")
    print("=" * 80)

    d = 128
    Nq = 32
    Nd = 128
    B_total = 100000
    results = []

    # Single GPU test
    device0 = "cuda:0"
    Q0 = torch.randn(Nq, d, dtype=torch.float16, device=device0)
    D0 = torch.randn(B_total, Nd, d, dtype=torch.float16, device=device0)

    print(f"\n  Single GPU (cuda:0), B={B_total}:")
    timing_1gpu = cuda_event_timer(lambda: flash_maxsim_v2_multiquery(Q0, D0), warmup=5, repeat=30)
    tput_1 = B_total / (timing_1gpu["mean_ms"] / 1000)
    print(f"    V2-MQ: {timing_1gpu['mean_ms']:8.3f} ms | {tput_1/1e6:.2f}M docs/s")
    results.append({"gpus": 1, "mean_ms": timing_1gpu["mean_ms"], "throughput": tput_1})

    del D0
    torch.cuda.empty_cache()

    # Two GPU test: split batch across GPUs
    try:
        device1 = "cuda:1"
        B_half = B_total // 2
        Q1 = torch.randn(Nq, d, dtype=torch.float16, device=device1)
        D0_half = torch.randn(B_half, Nd, d, dtype=torch.float16, device=device0)
        D1_half = torch.randn(B_half, Nd, d, dtype=torch.float16, device=device1)
        Q0 = torch.randn(Nq, d, dtype=torch.float16, device=device0)

        print(f"\n  Two GPUs (cuda:0 + cuda:1), B_per_gpu={B_half}:")

        # Warm up both
        for _ in range(5):
            flash_maxsim_v2_multiquery(Q0, D0_half)
            flash_maxsim_v2_multiquery(Q1, D1_half)
        torch.cuda.synchronize()

        # Time concurrent execution
        start_events_0 = [torch.cuda.Event(enable_timing=True) for _ in range(30)]
        end_events_0 = [torch.cuda.Event(enable_timing=True) for _ in range(30)]
        start_events_1 = [torch.cuda.Event(enable_timing=True) for _ in range(30)]
        end_events_1 = [torch.cuda.Event(enable_timing=True) for _ in range(30)]

        for i in range(30):
            # Launch on both GPUs concurrently
            with torch.cuda.device(device0):
                start_events_0[i].record()
                flash_maxsim_v2_multiquery(Q0, D0_half)
                end_events_0[i].record()
            with torch.cuda.device(device1):
                start_events_1[i].record()
                flash_maxsim_v2_multiquery(Q1, D1_half)
                end_events_1[i].record()

        torch.cuda.synchronize()

        times_0 = [s.elapsed_time(e) for s, e in zip(start_events_0, end_events_0)]
        times_1 = [s.elapsed_time(e) for s, e in zip(start_events_1, end_events_1)]
        # Wall time is max of the two
        wall_times = [max(t0, t1) for t0, t1 in zip(times_0, times_1)]
        mean_wall_ms = sum(wall_times) / len(wall_times)

        tput_2 = B_total / (mean_wall_ms / 1000)
        scaling = tput_2 / tput_1

        print(f"    GPU 0 avg: {sum(times_0)/len(times_0):8.3f} ms for {B_half} docs")
        print(f"    GPU 1 avg: {sum(times_1)/len(times_1):8.3f} ms for {B_half} docs")
        print(f"    Wall time: {mean_wall_ms:8.3f} ms for {B_total} docs total")
        print(f"    V2-MQ 2GPU: {tput_2/1e6:.2f}M docs/s | Scaling: {scaling:.2f}x")
        results.append({"gpus": 2, "mean_ms": mean_wall_ms, "throughput": tput_2,
                        "scaling_factor": scaling,
                        "gpu0_ms": sum(times_0)/len(times_0),
                        "gpu1_ms": sum(times_1)/len(times_1)})

        del D0_half, D1_half, Q0, Q1
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"    2 GPU test failed: {e}")

    with open(os.path.join(OUTPUT_DIR, "exp6_multi_gpu.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


# =============================================================================
# Experiment 7: Tile Size Ablation
# =============================================================================
def exp_tile_size_ablation(device="cuda:0"):
    """Ablation study on tile sizes for V2-MQ kernel."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 7: Tile Size Ablation (V2-MQ)")
    print("=" * 80)

    import triton

    d = 128
    Nq = 32
    Nd = 128
    B = 10000
    results = []

    Q = torch.randn(Nq, d, dtype=torch.float16, device=device)
    D = torch.randn(B, Nd, d, dtype=torch.float16, device=device)

    # Try different BLOCK_Nq values
    block_nq_values = [4, 8, 16, 32]
    block_nd_values = [32, 64, 128]

    for BLOCK_Nq in block_nq_values:
        for BLOCK_Nd in block_nd_values:
            try:
                # Create custom call with specific tile sizes
                from flash_maxsim_v2 import _flash_maxsim_v2_multiquery_kernel

                scores = torch.zeros(B, dtype=torch.float32, device=device)
                num_q_blocks = (Nq + BLOCK_Nq - 1) // BLOCK_Nq
                grid = (B, num_q_blocks)

                def run_tiled():
                    scores.zero_()
                    _flash_maxsim_v2_multiquery_kernel[grid](
                        Q, D, scores,
                        Nq, Nd, d,
                        BLOCK_Nq=BLOCK_Nq,
                        BLOCK_Nd=BLOCK_Nd,
                    )

                timing = cuda_event_timer(run_tiled, warmup=5, repeat=30)
                tput = B / (timing["mean_ms"] / 1000)
                io_bytes = Nq * d * 2 + B * Nd * d * 2 + B * 4
                bw = io_bytes / (timing["mean_ms"] / 1000) / 1e9
                print(f"  BQ={BLOCK_Nq:2d}, BN={BLOCK_Nd:3d}: {timing['mean_ms']:8.3f} ms | {tput/1e6:.2f}M docs/s | {bw:.1f} GB/s")
                results.append({"BLOCK_Nq": BLOCK_Nq, "BLOCK_Nd": BLOCK_Nd,
                               "mean_ms": timing["mean_ms"], "throughput": tput,
                               "bw_gb_s": bw, **timing})
            except Exception as e:
                print(f"  BQ={BLOCK_Nq:2d}, BN={BLOCK_Nd:3d}: ERROR {e}")

    torch.cuda.empty_cache()

    with open(os.path.join(OUTPUT_DIR, "exp7_tile_ablation.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


# =============================================================================
# Experiment 8: Large-Scale Batch Scaling (100 to 1M docs)
# =============================================================================
def exp_batch_scaling(device="cuda:0"):
    """Throughput vs batch size from 100 to 1M."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 8: Batch Scaling (100 to 1M docs)")
    print("=" * 80)

    d = 128
    Nq = 32
    Nd = 128
    results = []

    batch_sizes = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]

    for B in batch_sizes:
        mem_needed = B * Nd * d * 2
        if mem_needed > 60e9:
            print(f"\n  B={B}: SKIP ({mem_needed/1e9:.1f} GB needed)")
            continue

        print(f"\n  B={B} ({mem_needed/1e9:.1f} GB):")
        try:
            Q = torch.randn(Nq, d, dtype=torch.float16, device=device)
            D = torch.randn(B, Nd, d, dtype=torch.float16, device=device)

            timing = cuda_event_timer(lambda: flash_maxsim_v2_multiquery(Q, D),
                                      warmup=3, repeat=20)
            tput = B / (timing["mean_ms"] / 1000)
            io_bytes = Nq * d * 2 + B * Nd * d * 2 + B * 4
            bw = io_bytes / (timing["mean_ms"] / 1000) / 1e9
            print(f"    V2-MQ: {timing['mean_ms']:8.3f} ms | {tput/1e6:.2f}M docs/s | {bw:.1f} GB/s")
            results.append({"method": "flash_v2mq", "B": B, "mean_ms": timing["mean_ms"],
                           "throughput": tput, "bw_gb_s": bw, **timing})

            del D
            torch.cuda.empty_cache()
        except (torch.cuda.OutOfMemoryError, Exception) as e:
            print(f"    ERROR: {e}")
            torch.cuda.empty_cache()

    with open(os.path.join(OUTPUT_DIR, "exp8_batch_scaling.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


# =============================================================================
# Experiment 9: PQ Scaling with dimensions
# =============================================================================
def exp_pq_scaling(device="cuda:0"):
    """PQ performance with different configurations."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 9: TileMaxSimPQ Scaling")
    print("=" * 80)

    results = []
    Nq = 32
    B = 10000

    configs = [
        # (M, K, dsub, Nd)
        (8, 256, 8, 128),    # d=64, smaller
        (16, 256, 8, 128),   # d=128, standard
        (32, 256, 8, 128),   # d=256, larger
        (16, 256, 8, 64),    # shorter docs
        (16, 256, 8, 256),   # longer docs
        (16, 256, 8, 512),   # very long docs
    ]

    for M, K, dsub, Nd in configs:
        d = M * dsub
        print(f"\n  M={M}, K={K}, dsub={dsub}, d={d}, Nd={Nd}:")

        codebook = torch.randn(M, K, dsub, dtype=torch.float16, device=device)
        Q = torch.randn(Nq, d, dtype=torch.float16, device=device)
        codes = torch.randint(0, K, (B, Nd, M), dtype=torch.uint8, device=device)
        pqsim = TileMaxSimPQ(codebook)

        try:
            timing = cuda_event_timer(lambda: pqsim.score_batch(Q, codes), warmup=5, repeat=30)
            tput = B / (timing["mean_ms"] / 1000)
            pq_io = Nq * M * K * 4 + B * Nd * M + B * Nq * 4
            bw = pq_io / (timing["mean_ms"] / 1000) / 1e9
            print(f"    TileMaxSimPQ: {timing['mean_ms']:8.3f} ms | {tput/1e6:.2f}M docs/s | {bw:.1f} GB/s")
            results.append({"method": "flash_pqsim", "M": M, "K": K, "dsub": dsub,
                           "d": d, "Nd": Nd, "mean_ms": timing["mean_ms"],
                           "throughput": tput, "bw_gb_s": bw, **timing})
        except Exception as e:
            print(f"    TileMaxSimPQ: ERROR {e}")

        torch.cuda.empty_cache()

    with open(os.path.join(OUTPUT_DIR, "exp9_pq_scaling.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


# =============================================================================
# Experiment 10: Correctness Verification at Scale
# =============================================================================
def exp_correctness(device="cuda:0"):
    """Verify correctness across all configs."""
    print("\n" + "=" * 80)
    print("EXPERIMENT 10: Correctness Verification")
    print("=" * 80)

    results = []
    torch.manual_seed(42)

    configs = [
        (8, 32, 64, 100),
        (16, 64, 128, 100),
        (32, 128, 128, 100),
        (32, 128, 256, 100),
        (32, 256, 128, 100),
        (64, 128, 128, 100),
        (32, 128, 384, 50),
        (32, 128, 768, 20),
    ]

    for Nq, Nd, d, B in configs:
        Q = torch.randn(Nq, d, dtype=torch.float16, device=device)
        D = torch.randn(B, Nd, d, dtype=torch.float16, device=device)

        ref = pytorch_maxsim_naive(Q, D)
        v1 = flash_maxsim_batch(Q, D)

        try:
            v2mq = flash_maxsim_v2_multiquery(Q, D)
            diff_v2mq = (v2mq - ref).abs().max().item()
        except Exception:
            diff_v2mq = float('nan')

        diff_v1 = (v1 - ref).abs().max().item()

        ok = "PASS" if max(diff_v1, diff_v2mq if not math.isnan(diff_v2mq) else 0) < 1.0 else "FAIL"
        print(f"  Nq={Nq:2d}, Nd={Nd:3d}, d={d:3d}, B={B:3d}: V1 diff={diff_v1:.4f}, V2MQ diff={diff_v2mq:.4f} [{ok}]")
        results.append({"Nq": Nq, "Nd": Nd, "d": d, "B": B,
                       "diff_v1": diff_v1, "diff_v2mq": diff_v2mq, "status": ok})

    with open(os.path.join(OUTPUT_DIR, "exp10_correctness.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=int, default=0, help="Run specific experiment (0=all)")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    print(f"GPU 0: {torch.cuda.get_device_name(0)}")
    if torch.cuda.device_count() > 1:
        print(f"GPU 1: {torch.cuda.get_device_name(1)}")
    print(f"Total GPUs: {torch.cuda.device_count()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")

    all_results = {}

    if args.exp == 0 or args.exp == 10:
        all_results["correctness"] = exp_correctness(args.device)
    if args.exp == 0 or args.exp == 1:
        all_results["dim_scaling"] = exp_embedding_dim_scaling(args.device)
    if args.exp == 0 or args.exp == 2:
        all_results["query_scaling"] = exp_query_token_scaling(args.device)
    if args.exp == 0 or args.exp == 3:
        all_results["doc_scaling"] = exp_doc_token_scaling(args.device)
    if args.exp == 0 or args.exp == 4:
        all_results["bandwidth"] = exp_bandwidth_utilization(args.device)
    if args.exp == 0 or args.exp == 5:
        all_results["precision"] = exp_precision_comparison(args.device)
    if args.exp == 0 or args.exp == 6:
        all_results["multi_gpu"] = exp_multi_gpu_scaling()
    if args.exp == 0 or args.exp == 7:
        all_results["tile_ablation"] = exp_tile_size_ablation(args.device)
    if args.exp == 0 or args.exp == 8:
        all_results["batch_scaling"] = exp_batch_scaling(args.device)
    if args.exp == 0 or args.exp == 9:
        all_results["pq_scaling"] = exp_pq_scaling(args.device)

    print("\n\nAll experiments complete!")
    print(f"Results saved to {OUTPUT_DIR}/")
