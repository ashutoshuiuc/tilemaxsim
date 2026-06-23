"""
Detailed GPU profiling for TileMaxSim kernels.
Uses torch.profiler to get per-kernel breakdown of:
- Achieved FLOPS
- Memory bandwidth
- Kernel utilization
- SM occupancy
"""

import torch
import torch.profiler
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from flash_maxsim_kernel import flash_maxsim_batch, pytorch_maxsim_naive, pytorch_maxsim_loop
from flash_maxsim_v2 import flash_maxsim_v2, flash_maxsim_v2_multiquery
from flash_pqsim_kernel import TileMaxSimPQ

RESULTS_DIR = Path(__file__).parent.parent / "experiment_results"
RESULTS_DIR.mkdir(exist_ok=True)


def profile_with_torch_profiler(fn, name, warmup=10, active=5):
    """Profile using torch.profiler and extract key metrics."""
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
        with_flops=True,
    ) as prof:
        for _ in range(active):
            fn()
            torch.cuda.synchronize()

    # Extract kernel-level info
    events = prof.key_averages()
    kernels = []

    for evt in events:
        if evt.device_type == torch.autograd.DeviceType.CUDA or (hasattr(evt, 'cuda_time_total') and evt.cuda_time_total > 0):
            kernels.append({
                "name": evt.key[:80],
                "cuda_time_total_us": evt.cuda_time_total,
                "cuda_time_avg_us": evt.cuda_time_total / max(evt.count, 1),
                "count": evt.count,
                "flops": evt.flops if hasattr(evt, 'flops') else 0,
                "cpu_time_total_us": evt.cpu_time_total,
            })

    # Sort by CUDA time
    kernels.sort(key=lambda x: x["cuda_time_total_us"], reverse=True)

    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=15)

    return {
        "name": name,
        "table": table,
        "kernels": kernels[:20],
        "total_cuda_us": sum(k["cuda_time_total_us"] for k in kernels),
    }


def benchmark_with_events(fn, name, warmup=10, repeat=100):
    """Precise timing with CUDA events + bandwidth calculation."""
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
        "name": name,
        "mean_ms": sum(times_ms) / len(times_ms),
        "min_ms": min(times_ms),
        "median_ms": sorted(times_ms)[len(times_ms) // 2],
        "std_ms": (sum((t - sum(times_ms)/len(times_ms))**2 for t in times_ms) / len(times_ms)) ** 0.5,
        "p95_ms": sorted(times_ms)[int(0.95 * len(times_ms))],
        "p99_ms": sorted(times_ms)[int(0.99 * len(times_ms))],
    }


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    device = "cuda:0"

    print("=" * 80)
    print("Detailed GPU Profiling for TileMaxSim")
    print("=" * 80)
    print(f"Device: {torch.cuda.get_device_name(device)}")

    # H100 specs
    H100_PEAK_BW_GBs = 3350  # GB/s
    H100_PEAK_FP16_TFLOPS = 1979

    results = {}

    # Standard config
    Nq, Nd, d = 32, 128, 128
    configs = [
        (1000, "B=1K"),
        (10000, "B=10K"),
        (100000, "B=100K"),
    ]

    for B, label in configs:
        print(f"\n{'='*60}")
        print(f"Config: Nq={Nq}, Nd={Nd}, d={d}, {label}")
        print(f"{'='*60}")

        Q = torch.randn(Nq, d, dtype=torch.float16, device=device)
        D = torch.randn(B, Nd, d, dtype=torch.float16, device=device)

        # IO calculations
        flash_io_bytes = Nq * d * 2 + B * Nd * d * 2 + B * 4  # Q read + D read + output write
        naive_io_bytes = flash_io_bytes + 2 * B * Nq * Nd * 4  # + similarity matrix RW
        total_flops = B * Nq * Nd * (2 * d + 1)

        methods = [
            ("V2-MQ", lambda: flash_maxsim_v2_multiquery(Q, D)),
            ("V1-Batch", lambda: flash_maxsim_batch(Q, D)),
        ]
        if B <= 50000:
            methods.append(("PT-Naive", lambda: pytorch_maxsim_naive(Q, D)))

        for method_name, fn in methods:
            try:
                # Timing
                timing = benchmark_with_events(fn, method_name, warmup=10, repeat=50)

                # Calculate achieved metrics
                elapsed_s = timing["mean_ms"] / 1000
                io_bytes = flash_io_bytes if "V2" in method_name or "V1" in method_name else naive_io_bytes
                achieved_bw = io_bytes / elapsed_s / 1e9
                achieved_flops = total_flops / elapsed_s / 1e12
                bw_util = achieved_bw / H100_PEAK_BW_GBs * 100
                throughput = B / elapsed_s

                timing.update({
                    "achieved_bw_gb_s": achieved_bw,
                    "bw_utilization_pct": bw_util,
                    "achieved_tflops": achieved_flops,
                    "throughput_docs_per_s": throughput,
                    "io_bytes": io_bytes,
                    "total_flops": total_flops,
                    "arithmetic_intensity": total_flops / io_bytes,
                })

                print(f"\n  {method_name}:")
                print(f"    Latency: {timing['mean_ms']:.3f}ms (p95={timing['p95_ms']:.3f}, p99={timing['p99_ms']:.3f})")
                print(f"    Throughput: {throughput/1e6:.2f}M docs/s")
                print(f"    Achieved BW: {achieved_bw:.1f} GB/s ({bw_util:.1f}% of {H100_PEAK_BW_GBs} GB/s peak)")
                print(f"    Achieved FLOPS: {achieved_flops:.2f} TFLOP/s ({achieved_flops/H100_PEAK_FP16_TFLOPS*100:.2f}% of peak)")
                print(f"    Arithmetic Intensity: {total_flops/io_bytes:.1f} FLOP/byte")

                # Profile kernel breakdown (only for B=10K to keep profiling fast)
                if B == 10000:
                    prof_result = profile_with_torch_profiler(fn, f"{method_name}_{label}")
                    print(f"\n    Kernel breakdown:")
                    print(prof_result["table"])
                    timing["kernel_breakdown"] = prof_result["kernels"]

                results[f"{label}_{method_name}"] = timing

            except Exception as e:
                print(f"  {method_name}: ERROR {e}")

        torch.cuda.empty_cache()

    # PQ profiling
    print(f"\n{'='*60}")
    print("TileMaxSimPQ Profiling")
    print(f"{'='*60}")

    M, K, dsub = 16, 256, 8
    d = M * dsub
    codebook = torch.randn(M, K, dsub, dtype=torch.float16, device=device)
    Q_pq = torch.randn(Nq, d, dtype=torch.float16, device=device)
    pqsim = TileMaxSimPQ(codebook)

    for B, label in [(10000, "B=10K"), (100000, "B=100K")]:
        codes = torch.randint(0, K, (B, Nd, M), dtype=torch.uint8, device=device)

        timing = benchmark_with_events(
            lambda: pqsim.score_batch(Q_pq, codes),
            f"TileMaxSimPQ_{label}", warmup=10, repeat=50
        )

        pq_io = Nq * M * K * 4 + B * Nd * M + B * Nq * 4
        elapsed_s = timing["mean_ms"] / 1000
        achieved_bw = pq_io / elapsed_s / 1e9
        throughput = B / elapsed_s

        timing.update({
            "achieved_bw_gb_s": achieved_bw,
            "throughput_docs_per_s": throughput,
        })

        print(f"\n  {label}:")
        print(f"    Latency: {timing['mean_ms']:.3f}ms")
        print(f"    Throughput: {throughput/1e6:.2f}M docs/s")
        print(f"    Achieved BW: {achieved_bw:.1f} GB/s")

        results[f"PQ_{label}"] = timing
        torch.cuda.empty_cache()

    # Save
    out_path = RESULTS_DIR / "detailed_profiling.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
