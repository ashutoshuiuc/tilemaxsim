"""
Profile TileMaxSim V2-MQ kernel using torch.profiler (CUPTI-based).
Extracts occupancy, kernel details, and memory throughput.
"""

import torch
import torch.cuda
import os
import sys
import json
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

sys.path.insert(0, str(Path(__file__).parent))

PROJECT_ROOT = Path(__file__).parent.parent
from flash_maxsim_v2 import flash_maxsim_v2_multiquery

device = "cuda:0"
d = 128
Nq = 32
Nd = 128


def profile_kernel(B, num_warmup=10, num_active=5):
    """Profile the V2-MQ kernel using torch.profiler."""
    Q = torch.randn(Nq, d, dtype=torch.float16, device=device)
    D = torch.randn(B, Nd, d, dtype=torch.float16, device=device)

    # Warmup
    for _ in range(num_warmup):
        flash_maxsim_v2_multiquery(Q, D)
    torch.cuda.synchronize()

    # Profile
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=True,
        schedule=torch.profiler.schedule(
            wait=0, warmup=2, active=num_active
        ),
    ) as prof:
        for step in range(2 + num_active):
            flash_maxsim_v2_multiquery(Q, D)
            torch.cuda.synchronize()
            prof.step()

    # Extract kernel events
    events = prof.key_averages()
    result = {
        "B": B,
        "kernels": [],
    }

    print(f"\n{'='*80}")
    print(f"Profile for B={B}")
    print(f"{'='*80}")
    print(events.table(sort_by="cuda_time_total", row_limit=20))

    for event in events:
        if event.device_type == torch.autograd.DeviceType.CUDA or event.cuda_time_total > 0:
            result["kernels"].append({
                "name": event.key,
                "cuda_time_us": event.cuda_time_total / max(event.count, 1),
                "count": event.count,
                "cpu_time_us": event.cpu_time_total / max(event.count, 1),
            })

    return result


def profile_with_memory_events(B):
    """Profile with detailed memory info."""
    Q = torch.randn(Nq, d, dtype=torch.float16, device=device)
    D = torch.randn(B, Nd, d, dtype=torch.float16, device=device)

    # Warmup
    for _ in range(10):
        flash_maxsim_v2_multiquery(Q, D)
    torch.cuda.synchronize()

    # Measure kernel launch overhead
    import time
    times_cpu = []
    times_gpu = []
    for _ in range(50):
        torch.cuda.synchronize()
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)

        cpu_start = time.perf_counter()
        start_evt.record()
        flash_maxsim_v2_multiquery(Q, D)
        end_evt.record()
        torch.cuda.synchronize()
        cpu_end = time.perf_counter()

        times_cpu.append((cpu_end - cpu_start) * 1000)  # ms
        times_gpu.append(start_evt.elapsed_time(end_evt))

    import numpy as np
    cpu_times = np.array(times_cpu)
    gpu_times = np.array(times_gpu)

    # IO calculations
    io_bytes = Nq * d * 2 + B * Nd * d * 2 + B * 4
    flops = B * Nq * Nd * (2 * d)

    gpu_mean = float(np.mean(gpu_times))
    bw_gbs = io_bytes / (gpu_mean / 1000) / 1e9
    tflops = flops / (gpu_mean / 1000) / 1e12
    ai = flops / io_bytes

    result = {
        "B": B,
        "gpu_mean_ms": gpu_mean,
        "gpu_std_ms": float(np.std(gpu_times)),
        "cpu_mean_ms": float(np.mean(cpu_times)),
        "launch_overhead_ms": float(np.mean(cpu_times) - np.mean(gpu_times)),
        "io_bytes": io_bytes,
        "io_gb": io_bytes / 1e9,
        "flops": flops,
        "achieved_bw_gbs": bw_gbs,
        "achieved_bw_pct": bw_gbs / 3350 * 100,
        "achieved_tflops": tflops,
        "arithmetic_intensity": ai,
        "throughput_docs_per_s": B / (gpu_mean / 1000),
    }

    print(f"\nDetailed analysis for B={B}:")
    print(f"  GPU time: {gpu_mean:.3f} +/- {float(np.std(gpu_times)):.3f} ms")
    print(f"  CPU time: {float(np.mean(cpu_times)):.3f} ms (launch overhead: {result['launch_overhead_ms']:.3f} ms)")
    print(f"  IO: {io_bytes/1e9:.3f} GB (Q={Nq*d*2/1e6:.1f}MB + D={B*Nd*d*2/1e9:.3f}GB + Out={B*4/1e6:.1f}MB)")
    print(f"  Arithmetic Intensity: {ai:.1f} FLOP/byte (H100 crossover: 591)")
    print(f"  Achieved BW: {bw_gbs:.0f} GB/s ({bw_gbs/3350*100:.1f}% peak)")
    print(f"  Achieved Compute: {tflops:.1f} TFLOP/s ({tflops/1979*100:.1f}% peak)")
    print(f"  Throughput: {B/(gpu_mean/1000):.2e} docs/s")

    return result


if __name__ == "__main__":
    all_results = {}

    # Profile at different batch sizes
    for B in [1000, 10000, 100000]:
        all_results[f"profile_B{B}"] = profile_kernel(B)

    for B in [1000, 10000, 50000, 100000]:
        all_results[f"detailed_B{B}"] = profile_with_memory_events(B)

    # Save results
    out_path = PROJECT_ROOT / "experiment_results" / "cycle3_profiling.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")
