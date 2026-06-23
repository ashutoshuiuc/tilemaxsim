"""
Roofline model analysis for MaxSim operations on H100 GPU.

Computes theoretical performance bounds and compares against measured throughput.
Generates data for roofline plots (actual plotting done via matplotlib or LaTeX pgfplots).
"""

import json
import os
import sys
import math

# H100 SXM specifications
H100_SPECS = {
    "name": "NVIDIA H100 SXM",
    "peak_fp16_tensor_tflops": 1979,
    "peak_fp32_tensor_tflops": 989.5,
    "peak_fp16_cuda_tflops": 267.6,
    "peak_fp32_cuda_tflops": 133.8,
    "hbm3_bandwidth_tb_s": 3.35,
    "sram_per_sm_kb": 228,
    "num_sms": 132,
    "total_sram_mb": 228 * 132 / 1024,  # ~29.4 MB
    "hbm_capacity_gb": 80,
    "l2_cache_mb": 50,
}


def compute_roofline_point(flops, io_bytes, specs=H100_SPECS):
    """
    Compute roofline model metrics for a given operation.

    Returns:
        dict with arithmetic intensity, theoretical throughput limits,
        and bottleneck identification.
    """
    peak_flops = specs["peak_fp16_tensor_tflops"] * 1e12  # FLOP/s
    peak_bw = specs["hbm3_bandwidth_tb_s"] * 1e12  # bytes/s

    ai = flops / io_bytes  # FLOP/byte
    crossover_ai = peak_flops / peak_bw  # ~590 for H100

    # Roofline: min(peak_compute, ai * peak_bandwidth)
    compute_bound_perf = peak_flops  # FLOP/s
    memory_bound_perf = ai * peak_bw  # FLOP/s
    roofline_perf = min(compute_bound_perf, memory_bound_perf)

    # Time bounds
    time_compute = flops / peak_flops
    time_memory = io_bytes / peak_bw
    min_time = max(time_compute, time_memory)

    return {
        "flops": flops,
        "io_bytes": io_bytes,
        "arithmetic_intensity": ai,
        "crossover_ai": crossover_ai,
        "is_memory_bound": ai < crossover_ai,
        "roofline_perf_tflops": roofline_perf / 1e12,
        "time_compute_us": time_compute * 1e6,
        "time_memory_us": time_memory * 1e6,
        "min_time_us": min_time * 1e6,
        "bottleneck": "memory" if time_memory > time_compute else "compute",
    }


def maxsim_naive_analysis(Nq, Nd, d, B=1, dtype_bytes=2):
    """
    Analyze naive MaxSim: materialize full similarity matrix.

    IO: read Q(Nq*d*dtype) + read D(B*Nd*d*dtype) + write S(B*Nq*Nd*4) + read S(B*Nq*Nd*4)
    FLOPs: B * Nq * Nd * (2d) for matmul + B * Nq * Nd for max
    """
    flops = B * Nq * Nd * (2 * d + 1)
    io_bytes = (Nq * d * dtype_bytes +
                B * Nd * d * dtype_bytes +
                2 * B * Nq * Nd * 4)  # write + read sim matrix in fp32
    return compute_roofline_point(flops, io_bytes)


def maxsim_flash_analysis(Nq, Nd, d, B=1, dtype_bytes=2):
    """
    Analyze TileMaxSim: fused matmul + max, no sim matrix materialization.

    IO: read Q(Nq*d*dtype) + read D(B*Nd*d*dtype) + write out(B*Nq*4)
    FLOPs: same as naive (compute is the same, only IO differs)
    """
    flops = B * Nq * Nd * (2 * d + 1)
    io_bytes = (Nq * d * dtype_bytes +
                B * Nd * d * dtype_bytes +
                B * Nq * 4)
    return compute_roofline_point(flops, io_bytes)


def pqsim_analysis(Nq, Nd, M, K, dsub, B=1):
    """
    Analyze TileMaxSimPQ: lookup-table based scoring.

    IO: build table (Nq*M*dsub*2 + M*K*dsub*2) + read codes (B*Nd*M) + write output (B*Nq*4)
         + read table (Nq*M*K*4, but fits in SRAM so this is "free")
    FLOPs: Nq*M*K*2*dsub (table build) + B*Nq*Nd*M (lookups + adds)

    Key insight: the distance table (Nq*M*K*4 bytes) fits in SRAM for typical params.
    Nq=32, M=16, K=256 -> 32*16*256*4 = 512 KB. This fits in L2 cache (50MB on H100)
    and partially in SRAM (228KB per SM * 132 SMs = 29.4 MB total).
    """
    d = M * dsub
    # Table build FLOPs
    table_flops = Nq * M * K * (2 * dsub)
    # Scoring FLOPs (lookup + accumulate)
    score_flops = B * Nq * Nd * (2 * M)  # M lookups + M-1 adds per score
    total_flops = table_flops + score_flops

    # IO for table build
    table_io = (Nq * d * 2 +  # read Q
                M * K * dsub * 2 +  # read codebook
                Nq * M * K * 4)  # write table

    # IO for scoring (table in SRAM/L2, only read codes from HBM)
    score_io = (B * Nd * M +  # read codes (1 byte each)
                B * Nq * 4)  # write output

    total_io = table_io + score_io

    return compute_roofline_point(total_flops, total_io)


def pqsim_decompress_analysis(Nq, Nd, M, K, dsub, B=1):
    """
    Analyze traditional decompress+score approach.

    IO: read codes (B*Nd*M) + read codebook (M*K*dsub*2) + write decompressed (B*Nd*d*2)
        + then do standard MaxSim on decompressed vectors
    """
    d = M * dsub
    # Decompression IO
    decomp_io = (B * Nd * M +  # read codes
                 M * K * dsub * 2 +  # read codebook
                 B * Nd * d * 2)  # write decompressed vectors

    # Then standard naive MaxSim IO
    maxsim_io = (Nq * d * 2 +
                 B * Nd * d * 2 +
                 2 * B * Nq * Nd * 4)

    total_io = decomp_io + maxsim_io

    # FLOPs: decompression is just gather (negligible) + MaxSim
    total_flops = B * Nq * Nd * (2 * d + 1)

    return compute_roofline_point(total_flops, total_io)


def full_analysis():
    """Run complete roofline analysis and print results."""
    print("=" * 80)
    print("Roofline Analysis: MaxSim Operations on H100 SXM")
    print("=" * 80)

    specs = H100_SPECS
    crossover = specs["peak_fp16_tensor_tflops"] * 1e12 / (specs["hbm3_bandwidth_tb_s"] * 1e12)
    print(f"\nH100 Crossover AI: {crossover:.1f} FLOP/byte")
    print(f"(Operations with AI < {crossover:.0f} are memory-bound)\n")

    results = {}

    # Standard MaxSim analysis
    configs = [
        (32, 64, 128, "Nq=32,Nd=64"),
        (32, 128, 128, "Nq=32,Nd=128"),
        (32, 256, 128, "Nq=32,Nd=256"),
        (64, 128, 128, "Nq=64,Nd=128"),
    ]

    batch_sizes = [1, 100, 1000, 10000, 100000]

    print("MaxSim Analysis (per-document, B documents):")
    print("-" * 80)
    print(f"{'Config':<20} {'B':>8} {'Method':<12} {'AI':>8} {'Bound':>8} {'Min Time':>12} {'Peak GFLOP/s':>14}")
    print("-" * 80)

    for label, Nq, Nd, d in [("Nq=32,Nd=128", 32, 128, 128)]:
        for B in batch_sizes:
            naive = maxsim_naive_analysis(Nq, Nd, d, B)
            flash = maxsim_flash_analysis(Nq, Nd, d, B)

            print(f"{label:<20} {B:>8} {'Naive':<12} {naive['arithmetic_intensity']:>8.1f} "
                  f"{'MEM' if naive['is_memory_bound'] else 'CMP':>8} "
                  f"{naive['min_time_us']:>10.2f}us {naive['roofline_perf_tflops']*1000:>12.1f}")
            print(f"{'':<20} {'':<8} {'Flash':<12} {flash['arithmetic_intensity']:>8.1f} "
                  f"{'MEM' if flash['is_memory_bound'] else 'CMP':>8} "
                  f"{flash['min_time_us']:>10.2f}us {flash['roofline_perf_tflops']*1000:>12.1f}")

            results[f"{label}_B{B}_naive"] = naive
            results[f"{label}_B{B}_flash"] = flash

    # PQ Analysis
    print(f"\n\nTileMaxSimPQ Analysis:")
    print("-" * 80)
    M, K, dsub = 16, 256, 8
    d = M * dsub

    for B in [1, 100, 1000, 10000, 100000]:
        Nq, Nd = 32, 128
        pq_flash = pqsim_analysis(Nq, Nd, M, K, dsub, B)
        pq_decomp = pqsim_decompress_analysis(Nq, Nd, M, K, dsub, B)

        print(f"B={B:>6}: TileMaxSimPQ  AI={pq_flash['arithmetic_intensity']:>6.1f}  "
              f"{'MEM' if pq_flash['is_memory_bound'] else 'CMP'}  "
              f"min={pq_flash['min_time_us']:>10.2f}us")
        print(f"{'':>8} Decomp+Score AI={pq_decomp['arithmetic_intensity']:>6.1f}  "
              f"{'MEM' if pq_decomp['is_memory_bound'] else 'CMP'}  "
              f"min={pq_decomp['min_time_us']:>10.2f}us")
        print(f"{'':>8} IO reduction: {pq_decomp['io_bytes']/pq_flash['io_bytes']:.1f}x")

        results[f"pq_B{B}_flash"] = pq_flash
        results[f"pq_B{B}_decomp"] = pq_decomp

    # IO reduction summary
    print(f"\n\nIO Reduction Summary (Flash vs Naive):")
    print("-" * 60)
    for Nq, Nd, d, label in configs:
        naive = maxsim_naive_analysis(Nq, Nd, d, 10000)
        flash = maxsim_flash_analysis(Nq, Nd, d, 10000)
        reduction = naive["io_bytes"] / flash["io_bytes"]
        print(f"  {label}: {reduction:.1f}x IO reduction, "
              f"{naive['min_time_us']/flash['min_time_us']:.1f}x theoretical speedup")

    # Save results
    output_dir = os.path.join(os.path.dirname(__file__), "..", "roofline_results")
    os.makedirs(output_dir, exist_ok=True)

    # Convert to serializable
    serializable = {}
    for k, v in results.items():
        serializable[k] = {kk: float(vv) if isinstance(vv, (int, float)) else vv
                          for kk, vv in v.items()}

    with open(os.path.join(output_dir, "roofline_analysis.json"), 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {output_dir}/roofline_analysis.json")

    return results


if __name__ == "__main__":
    full_analysis()
