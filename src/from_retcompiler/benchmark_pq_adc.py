"""
Benchmark: Fused PQ ADC kernel vs unfused PyTorch baseline.

Measures wall-clock latency for the generated fused decompress_score kernel
against the standard two-step approach (decompress, then matmul).

Usage:
    python src/benchmark_pq_adc.py [--device cuda:0]
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

import triton
import triton.language as tl

PROJECT_ROOT = Path(__file__).parent.parent


@triton.jit
def pq_adc_fused_kernel(
    Q_ptr, Codes_ptr, Codebook_ptr, Out_ptr,
    Table_ptr,
    n_queries, n_docs, dim: tl.constexpr,
    n_subvectors: tl.constexpr, sub_dim: tl.constexpr,
    n_centroids: tl.constexpr,
    stride_qn, stride_qd,
    stride_cn, stride_cs,
    stride_cbs, stride_cbc, stride_cbd,
    stride_on, stride_od,
    stride_tn, stride_ts, stride_tc,
    TILE_D: tl.constexpr = 64,
):
    """Fused PQ ADC: lookup precomputed distance table and accumulate per doc."""
    pid_q = tl.program_id(0)
    pid_d = tl.program_id(1)

    d_start = pid_d * TILE_D
    d_offsets = d_start + tl.arange(0, TILE_D)
    d_mask = d_offsets < n_docs

    scores = tl.zeros([TILE_D], dtype=tl.float32)

    for sv in tl.static_range(n_subvectors):
        # Load codes for this tile of docs
        codes = tl.load(
            Codes_ptr + d_offsets * stride_cn + sv * stride_cs,
            mask=d_mask,
            other=0,
        )

        # Gather from precomputed distance table: table[pid_q, sv, code]
        partial = tl.load(
            Table_ptr + pid_q * stride_tn + sv * stride_ts + codes * stride_tc,
            mask=d_mask,
            other=0.0,
        )
        scores += partial

    out_ptrs = Out_ptr + pid_q * stride_on + d_offsets * stride_od
    tl.store(out_ptrs, scores, mask=d_mask)


def pytorch_pq_adc_unfused(queries, codes, codebook):
    """Unfused PQ ADC: decompress all docs, then matmul."""
    n_docs, n_subvectors = codes.shape
    _, n_centroids, sub_dim = codebook[0].shape if codebook.dim() == 3 else (None, None, None)

    n_subvectors_actual = codebook.shape[0]
    sub_dim_actual = codebook.shape[2]

    # Step 1: Decompress - gather centroids for each doc
    decompressed = torch.zeros(n_docs, n_subvectors_actual * sub_dim_actual,
                               device=queries.device, dtype=queries.dtype)
    for sv in range(n_subvectors_actual):
        sv_codes = codes[:, sv].long()
        centroids = codebook[sv][sv_codes]  # [n_docs, sub_dim]
        decompressed[:, sv * sub_dim_actual:(sv + 1) * sub_dim_actual] = centroids

    # Step 2: Score via matmul
    scores = queries @ decompressed.T  # [n_queries, n_docs]
    return scores


def pytorch_pq_adc_table(queries, codes, codebook):
    """Optimized unfused: precompute distance tables, then lookup."""
    n_queries, dim = queries.shape
    n_docs, n_subvectors = codes.shape
    _, n_centroids, sub_dim = codebook.shape

    # Precompute distance tables: [n_queries, n_subvectors, 256]
    tables = torch.zeros(n_queries, n_subvectors, n_centroids,
                         device=queries.device, dtype=torch.float32)
    for sv in range(n_subvectors):
        q_sub = queries[:, sv * sub_dim:(sv + 1) * sub_dim]  # [n_queries, sub_dim]
        tables[:, sv, :] = q_sub.float() @ codebook[sv].float().T  # [n_queries, 256]

    # Lookup and accumulate
    scores = torch.zeros(n_queries, n_docs, device=queries.device, dtype=torch.float32)
    for sv in range(n_subvectors):
        sv_codes = codes[:, sv].long()  # [n_docs]
        scores += tables[:, sv][:, sv_codes]  # broadcasting gather

    return scores


def run_fused_kernel(queries, codes, codebook, n_docs):
    """Run the fused PQ ADC: precompute table + Triton lookup kernel.

    The 'fusion' eliminates materializing decompressed document vectors.
    Instead: table[q, sv, c] = dot(q[sv_slice], codebook[sv, c]),
    then for each doc, accumulate table[q, sv, code[doc, sv]] over subvectors.
    """
    n_queries = queries.shape[0]
    dim = queries.shape[1]
    n_subvectors = codebook.shape[0]
    n_centroids = codebook.shape[1]
    sub_dim = codebook.shape[2]

    # Step 1: Precompute distance tables [n_queries, n_subvectors, n_centroids]
    table = torch.zeros(n_queries, n_subvectors, n_centroids,
                        device=queries.device, dtype=torch.float32)
    for sv in range(n_subvectors):
        q_sub = queries[:, sv * sub_dim:(sv + 1) * sub_dim].float()
        table[:, sv, :] = q_sub @ codebook[sv].float().T

    # Step 2: Triton kernel for lookup+accumulate (avoids materializing N_docs × dim)
    out = torch.zeros(n_queries, n_docs, device=queries.device, dtype=torch.float32)

    TILE_D = 64
    grid = (n_queries, (n_docs + TILE_D - 1) // TILE_D)

    pq_adc_fused_kernel[grid](
        queries, codes, codebook, out,
        table,
        n_queries, n_docs, dim, n_subvectors, sub_dim, n_centroids,
        queries.stride(0), queries.stride(1),
        codes.stride(0), codes.stride(1),
        codebook.stride(0), codebook.stride(1), codebook.stride(2),
        out.stride(0), out.stride(1),
        table.stride(0), table.stride(1), table.stride(2),
        TILE_D=TILE_D,
    )
    return out


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
        "max_ms": max(times),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = args.device
    torch.manual_seed(42)

    # PQ parameters
    dim = 128
    n_subvectors = 8
    sub_dim = dim // n_subvectors  # 16
    n_centroids = 256
    n_queries = 32

    # Test at multiple scales
    doc_counts = [1024, 8192, 65536, 262144, 1000000]

    results = []

    print("=" * 70)
    print("PQ ADC Benchmark: Fused Triton vs Unfused PyTorch")
    print(f"dim={dim}, n_subvectors={n_subvectors}, sub_dim={sub_dim}, n_queries={n_queries}")
    print("=" * 70)

    # Create query and codebook (shared across scales)
    queries = F.normalize(torch.randn(n_queries, dim, device=device, dtype=torch.float16), dim=-1)
    codebook = F.normalize(
        torch.randn(n_subvectors, n_centroids, sub_dim, device=device, dtype=torch.float16),
        dim=-1,
    )

    for n_docs in doc_counts:
        print(f"\n--- n_docs = {n_docs:,} ---")

        # Create random PQ codes
        codes = torch.randint(0, n_centroids, (n_docs, n_subvectors),
                              device=device, dtype=torch.uint8)

        # Memory for decompressed vectors (what unfused needs)
        decompressed_bytes = n_docs * dim * 2  # fp16
        codes_bytes = n_docs * n_subvectors  # uint8
        print(f"  Decompressed vector memory: {decompressed_bytes / 1e6:.1f} MB")
        print(f"  PQ codes memory: {codes_bytes / 1e6:.1f} MB")
        print(f"  IO reduction: {decompressed_bytes / codes_bytes:.1f}x")

        # Benchmark unfused (table-based, more fair comparison)
        try:
            timing_unfused = bench_fn(
                lambda: pytorch_pq_adc_table(queries, codes, codebook),
                device=device,
            )
            print(f"  Unfused (table): {timing_unfused['median_ms']:.3f} ms")
        except RuntimeError as e:
            if "out of memory" in str(e):
                timing_unfused = {"median_ms": float("inf"), "mean_ms": float("inf")}
                print(f"  Unfused (table): OOM")
                torch.cuda.empty_cache()
            else:
                raise

        # Benchmark fused kernel
        try:
            timing_fused = bench_fn(
                lambda: run_fused_kernel(queries, codes, codebook, n_docs),
                device=device,
            )
            print(f"  Fused (Triton):  {timing_fused['median_ms']:.3f} ms")
        except RuntimeError as e:
            if "out of memory" in str(e):
                timing_fused = {"median_ms": float("inf"), "mean_ms": float("inf")}
                print(f"  Fused (Triton):  OOM")
                torch.cuda.empty_cache()
            else:
                raise

        # Speedup
        if timing_unfused["median_ms"] != float("inf") and timing_fused["median_ms"] != float("inf"):
            speedup = timing_unfused["median_ms"] / timing_fused["median_ms"]
            print(f"  Speedup: {speedup:.2f}x")
        else:
            speedup = None

        # Correctness check (at small scale)
        if n_docs <= 8192:
            out_fused = run_fused_kernel(queries, codes, codebook, n_docs)
            out_unfused = pytorch_pq_adc_table(queries, codes, codebook)
            max_diff = (out_fused - out_unfused).abs().max().item()
            print(f"  Max absolute diff: {max_diff:.6f}")
            correct = max_diff < 0.01
        else:
            correct = None

        results.append({
            "n_docs": n_docs,
            "n_queries": n_queries,
            "unfused_median_ms": timing_unfused["median_ms"],
            "fused_median_ms": timing_fused["median_ms"],
            "speedup": speedup,
            "correct": correct,
            "io_reduction": decompressed_bytes / codes_bytes,
        })

        torch.cuda.empty_cache()

    # Save results
    out_path = PROJECT_ROOT / "final_results" / "pq_adc_benchmark.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return results


if __name__ == "__main__":
    main()
