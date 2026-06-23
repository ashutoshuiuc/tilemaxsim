"""
Review cycle experiments for TileMaxSim.
Address reviewer weaknesses: bandwidth breakdown, profiling, integration path.
"""
import torch
import torch.nn.functional as F
import time
import json
import numpy as np
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "tracker"


def bandwidth_breakdown_analysis():
    """W2: Detailed breakdown of where bandwidth is lost."""
    device = 'cuda:0'
    results = {}

    # Measure raw memory bandwidth with simple copy
    for size_mb in [1, 10, 100, 1000]:
        n = size_mb * 1024 * 1024 // 4  # float32
        a = torch.randn(n, device=device)
        b = torch.empty_like(a)

        torch.cuda.synchronize()
        times = []
        for _ in range(20):
            torch.cuda.synchronize()
            start = time.perf_counter()
            b.copy_(a)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

        bw = (size_mb * 2) / (np.mean(times) * 1000)  # GB/s (read + write)
        results[f'raw_copy_{size_mb}MB'] = {'bandwidth_GBs': round(bw, 1), 'time_ms': round(np.mean(times)*1000, 4)}

    # Measure matmul bandwidth
    for n in [1000, 10000, 100000]:
        A = torch.randn(32, 128, device=device, dtype=torch.float16)
        B = torch.randn(n, 128, device=device, dtype=torch.float16)

        torch.cuda.synchronize()
        times = []
        for _ in range(20):
            torch.cuda.synchronize()
            start = time.perf_counter()
            C = torch.matmul(A, B.T)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

        bytes_read = (32 * 128 + n * 128) * 2  # FP16
        bw = bytes_read / (np.mean(times) * 1e9)  # GB/s
        results[f'matmul_n{n}'] = {'bandwidth_GBs': round(bw, 1), 'time_ms': round(np.mean(times)*1000, 4)}

    # Measure MaxSim (loop over query tokens) bandwidth
    for K in [1000, 10000, 100000]:
        Q = torch.randn(1, 32, 128, device=device, dtype=torch.float16)
        D = torch.randn(K, 128, 128, device=device, dtype=torch.float16)
        D_flat = D.reshape(K * 128, 128).float().T

        torch.cuda.synchronize()
        times = []
        for _ in range(10):
            torch.cuda.synchronize()
            start = time.perf_counter()
            scores = torch.zeros(1, K, device=device, dtype=torch.float32)
            for i in range(32):
                q_i = Q[:, i, :].float()
                sim = torch.matmul(q_i, D_flat).reshape(1, K, 128)
                scores += sim.max(dim=-1).values
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

        bytes_total = (32 * 128 * 2) + (K * 128 * 128 * 2) * 32  # Q read once + D read 32 times
        bytes_optimal = (32 * 128 * 2) + (K * 128 * 128 * 2)  # D read once
        achieved_bw = bytes_total / (np.mean(times) * 1e9)
        optimal_bw = bytes_optimal / (np.mean(times) * 1e9)

        results[f'maxsim_K{K}'] = {
            'time_ms': round(np.mean(times)*1000, 2),
            'achieved_bw_GBs': round(achieved_bw, 1),
            'if_D_read_once_GBs': round(optimal_bw, 1),
            'D_reread_factor': 32,
        }

    return results


def kernel_launch_overhead():
    """Measure kernel launch overhead for different operation counts."""
    device = 'cuda:0'
    results = {}

    # Single large op vs many small ops
    A = torch.randn(32, 128, device=device, dtype=torch.float16)
    B = torch.randn(10000, 128, device=device, dtype=torch.float16)

    # Single matmul
    torch.cuda.synchronize()
    times = []
    for _ in range(50):
        torch.cuda.synchronize()
        start = time.perf_counter()
        C = torch.matmul(A, B.T)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    results['single_matmul'] = {'time_us': round(np.mean(times)*1e6, 1), 'launches': 1}

    # 32 sequential matmuls (simulating loop-over-tokens)
    torch.cuda.synchronize()
    times = []
    for _ in range(20):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for i in range(32):
            C = torch.matmul(A[i:i+1], B.T)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    results['32_sequential_matmuls'] = {'time_us': round(np.mean(times)*1e6, 1), 'launches': 32}

    # Overhead per launch
    overhead_per_launch = (results['32_sequential_matmuls']['time_us'] - results['single_matmul']['time_us']) / 31
    results['estimated_launch_overhead_us'] = round(overhead_per_launch, 2)

    return results


def integration_path_analysis():
    """Describe how TileMaxSim integrates into ColBERT pipeline."""
    return {
        'pipeline_stages': [
            'Query encoding (BERT forward pass)',
            'Candidate generation (centroid matching / ANN)',
            'Candidate filtering (centroid pruning)',
            'Full scoring (TileMaxSim replaces this)',
            'Top-k selection',
        ],
        'integration_points': {
            'PLAID': 'Replace deferred scoring stage with TileMaxSim GPU kernel',
            'WARP': 'Replace Stage 3 (full scoring) which dominates 60-80% of latency',
            'GEM': 'Use as scoring function in graph traversal verification step',
            'standalone': 'Direct GPU-resident scoring for pre-filtered candidates',
        },
        'requirements': {
            'GPU_memory': '8.8M MS MARCO compressed = 33.6GB (fits 1xH100)',
            'data_format': 'Contiguous FP16 tensors, padded to fixed doc length',
            'index_format': 'ColBERTv2 residual-compressed or full FP16',
        },
    }


def main():
    print("=" * 70)
    print("TileMaxSim Review Cycle Experiments")
    print("=" * 70)

    all_results = {}

    print("\n--- Bandwidth Breakdown ---")
    bw = bandwidth_breakdown_analysis()
    all_results['bandwidth_breakdown'] = bw
    for k, v in bw.items():
        print(f"  {k}: {v}")

    print("\n--- Kernel Launch Overhead ---")
    launch = kernel_launch_overhead()
    all_results['kernel_launch'] = launch
    for k, v in launch.items():
        print(f"  {k}: {v}")

    print("\n--- Integration Path ---")
    integration = integration_path_analysis()
    all_results['integration'] = integration
    print(json.dumps(integration, indent=2))

    # Save
    out = RESULTS_DIR / "review_experiments.json"
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == '__main__':
    main()
