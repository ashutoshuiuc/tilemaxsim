"""
TileMaxSim V2: Optimized Triton Kernel with better tiling and vectorization.

Key optimizations over V1:
1. Process multiple query tokens per program (BLOCK_Nq) to increase data reuse of D tiles
2. Use larger BLOCK_Nd tiles for better memory coalescing
3. Fuse the sum-reduction into the same kernel (avoid separate reduce kernel)
4. Use tensor core-friendly tile sizes (multiples of 16)

This is the "fused" version that computes the full MaxSim score in a single kernel launch.
"""

import torch
import triton
import triton.language as tl
import time


@triton.jit
def _flash_maxsim_v2_kernel(
    Q_ptr,        # (Nq, d) query embeddings
    D_ptr,        # (B, Nd, d) document embeddings
    Out_ptr,      # (B,) final MaxSim scores (sum of per-token maxes)
    Nq,           # number of query tokens
    Nd,           # number of document tokens per document
    d: tl.constexpr,  # embedding dimension
    BLOCK_Nd: tl.constexpr,  # tile size for document tokens
):
    """
    Each program handles one document (batch element).
    Iterates over all query tokens and document tiles.
    Accumulates the sum of per-query-token maxima in a register.
    """
    batch_idx = tl.program_id(0)

    # Accumulator for final score (sum of maxes)
    score_acc = tl.zeros([], dtype=tl.float32)

    # Process each query token
    for q_idx in range(Nq):
        # Load query token: Q[q_idx, :] -> (d,)
        q_offsets = tl.arange(0, d)
        q_vec = tl.load(Q_ptr + q_idx * d + q_offsets).to(tl.float32)

        # Track max similarity for this query token
        running_max = tl.full([], value=float('-inf'), dtype=tl.float32)

        # Iterate over document token tiles
        for d_start in range(0, Nd, BLOCK_Nd):
            d_indices = d_start + tl.arange(0, BLOCK_Nd)
            d_mask = d_indices < Nd

            # Load D[batch_idx, d_indices, :] -> (BLOCK_Nd, d)
            d_ptrs = D_ptr + batch_idx * Nd * d + d_indices[:, None] * d + q_offsets[None, :]
            d_mask_2d = d_mask[:, None]
            d_tile = tl.load(d_ptrs, mask=d_mask_2d, other=0.0).to(tl.float32)

            # Dot products: (BLOCK_Nd, d) @ (d,) -> (BLOCK_Nd,)
            dots = tl.sum(d_tile * q_vec[None, :], axis=1)

            # Update running max
            dots = tl.where(d_mask, dots, float('-inf'))
            tile_max = tl.max(dots, axis=0)
            running_max = tl.maximum(running_max, tile_max)

        # Add this query token's max to the score
        score_acc += running_max

    # Store final score
    tl.store(Out_ptr + batch_idx, score_acc)


@triton.jit
def _flash_maxsim_v2_multiquery_kernel(
    Q_ptr,        # (Nq, d) query embeddings
    D_ptr,        # (B, Nd, d) document embeddings
    Out_ptr,      # (B,) final MaxSim scores
    Nq,
    Nd,
    d: tl.constexpr,
    BLOCK_Nq: tl.constexpr,  # number of query tokens processed together
    BLOCK_Nd: tl.constexpr,  # tile size for document tokens
):
    """
    Optimized: processes BLOCK_Nq query tokens simultaneously per document.
    This allows reusing document tiles across multiple query tokens,
    reducing total HBM reads by factor of BLOCK_Nq.
    """
    batch_idx = tl.program_id(0)
    q_block_idx = tl.program_id(1)  # which block of query tokens

    q_start = q_block_idx * BLOCK_Nq
    q_indices = q_start + tl.arange(0, BLOCK_Nq)
    q_mask = q_indices < Nq

    # Load BLOCK_Nq query tokens: (BLOCK_Nq, d)
    k_offsets = tl.arange(0, d)
    q_ptrs = Q_ptr + q_indices[:, None] * d + k_offsets[None, :]
    q_tile = tl.load(q_ptrs, mask=q_mask[:, None], other=0.0).to(tl.float32)

    # Running max per query token: (BLOCK_Nq,)
    running_maxes = tl.full([BLOCK_Nq], value=float('-inf'), dtype=tl.float32)

    # Iterate over document token tiles
    for d_start in range(0, Nd, BLOCK_Nd):
        d_indices = d_start + tl.arange(0, BLOCK_Nd)
        d_mask_1d = d_indices < Nd

        # Load D[batch_idx, d_indices, :] -> (BLOCK_Nd, d)
        d_ptrs = D_ptr + batch_idx * Nd * d + d_indices[:, None] * d + k_offsets[None, :]
        d_tile = tl.load(d_ptrs, mask=d_mask_1d[:, None], other=0.0).to(tl.float32)

        # Compute similarity: (BLOCK_Nq, d) @ (d, BLOCK_Nd) -> (BLOCK_Nq, BLOCK_Nd)
        # Using manual dot product via broadcasting
        # q_tile: (BLOCK_Nq, d), d_tile: (BLOCK_Nd, d)
        # sim[i,j] = sum_k q_tile[i,k] * d_tile[j,k]
        sim = tl.dot(q_tile, tl.trans(d_tile))  # (BLOCK_Nq, BLOCK_Nd)

        # Mask invalid document positions
        sim = tl.where(d_mask_1d[None, :], sim, float('-inf'))

        # Max over document dimension for each query token
        tile_maxes = tl.max(sim, axis=1)  # (BLOCK_Nq,)
        running_maxes = tl.maximum(running_maxes, tile_maxes)

    # Mask invalid query tokens and sum
    running_maxes = tl.where(q_mask, running_maxes, 0.0)

    # Atomic add to output (multiple query blocks contribute to same doc score)
    tl.atomic_add(Out_ptr + batch_idx, tl.sum(running_maxes, axis=0))


@triton.jit
def _flash_maxsim_v2_mq_dimtile_kernel(
    Q_ptr,        # (Nq, d) query embeddings
    D_ptr,        # (B, Nd, d) document embeddings
    Out_ptr,      # (B,) final MaxSim scores
    Nq,
    Nd,
    d,            # NOT constexpr -- we tile over it
    BLOCK_Nq: tl.constexpr,
    BLOCK_Nd: tl.constexpr,
    BLOCK_d: tl.constexpr,   # tile size for embedding dimension
):
    """
    V2-MQ with dimension tiling: supports arbitrary d by tiling over the
    embedding dimension. For each doc-token tile, we accumulate partial
    dot products across d-tiles before computing the max.
    """
    batch_idx = tl.program_id(0)
    q_block_idx = tl.program_id(1)

    q_start = q_block_idx * BLOCK_Nq
    q_indices = q_start + tl.arange(0, BLOCK_Nq)
    q_mask = q_indices < Nq

    # Running max per query token: (BLOCK_Nq,)
    running_maxes = tl.full([BLOCK_Nq], value=float('-inf'), dtype=tl.float32)

    # Iterate over document token tiles
    for nd_start in range(0, Nd, BLOCK_Nd):
        nd_indices = nd_start + tl.arange(0, BLOCK_Nd)
        nd_mask = nd_indices < Nd

        # Accumulate dot products across d-tiles: (BLOCK_Nq, BLOCK_Nd)
        sim_acc = tl.zeros([BLOCK_Nq, BLOCK_Nd], dtype=tl.float32)

        for k_start in range(0, d, BLOCK_d):
            k_offsets = k_start + tl.arange(0, BLOCK_d)
            k_mask = k_offsets < d

            # Load Q tile: (BLOCK_Nq, BLOCK_d)
            q_ptrs = Q_ptr + q_indices[:, None] * d + k_offsets[None, :]
            q_tile = tl.load(q_ptrs, mask=q_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)

            # Load D tile: (BLOCK_Nd, BLOCK_d)
            d_ptrs = D_ptr + batch_idx * Nd * d + nd_indices[:, None] * d + k_offsets[None, :]
            d_tile = tl.load(d_ptrs, mask=nd_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)

            # Accumulate partial dot products
            sim_acc += tl.dot(q_tile, tl.trans(d_tile))

        # Now sim_acc has the full dot products. Mask and take max.
        sim_acc = tl.where(nd_mask[None, :], sim_acc, float('-inf'))
        tile_maxes = tl.max(sim_acc, axis=1)  # (BLOCK_Nq,)
        running_maxes = tl.maximum(running_maxes, tile_maxes)

    # Mask invalid query tokens and sum
    running_maxes = tl.where(q_mask, running_maxes, 0.0)
    tl.atomic_add(Out_ptr + batch_idx, tl.sum(running_maxes, axis=0))


def flash_maxsim_v2(Q: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """
    TileMaxSim V2: single-kernel fused MaxSim scoring.
    One program per document, loops over all query tokens internally.

    Args:
        Q: (Nq, d) query embeddings, fp16
        D: (B, Nd, d) document embeddings, fp16

    Returns:
        scores: (B,) MaxSim scores
    """
    B, Nd, d = D.shape
    Nq = Q.shape[0]

    scores = torch.empty(B, dtype=torch.float32, device=Q.device)

    BLOCK_Nd = min(triton.next_power_of_2(Nd), 256)

    # Use 1D grid: one program per document
    MAX_GRID = 65535
    for chunk_start in range(0, B, MAX_GRID):
        chunk_end = min(chunk_start + MAX_GRID, B)
        chunk_B = chunk_end - chunk_start

        _flash_maxsim_v2_kernel[(chunk_B,)](
            Q, D[chunk_start:chunk_end], scores[chunk_start:chunk_end],
            Nq, Nd, d,
            BLOCK_Nd=BLOCK_Nd,
        )

    return scores


def flash_maxsim_v2_multiquery(Q: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """
    TileMaxSim V2 with multi-query tiling.
    Processes BLOCK_Nq query tokens together to reuse D tiles.
    Automatically selects the optimal kernel variant:
    - For d <= 128: uses the single-d kernel (no dimension tiling)
    - For d > 128: uses dimension-tiled kernel that tiles over d

    Args:
        Q: (Nq, d) query embeddings, fp16
        D: (B, Nd, d) document embeddings, fp16

    Returns:
        scores: (B,) MaxSim scores
    """
    B, Nd, d = D.shape
    Nq = Q.shape[0]

    scores = torch.zeros(B, dtype=torch.float32, device=Q.device)

    # Choose BLOCK_Nq: want it to be power of 2 and <= Nq
    # Setting BLOCK_Nq=Nq processes all query tokens in one pass,
    # minimizing document re-reads (D read exactly once).
    BLOCK_Nq = min(32, triton.next_power_of_2(Nq))
    num_q_blocks = (Nq + BLOCK_Nq - 1) // BLOCK_Nq

    # Select kernel based on d
    use_dimtile = d > 128

    if use_dimtile:
        # Dimension-tiled kernel for large d
        BLOCK_d = 128  # tile over embedding dimension in chunks of 128
        BLOCK_Nd = min(triton.next_power_of_2(Nd), 32)  # smaller doc tiles to fit
    else:
        BLOCK_Nd = min(triton.next_power_of_2(Nd), 64)

    MAX_GRID = 65535
    for chunk_start in range(0, B, MAX_GRID):
        chunk_end = min(chunk_start + MAX_GRID, B)
        chunk_B = chunk_end - chunk_start

        # Reset scores for this chunk (atomic_add accumulates)
        scores[chunk_start:chunk_end].zero_()

        grid = (chunk_B, num_q_blocks)

        if use_dimtile:
            _flash_maxsim_v2_mq_dimtile_kernel[grid](
                Q, D[chunk_start:chunk_end], scores[chunk_start:chunk_end],
                Nq, Nd, d,
                BLOCK_Nq=BLOCK_Nq,
                BLOCK_Nd=BLOCK_Nd,
                BLOCK_d=BLOCK_d,
            )
        else:
            _flash_maxsim_v2_multiquery_kernel[grid](
                Q, D[chunk_start:chunk_end], scores[chunk_start:chunk_end],
                Nq, Nd, d,
                BLOCK_Nq=BLOCK_Nq,
                BLOCK_Nd=BLOCK_Nd,
            )

    return scores


def benchmark_v2(device="cuda:0"):
    """Benchmark V2 kernels against V1 and baselines."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from flash_maxsim_kernel import (
        flash_maxsim_batch, pytorch_maxsim_naive, pytorch_maxsim_loop,
    )

    d = 128
    Nq = 32

    print("=" * 80)
    print("TileMaxSim V2 Benchmarks")
    print("=" * 80)
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print()

    configs = [
        (128, 1000, "Nd=128, B=1K"),
        (128, 10000, "Nd=128, B=10K"),
        (128, 100000, "Nd=128, B=100K"),
        (256, 1000, "Nd=256, B=1K"),
        (256, 10000, "Nd=256, B=10K"),
        (256, 100000, "Nd=256, B=100K"),
    ]

    for Nd, B, label in configs:
        Q = torch.randn(Nq, d, dtype=torch.float16, device=device)
        D = torch.randn(B, Nd, d, dtype=torch.float16, device=device)

        print(f"\n{label}:")

        # Warmup and time each method
        def time_fn(fn, warmup=5, repeat=20):
            for _ in range(warmup):
                fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(repeat):
                fn()
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / repeat

        methods = {}

        # TileMaxSim V1
        try:
            t = time_fn(lambda: flash_maxsim_batch(Q, D))
            methods["V1"] = t
            print(f"  TileMaxSim V1:         {t*1000:8.3f} ms | {B/t:12.0f} docs/s")
        except Exception as e:
            print(f"  TileMaxSim V1:         ERROR: {e}")

        # TileMaxSim V2 (single kernel)
        try:
            t = time_fn(lambda: flash_maxsim_v2(Q, D))
            methods["V2"] = t
            print(f"  TileMaxSim V2:         {t*1000:8.3f} ms | {B/t:12.0f} docs/s")
        except Exception as e:
            print(f"  TileMaxSim V2:         ERROR: {e}")

        # TileMaxSim V2 multi-query
        try:
            t = time_fn(lambda: flash_maxsim_v2_multiquery(Q, D))
            methods["V2-MQ"] = t
            print(f"  TileMaxSim V2-MQ:      {t*1000:8.3f} ms | {B/t:12.0f} docs/s")
        except Exception as e:
            print(f"  TileMaxSim V2-MQ:      ERROR: {e}")

        # PyTorch Loop baseline
        if B <= 100000:
            try:
                t = time_fn(lambda: pytorch_maxsim_loop(Q, D))
                methods["PT-Loop"] = t
                print(f"  PyTorch Loop:           {t*1000:8.3f} ms | {B/t:12.0f} docs/s")
            except Exception as e:
                print(f"  PyTorch Loop:           ERROR: {e}")

        # PyTorch Naive baseline
        if B <= 50000:
            try:
                t = time_fn(lambda: pytorch_maxsim_naive(Q, D))
                methods["PT-Naive"] = t
                print(f"  PyTorch Naive:          {t*1000:8.3f} ms | {B/t:12.0f} docs/s")
            except torch.cuda.OutOfMemoryError:
                print(f"  PyTorch Naive:          OOM")
                torch.cuda.empty_cache()

        # Speedups
        best_flash = min(methods.get("V1", float('inf')),
                        methods.get("V2", float('inf')),
                        methods.get("V2-MQ", float('inf')))
        for name, t in methods.items():
            if name.startswith("PT"):
                speedup = t / best_flash
                print(f"  -> Best TileMaxSim vs {name}: {speedup:.1f}x")

        # Correctness check
        if B <= 1000:
            s_v2 = flash_maxsim_v2(Q, D)
            s_v1 = flash_maxsim_batch(Q, D)
            s_naive = pytorch_maxsim_naive(Q, D)
            diff_v2_naive = (s_v2 - s_naive).abs().max().item()
            diff_v1_naive = (s_v1 - s_naive).abs().max().item()
            print(f"  Correctness: V2 vs Naive max_diff={diff_v2_naive:.4f}, V1 vs Naive max_diff={diff_v1_naive:.4f}")

        torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    benchmark_v2(args.device)
