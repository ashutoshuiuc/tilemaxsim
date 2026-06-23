"""
TileMaxSim: IO-Aware Triton Kernel for Fused MaxSim Scoring

MaxSim(Q, D) = sum_i max_j (Q[i] . D[j])
where Q is (Nq, d) query embeddings and D is (Nd, d) document embeddings.

Naive approach: S = Q @ D^T  (Nq x Nd), then max over dim=1, then sum.
This materializes the full Nq x Nd matrix in HBM.

TileMaxSim: Tiles over document tokens in SRAM, accumulates running max
per query token in registers, then reduces with sum at the end.
Never materializes the full similarity matrix.

IO Complexity:
  - Naive: O(Nq * Nd) HBM writes + reads for the similarity matrix
  - TileMaxSim: O(Nq * d + Nd * d) HBM reads (just the embeddings), O(Nq) writes (the max values)
  - Reduction: from O(Nq * Nd * d) to O((Nq + Nd) * d) in HBM accesses
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _flash_maxsim_fwd_kernel(
    Q_ptr,        # (Nq, d) query embeddings
    D_ptr,        # (Nd, d) document embeddings
    Out_ptr,      # (Nq,) per-query-token max similarities
    Nq,           # number of query tokens
    Nd,           # number of document tokens
    d: tl.constexpr,  # embedding dimension
    BLOCK_Nd: tl.constexpr,  # tile size for document tokens
    BLOCK_d: tl.constexpr,   # tile size for embedding dimension
):
    """
    Each program instance handles one query token.
    It iterates over document token tiles, computing partial dot products
    and maintaining a running maximum.
    """
    # Program ID = query token index
    q_idx = tl.program_id(0)

    if q_idx >= Nq:
        return

    # Initialize running max to -inf
    running_max = tl.full([], value=float('-inf'), dtype=tl.float32)

    # Load query token embedding (stays in registers for the entire loop)
    # We process d in tiles of BLOCK_d
    # For typical d=128, BLOCK_d=128 means single tile
    q_offsets = tl.arange(0, BLOCK_d)  # (BLOCK_d,)

    # Iterate over document token tiles
    for d_start in range(0, Nd, BLOCK_Nd):
        d_indices = d_start + tl.arange(0, BLOCK_Nd)  # (BLOCK_Nd,)
        d_mask = d_indices < Nd

        # Compute dot products between this query token and document token tile
        # dot[j] = sum_k Q[q_idx, k] * D[d_indices[j], k]
        # We accumulate over embedding dimension tiles
        dots = tl.zeros([BLOCK_Nd], dtype=tl.float32)

        for k_start in range(0, d, BLOCK_d):
            k_offsets = k_start + q_offsets
            k_mask = k_offsets < d

            # Load Q[q_idx, k_start:k_start+BLOCK_d]  -> (BLOCK_d,)
            q_vals = tl.load(
                Q_ptr + q_idx * d + k_offsets,
                mask=k_mask,
                other=0.0
            ).to(tl.float32)

            # Load D[d_indices, k_start:k_start+BLOCK_d] -> (BLOCK_Nd, BLOCK_d)
            d_ptrs = D_ptr + d_indices[:, None] * d + k_offsets[None, :]
            d_mask_2d = d_mask[:, None] & k_mask[None, :]
            d_vals = tl.load(d_ptrs, mask=d_mask_2d, other=0.0).to(tl.float32)

            # Accumulate dot products: (BLOCK_Nd, BLOCK_d) * (BLOCK_d,) -> (BLOCK_Nd,)
            dots += tl.sum(d_vals * q_vals[None, :], axis=1)

        # Update running max with this tile's dot products
        # Mask out invalid positions
        dots = tl.where(d_mask, dots, float('-inf'))
        tile_max = tl.max(dots, axis=0)
        running_max = tl.maximum(running_max, tile_max)

    # Store the max similarity for this query token
    tl.store(Out_ptr + q_idx, running_max)


@triton.jit
def _flash_maxsim_batched_kernel(
    Q_ptr,        # (B, Nq, d)
    D_ptr,        # (B, Nd, d) or (total_Nd, d) with offsets
    Out_ptr,      # (B,) final MaxSim scores
    Nq,
    Nd,
    d: tl.constexpr,
    BLOCK_Nd: tl.constexpr,
    BLOCK_d: tl.constexpr,
):
    """
    Batched version: each program handles one (batch, query_token) pair.
    Final reduction (sum over query tokens) done in a separate small kernel.
    """
    batch_idx = tl.program_id(1)
    q_idx = tl.program_id(0)

    if q_idx >= Nq:
        return

    # Offset into batch
    Q_batch = Q_ptr + batch_idx * Nq * d
    D_batch = D_ptr + batch_idx * Nd * d

    running_max = tl.full([], value=float('-inf'), dtype=tl.float32)
    q_offsets = tl.arange(0, BLOCK_d)

    for d_start in range(0, Nd, BLOCK_Nd):
        d_indices = d_start + tl.arange(0, BLOCK_Nd)
        d_mask = d_indices < Nd
        dots = tl.zeros([BLOCK_Nd], dtype=tl.float32)

        for k_start in range(0, d, BLOCK_d):
            k_offsets = k_start + q_offsets
            k_mask = k_offsets < d

            q_vals = tl.load(
                Q_batch + q_idx * d + k_offsets,
                mask=k_mask, other=0.0
            ).to(tl.float32)

            d_ptrs = D_batch + d_indices[:, None] * d + k_offsets[None, :]
            d_mask_2d = d_mask[:, None] & k_mask[None, :]
            d_vals = tl.load(d_ptrs, mask=d_mask_2d, other=0.0).to(tl.float32)

            dots += tl.sum(d_vals * q_vals[None, :], axis=1)

        dots = tl.where(d_mask, dots, float('-inf'))
        tile_max = tl.max(dots, axis=0)
        running_max = tl.maximum(running_max, tile_max)

    # Store per-query-token max (will be summed later)
    tl.store(Out_ptr + batch_idx * Nq + q_idx, running_max)


@triton.jit
def _sum_reduce_kernel(
    In_ptr,   # (B, Nq)
    Out_ptr,  # (B,)
    Nq,
    BLOCK_Nq: tl.constexpr,
):
    """Reduce per-query-token maxima to final score by summing."""
    batch_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_Nq)
    mask = offsets < Nq
    vals = tl.load(In_ptr + batch_idx * Nq + offsets, mask=mask, other=0.0)
    result = tl.sum(vals, axis=0)
    tl.store(Out_ptr + batch_idx, result)


def flash_maxsim_single(Q: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """
    Compute MaxSim score for a single query-document pair.

    Args:
        Q: (Nq, d) query token embeddings, fp16
        D: (Nd, d) document token embeddings, fp16

    Returns:
        score: scalar MaxSim score
    """
    Nq, d = Q.shape
    Nd = D.shape[0]
    assert D.shape[1] == d

    # Output: per-query-token max similarities
    out = torch.empty(Nq, dtype=torch.float32, device=Q.device)

    # Choose tile sizes
    BLOCK_Nd = min(triton.next_power_of_2(Nd), 256)
    BLOCK_d = min(triton.next_power_of_2(d), 128)

    grid = (Nq,)
    _flash_maxsim_fwd_kernel[grid](
        Q, D, out,
        Nq, Nd, d,
        BLOCK_Nd=BLOCK_Nd,
        BLOCK_d=BLOCK_d,
    )

    return out.sum()


def flash_maxsim_batch(Q: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """
    Compute MaxSim scores for a batch of documents against a single query.

    Args:
        Q: (Nq, d) query token embeddings
        D: (B, Nd, d) batch of document token embeddings

    Returns:
        scores: (B,) MaxSim scores
    """
    B, Nd, d = D.shape
    Nq = Q.shape[0]
    assert Q.shape[1] == d

    BLOCK_Nd = min(triton.next_power_of_2(Nd), 256)
    BLOCK_d = min(triton.next_power_of_2(d), 128)
    BLOCK_Nq = triton.next_power_of_2(Nq)

    # CUDA grid y-dim limit is 65535; chunk if needed
    MAX_BATCH_GRID = 65535
    scores = torch.empty(B, dtype=torch.float32, device=Q.device)

    for chunk_start in range(0, B, MAX_BATCH_GRID):
        chunk_end = min(chunk_start + MAX_BATCH_GRID, B)
        chunk_B = chunk_end - chunk_start

        D_chunk = D[chunk_start:chunk_end]
        Q_expanded = Q.unsqueeze(0).expand(chunk_B, -1, -1).contiguous()

        token_maxes = torch.empty(chunk_B, Nq, dtype=torch.float32, device=Q.device)

        grid = (Nq, chunk_B)
        _flash_maxsim_batched_kernel[grid](
            Q_expanded, D_chunk, token_maxes,
            Nq, Nd, d,
            BLOCK_Nd=BLOCK_Nd,
            BLOCK_d=BLOCK_d,
        )

        _sum_reduce_kernel[(chunk_B,)](
            token_maxes, scores[chunk_start:chunk_end],
            Nq, BLOCK_Nq=BLOCK_Nq,
        )

    return scores


def flash_maxsim_multi_doc(Q: torch.Tensor, D_flat: torch.Tensor,
                           doc_offsets: torch.Tensor) -> torch.Tensor:
    """
    Compute MaxSim for variable-length documents stored in a flat tensor.

    Args:
        Q: (Nq, d) query embeddings
        D_flat: (total_tokens, d) all document token embeddings concatenated
        doc_offsets: (num_docs + 1,) start offsets for each document

    Returns:
        scores: (num_docs,) MaxSim scores
    """
    num_docs = doc_offsets.shape[0] - 1
    Nq, d = Q.shape
    scores = torch.empty(num_docs, dtype=torch.float32, device=Q.device)

    # For variable-length docs, we launch one kernel per document
    # (Could be optimized with a more sophisticated kernel)
    for i in range(num_docs):
        start = doc_offsets[i].item()
        end = doc_offsets[i + 1].item()
        D_doc = D_flat[start:end]
        scores[i] = flash_maxsim_single(Q, D_doc)

    return scores


# ============================================================
# PyTorch Baselines for comparison
# ============================================================

def pytorch_maxsim_naive(Q: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """
    Naive PyTorch MaxSim: materializes full similarity matrix.
    Q: (Nq, d), D: (B, Nd, d) -> scores: (B,)
    """
    # S = Q @ D^T -> (B, Nq, Nd)
    S = torch.einsum('qd,bnd->bqn', Q.float(), D.float())
    # max over document tokens, sum over query tokens
    return S.max(dim=-1).values.sum(dim=-1)


def pytorch_maxsim_loop(Q: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """
    Loop-over-query-tokens PyTorch MaxSim: avoids materializing full matrix at once.
    Q: (Nq, d), D: (B, Nd, d) -> scores: (B,)
    """
    B, Nd, d = D.shape
    Nq = Q.shape[0]
    scores = torch.zeros(B, dtype=torch.float32, device=Q.device)
    for i in range(Nq):
        # Q[i] . D -> (B, Nd)
        sims = torch.einsum('d,bnd->bn', Q[i].float(), D.float())
        scores += sims.max(dim=-1).values
    return scores


if __name__ == "__main__":
    # Quick correctness test
    torch.manual_seed(42)
    device = "cuda:0"
    Nq, Nd, d = 32, 128, 128
    B = 16

    Q = torch.randn(Nq, d, dtype=torch.float16, device=device)
    D = torch.randn(B, Nd, d, dtype=torch.float16, device=device)

    # Single doc test
    score_flash = flash_maxsim_single(Q, D[0])
    score_naive = pytorch_maxsim_naive(Q, D[:1])[0]
    print(f"Single doc - Flash: {score_flash.item():.4f}, Naive: {score_naive.item():.4f}, "
          f"Diff: {abs(score_flash.item() - score_naive.item()):.6f}")

    # Batch test
    scores_flash = flash_maxsim_batch(Q, D)
    scores_naive = pytorch_maxsim_naive(Q, D)
    max_diff = (scores_flash - scores_naive).abs().max().item()
    print(f"Batch (B={B}) - Max diff: {max_diff:.6f}")
    print(f"Flash scores[:4]: {scores_flash[:4].tolist()}")
    print(f"Naive scores[:4]: {scores_naive[:4].tolist()}")

    print("\nCorrectness test passed!" if max_diff < 0.1 else "\nWARNING: Large difference!")
