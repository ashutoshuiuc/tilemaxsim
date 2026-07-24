"""
TileMaxSimPQ: Fused PQ Decompression + MaxSim Scoring Kernel

Product Quantization (PQ) compresses each d-dimensional vector into M sub-vector codes,
each indexing into a codebook of K centroids per sub-quantizer.

Standard approach (PLAID/WARP): Decompress -> Score
  1. For each document token, look up M centroids and concatenate to get approximate vector
  2. Compute dot product with query token

TileMaxSimPQ approach: Pre-compute lookup table, score via table lookup
  1. Pre-compute distance table: T[m, k] = Q[i, m*dsub:(m+1)*dsub] . C[m, k] for all query tokens
  2. For each document token (represented as M codes), score = sum_m T[m, code[m]]
  3. Fuse the lookup + sum + max-reduction in a single kernel pass

This avoids ever decompressing document vectors. The lookup table fits in shared memory:
  K=256 centroids * M=16 sub-quantizers * 4 bytes = 16KB per query token << H100's 228KB SRAM.

IO Complexity:
  - Decompress+Score: O(Nd * M * dsub) reads for decompression + O(Nq * Nd * d) for scoring
  - TileMaxSimPQ: O(Nq * M * K * dsub) for table build + O(Nq * Nd * M) for scoring
  - For Nd >> K (typical: Nd=100+, K=256), table approach does far less HBM I/O
"""

import torch
import triton
import triton.language as tl
import math


@triton.jit
def _build_distance_table_kernel(
    Q_ptr,          # (Nq, d) query embeddings
    Codebook_ptr,   # (M, K, dsub) PQ codebook
    Table_ptr,      # (Nq, M, K) output distance table
    Nq,
    M: tl.constexpr,
    K: tl.constexpr,
    dsub: tl.constexpr,
):
    """
    Build the query-centroid distance table.
    T[q, m, k] = dot(Q[q, m*dsub:(m+1)*dsub], Codebook[m, k, :])

    Each program handles one (query_token, sub_quantizer) pair.
    """
    q_idx = tl.program_id(0)
    m_idx = tl.program_id(1)

    if q_idx >= Nq:
        return

    # Load query sub-vector: Q[q_idx, m_idx*dsub : (m_idx+1)*dsub]
    sub_offsets = tl.arange(0, dsub)
    q_sub = tl.load(
        Q_ptr + q_idx * (M * dsub) + m_idx * dsub + sub_offsets
    ).to(tl.float32)

    # For each centroid k, compute dot product
    for k in range(K):
        # Load codebook[m_idx, k, :]
        cb_vals = tl.load(
            Codebook_ptr + m_idx * K * dsub + k * dsub + sub_offsets
        ).to(tl.float32)

        dot = tl.sum(q_sub * cb_vals, axis=0)
        tl.store(Table_ptr + q_idx * M * K + m_idx * K + k, dot)


@triton.jit
def _flash_pqsim_score_kernel(
    Table_ptr,      # (Nq, M, K) distance table
    Codes_ptr,      # (Nd, M) PQ codes (uint8)
    Out_ptr,        # (Nq,) per-query-token max similarities
    Nq,
    Nd,
    M: tl.constexpr,
    K: tl.constexpr,
    BLOCK_Nd: tl.constexpr,
):
    """
    Score documents using the pre-computed distance table.
    For each query token q_idx:
      max_j sum_m Table[q_idx, m, Codes[j, m]]

    Each program handles one query token, iterating over document tiles.
    """
    q_idx = tl.program_id(0)

    if q_idx >= Nq:
        return

    running_max = tl.full([], value=float('-inf'), dtype=tl.float32)

    for d_start in range(0, Nd, BLOCK_Nd):
        # int64 so d_indices * M cannot overflow signed int32 for large Nd
        d_indices = (d_start + tl.arange(0, BLOCK_Nd)).to(tl.int64)
        d_mask = d_indices < Nd

        # Accumulate score for each document in this tile
        scores = tl.zeros([BLOCK_Nd], dtype=tl.float32)

        for m in range(M):
            # Load codes for sub-quantizer m: Codes[d_indices, m]
            codes = tl.load(
                Codes_ptr + d_indices * M + m,
                mask=d_mask,
                other=0
            ).to(tl.int32)

            # Look up table values: Table[q_idx, m, codes]
            table_vals = tl.load(
                Table_ptr + q_idx * M * K + m * K + codes,
                mask=d_mask,
                other=0.0
            )

            scores += table_vals

        scores = tl.where(d_mask, scores, float('-inf'))
        tile_max = tl.max(scores, axis=0)
        running_max = tl.maximum(running_max, tile_max)

    tl.store(Out_ptr + q_idx, running_max)


@triton.jit
def _flash_pqsim_batched_score_kernel(
    Table_ptr,      # (Nq, M, K) distance table
    Codes_ptr,      # (B, Nd, M) PQ codes (uint8)
    Out_ptr,        # (B, Nq) per-query-token max similarities
    Nq,
    Nd,
    M: tl.constexpr,
    K: tl.constexpr,
    BLOCK_Nd: tl.constexpr,
):
    """Batched version: handles multiple documents."""
    q_idx = tl.program_id(0)
    batch_idx = tl.program_id(1).to(tl.int64)

    if q_idx >= Nq:
        return

    running_max = tl.full([], value=float('-inf'), dtype=tl.float32)

    for d_start in range(0, Nd, BLOCK_Nd):
        # int64 to avoid signed int32 overflow in batch_idx * Nd * M on large batches
        d_indices = (d_start + tl.arange(0, BLOCK_Nd)).to(tl.int64)
        d_mask = d_indices < Nd

        scores = tl.zeros([BLOCK_Nd], dtype=tl.float32)

        for m in range(M):
            codes = tl.load(
                Codes_ptr + batch_idx * Nd * M + d_indices * M + m,
                mask=d_mask,
                other=0
            ).to(tl.int32)

            table_vals = tl.load(
                Table_ptr + q_idx * M * K + m * K + codes,
                mask=d_mask,
                other=0.0
            )

            scores += table_vals

        scores = tl.where(d_mask, scores, float('-inf'))
        tile_max = tl.max(scores, axis=0)
        running_max = tl.maximum(running_max, tile_max)

    tl.store(Out_ptr + batch_idx * Nq + q_idx, running_max)


class TileMaxSimPQ:
    """
    High-level interface for TileMaxSimPQ scoring.
    """

    def __init__(self, codebook: torch.Tensor):
        """
        Args:
            codebook: (M, K, dsub) PQ codebook centroids, fp16
        """
        self.codebook = codebook.contiguous()
        self.M, self.K, self.dsub = codebook.shape
        self.d = self.M * self.dsub

    def build_distance_table(self, Q: torch.Tensor) -> torch.Tensor:
        """
        Build query-centroid distance table.

        Args:
            Q: (Nq, d) query embeddings

        Returns:
            table: (Nq, M, K) distance table
        """
        Nq = Q.shape[0]
        assert Q.shape[1] == self.d

        table = torch.empty(Nq, self.M, self.K, dtype=torch.float32, device=Q.device)

        grid = (Nq, self.M)
        _build_distance_table_kernel[grid](
            Q, self.codebook, table,
            Nq, self.M, self.K, self.dsub,
        )

        return table

    def score_batch(self, Q: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        """
        Score a batch of PQ-compressed documents against a query.

        Args:
            Q: (Nq, d) query embeddings
            codes: (B, Nd, M) PQ codes (uint8 or int32)

        Returns:
            scores: (B,) MaxSim scores
        """
        B, Nd, M = codes.shape
        Nq = Q.shape[0]
        assert M == self.M

        # Build distance table
        table = self.build_distance_table(Q)

        # Score with chunking for large batches (grid y-dim limit = 65535)
        BLOCK_Nd = min(triton.next_power_of_2(Nd), 256)
        codes_int = codes.to(torch.int32).contiguous()

        MAX_BATCH_GRID = 65535
        scores = torch.empty(B, dtype=torch.float32, device=Q.device)

        for chunk_start in range(0, B, MAX_BATCH_GRID):
            chunk_end = min(chunk_start + MAX_BATCH_GRID, B)
            chunk_B = chunk_end - chunk_start

            token_maxes = torch.empty(chunk_B, Nq, dtype=torch.float32, device=Q.device)

            grid = (Nq, chunk_B)
            _flash_pqsim_batched_score_kernel[grid](
                table, codes_int[chunk_start:chunk_end], token_maxes,
                Nq, Nd, self.M, self.K, BLOCK_Nd=BLOCK_Nd,
            )

            scores[chunk_start:chunk_end] = token_maxes.sum(dim=1)

        return scores

    def score_single(self, Q: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        """
        Score a single PQ-compressed document.

        Args:
            Q: (Nq, d) query embeddings
            codes: (Nd, M) PQ codes

        Returns:
            score: scalar MaxSim score
        """
        Nq = Q.shape[0]
        Nd = codes.shape[0]

        table = self.build_distance_table(Q)

        out = torch.empty(Nq, dtype=torch.float32, device=Q.device)
        BLOCK_Nd = min(triton.next_power_of_2(Nd), 256)

        codes_int = codes.to(torch.int32).contiguous()

        _flash_pqsim_score_kernel[(Nq,)](
            table, codes_int, out,
            Nq, Nd, self.M, self.K, BLOCK_Nd=BLOCK_Nd,
        )

        return out.sum()


def pytorch_pqsim_baseline(Q: torch.Tensor, codebook: torch.Tensor,
                            codes: torch.Tensor) -> torch.Tensor:
    """
    PyTorch baseline: decompress + score.

    Args:
        Q: (Nq, d) query embeddings
        codebook: (M, K, dsub) PQ codebook
        codes: (B, Nd, M) PQ codes

    Returns:
        scores: (B,) MaxSim scores
    """
    M, K, dsub = codebook.shape
    B, Nd, _ = codes.shape
    Nq, d = Q.shape

    # Decompress: gather centroids for each code
    # codes: (B, Nd, M) -> indices into codebook
    D_approx = torch.zeros(B, Nd, d, dtype=torch.float32, device=Q.device)

    for m in range(M):
        # codes[:, :, m] -> (B, Nd) indices into codebook[m]
        c = codes[:, :, m].long()  # (B, Nd)
        # codebook[m, c] -> (B, Nd, dsub)
        sub_vecs = codebook[m][c]  # (B, Nd, dsub)
        D_approx[:, :, m * dsub:(m + 1) * dsub] = sub_vecs.float()

    # MaxSim scoring
    S = torch.einsum('qd,bnd->bqn', Q.float(), D_approx)
    return S.max(dim=-1).values.sum(dim=-1)


if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cuda:0"

    # PQ parameters
    M = 16       # sub-quantizers
    K = 256      # centroids per sub-quantizer
    dsub = 8     # sub-vector dimension
    d = M * dsub  # = 128
    Nq = 32      # query tokens
    Nd = 128     # document tokens per doc
    B = 8        # batch size

    # Create random codebook and codes
    codebook = torch.randn(M, K, dsub, dtype=torch.float16, device=device)
    Q = torch.randn(Nq, d, dtype=torch.float16, device=device)
    codes = torch.randint(0, K, (B, Nd, M), dtype=torch.uint8, device=device)

    # TileMaxSimPQ
    pqsim = TileMaxSimPQ(codebook)
    scores_flash = pqsim.score_batch(Q, codes)

    # PyTorch baseline
    scores_baseline = pytorch_pqsim_baseline(Q, codebook, codes)

    max_diff = (scores_flash - scores_baseline).abs().max().item()
    print(f"PQ Batch (B={B}) - Max diff: {max_diff:.6f}")
    print(f"Flash scores[:4]: {scores_flash[:4].tolist()}")
    print(f"Baseline scores[:4]: {scores_baseline[:4].tolist()}")
    print("\nCorrectness test passed!" if max_diff < 1.0 else "\nWARNING: Large difference!")
