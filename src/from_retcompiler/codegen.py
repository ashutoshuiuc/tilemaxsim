"""
Triton Code Generation

Generates Triton GPU kernels from optimized (fused + tiled) IR.
Each fused operation maps to a Triton kernel function.

The code generator:
1. Takes a fused IR node + TileConfig
2. Emits a Triton kernel as a Python string
3. Can optionally compile and return a callable

Supported fused patterns:
- score_max_reduce_sum_reduce (full MaxSim)
- score_max_reduce
- decompress_score
- sparse_lookup_score
- score_topk
- sum_reduce_topk
- Individual ops as fallback
"""

from __future__ import annotations

import textwrap
from typing import Dict, Optional

from .ir import FusedOp, IRNode, OpType, PipelineGraph, ScoreMode
from .tiling import TileConfig


# ---------------------------------------------------------------------------
# Triton kernel templates
# ---------------------------------------------------------------------------

def gen_maxsim_fused_kernel(tile: TileConfig) -> str:
    """
    Generate a Triton kernel for fused MaxSim:
    score(maxsim) + max_reduce(doc_tokens) + sum_reduce(query_tokens)

    For each (query, doc) pair:
      score = sum over query_tokens of max over doc_tokens of dot(q_tok, d_tok)
    """
    return textwrap.dedent(f"""\
    import triton
    import triton.language as tl

    @triton.jit
    def maxsim_fused_kernel(
        Q_ptr,          # [n_queries, query_tokens, dim]
        D_ptr,          # [n_docs, doc_tokens, dim]
        Out_ptr,        # [n_queries, n_docs]
        n_queries,
        query_tokens,
        n_docs,
        doc_tokens,
        dim,
        stride_qq, stride_qt, stride_qd,    # Q strides
        stride_dd, stride_dt, stride_ddim,   # D strides
        stride_oq, stride_od,               # Out strides
        TILE_QT: tl.constexpr = {tile.tile_qt},
        TILE_D: tl.constexpr = {tile.tile_d},
        TILE_DT: tl.constexpr = {tile.tile_dt},
        TILE_DIM: tl.constexpr = {tile.tile_dim},
    ):
        # Program IDs: one block per (query, doc_tile)
        pid_q = tl.program_id(0)
        pid_d = tl.program_id(1)

        # Document indices for this tile
        d_start = pid_d * TILE_D
        d_offsets = d_start + tl.arange(0, TILE_D)
        d_mask = d_offsets < n_docs

        # Accumulator: sum over query tokens of max-dot-over-doc-tokens
        # Shape: [TILE_D]
        acc = tl.zeros([TILE_D], dtype=tl.float32)

        # Loop over query tokens
        for qt in range(0, query_tokens, TILE_QT):
            qt_offsets = qt + tl.arange(0, TILE_QT)
            qt_mask = qt_offsets < query_tokens

            # For each query token, compute max dot product over doc tokens
            # max_dots shape: [TILE_QT, TILE_D]
            max_dots = tl.full([TILE_QT, TILE_D], float('-inf'), dtype=tl.float32)

            # Loop over doc token tiles
            for dt in range(0, doc_tokens, TILE_DT):
                dt_offsets = dt + tl.arange(0, TILE_DT)
                dt_mask = dt_offsets < doc_tokens

                # Load query token embeddings: [TILE_QT, TILE_DIM]
                q_ptrs = (Q_ptr
                          + pid_q * stride_qq
                          + qt_offsets[:, None] * stride_qt
                          + tl.arange(0, TILE_DIM)[None, :] * stride_qd)
                q = tl.load(q_ptrs, mask=qt_mask[:, None], other=0.0)

                # For each doc in tile, load doc token embeddings and compute dots
                # We iterate over docs to manage memory
                # Load doc embeddings: [TILE_D, TILE_DT, TILE_DIM]
                # Process as [TILE_DT, TILE_DIM] per doc tile
                for d_inner in range(TILE_D):
                    d_idx = d_start + d_inner
                    if d_idx < n_docs:
                        d_ptrs = (D_ptr
                                  + d_idx * stride_dd
                                  + dt_offsets[:, None] * stride_dt
                                  + tl.arange(0, TILE_DIM)[None, :] * stride_ddim)
                        d = tl.load(d_ptrs, mask=dt_mask[:, None], other=0.0)

                        # dots: [TILE_QT, TILE_DT] = Q @ D^T
                        dots = tl.dot(q, tl.trans(d))

                        # Max over doc tokens: [TILE_QT]
                        max_dt = tl.max(dots, axis=1)

                        # Update running max
                        max_dots[:, d_inner] = tl.maximum(
                            max_dots[:, d_inner], max_dt
                        )

            # Sum over query tokens: [TILE_D]
            acc += tl.sum(max_dots, axis=0)

        # Store output
        out_ptrs = Out_ptr + pid_q * stride_oq + d_offsets * stride_od
        tl.store(out_ptrs, acc, mask=d_mask)
    """)


def gen_maxsim_optimized_kernel(tile: TileConfig) -> str:
    """
    Generate an optimized MaxSim kernel that processes docs in tiles
    and uses shared memory for query embeddings.
    """
    return textwrap.dedent(f"""\
    import triton
    import triton.language as tl

    @triton.jit
    def maxsim_optimized_kernel(
        Q_ptr,          # [n_queries, query_tokens, dim]
        D_ptr,          # [n_docs, doc_tokens, dim]
        Out_ptr,        # [n_queries, n_docs]
        n_queries,
        query_tokens,
        n_docs,
        doc_tokens,
        dim,
        stride_qq, stride_qt, stride_qd,
        stride_dd, stride_dt, stride_ddim,
        stride_oq, stride_od,
        TILE_QT: tl.constexpr = {tile.tile_qt},
        TILE_D: tl.constexpr = {tile.tile_d},
        TILE_DT: tl.constexpr = {tile.tile_dt},
        DIM: tl.constexpr = {tile.tile_dim},
    ):
        pid_q = tl.program_id(0)
        pid_d_tile = tl.program_id(1)
        d_start = pid_d_tile * TILE_D

        # Accumulator for scores: [TILE_D]
        scores = tl.zeros([TILE_D], dtype=tl.float32)

        # Iterate over query tokens
        for qt_start in range(0, query_tokens, TILE_QT):
            n_qt = tl.minimum(TILE_QT, query_tokens - qt_start)
            qt_offs = qt_start + tl.arange(0, TILE_QT)
            qt_mask = qt_offs < query_tokens

            # Load query embeddings [TILE_QT, DIM]
            q = tl.load(
                Q_ptr + pid_q * stride_qq + qt_offs[:, None] * stride_qt
                + tl.arange(0, DIM)[None, :] * stride_qd,
                mask=qt_mask[:, None],
                other=0.0,
            )

            # For each doc in tile
            d_offs = d_start + tl.arange(0, TILE_D)
            d_mask = d_offs < n_docs

            # max over doc tokens for this query token set: [TILE_QT, TILE_D]
            best = tl.full([TILE_QT, TILE_D], float('-inf'), dtype=tl.float32)

            for dt_start in range(0, doc_tokens, TILE_DT):
                dt_offs = dt_start + tl.arange(0, TILE_DT)
                dt_mask = dt_offs < doc_tokens

                # Load doc embeddings for all docs in tile at these doc_token positions
                # We process one doc at a time within the tile
                for di in range(TILE_D):
                    d_idx = d_start + di
                    if d_idx < n_docs:
                        d = tl.load(
                            D_ptr + d_idx * stride_dd
                            + dt_offs[:, None] * stride_dt
                            + tl.arange(0, DIM)[None, :] * stride_ddim,
                            mask=dt_mask[:, None],
                            other=0.0,
                        )  # [TILE_DT, DIM]

                        # Compute dot products: [TILE_QT, TILE_DT]
                        dots = tl.dot(q, tl.trans(d))
                        # Max over doc tokens
                        mx = tl.max(dots, axis=1)  # [TILE_QT]
                        best[:, di] = tl.maximum(best[:, di], mx)

            # Sum over query tokens -> scores for this doc tile
            scores += tl.sum(tl.where(qt_mask[:, None], best, 0.0), axis=0)

        # Write output
        d_offs = d_start + tl.arange(0, TILE_D)
        tl.store(
            Out_ptr + pid_q * stride_oq + d_offs * stride_od,
            scores,
            mask=d_offs < n_docs,
        )
    """)


def gen_dot_score_kernel(tile: TileConfig) -> str:
    """Generate a Triton kernel for dense dot-product scoring."""
    return textwrap.dedent(f"""\
    import triton
    import triton.language as tl

    @triton.jit
    def dot_score_kernel(
        Q_ptr,          # [n_queries, dim]
        D_ptr,          # [n_docs, dim]
        Out_ptr,        # [n_queries, n_docs]
        n_queries,
        n_docs,
        dim,
        stride_qn, stride_qd,
        stride_dn, stride_dd,
        stride_on, stride_od,
        TILE_D: tl.constexpr = {tile.tile_d},
        TILE_DIM: tl.constexpr = {tile.tile_dim},
    ):
        pid_q = tl.program_id(0)
        pid_d = tl.program_id(1)

        d_start = pid_d * TILE_D
        d_offsets = d_start + tl.arange(0, TILE_D)
        d_mask = d_offsets < n_docs

        # Accumulate dot product over dim chunks
        acc = tl.zeros([TILE_D], dtype=tl.float32)

        for dim_start in range(0, dim, TILE_DIM):
            dim_offs = dim_start + tl.arange(0, TILE_DIM)
            dim_mask = dim_offs < dim

            # Load query: [TILE_DIM]
            q = tl.load(
                Q_ptr + pid_q * stride_qn + dim_offs * stride_qd,
                mask=dim_mask,
                other=0.0,
            )

            # Load docs: [TILE_D, TILE_DIM]
            d = tl.load(
                D_ptr + d_offsets[:, None] * stride_dn + dim_offs[None, :] * stride_dd,
                mask=d_mask[:, None] & dim_mask[None, :],
                other=0.0,
            )

            # Dot product: [TILE_D]
            acc += tl.sum(d * q[None, :], axis=1)

        # Store
        out_ptrs = Out_ptr + pid_q * stride_on + d_offsets * stride_od
        tl.store(out_ptrs, acc, mask=d_mask)
    """)


def gen_sparse_score_kernel(tile: TileConfig) -> str:
    """Generate a Triton kernel for sparse scoring (SPLADE-style)."""
    return textwrap.dedent(f"""\
    import triton
    import triton.language as tl

    @triton.jit
    def sparse_score_kernel(
        Q_weights_ptr,    # [n_queries, n_query_terms] - sparse weights
        Q_indices_ptr,    # [n_queries, n_query_terms] - term IDs
        D_weights_ptr,    # [n_docs, max_terms] - sparse doc weights
        D_indices_ptr,    # [n_docs, max_terms] - term IDs
        D_lengths_ptr,    # [n_docs] - actual number of terms per doc
        Out_ptr,          # [n_queries, n_docs]
        n_queries,
        n_docs,
        n_query_terms,
        max_doc_terms,
        stride_qwn, stride_qwt,
        stride_qin, stride_qit,
        stride_dwn, stride_dwt,
        stride_din, stride_dit,
        stride_on, stride_od,
        TILE_D: tl.constexpr = {tile.tile_d},
        TILE_QT: tl.constexpr = {tile.tile_qt},
    ):
        pid_q = tl.program_id(0)
        pid_d = tl.program_id(1)

        d_start = pid_d * TILE_D
        d_offsets = d_start + tl.arange(0, TILE_D)
        d_mask = d_offsets < n_docs

        # Score accumulator: [TILE_D]
        scores = tl.zeros([TILE_D], dtype=tl.float32)

        # For each query term
        for qt in range(0, n_query_terms, TILE_QT):
            qt_offs = qt + tl.arange(0, TILE_QT)
            qt_mask = qt_offs < n_query_terms

            # Load query term weights and IDs
            q_w = tl.load(
                Q_weights_ptr + pid_q * stride_qwn + qt_offs * stride_qwt,
                mask=qt_mask, other=0.0
            )
            q_idx = tl.load(
                Q_indices_ptr + pid_q * stride_qin + qt_offs * stride_qit,
                mask=qt_mask, other=-1
            )

            # For each doc in tile, check if any doc term matches query terms
            # This is a simplified version; real implementation would use
            # inverted index for efficiency
            for di in range(TILE_D):
                d_idx = d_start + di
                if d_idx < n_docs:
                    d_len = tl.load(D_lengths_ptr + d_idx)
                    for dt in range(max_doc_terms):
                        if dt < d_len:
                            d_term = tl.load(D_indices_ptr + d_idx * stride_din + dt * stride_dit)
                            d_w = tl.load(D_weights_ptr + d_idx * stride_dwn + dt * stride_dwt)
                            # Check matches
                            for qti in range(TILE_QT):
                                if qt + qti < n_query_terms:
                                    if d_term == q_idx[qti]:
                                        scores[di] += q_w[qti] * d_w

        # Store
        out_ptrs = Out_ptr + pid_q * stride_on + d_offsets * stride_od
        tl.store(out_ptrs, scores, mask=d_mask)
    """)


def gen_decompress_score_kernel(tile: TileConfig) -> str:
    """Generate a fused decompress+score kernel (PQ ADC - asymmetric distance computation)."""
    return textwrap.dedent(f"""\
    import triton
    import triton.language as tl

    @triton.jit
    def decompress_score_kernel(
        Q_ptr,            # [n_queries, dim] - float16 query vectors
        Codes_ptr,        # [n_docs, n_subvectors] - uint8 PQ codes
        Codebook_ptr,     # [n_subvectors, 256, sub_dim] - PQ codebook
        Out_ptr,          # [n_queries, n_docs] - float32 scores
        n_queries,
        n_docs,
        dim,
        n_subvectors,
        sub_dim,          # = dim // n_subvectors
        stride_qn, stride_qd,
        stride_cn, stride_cs,
        stride_cbs, stride_cbc, stride_cbd,
        stride_on, stride_od,
        TILE_D: tl.constexpr = {tile.tile_d},
        N_SUB: tl.constexpr = 8,
        SUB_DIM: tl.constexpr = 16,
    ):
        pid_q = tl.program_id(0)
        pid_d = tl.program_id(1)

        d_start = pid_d * TILE_D
        d_offsets = d_start + tl.arange(0, TILE_D)
        d_mask = d_offsets < n_docs

        # Precompute distance tables: for each subvector, compute dot product
        # of query sub-vector with all 256 centroids
        # This gives us a [N_SUB, 256] lookup table
        # Then scoring is just: sum over subvectors of table[code[sv]]

        scores = tl.zeros([TILE_D], dtype=tl.float32)

        for sv in range(N_SUB):
            # Load query subvector: [SUB_DIM]
            dim_offs = sv * SUB_DIM + tl.arange(0, SUB_DIM)
            q_sub = tl.load(
                Q_ptr + pid_q * stride_qn + dim_offs * stride_qd,
                mask=dim_offs < dim,
                other=0.0,
            )

            # Load PQ codes for this subvector: [TILE_D]
            codes = tl.load(
                Codes_ptr + d_offsets * stride_cn + sv * stride_cs,
                mask=d_mask,
                other=0,
            )

            # For each doc, look up the centroid and compute partial dot product
            # This is a gather operation: codebook[sv, code, :]
            for di in range(TILE_D):
                d_idx = d_start + di
                if d_idx < n_docs:
                    code_val = tl.load(Codes_ptr + d_idx * stride_cn + sv * stride_cs)
                    # Load centroid: [SUB_DIM]
                    centroid = tl.load(
                        Codebook_ptr + sv * stride_cbs + code_val * stride_cbc
                        + tl.arange(0, SUB_DIM) * stride_cbd
                    )
                    # Partial dot product
                    scores[di] += tl.sum(q_sub * centroid)

        # Store
        out_ptrs = Out_ptr + pid_q * stride_on + d_offsets * stride_od
        tl.store(out_ptrs, scores, mask=d_mask)
    """)


# ---------------------------------------------------------------------------
# Code generator dispatcher
# ---------------------------------------------------------------------------

def generate_kernel(
    node: IRNode,
    tile: TileConfig,
) -> str:
    """Generate a Triton kernel string for the given IR node and tile config."""
    if node.op_type == OpType.FUSED:
        fusion_name = node.attrs.get("fusion_name", "")
        if fusion_name == "score_max_reduce_sum_reduce":
            return gen_maxsim_optimized_kernel(tile)
        elif fusion_name == "score_max_reduce":
            return gen_maxsim_fused_kernel(tile)
        elif fusion_name == "decompress_score":
            return gen_decompress_score_kernel(tile)
        elif fusion_name == "sparse_lookup_score":
            return gen_sparse_score_kernel(tile)
        elif fusion_name in ("score_topk", "sum_reduce_topk"):
            return gen_dot_score_kernel(tile)
        else:
            return gen_dot_score_kernel(tile)
    elif node.op_type == OpType.SCORE:
        if hasattr(node, 'mode') and node.mode == ScoreMode.MAXSIM:
            return gen_maxsim_fused_kernel(tile)
        else:
            return gen_dot_score_kernel(tile)
    else:
        return f"# No specialized kernel for {node.op_type.value}; use PyTorch fallback\n"


def generate_pipeline(
    graph: PipelineGraph,
    tile_configs: Dict[str, TileConfig],
) -> str:
    """Generate complete Triton code for all kernels in a pipeline."""
    parts = [
        "# Auto-generated by RetrieverCompiler",
        "# Pipeline: " + graph.summary(),
        "",
        "import torch",
        "import triton",
        "import triton.language as tl",
        "",
    ]

    for node in graph.topo_order():
        if node.name in tile_configs:
            parts.append(f"# --- {node.name} ---")
            kernel_code = generate_kernel(node, tile_configs[node.name])
            # Remove the import lines since we already have them at the top
            lines = kernel_code.split("\n")
            lines = [l for l in lines if not l.startswith("import ")]
            parts.append("\n".join(lines))
            parts.append("")

    return "\n".join(parts)


def compile_pipeline(
    graph: PipelineGraph,
    tile_configs: Dict[str, TileConfig],
    output_path: Optional[str] = None,
) -> str:
    """Generate and optionally save the compiled pipeline code."""
    code = generate_pipeline(graph, tile_configs)
    if output_path:
        with open(output_path, "w") as f:
            f.write(code)
    return code
