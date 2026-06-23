"""
Auto-Tiling Pass

Given hardware parameters (SRAM size, bandwidth, compute throughput) and
operation characteristics, automatically determine optimal tile sizes for
each fused kernel.

The approach is analytical (not autotuning): we model the working set size
and compute-to-memory ratio for each tile configuration and select the one
that maximizes compute utilization while fitting in SRAM.

Key hardware parameters for H100:
  - L2 cache: 50 MB
  - Shared memory per SM: 228 KB
  - Memory bandwidth: 3.35 TB/s
  - FP16 compute: 989 TFLOPS (with sparsity)
  - SMs: 132

Tiling dimensions for retrieval operations:
  - TILE_Q: number of queries processed per block
  - TILE_QT: number of query tokens per tile
  - TILE_D: number of documents per tile
  - TILE_DT: number of document tokens per tile
  - TILE_DIM: embedding dimension tile (usually full dim fits)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .ir import FusedOp, IRNode, OpType, PipelineGraph


# ---------------------------------------------------------------------------
# Hardware model
# ---------------------------------------------------------------------------

@dataclass
class HardwareConfig:
    """Hardware parameters for tiling decisions."""
    sram_bytes: int = 228 * 1024       # Shared memory per SM (H100)
    l2_bytes: int = 50 * 1024 * 1024   # L2 cache size
    bandwidth_gb_s: float = 3350.0     # HBM bandwidth in GB/s
    compute_tflops: float = 989.0      # FP16 TFLOPS
    n_sms: int = 132                   # Number of SMs
    warp_size: int = 32
    max_threads_per_sm: int = 2048

    def compute_intensity_threshold(self) -> float:
        """Compute intensity (FLOPS/byte) needed to be compute-bound."""
        return (self.compute_tflops * 1e12) / (self.bandwidth_gb_s * 1e9)

    @classmethod
    def h100(cls) -> HardwareConfig:
        return cls()

    @classmethod
    def a100(cls) -> HardwareConfig:
        return cls(
            sram_bytes=192 * 1024,
            l2_bytes=40 * 1024 * 1024,
            bandwidth_gb_s=2039.0,
            compute_tflops=312.0,
            n_sms=108,
        )


# ---------------------------------------------------------------------------
# Tile configuration
# ---------------------------------------------------------------------------

@dataclass
class TileConfig:
    """Tile sizes for a kernel."""
    tile_q: int = 1          # queries per block
    tile_qt: int = 32        # query tokens per tile
    tile_d: int = 64         # documents per tile
    tile_dt: int = 32        # document tokens per tile
    tile_dim: int = 128      # embedding dimension tile
    n_warps: int = 4         # warps per block
    n_stages: int = 2        # pipeline stages for software pipelining

    def sram_usage(self, dtype_bytes: int = 2) -> int:
        """Estimate shared memory usage in bytes."""
        # Query tile: tile_qt * tile_dim
        q_buf = self.tile_qt * self.tile_dim * dtype_bytes
        # Doc tile: tile_dt * tile_dim
        d_buf = self.tile_dt * self.tile_dim * dtype_bytes
        # Score accumulator: tile_qt * tile_d (kept in registers ideally)
        # Double buffer for pipelining
        return (q_buf + d_buf) * self.n_stages

    def compute_flops(self) -> int:
        """Estimate FLOPs per tile (for MaxSim-like ops)."""
        # For each (qt, d) pair: dot product over dim, then max over dt
        return 2 * self.tile_qt * self.tile_d * self.tile_dt * self.tile_dim

    def memory_bytes(self, dtype_bytes: int = 2) -> int:
        """Estimate memory traffic per tile."""
        q_read = self.tile_qt * self.tile_dim * dtype_bytes
        d_read = self.tile_d * self.tile_dt * self.tile_dim * dtype_bytes
        score_write = self.tile_q * self.tile_d * 4  # float32 output
        return q_read + d_read + score_write

    def compute_intensity(self, dtype_bytes: int = 2) -> float:
        """FLOPS per byte of memory traffic."""
        mem = self.memory_bytes(dtype_bytes)
        if mem == 0:
            return float('inf')
        return self.compute_flops() / mem


# ---------------------------------------------------------------------------
# Auto-tiling algorithms
# ---------------------------------------------------------------------------

def auto_tile_maxsim(
    hw: HardwareConfig,
    n_queries: int = 1,
    query_tokens: int = 32,
    n_docs: int = 8192,
    doc_tokens: int = 180,
    dim: int = 128,
    dtype_bytes: int = 2,
) -> TileConfig:
    """
    Determine optimal tile sizes for a MaxSim (fused score+max_reduce+sum_reduce) kernel.

    Strategy:
    1. tile_dim = dim (embedding dimension usually fits entirely)
    2. tile_qt = query_tokens (process all query tokens per block)
    3. Maximize tile_d subject to SRAM budget
    4. Choose tile_dt to maximize compute intensity
    """
    sram_budget = hw.sram_bytes
    ci_threshold = hw.compute_intensity_threshold()

    # Full embedding dimension in one tile
    tile_dim = dim

    # All query tokens (usually 32) in one tile
    tile_qt = min(query_tokens, 64)

    # Find maximum tile_dt that fits in SRAM with reasonable tile_d
    best_config = None
    best_ci = 0.0

    for tile_dt in [16, 32, 64, 128]:
        if tile_dt > doc_tokens:
            continue
        for tile_d in [16, 32, 64, 128, 256]:
            if tile_d > n_docs:
                continue

            config = TileConfig(
                tile_q=1,
                tile_qt=tile_qt,
                tile_d=tile_d,
                tile_dt=tile_dt,
                tile_dim=tile_dim,
            )

            sram = config.sram_usage(dtype_bytes)
            if sram > sram_budget:
                continue

            ci = config.compute_intensity(dtype_bytes)

            # Prefer configs that are compute-bound (ci > threshold)
            # Among compute-bound configs, prefer larger tiles (more work per block)
            score = ci * config.compute_flops()
            if ci > best_ci or (ci >= ci_threshold and config.compute_flops() > (best_config.compute_flops() if best_config else 0)):
                best_ci = ci
                best_config = config

    if best_config is None:
        # Fallback to conservative tile sizes
        best_config = TileConfig(
            tile_q=1, tile_qt=32, tile_d=32, tile_dt=32, tile_dim=dim
        )

    return best_config


def auto_tile_dot(
    hw: HardwareConfig,
    n_queries: int = 1,
    n_docs: int = 8192,
    dim: int = 768,
    dtype_bytes: int = 2,
) -> TileConfig:
    """
    Determine optimal tile sizes for a dot-product scoring kernel.

    This is essentially a GEMV/GEMM tiling problem.
    """
    sram_budget = hw.sram_bytes

    best_config = None
    best_throughput = 0.0

    for tile_d in [64, 128, 256, 512, 1024]:
        if tile_d > n_docs:
            continue
        for tile_dim_chunk in [64, 128, 256]:
            if tile_dim_chunk > dim:
                tile_dim_chunk = dim

            # SRAM: query chunk + doc chunk
            sram = (tile_dim_chunk + tile_d * tile_dim_chunk) * dtype_bytes * 2
            if sram > sram_budget:
                continue

            flops = 2 * tile_d * tile_dim_chunk
            mem = (tile_dim_chunk + tile_d * tile_dim_chunk) * dtype_bytes
            ci = flops / mem if mem > 0 else 0

            # Throughput proxy: total work / max(compute_time, memory_time)
            compute_time = flops / (hw.compute_tflops * 1e12)
            memory_time = mem / (hw.bandwidth_gb_s * 1e9)
            throughput = flops / max(compute_time, memory_time)

            if throughput > best_throughput:
                best_throughput = throughput
                best_config = TileConfig(
                    tile_q=1,
                    tile_qt=1,
                    tile_d=tile_d,
                    tile_dt=1,
                    tile_dim=tile_dim_chunk,
                )

    if best_config is None:
        best_config = TileConfig(tile_q=1, tile_d=128, tile_dim=dim)

    return best_config


def auto_tile_sparse(
    hw: HardwareConfig,
    n_queries: int = 1,
    vocab_size: int = 30522,
    avg_postings: int = 100,
    dtype_bytes: int = 2,
) -> TileConfig:
    """
    Determine tile sizes for sparse scoring kernel.
    Key dimension: number of query terms processed per block.
    """
    # Sparse scoring is memory-bound; tile over query terms
    return TileConfig(
        tile_q=1,
        tile_qt=64,  # query terms per block
        tile_d=256,   # docs to accumulate per block
        tile_dt=1,
        tile_dim=1,
    )


def auto_tile(
    graph: PipelineGraph,
    hw: Optional[HardwareConfig] = None,
    **kwargs,
) -> Dict[str, TileConfig]:
    """
    Auto-tile all operations in a pipeline graph.

    Returns a mapping from node name to TileConfig.
    """
    if hw is None:
        hw = HardwareConfig.h100()

    tile_configs = {}
    for node in graph.topo_order():
        if node.op_type == OpType.FUSED:
            fusion_name = node.attrs.get("fusion_name", "")
            if "maxsim" in fusion_name or "score_max_reduce" in fusion_name:
                tile_configs[node.name] = auto_tile_maxsim(hw, **kwargs)
            elif "decompress_score" in fusion_name:
                tile_configs[node.name] = auto_tile_dot(hw, **kwargs)
            elif "sparse" in fusion_name:
                tile_configs[node.name] = auto_tile_sparse(hw, **kwargs)
            else:
                tile_configs[node.name] = auto_tile_dot(hw, **kwargs)
        elif node.op_type == OpType.SCORE:
            if hasattr(node, 'mode'):
                from .ir import ScoreMode
                if node.mode == ScoreMode.MAXSIM:
                    tile_configs[node.name] = auto_tile_maxsim(hw, **kwargs)
                else:
                    tile_configs[node.name] = auto_tile_dot(hw, **kwargs)
            else:
                tile_configs[node.name] = auto_tile_dot(hw, **kwargs)
        elif node.op_type in (OpType.TOPK, OpType.MAX_REDUCE, OpType.SUM_REDUCE):
            tile_configs[node.name] = TileConfig(tile_d=256)

    return tile_configs


def tiling_report(tile_configs: Dict[str, TileConfig], hw: HardwareConfig) -> str:
    """Generate a human-readable report of tiling decisions."""
    lines = ["Tiling Report", "=" * 60]
    ci_threshold = hw.compute_intensity_threshold()
    lines.append(f"Hardware compute intensity threshold: {ci_threshold:.1f} FLOPS/byte")
    lines.append("")

    for name, config in tile_configs.items():
        ci = config.compute_intensity()
        bound = "compute-bound" if ci >= ci_threshold else "memory-bound"
        lines.append(f"  {name}:")
        lines.append(f"    Tiles: Q={config.tile_q}, QT={config.tile_qt}, "
                      f"D={config.tile_d}, DT={config.tile_dt}, DIM={config.tile_dim}")
        lines.append(f"    SRAM usage: {config.sram_usage() / 1024:.1f} KB")
        lines.append(f"    Compute intensity: {ci:.1f} FLOPS/byte ({bound})")
        lines.append(f"    FLOPs/tile: {config.compute_flops():,}")
        lines.append("")

    return "\n".join(lines)
