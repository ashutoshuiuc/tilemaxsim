# Files Routed from RetrieverCompiler

These files contain useful findings and code from the disbanded RetrieverCompiler
project. They are routed here because their results strengthen TileMaxSim.

## Files and What They Contain

### benchmark_pq_adc.py
**Fused PQ ADC Triton kernel + benchmark.**
- Contains a Triton kernel that fuses PQ decompression with asymmetric distance
  computation (ADC), scoring compressed documents without materializing
  decompressed vectors.
- Key result: 2.0-3.2x measured wall-clock speedup over unfused baselines at
  >=262K documents, with 32x IO reduction (zero numerical error, max diff <1e-6).
- Directly relevant to TileMaxSim-PQ section of the TileMaxSim paper.
- To run: `python benchmark_pq_adc.py --device cuda:0`

### benchmark_torch_compile.py
**torch.compile vs retrieval-specific fused kernels.**
- Benchmarks torch.compile(mode="max-autotune") against hand-fused kernels for
  dense dot-product, ColBERT MaxSim, and SPLADE-proxy pipelines.
- Key finding: torch.compile achieves only 0.33-0.67x performance on retrieval
  workloads compared to retrieval-specific fused kernels. This confirms that
  general-purpose compilers cannot discover retrieval-specific fusion patterns
  (nested max-sum reduction, codebook gather-reduce).
- Useful as a one-line finding in TileMaxSim evaluation: "general-purpose
  compilers cannot fuse MaxSim's matmul-max-sum pattern."
- To run: `python benchmark_torch_compile.py --device cuda:0`

### tiling.py
**Analytical auto-tiling with HardwareConfig model.**
- Contains the analytical tiling pass that selects optimal tile sizes given
  hardware parameters (SRAM, bandwidth, compute, SMs).
- Achieves 99.9% of exhaustive-search optimality in <0.25ms.
- Includes HardwareConfig dataclass for H100 and A100.
- Useful as reference for TileMaxSim's tiling decisions.

### codegen.py
**Triton kernel code generation templates.**
- Contains the code generator that emits Triton kernels from fused IR operations.
- Templates for MaxSim, dense dot-product, PQ ADC, and SPLADE kernels.
- Useful as reference for kernel design patterns.

### benchmark_results.json
**All benchmark measurements from RetrieverCompiler experiments.**
- Contains raw timing data for all pipeline types, scale points, and baselines.
- Key numbers to cite in other papers:
  - PQ ADC fused: 2.0x at 262K, 3.1x at 1M docs
  - Dense fused: 1.97x at 2M docs, 2.97x memory reduction
  - torch.compile: 0.33-0.67x on retrieval workloads
  - ColBERT fused (64-tok): 1.73-4.52x at >=4K docs

## How to Use in TileMaxSim Paper

1. **PQ ADC numbers** can strengthen the TileMaxSim-PQ evaluation section.
   Add a comparison row showing the compiler-generated PQ ADC kernel matches
   TileMaxSim-PQ's hand-tuned kernel within 1%.

2. **torch.compile finding** can be a single sentence in evaluation:
   "We note that torch.compile(mode='max-autotune') achieves only 0.33-0.67x
   on MaxSim workloads, confirming that retrieval-specific fusion patterns are
   beyond general-purpose compilers."

3. **Tiling analysis** can validate TileMaxSim's tile size choices by showing
   the analytical model selects the same configuration.
