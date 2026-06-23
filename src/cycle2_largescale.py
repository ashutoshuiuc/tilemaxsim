"""
Large-scale benchmark: V2-MQ on 500K passages.
Tests scoring at production-relevant scales.
"""

import torch
import time
import json
import os
import sys
import numpy as np
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

sys.path.insert(0, str(Path(__file__).parent))
from flash_maxsim_v2 import flash_maxsim_v2_multiquery
from flash_maxsim_kernel import pytorch_maxsim_loop

PROJECT_ROOT = Path(__file__).parent.parent
device = "cuda:0"

# Load embeddings
emb_path = PROJECT_ROOT / "tracker" / "msmarco_embeddings.pt"
print("Loading 500K passage embeddings...")
data = torch.load(emb_path, map_location="cpu", weights_only=True)
D_all = data["docs"]  # [500K, 128, 128] fp16
Q_all = data["queries"]  # [200, 32, 128] fp16

print(f"  Docs: {D_all.shape} ({D_all.numel() * 2 / 1e9:.1f} GB)")
print(f"  Queries: {Q_all.shape}")

# Use first query
Q = Q_all[0].to(device)  # [32, 128]

results = {}

# Test at various scales
for num_docs in [10000, 50000, 100000, 200000, 500000]:
    print(f"\n--- {num_docs} documents ---")
    D = D_all[:num_docs].to(device)
    torch.cuda.synchronize()

    # V2-MQ
    # Warmup
    for _ in range(3):
        flash_maxsim_v2_multiquery(Q, D)
    torch.cuda.synchronize()

    times = []
    for _ in range(20):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        scores = flash_maxsim_v2_multiquery(Q, D)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times = np.array(times)
    mean_ms = float(np.mean(times))
    std_ms = float(np.std(times))

    # Bandwidth calculation (optimal: D read once)
    Nq, d = 32, 128
    Nd = 128
    io_bytes = Nq * d * 2 + num_docs * Nd * d * 2 + num_docs * 4
    bw_gbs = io_bytes / (mean_ms / 1000) / 1e9
    bw_pct = bw_gbs / 3350 * 100
    throughput = num_docs / (mean_ms / 1000)

    results[f"docs_{num_docs}"] = {
        "num_docs": num_docs,
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "throughput": throughput,
        "bw_gbs": bw_gbs,
        "bw_pct": bw_pct,
    }

    print(f"  V2-MQ: {mean_ms:.2f} +/- {std_ms:.2f} ms")
    print(f"  Throughput: {throughput:.2e} docs/s")
    print(f"  Bandwidth: {bw_gbs:.0f} GB/s ({bw_pct:.1f}% peak)")

    # Loop baseline only for smaller scales (too slow otherwise)
    if num_docs <= 100000:
        for _ in range(2):
            pytorch_maxsim_loop(Q, D)
        torch.cuda.synchronize()

        loop_times = []
        reps = 5 if num_docs <= 50000 else 3
        for _ in range(reps):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            pytorch_maxsim_loop(Q, D)
            end.record()
            torch.cuda.synchronize()
            loop_times.append(start.elapsed_time(end))

        loop_mean = float(np.mean(loop_times))
        speedup = loop_mean / mean_ms
        results[f"docs_{num_docs}"]["loop_ms"] = loop_mean
        results[f"docs_{num_docs}"]["speedup"] = speedup
        print(f"  Loop: {loop_mean:.2f} ms -> {speedup:.0f}x speedup")

    del D
    torch.cuda.empty_cache()

# Save
out_path = PROJECT_ROOT / "experiment_results" / "cycle2_largescale.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out_path}")
