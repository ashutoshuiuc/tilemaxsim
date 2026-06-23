"""Quick script for nsys profiling."""
import torch
import sys
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, str(Path(__file__).parent))

from flash_maxsim_v2 import flash_maxsim_v2_multiquery

device = "cuda:0"
Nq, d, Nd, B = 32, 128, 128, 100000

Q = torch.randn(Nq, d, dtype=torch.float16, device=device)
D = torch.randn(B, Nd, d, dtype=torch.float16, device=device)

# Warmup
for _ in range(10):
    flash_maxsim_v2_multiquery(Q, D)
torch.cuda.synchronize()

# Profile range
torch.cuda.nvtx.range_push("TileMaxSim_V2MQ")
for _ in range(5):
    scores = flash_maxsim_v2_multiquery(Q, D)
torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()

print(f"Done. Scores shape: {scores.shape}, mean: {scores.mean():.4f}")
