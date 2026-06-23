"""
Cycle 6: Re-run scaling experiments with BQ=32 kernel to update stale tables.
Covers: query token scaling, doc token scaling, precision comparison.
"""

import torch
import torch.nn.functional as F
import numpy as np
import os, sys
from pathlib import Path
import json

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, str(Path(__file__).parent))

PROJECT_ROOT = Path(__file__).parent.parent

from flash_maxsim_v2 import flash_maxsim_v2_multiquery
from flash_maxsim_kernel import pytorch_maxsim_naive

device = "cuda:0"

def time_fn(fn, warmup=10, repeat=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return np.array(times)

results = {}

# 1. Query token scaling
print("=" * 60)
print("QUERY TOKEN SCALING (Nd=128, d=128, B=10K)")
print("=" * 60)
d, Nd, B = 128, 128, 10000
results["query_scaling"] = {}
for Nq in [8, 16, 32, 64]:
    Q = torch.randn(Nq, d, dtype=torch.float16, device=device)
    D = torch.randn(B, Nd, d, dtype=torch.float16, device=device)

    times = time_fn(lambda: flash_maxsim_v2_multiquery(Q, D))
    mean = float(np.mean(times))
    tp = B / (mean / 1000)

    times_naive = time_fn(lambda: pytorch_maxsim_naive(Q, D), warmup=5, repeat=15)
    mean_naive = float(np.mean(times_naive))
    tp_naive = B / (mean_naive / 1000)

    results["query_scaling"][f"Nq{Nq}"] = {
        "v2mq_ms": mean, "v2mq_throughput": tp,
        "naive_ms": mean_naive, "naive_throughput": tp_naive,
    }
    print(f"  Nq={Nq:>2}: V2-MQ={mean:.3f}ms ({tp:.1e} M/s) | Naive={mean_naive:.3f}ms ({tp_naive:.1e} M/s)")
    torch.cuda.empty_cache()

# 2. Document token scaling
print("\n" + "=" * 60)
print("DOC TOKEN SCALING (Nq=32, d=128, B=10K)")
print("=" * 60)
Nq = 32
results["doc_scaling"] = {}
for Nd in [32, 64, 128, 256, 512]:
    Q = torch.randn(Nq, d, dtype=torch.float16, device=device)
    D = torch.randn(B, Nd, d, dtype=torch.float16, device=device)

    times = time_fn(lambda: flash_maxsim_v2_multiquery(Q, D))
    mean = float(np.mean(times))
    tp = B / (mean / 1000)
    io = Nq * d * 2 + B * Nd * d * 2 + B * 4
    bw = io / (mean / 1000) / 1e9

    try:
        times_naive = time_fn(lambda: pytorch_maxsim_naive(Q, D), warmup=3, repeat=10)
        tp_naive = B / (float(np.mean(times_naive)) / 1000)
    except:
        tp_naive = None
        torch.cuda.empty_cache()

    results["doc_scaling"][f"Nd{Nd}"] = {
        "v2mq_ms": mean, "v2mq_throughput": tp, "bw_gbs": bw,
        "naive_throughput": tp_naive,
    }
    print(f"  Nd={Nd:>3}: V2-MQ={mean:.3f}ms ({tp/1e6:.1f}M/s, BW={bw:.0f} GB/s) | Naive={tp_naive/1e6:.1f}M/s" if tp_naive else f"  Nd={Nd:>3}: V2-MQ={mean:.3f}ms ({tp/1e6:.1f}M/s, BW={bw:.0f} GB/s)")
    torch.cuda.empty_cache()

# 3. Precision comparison
print("\n" + "=" * 60)
print("PRECISION COMPARISON (Nq=32, Nd=128, d=128, B=10K)")
print("=" * 60)
Nq, Nd, d = 32, 128, 128
results["precision"] = {}
for dtype_name, dtype in [("FP32", torch.float32), ("FP16", torch.float16), ("BF16", torch.bfloat16)]:
    Q = torch.randn(Nq, d, dtype=dtype, device=device)
    D = torch.randn(B, Nd, d, dtype=dtype, device=device)

    times = time_fn(lambda: flash_maxsim_v2_multiquery(Q, D))
    mean = float(np.mean(times))
    tp = B / (mean / 1000)

    try:
        times_naive = time_fn(lambda: pytorch_maxsim_naive(Q, D), warmup=5, repeat=15)
        tp_naive = B / (float(np.mean(times_naive)) / 1000)
    except:
        tp_naive = None
        torch.cuda.empty_cache()

    results["precision"][dtype_name] = {
        "v2mq_ms": mean, "v2mq_throughput": tp,
        "naive_ms": float(np.mean(times_naive)) if tp_naive else None,
        "naive_throughput": tp_naive,
    }
    print(f"  {dtype_name}: V2-MQ={mean:.3f}ms ({tp/1e6:.1f}M/s) | Naive={tp_naive/1e6:.1f}M/s" if tp_naive else f"  {dtype_name}: V2-MQ={mean:.3f}ms ({tp/1e6:.1f}M/s)")
    torch.cuda.empty_cache()

# Save
out = PROJECT_ROOT / "experiment_results" / "cycle6_scaling.json"
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {out}")
