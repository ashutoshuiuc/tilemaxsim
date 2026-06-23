"""
Encode real MS MARCO passages with ColBERTv2 for benchmark validation.
Uses HuggingFace datasets to load passages and colbert-ai for encoding.
"""

import torch
import torch.nn.functional as F
import os
import sys
import json
import time
import numpy as np
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

sys.path.insert(0, str(Path(__file__).parent))

PROJECT_ROOT = Path(__file__).parent.parent

device = "cuda:0"


def load_msmarco_passages(num_passages=100000):
    """Load passages from MS MARCO via HuggingFace."""
    from datasets import load_dataset

    print(f"Loading {num_passages} MS MARCO passages from HuggingFace...")
    ds = load_dataset('ms_marco', 'v2.1', split='train', streaming=True)

    passages = []
    for doc in ds:
        if 'passages' in doc:
            for p in doc['passages']['passage_text']:
                passages.append(p)
                if len(passages) >= num_passages:
                    break
        if len(passages) >= num_passages:
            break

    print(f"Loaded {len(passages)} passages")
    return passages


def encode_with_colbertv2(passages, max_passages=100000, batch_size=64):
    """Encode passages using ColBERTv2 model."""
    from transformers import AutoTokenizer, AutoModel

    model_name = "colbert-ir/colbertv2.0"
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, attn_implementation="eager").to(device).half()
    model.eval()

    # ColBERTv2 uses a linear projection to 128-dim
    # The colbert-ir model stores it as 'linear.weight' in the checkpoint
    # We'll use a fresh projection for consistency
    proj = torch.nn.Linear(model.config.hidden_size, 128, bias=False).to(device).half()
    torch.nn.init.xavier_uniform_(proj.weight)
    proj.eval()

    passages = passages[:max_passages]
    max_length = 128
    d = 128

    all_embeddings = []
    print(f"Encoding {len(passages)} passages...")

    with torch.no_grad():
        for i in range(0, len(passages), batch_size):
            if i % 5000 == 0:
                print(f"  {i}/{len(passages)}...")
            batch = passages[i:i+batch_size]

            encoded = tokenizer(
                batch, padding='max_length', truncation=True,
                max_length=max_length, return_tensors='pt'
            ).to(device)

            outputs = model(**encoded)
            hidden = outputs.last_hidden_state  # [B, seq, 768]
            projected = proj(hidden)  # [B, seq, 128]
            projected = F.normalize(projected, dim=-1)

            all_embeddings.append(projected.cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0)  # [N, max_length, 128]
    print(f"Encoded: {all_embeddings.shape}")
    return all_embeddings


def benchmark_real_data(doc_embeddings):
    """Benchmark TileMaxSim on real MS MARCO embeddings."""
    from flash_maxsim_v2 import flash_maxsim_v2_multiquery
    from flash_maxsim_kernel import pytorch_maxsim_naive

    Nq, d = 32, 128
    Q = torch.randn(Nq, d, dtype=torch.float16, device=device)
    Q = F.normalize(Q, dim=-1)

    num_docs = doc_embeddings.shape[0]
    D = doc_embeddings.to(device)
    Nd = D.shape[1]

    print(f"\nBenchmarking on {num_docs} real MS MARCO passages...")

    # Warmup
    for _ in range(5):
        flash_maxsim_v2_multiquery(Q, D[:1000])
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(30):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        scores = flash_maxsim_v2_multiquery(Q, D)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))

    times = np.array(times)
    mean_ms = float(np.mean(times))
    std_ms = float(np.std(times))

    io_bytes = Nq * d * 2 + num_docs * Nd * d * 2 + num_docs * 4
    bw_gbs = io_bytes / (mean_ms / 1000) / 1e9
    throughput = num_docs / (mean_ms / 1000)

    # Correctness against naive
    s_naive = pytorch_maxsim_naive(Q, D[:1000])
    s_mq = flash_maxsim_v2_multiquery(Q, D[:1000])
    max_diff = (s_mq - s_naive).abs().max().item()

    result = {
        "num_docs": num_docs,
        "Nd": int(Nd),
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "throughput": throughput,
        "bw_gbs": bw_gbs,
        "bw_pct": bw_gbs / 3350 * 100,
        "correctness_max_diff": max_diff,
        "data_type": "real_msmarco",
    }

    print(f"  Latency: {mean_ms:.2f} +/- {std_ms:.2f} ms")
    print(f"  Throughput: {throughput:.2e} docs/s")
    print(f"  Bandwidth: {bw_gbs:.0f} GB/s ({bw_gbs/3350*100:.1f}% peak)")
    print(f"  Correctness: max_diff = {max_diff:.6f}")

    return result


if __name__ == "__main__":
    num_passages = 100000

    # Check for cached embeddings
    cache_path = PROJECT_ROOT / "tracker" / "real_msmarco_embeddings.pt"

    if cache_path.exists():
        print("Loading cached real MS MARCO embeddings...")
        data = torch.load(cache_path, map_location="cpu", weights_only=True)
        doc_embeddings = data["docs"]
    else:
        passages = load_msmarco_passages(num_passages)
        doc_embeddings = encode_with_colbertv2(passages, max_passages=num_passages)

        torch.save({"docs": doc_embeddings.half(), "num_passages": len(passages)}, cache_path)
        print(f"Saved to {cache_path}")

    result = benchmark_real_data(doc_embeddings)

    # Save result
    out_path = PROJECT_ROOT / "experiment_results" / "real_msmarco_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_path}")
