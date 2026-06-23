"""
Full-scale TileMaxSim benchmark on MS MARCO passages.

Steps:
1. Download MS MARCO passages and queries
2. Encode passages and queries with a ColBERT-style model (BERT-base, 128-dim)
3. Benchmark TileMaxSim kernels at realistic scale
4. Compare against PyTorch baselines
5. Save results for paper
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import json
import os
import sys
from pathlib import Path

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent))

RESULTS_DIR = Path(__file__).parent.parent / "tracker"
DATA_DIR = Path(os.environ.get("DATA_ROOT", str(Path(__file__).parent.parent.parent / "datasets")))


def encode_texts_colbert_style(texts, model, tokenizer, device, max_length=128, dim=128, batch_size=256):
    """Encode texts into ColBERT-style per-token embeddings."""
    all_embeddings = []
    all_lengths = []

    model.eval()
    # Simple linear projection to 128-dim
    proj = torch.nn.Linear(model.config.hidden_size, dim).to(device).half()
    proj.eval()

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        encoded = tokenizer(batch, padding=True, truncation=True,
                          max_length=max_length, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model(**encoded)
            hidden = outputs.last_hidden_state  # [B, seq, hidden]
            projected = proj(hidden)  # [B, seq, dim]
            projected = F.normalize(projected, dim=-1)

        mask = encoded['attention_mask']
        for b in range(projected.shape[0]):
            length = mask[b].sum().item()
            all_embeddings.append(projected[b, :length].cpu())
            all_lengths.append(length)

        if (i // batch_size) % 50 == 0:
            print(f"  Encoded {i+len(batch)}/{len(texts)} texts")

    return all_embeddings, all_lengths


def pad_embeddings(embeddings, lengths, max_tokens, dim):
    """Pad variable-length embeddings to fixed size tensor."""
    n = len(embeddings)
    padded = torch.zeros(n, max_tokens, dim)
    for i, (emb, length) in enumerate(zip(embeddings, lengths)):
        actual = min(length, max_tokens)
        padded[i, :actual] = emb[:actual]
    return padded


def benchmark_maxsim_methods(query_embs, doc_embs, device, n_warmup=3, n_runs=10):
    """Benchmark different MaxSim implementations at scale."""
    Q = query_embs.to(device)
    D = doc_embs.to(device)
    B, n, d = Q.shape
    K, m, _ = D.shape

    results = {}

    # Method 1: Loop over query tokens (best for large K from our experiments)
    def loop_query_tokens():
        scores = torch.zeros(B, K, device=device, dtype=torch.float32)
        D_flat = D.reshape(K * m, d).float().T  # [d, K*m]
        for i in range(n):
            q_i = Q[:, i, :].float()
            sim_flat = torch.matmul(q_i, D_flat)
            sim = sim_flat.reshape(B, K, m)
            scores += sim.max(dim=-1).values
        return scores

    # Method 2: Batched matmul
    def batched_matmul(doc_batch_size=2000):
        scores = torch.zeros(B, K, device=device, dtype=torch.float32)
        for start in range(0, K, doc_batch_size):
            end = min(start + doc_batch_size, K)
            D_chunk = D[start:end].float()
            sim = torch.matmul(Q.float().unsqueeze(1),
                             D_chunk.unsqueeze(0).transpose(-1, -2))
            scores[:, start:end] = sim.max(dim=-1).values.sum(dim=-1)
        return scores

    # Method 3: TileMaxSim V2-MQ (import from our kernel)
    try:
        from flash_maxsim_v2 import flash_maxsim_v2_mq
        has_v2 = True
    except:
        has_v2 = False

    methods = {
        'loop_query_tokens': loop_query_tokens,
        'batched_matmul': batched_matmul,
    }
    if has_v2:
        methods['flash_maxsim_v2_mq'] = lambda: flash_maxsim_v2_mq(Q.half(), D.half())

    for name, fn in methods.items():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Warmup
        for _ in range(n_warmup):
            try:
                _ = fn()
                torch.cuda.synchronize()
            except Exception as e:
                print(f"  {name} warmup failed: {e}")
                break
        else:
            # Benchmark
            times = []
            for _ in range(n_runs):
                torch.cuda.synchronize()
                start = time.perf_counter()
                _ = fn()
                torch.cuda.synchronize()
                times.append(time.perf_counter() - start)

            results[name] = {
                'mean_ms': sum(times) / len(times) * 1000,
                'min_ms': min(times) * 1000,
                'std_ms': np.std(times) * 1000,
                'docs_per_sec': K * B / (sum(times) / len(times)),
                'total_docs': K,
                'batch_size': B,
            }
            print(f"  {name}: {results[name]['mean_ms']:.2f}ms, "
                  f"{results[name]['docs_per_sec']:.0f} docs/s")

    return results


def main():
    print("=" * 80)
    print("TileMaxSim Full-Scale MS MARCO Benchmark")
    print("=" * 80)

    device = 'cuda:0'
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / (1024**3):.1f} GB")

    # Step 1: Load MS MARCO passages
    print("\n--- Step 1: Loading MS MARCO passages ---")
    from datasets import load_dataset

    # Load passages - use the passage collection
    print("Loading passages...")
    try:
        # Try loading passage ranking dataset
        ds = load_dataset('microsoft/ms_marco', 'v2.1', split='train', streaming=True)
        passages = []
        queries = []
        seen_passages = set()

        for i, ex in enumerate(ds):
            # Extract unique passages
            if 'passages' in ex:
                for p in ex['passages']['passage_text']:
                    if len(p) > 20 and p not in seen_passages:
                        passages.append(p)
                        seen_passages.add(p)
            if 'query' in ex:
                queries.append(ex['query'])

            # Collect enough data for meaningful benchmark
            if len(passages) >= 500000:  # 500K passages
                break
            if i % 10000 == 0 and i > 0:
                print(f"  Processed {i} examples, {len(passages)} unique passages, {len(queries)} queries")

    except Exception as e:
        print(f"Error loading MS MARCO: {e}")
        print("Falling back to synthetic data at realistic scale...")
        passages = [f"This is passage number {i} about topic {i % 1000}" for i in range(500000)]
        queries = [f"What is topic {i % 1000}?" for i in range(1000)]

    print(f"Collected {len(passages)} passages and {len(queries)} queries")

    # Subsample queries
    queries = queries[:1000]

    # Step 2: Encode with BERT (ColBERT-style)
    print("\n--- Step 2: Encoding with BERT (ColBERT-style) ---")
    from transformers import AutoTokenizer, AutoModel

    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).half()

    dim = 128
    max_doc_tokens = 128
    max_query_tokens = 32

    # Encode passages in batches
    print("Encoding passages...")
    n_passages_to_encode = min(len(passages), 100000)  # 100K for GPU memory
    passage_subset = passages[:n_passages_to_encode]

    doc_embs, doc_lens = encode_texts_colbert_style(
        passage_subset, model, tokenizer, device,
        max_length=max_doc_tokens, dim=dim, batch_size=512
    )
    print(f"Encoded {len(doc_embs)} passages, avg length: {np.mean(doc_lens):.1f} tokens")

    # Encode queries
    print("Encoding queries...")
    query_subset = queries[:100]  # 100 queries for benchmark
    query_embs, query_lens = encode_texts_colbert_style(
        query_subset, model, tokenizer, device,
        max_length=max_query_tokens, dim=dim, batch_size=128
    )
    print(f"Encoded {len(query_embs)} queries, avg length: {np.mean(query_lens):.1f} tokens")

    # Pad to fixed tensors
    doc_padded = pad_embeddings(doc_embs, doc_lens, max_doc_tokens, dim)
    query_padded = pad_embeddings(query_embs, query_lens, max_query_tokens, dim)

    # Free model memory
    del model
    torch.cuda.empty_cache()

    # Step 3: Run benchmarks at various scales
    print("\n--- Step 3: Running TileMaxSim benchmarks ---")

    all_results = {}

    # Test at different scales
    scales = [1000, 5000, 10000, 50000, 100000]

    for n_docs in scales:
        if n_docs > len(doc_embs):
            continue

        print(f"\n  Scale: {n_docs} documents")
        D = doc_padded[:n_docs].half()
        torch.cuda.empty_cache()

        # Single query benchmark
        Q_single = query_padded[:1].half()
        print(f"  -- Single query (1 x {n_docs} docs)")
        try:
            res_single = benchmark_maxsim_methods(Q_single, D, device, n_warmup=3, n_runs=10)
            all_results[f'single_q_{n_docs}_docs'] = res_single
        except Exception as e:
            print(f"    Error: {e}")
            torch.cuda.empty_cache()

        # Batch query benchmark
        for batch_size in [8, 32]:
            if batch_size > len(query_embs):
                continue
            Q_batch = query_padded[:batch_size].half()
            print(f"  -- Batch {batch_size} queries x {n_docs} docs")
            try:
                res_batch = benchmark_maxsim_methods(Q_batch, D, device, n_warmup=3, n_runs=5)
                all_results[f'batch{batch_size}_q_{n_docs}_docs'] = res_batch
            except Exception as e:
                print(f"    Error: {e}")
                torch.cuda.empty_cache()

    # Step 4: Multi-GPU test
    if torch.cuda.device_count() >= 2:
        print("\n--- Step 4: Multi-GPU scaling ---")
        n_docs = min(50000, len(doc_embs))
        D = doc_padded[:n_docs].half()
        Q = query_padded[:32].half()

        # Single GPU
        torch.cuda.synchronize()
        Q_gpu0 = Q.to('cuda:0')
        D_gpu0 = D.to('cuda:0')

        times_1gpu = []
        D_flat = D_gpu0.reshape(n_docs * max_doc_tokens, dim).float().T
        for _ in range(3):  # warmup
            scores = torch.zeros(32, n_docs, device='cuda:0', dtype=torch.float32)
            for i in range(max_query_tokens):
                q_i = Q_gpu0[:, i, :].float()
                sim = torch.matmul(q_i, D_flat).reshape(32, n_docs, max_doc_tokens)
                scores += sim.max(dim=-1).values
            torch.cuda.synchronize()

        for _ in range(5):
            torch.cuda.synchronize()
            start = time.perf_counter()
            scores = torch.zeros(32, n_docs, device='cuda:0', dtype=torch.float32)
            for i in range(max_query_tokens):
                q_i = Q_gpu0[:, i, :].float()
                sim = torch.matmul(q_i, D_flat).reshape(32, n_docs, max_doc_tokens)
                scores += sim.max(dim=-1).values
            torch.cuda.synchronize()
            times_1gpu.append(time.perf_counter() - start)

        # Two GPUs (split documents)
        half = n_docs // 2
        D_gpu0_half = D[:half].to('cuda:0')
        D_gpu1_half = D[half:n_docs].to('cuda:1')
        Q_gpu0 = Q.to('cuda:0')
        Q_gpu1 = Q.to('cuda:1')

        D_flat0 = D_gpu0_half.reshape(half * max_doc_tokens, dim).float().T
        D_flat1 = D_gpu1_half.reshape((n_docs - half) * max_doc_tokens, dim).float().T

        for _ in range(3):  # warmup
            scores0 = torch.zeros(32, half, device='cuda:0', dtype=torch.float32)
            scores1 = torch.zeros(32, n_docs - half, device='cuda:1', dtype=torch.float32)
            for i in range(max_query_tokens):
                sim0 = torch.matmul(Q_gpu0[:, i, :].float(), D_flat0).reshape(32, half, max_doc_tokens)
                sim1 = torch.matmul(Q_gpu1[:, i, :].float(), D_flat1).reshape(32, n_docs - half, max_doc_tokens)
                scores0 += sim0.max(dim=-1).values
                scores1 += sim1.max(dim=-1).values
            torch.cuda.synchronize()

        times_2gpu = []
        for _ in range(5):
            torch.cuda.synchronize()
            start = time.perf_counter()
            scores0 = torch.zeros(32, half, device='cuda:0', dtype=torch.float32)
            scores1 = torch.zeros(32, n_docs - half, device='cuda:1', dtype=torch.float32)
            for i in range(max_query_tokens):
                sim0 = torch.matmul(Q_gpu0[:, i, :].float(), D_flat0).reshape(32, half, max_doc_tokens)
                sim1 = torch.matmul(Q_gpu1[:, i, :].float(), D_flat1).reshape(32, n_docs - half, max_doc_tokens)
                scores0 += sim0.max(dim=-1).values
                scores1 += sim1.max(dim=-1).values
            torch.cuda.synchronize()
            times_2gpu.append(time.perf_counter() - start)

        t1 = np.mean(times_1gpu) * 1000
        t2 = np.mean(times_2gpu) * 1000
        print(f"  1 GPU: {t1:.1f}ms")
        print(f"  2 GPUs: {t2:.1f}ms")
        print(f"  Scaling: {t1/t2:.2f}x")

        all_results['multi_gpu'] = {
            '1_gpu_ms': t1,
            '2_gpu_ms': t2,
            'scaling_factor': t1 / t2,
            'n_docs': n_docs,
            'n_queries': 32,
        }

    # Step 5: Memory analysis
    print("\n--- Step 5: Memory analysis ---")
    mem_analysis = {}
    for n_docs in [10000, 50000, 100000, 500000, 1000000, 8800000]:
        # ColBERTv2 compressed: ~2 bits per dim per token
        compressed_gb = n_docs * max_doc_tokens * dim * 2 / 8 / (1024**3)  # 2-bit residual
        full_fp16_gb = n_docs * max_doc_tokens * dim * 2 / (1024**3)  # FP16
        mem_analysis[f'{n_docs}_docs'] = {
            'compressed_gb': round(compressed_gb, 2),
            'fp16_gb': round(full_fp16_gb, 2),
            'fits_1_h100': full_fp16_gb < 80,
            'fits_8_h100': full_fp16_gb < 640,
        }
        print(f"  {n_docs:>10} docs: compressed={compressed_gb:.2f}GB, FP16={full_fp16_gb:.2f}GB, "
              f"fits_1xH100={full_fp16_gb < 80}, fits_8xH100={full_fp16_gb < 640}")

    all_results['memory_analysis'] = mem_analysis

    # Step 6: Roofline comparison
    print("\n--- Step 6: Roofline analysis with real data ---")
    h100_bw = 3.35e12  # bytes/s
    h100_flops = 989e12  # FP16 TFLOPS

    roofline = {}
    for n_docs in scales:
        if n_docs > len(doc_embs):
            continue
        flops = max_query_tokens * n_docs * max_doc_tokens * dim * 2
        bytes_read = (max_query_tokens * dim + n_docs * max_doc_tokens * dim) * 2
        ai = flops / bytes_read
        compute_time = flops / h100_flops * 1000
        memory_time = bytes_read / h100_bw * 1000

        key = f'single_q_{n_docs}_docs'
        measured = all_results.get(key, {}).get('loop_query_tokens', {}).get('mean_ms', None)

        roofline[f'{n_docs}_docs'] = {
            'arithmetic_intensity': round(ai, 1),
            'compute_bound_ms': round(compute_time, 4),
            'memory_bound_ms': round(memory_time, 4),
            'measured_ms': round(measured, 2) if measured else None,
            'efficiency': round(memory_time / measured * 100, 1) if measured else None,
        }
        eff_str = f", efficiency={memory_time/measured*100:.1f}%" if measured else ""
        print(f"  {n_docs} docs: AI={ai:.1f}, mem_bound={memory_time:.3f}ms{eff_str}")

    all_results['roofline'] = roofline

    # Save results
    results_path = RESULTS_DIR / "full_scale_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
