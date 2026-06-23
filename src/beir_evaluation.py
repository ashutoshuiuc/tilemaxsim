"""
BEIR cross-domain evaluation with TileMaxSim.
Tests on SciFact, NFCorpus, and TREC-COVID to show speedup is consistent across domains.
Uses transformers directly (avoids colbert-ai compatibility issues).
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import os
import sys
import time
from pathlib import Path
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, str(Path(__file__).parent))

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "experiment_results"
DATA_ROOT = Path(os.environ.get("DATA_ROOT", str(PROJECT_ROOT.parent / "datasets")))
RESULTS_DIR.mkdir(exist_ok=True)


class ColBERTEncoder:
    """Minimal ColBERTv2 encoder using transformers directly."""

    def __init__(self, model_name="colbert-ir/colbertv2.0", dim=128, device="cuda:0"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.dim = dim
        self.device = device
        self.linear = torch.nn.Linear(self.model.config.hidden_size, dim, bias=False).to(device)
        try:
            from huggingface_hub import hf_hub_download
            import safetensors.torch
            ckpt_path = hf_hub_download(repo_id=model_name, filename="model.safetensors")
            state_dict = safetensors.torch.load_file(ckpt_path)
            for key in state_dict:
                if "linear" in key.lower() and state_dict[key].shape == (dim, self.model.config.hidden_size):
                    self.linear.weight.data = state_dict[key].to(device)
                    break
        except Exception:
            pass

    @torch.no_grad()
    def encode_queries(self, texts, max_length=32, batch_size=64):
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            encoded = self.tokenizer(
                ["[unused0] " + t for t in batch],
                padding=True, truncation=True, max_length=max_length,
                return_tensors='pt'
            ).to(self.device)
            outputs = self.model(**encoded)
            projected = self.linear(outputs.last_hidden_state)
            projected = F.normalize(projected, dim=-1)
            mask = encoded['attention_mask']
            for b in range(projected.shape[0]):
                length = mask[b].sum().item()
                all_embs.append(projected[b, :length].half().cpu())
        return all_embs

    @torch.no_grad()
    def encode_passages(self, texts, max_length=180, batch_size=64):
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            encoded = self.tokenizer(
                ["[unused1] " + t for t in batch],
                padding=True, truncation=True, max_length=max_length,
                return_tensors='pt'
            ).to(self.device)
            outputs = self.model(**encoded)
            projected = self.linear(outputs.last_hidden_state)
            projected = F.normalize(projected, dim=-1)
            mask = encoded['attention_mask']
            for b in range(projected.shape[0]):
                length = mask[b].sum().item()
                all_embs.append(projected[b, :length].half().cpu())
            if (i // batch_size) % 20 == 0 and i > 0:
                print(f"      Encoded {i+len(batch)}/{len(texts)}")
        return all_embs


def pytorch_maxsim_ref(Q, D):
    """Reference MaxSim: Q (Nq, d), D (B, Nd, d) -> (B,)"""
    S = torch.einsum('qd,bnd->bqn', Q.float(), D.float())
    return S.max(dim=-1).values.sum(dim=-1)


def flash_maxsim_score(Q, D):
    """TileMaxSim scoring."""
    from flash_maxsim_v2 import flash_maxsim_v2_multiquery
    from flash_maxsim_kernel import flash_maxsim_batch
    B, Nd, d = D.shape
    Nq = Q.shape[0]
    if d <= 128 and Nq >= 16:
        try:
            return flash_maxsim_v2_multiquery(Q.half(), D.half())
        except Exception:
            pass
    return flash_maxsim_batch(Q.half(), D.half())


def compute_metrics(rankings, qrels, k_values=[10, 100, 1000]):
    """Compute nDCG@k, Recall@k, MRR@10."""
    results = {}
    n_queries = 0
    mrr_sum = 0.0
    ndcg_sums = {k: 0.0 for k in k_values}
    recall_sums = {k: 0.0 for k in k_values}

    for qid, ranked_pids in rankings.items():
        if qid not in qrels:
            continue
        relevant = {pid: rel for pid, rel in qrels[qid].items() if rel > 0}
        if not relevant:
            continue
        n_queries += 1

        for rank, pid in enumerate(ranked_pids[:10], 1):
            if pid in relevant:
                mrr_sum += 1.0 / rank
                break

        for k in k_values:
            dcg = 0.0
            for rank, pid in enumerate(ranked_pids[:k], 1):
                if pid in relevant:
                    dcg += relevant[pid] / np.log2(rank + 1)
            ideal_rels = sorted(relevant.values(), reverse=True)[:k]
            idcg = sum(r / np.log2(rank + 1) for rank, r in enumerate(ideal_rels, 1))
            ndcg_sums[k] += dcg / idcg if idcg > 0 else 0.0
            retrieved = set(ranked_pids[:k])
            recall_sums[k] += len(set(relevant.keys()) & retrieved) / len(relevant)

    for k in k_values:
        results[f"nDCG@{k}"] = ndcg_sums[k] / n_queries if n_queries else 0
        results[f"Recall@{k}"] = recall_sums[k] / n_queries if n_queries else 0
    results["MRR@10"] = mrr_sum / n_queries if n_queries else 0
    results["n_queries"] = n_queries
    return results


def evaluate_beir_dataset(dataset_name, encoder, device, max_docs=50000):
    """Evaluate on a BEIR dataset."""
    from beir import util as beir_util
    from beir.datasets.data_loader import GenericDataLoader

    print(f"\n{'='*60}")
    print(f"BEIR Dataset: {dataset_name}")
    print(f"{'='*60}")

    data_path = os.path.join(str(DATA_ROOT / "beir"), dataset_name)
    if not os.path.exists(data_path):
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
        data_path = beir_util.download_and_unzip(url, os.path.dirname(data_path))

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    print(f"  Corpus size: {len(corpus)}")
    print(f"  Queries: {len(queries)}")
    print(f"  Qrels: {len(qrels)}")

    corpus_ids = list(corpus.keys())[:max_docs]
    corpus_texts = [
        (corpus[cid].get("title", "") + " " + corpus[cid].get("text", "")).strip()
        for cid in corpus_ids
    ]

    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]

    print(f"  Using {len(corpus_ids)} docs, {len(query_ids)} queries")

    # Encode corpus
    print("  Encoding corpus...")
    doc_embs = encoder.encode_passages(corpus_texts, max_length=180)

    max_nd = min(max(e.shape[0] for e in doc_embs), 180)
    d_dim = doc_embs[0].shape[1]
    D_padded = torch.zeros(len(doc_embs), max_nd, d_dim, dtype=torch.float16)
    for i, emb in enumerate(doc_embs):
        nd = min(emb.shape[0], max_nd)
        D_padded[i, :nd] = emb[:nd].half()
    D_gpu = D_padded.to(device)
    print(f"  Corpus encoded: {D_gpu.shape}")

    # Score each query
    rankings_ref = {}
    rankings_flash = {}
    total_ref_time = 0
    total_flash_time = 0

    for qi, qid in enumerate(query_ids):
        Q_list = encoder.encode_queries([queries[qid]])
        Q = Q_list[0].to(device)

        chunk_size = 4096
        ref_scores = []
        flash_scores = []

        for c_start in range(0, len(doc_embs), chunk_size):
            c_end = min(c_start + chunk_size, len(doc_embs))
            D_chunk = D_gpu[c_start:c_end]

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            rs = pytorch_maxsim_ref(Q, D_chunk)
            torch.cuda.synchronize()
            total_ref_time += time.perf_counter() - t0

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fs = flash_maxsim_score(Q, D_chunk)
            torch.cuda.synchronize()
            total_flash_time += time.perf_counter() - t0

            ref_scores.append(rs.cpu())
            flash_scores.append(fs.cpu())

        ref_scores = torch.cat(ref_scores)
        flash_scores = torch.cat(flash_scores)

        ref_order = ref_scores.argsort(descending=True).tolist()
        flash_order = flash_scores.argsort(descending=True).tolist()

        rankings_ref[qid] = [corpus_ids[i] for i in ref_order]
        rankings_flash[qid] = [corpus_ids[i] for i in flash_order]

        if (qi + 1) % 50 == 0:
            print(f"    Scored {qi+1}/{len(query_ids)} queries")

    qrels_converted = {}
    for qid in query_ids:
        if qid in qrels:
            qrels_converted[qid] = {pid: int(rel) for pid, rel in qrels[qid].items()}

    metrics_ref = compute_metrics(rankings_ref, qrels_converted)
    metrics_flash = compute_metrics(rankings_flash, qrels_converted)

    print(f"\n  {'Metric':<15} {'Reference':>10} {'TileMaxSim':>12}")
    print(f"  {'-'*40}")
    for metric in ["nDCG@10", "MRR@10", "Recall@100", "Recall@1000"]:
        if metric in metrics_ref:
            print(f"  {metric:<15} {metrics_ref[metric]:>10.4f} {metrics_flash[metric]:>12.4f}")

    speedup = total_ref_time / total_flash_time if total_flash_time > 0 else 0
    print(f"\n  Ref time: {total_ref_time:.2f}s, Flash time: {total_flash_time:.2f}s, Speedup: {speedup:.2f}x")

    del D_gpu
    torch.cuda.empty_cache()

    return {
        "dataset": dataset_name,
        "corpus_size": len(corpus_ids),
        "n_queries": len(query_ids),
        "ref_metrics": metrics_ref,
        "flash_metrics": metrics_flash,
        "timing": {
            "ref_total_s": total_ref_time,
            "flash_total_s": total_flash_time,
            "speedup": speedup,
        },
    }


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    device = "cuda:0"

    print("=" * 80)
    print("BEIR Cross-Domain Evaluation with TileMaxSim")
    print("=" * 80)

    encoder = ColBERTEncoder(device=device)

    datasets = ["scifact", "nfcorpus", "trec-covid"]
    all_results = {}

    for ds_name in datasets:
        try:
            result = evaluate_beir_dataset(ds_name, encoder, device, max_docs=50000)
            all_results[ds_name] = result
        except Exception as e:
            print(f"  ERROR on {ds_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[ds_name] = {"error": str(e)}

    out_path = RESULTS_DIR / "beir_evaluation.json"
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Summary
    print(f"\n{'='*80}")
    print("BEIR Cross-Domain Summary")
    print(f"{'='*80}")
    print(f"{'Dataset':<15} {'nDCG@10 (ref)':>14} {'nDCG@10 (flash)':>16} {'Speedup':>10}")
    for ds_name, result in all_results.items():
        if "error" in result:
            print(f"{ds_name:<15} {'ERROR':>14}")
        else:
            ref = result["ref_metrics"]["nDCG@10"]
            flash = result["flash_metrics"]["nDCG@10"]
            spd = result["timing"]["speedup"]
            print(f"{ds_name:<15} {ref:>14.4f} {flash:>16.4f} {spd:>9.2f}x")


if __name__ == "__main__":
    main()
