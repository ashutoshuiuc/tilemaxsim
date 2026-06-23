"""
Benchmark WARP scoring latency on BEIR datasets.

Measures per-stage latency using WARP's built-in ExecutionTracker,
isolating scoring/decompression time from query encoding and candidate generation.

Usage: python baselines/bench_warp.py --dataset beir.nfcorpus --nbits 2 --k 10
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
import json
import time
import argparse

import psutil
import torch

torch.set_num_threads(1)
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Path to a local checkout of the XTR/WARP repo. Override with the WARP_DIR env var.
WARP_DIR = os.environ.get("WARP_DIR", os.path.expanduser("~/xtr-warp"))
sys.path.insert(0, WARP_DIR)

from warp.engine.searcher import WARPSearcher
from warp.data.queries import WARPQueries
from warp.utils.tracker import ExecutionTracker
from utility.runner_utils import make_run_config


def run_benchmark(dataset, nbits, k, num_runs, num_threads, datasplit="dev"):
    config = {
        "collection": dataset.split(".")[0],
        "dataset": dataset.split(".")[1] if "." in dataset else dataset,
        "datasplit": datasplit,
        "nbits": nbits,
        "nprobe": 32,
        "runtime": None,
        "num_threads": num_threads,
        "document_top_k": k,
        "bound": 196,
    }

    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    torch.set_num_threads(num_threads)

    run_config = make_run_config(config)
    searcher = WARPSearcher(run_config)
    queries = WARPQueries(run_config)

    steps = ["Query Encoding", "Candidate Generation", "top-k Precompute",
             "Decompression", "Build Matrix"]

    all_trackers = []
    for run_idx in range(num_runs):
        print(f"\n--- Run {run_idx + 1}/{num_runs} ---")
        tracker = ExecutionTracker(name="WARP Benchmark", steps=steps)
        rankings = searcher.search_all(
            queries, k=k, batched=False, tracker=tracker, show_progress=True
        )
        metrics = rankings.evaluate(queries.qrels, k=k)
        print(f"  Metrics: {metrics}")

        tracker_dict = tracker.as_dict()
        all_trackers.append(tracker_dict)
        print(f"  Tracker: {json.dumps(tracker_dict, indent=2)}")

    return {
        "config": config,
        "trackers": all_trackers,
        "metrics": metrics,
        "num_runs": num_runs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="beir.nfcorpus")
    parser.add_argument("--nbits", type=int, default=2)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--datasplit", default="dev")
    args = parser.parse_args()

    print(f"WARP Benchmark: {args.dataset}, nbits={args.nbits}, k={args.k}")
    print(f"Threads: {args.num_threads}, Runs: {args.num_runs}")
    print(f"CPU: {psutil.cpu_count(logical=False)} physical cores")

    results = run_benchmark(
        args.dataset, args.nbits, args.k, args.num_runs,
        args.num_threads, args.datasplit,
    )

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "final_results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"warp_benchmark_{args.dataset.replace('.', '_')}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
