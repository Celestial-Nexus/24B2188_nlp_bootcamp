"""Task 2: Dense Retrieval (general + scientific).

Produces:
  predictions/val_general.trec      (162 val, top-100)
  predictions/t2_general.trec       (300 eval, top-100)
  predictions/t2_scientific.trec    (300 eval, top-100)

Caches dense train-claim top-100 for Task 4 hard-neg mining.
Saves timing/memory stats to cache/dense_stats.json.

Run from repo root:
  python code/task2_dense.py
"""

import json
import os
import pickle
import sys
import time

# Disable MPS (Apple Metal) — causes OOM on M3 when encoding 5183 docs.
# Force pure CPU execution for reproducibility and stability.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import faiss
import numpy as np
import psutil
import torch

# Override MPS availability so sentence-transformers picks CPU
if hasattr(torch.backends, "mps"):
    torch.backends.mps.enabled = False  # type: ignore[attr-defined]

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from code.utils import (
    build_qrels,
    doc_text,
    evaluate_trec,
    load_data,
    write_trec,
)

PRED_DIR  = "predictions"
CACHE_DIR = "cache"
os.makedirs(PRED_DIR,  exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

MODELS = {
    "general":    "BAAI/bge-base-en-v1.5",
    "scientific": "pritamdeka/S-PubMedBert-MS-MARCO",
}


def build_index(model, doc_texts, batch_size=8):
    """Encode docs and build a FAISS IndexFlatIP. Returns (index, embeddings).
    
    Note: batch_size=8 and show_progress_bar=False to avoid macOS Python 3.13
    multiprocessing/loky crash when encoding large corpora in child processes.
    """
    n = len(doc_texts)
    print(f"Encoding {n} docs (batch_size={batch_size}) …", flush=True)
    embeds = model.encode(
        doc_texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,   # no tqdm workers — avoids loky semaphore crash
        convert_to_numpy=True,
    )
    print(f"Done encoding → shape {embeds.shape}", flush=True)
    index = faiss.IndexFlatIP(embeds.shape[1])
    index.add(embeds)
    return index, embeds


def dense_retrieve(model, index, all_doc_ids, queries, k=100, batch_size=8):
    """Return {claim_id: [(doc_id, score), ...]} for list of (id, text) tuples."""
    q_texts = [q for _, q in queries]
    q_embeds = model.encode(
        q_texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    scores_all, idx_all = index.search(q_embeds, k)
    run = {}
    for i, (cid, _) in enumerate(queries):
        run[cid] = [
            (all_doc_ids[idx_all[i, j]], float(scores_all[i, j]))
            for j in range(k)
        ]
    return run


def measure_latency(model, index, all_doc_ids, queries, warmup=3):
    """Average per-query latency (ms) after a warm-up pass."""
    warm_texts = [q for _, q in queries[:warmup]]
    _ = model.encode(warm_texts, normalize_embeddings=True, convert_to_numpy=True)

    times = []
    for _, q in queries:
        t0 = time.perf_counter()
        qe = model.encode([q], normalize_embeddings=True, convert_to_numpy=True)
        index.search(qe, 100)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times))


def main():
    from sentence_transformers import SentenceTransformer

    corpus_list, corpus, all_doc_ids, train_claims, val_claims, eval_claims = load_data()
    doc_texts = [doc_text(d) for d in corpus_list]

    val_queries   = [(c["id"], c["claim"]) for c in val_claims]
    eval_queries  = [(c["id"], c["claim"]) for c in eval_claims]
    train_queries = [(c["id"], c["claim"]) for c in train_claims if c.get("evidence")]
    qrels_val     = build_qrels(val_claims)

    proc = psutil.Process(os.getpid())
    stats = {}

    for role, model_name in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Model ({role}): {model_name}")
        print(f"{'='*60}")

        model = SentenceTransformer(model_name, device="cpu")
        model.max_seq_length = 512

        mem_before = proc.memory_info().rss / 1e6

        t0         = time.perf_counter()
        index, _   = build_index(model, doc_texts)
        index_time = time.perf_counter() - t0

        mem_after  = proc.memory_info().rss / 1e6
        memory_mb  = mem_after - mem_before  # index overhead (model weights excluded)
        if torch.cuda.is_available():
            memory_mb += torch.cuda.max_memory_allocated() / 1e6

        print(f"Index built in {index_time:.1f}s  |  memory delta={memory_mb:.0f} MB")

        # Latency over val queries
        lat_ms = measure_latency(model, index, all_doc_ids, val_queries)
        print(f"Query latency (avg over {len(val_queries)} val queries): {lat_ms:.1f} ms")

        stats[role] = {
            "model": model_name,
            "index_time_sec": round(index_time, 2),
            "query_latency_ms": round(lat_ms, 2),
            "memory_mb": round(memory_mb, 1),
        }

        # -- Retrieval --
        # Val
        val_run = dense_retrieve(model, index, all_doc_ids, val_queries)
        # Eval
        eval_run = dense_retrieve(model, index, all_doc_ids, eval_queries)
        # Train (for Task 4 hard-neg mining)
        train_run = dense_retrieve(model, index, all_doc_ids, train_queries)

        if role == "general":
            write_trec(val_run,  os.path.join(PRED_DIR, "val_general.trec"),  "dense_general")
            write_trec(eval_run, os.path.join(PRED_DIR, "t2_general.trec"),   "dense_general")
            with open(os.path.join(CACHE_DIR, "dense_general_val_run.pkl"), "wb") as f:
                pickle.dump(val_run, f)
            with open(os.path.join(CACHE_DIR, "dense_general_train_run.pkl"), "wb") as f:
                pickle.dump(train_run, f)

            m = evaluate_trec(os.path.join(PRED_DIR, "val_general.trec"), qrels_val)
            stats[role]["val_ndcg10"] = round(m["ndcg_cut_10"], 4)
            print(f"Val metrics ({role}): {m}")

        else:  # scientific
            write_trec(eval_run, os.path.join(PRED_DIR, "t2_scientific.trec"), "dense_scientific")
            with open(os.path.join(CACHE_DIR, "dense_scientific_val_run.pkl"), "wb") as f:
                pickle.dump(val_run, f)
            with open(os.path.join(CACHE_DIR, "dense_scientific_train_run.pkl"), "wb") as f:
                pickle.dump(train_run, f)

            # Eval on val to pick best scientific model (not a required file)
            write_trec(val_run, os.path.join(CACHE_DIR, "val_scientific_tmp.trec"), "dense_sci")
            m = evaluate_trec(os.path.join(CACHE_DIR, "val_scientific_tmp.trec"), qrels_val)
            stats[role]["val_ndcg10"] = round(m["ndcg_cut_10"], 4)
            print(f"Val metrics ({role}): {m}")

        # Free GPU memory between models
        del model, index
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(os.path.join(CACHE_DIR, "dense_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print("\nDone. Written: val_general.trec, t2_general.trec, t2_scientific.trec")
    return stats


if __name__ == "__main__":
    main()
