"""Task 4: Domain Fine-Tuning with Random and Hard Negatives.

Produces:
  predictions/t4_random.trec  (300 eval, top-100)
  predictions/t4_hard.trec    (300 eval, top-100)

Checkpoints: checkpoints/random_neg/  and  checkpoints/hard_neg/
Val nDCG@10 per epoch is logged to checkpoints/<name>/eval_results.csv
(written by InformationRetrievalEvaluator — use for Q3 overfitting analysis).

Run from repo root:
  python code/task4_finetune.py
"""

import json
import os
import pickle
import random
import sys
import time

# Force CPU — Apple M3 Metal OOM on large batch matrix ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # fallback to CPU for unsupported ops

import faiss
import numpy as np
import psutil
import torch

# Disable MPS (Apple Metal) — causes OOM on M3 during large batch matrix ops.
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
SEED      = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

os.makedirs(PRED_DIR,  exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Training dataset builders
# ---------------------------------------------------------------------------

def build_random_neg_dataset(train_claims, corpus, all_doc_ids, n_neg=7):
    """(anchor, positive) pairs only — MNRL uses in-batch negatives."""
    from datasets import Dataset

    pairs = []
    for claim in train_claims:
        if not claim.get("evidence"):
            continue
        gold_ids = {e["doc_id"] for e in claim["evidence"]}
        neg_pool = [d for d in all_doc_ids if d not in gold_ids]
        for ev in claim["evidence"]:
            pos_text = doc_text(corpus[ev["doc_id"]])
            neg_sample = random.sample(neg_pool, n_neg)
            # One (anchor, positive, negative_0..n_neg-1) row per positive pair
            row = {
                "anchor":   claim["claim"],
                "positive": pos_text,
            }
            for i, did in enumerate(neg_sample):
                row[f"negative_{i}"] = doc_text(corpus[did])
            pairs.append(row)

    return Dataset.from_list(pairs)


def build_hard_neg_dataset(
    train_claims, corpus, all_doc_ids, bm25_train_run, dense_train_run,
    rank_lo=5, rank_hi=50, n_neg=7
):
    """Hard negatives mined from BM25 + dense top-100 on train claims."""
    from datasets import Dataset

    pairs = []
    for claim in train_claims:
        if not claim.get("evidence"):
            continue
        cid      = claim["id"]
        gold_ids = {e["doc_id"] for e in claim["evidence"]}

        bm25_top  = [did for did, _ in bm25_train_run.get(cid, [])]
        dense_top = [did for did, _ in dense_train_run.get(cid, [])]

        # Union of rank window, excluding gold docs
        cands = []
        for did in bm25_top[rank_lo - 1 : rank_hi] + dense_top[rank_lo - 1 : rank_hi]:
            if did not in gold_ids and did not in cands:
                cands.append(did)

        if len(cands) < n_neg:
            # Fall back to random negatives if not enough hard negs
            extra = [d for d in all_doc_ids if d not in gold_ids and d not in cands]
            cands += random.sample(extra, n_neg - len(cands))

        for ev in claim["evidence"]:
            pos_text   = doc_text(corpus[ev["doc_id"]])
            hard_sample = random.sample(cands, min(n_neg, len(cands)))
            row = {
                "anchor":   claim["claim"],
                "positive": pos_text,
            }
            for i, did in enumerate(hard_sample):
                row[f"negative_{i}"] = doc_text(corpus[did])
            pairs.append(row)

    return Dataset.from_list(pairs)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_biencoder(
    base_model_name, train_dataset, evaluator,
    output_dir, epochs=5, batch_size=16, lr=2e-5, warmup_steps=50
):
    """Fine-tune with SentenceTransformerTrainer. Returns trained model."""
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.losses import MultipleNegativesRankingLoss

    model = SentenceTransformer(base_model_name, device="cpu")
    model.max_seq_length = 256  # prevent OOM; also keeps memory sane on CPU

    loss = MultipleNegativesRankingLoss(model)

    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        warmup_steps=warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_val_cosine_ndcg@10",
        greater_is_better=True,
        seed=SEED,
        fp16=False,                      # not supported on Apple M3 CPU
        dataloader_drop_last=True,   # avoids tiny last batch issues with MNRL
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
        evaluator=evaluator,
    )
    trainer.train()

    # load_best_model_at_end=True means trainer.model is the best checkpoint
    return trainer.model


# ---------------------------------------------------------------------------
# Inference with a fine-tuned bi-encoder
# ---------------------------------------------------------------------------

def encode_and_retrieve(model, corpus_list, all_doc_ids, queries, k=100):
    """Encode corpus + queries, do FAISS search. Returns {cid: [(did, score)]}."""
    doc_texts = [doc_text(d) for d in corpus_list]

    print("  Encoding corpus …")
    doc_embeds = model.encode(
        doc_texts, batch_size=8, normalize_embeddings=True,
        show_progress_bar=False, convert_to_numpy=True
    )
    index = faiss.IndexFlatIP(doc_embeds.shape[1])
    index.add(doc_embeds)

    print("  Encoding queries …")
    q_texts  = [q for _, q in queries]
    q_embeds = model.encode(
        q_texts, batch_size=8, normalize_embeddings=True,
        show_progress_bar=False, convert_to_numpy=True
    )
    scores_all, idx_all = index.search(q_embeds, k)

    run = {}
    for i, (cid, _) in enumerate(queries):
        run[cid] = [
            (all_doc_ids[idx_all[i, j]], float(scores_all[i, j]))
            for j in range(k)
        ]
    return run


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from sentence_transformers.evaluation import InformationRetrievalEvaluator

    corpus_list, corpus, all_doc_ids, train_claims, val_claims, eval_claims = load_data()
    qrels_val    = build_qrels(val_claims)
    eval_queries = [(c["id"], c["claim"]) for c in eval_claims]
    val_queries  = [(c["id"], c["claim"]) for c in val_claims]

    # Load cached BM25 and dense train runs (produced by tasks 1 and 2)
    with open(os.path.join(CACHE_DIR, "bm25_train_run.pkl"), "rb") as f:
        bm25_train_run = pickle.load(f)
    with open(os.path.join(CACHE_DIR, "dense_general_train_run.pkl"), "rb") as f:
        dense_train_run = pickle.load(f)

    # InformationRetrievalEvaluator for val nDCG@10 after each epoch
    ir_queries  = {str(c["id"]): c["claim"]
                   for c in val_claims if c.get("evidence")}
    ir_corpus   = {str(d["doc_id"]): doc_text(d) for d in corpus_list}
    ir_relevant = {
        str(c["id"]): {str(e["doc_id"]) for e in c["evidence"]}
        for c in val_claims if c.get("evidence")
    }
    evaluator = InformationRetrievalEvaluator(
        queries=ir_queries,
        corpus=ir_corpus,
        relevant_docs=ir_relevant,
        ndcg_at_k=[10],
        name="val",
        write_csv=True,
        show_progress_bar=True,
    )

    BASE_MODEL = "BAAI/bge-base-en-v1.5"

    # ------------------------------------------------------------------
    # 4a — Random negatives
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("Task 4a: Fine-tuning with RANDOM negatives")
    print("="*60)

    rand_ds = build_random_neg_dataset(train_claims, corpus, all_doc_ids, n_neg=7)
    print(f"Random-neg training examples: {len(rand_ds)}")

    rand_model = train_biencoder(
        BASE_MODEL, rand_ds, evaluator,
        output_dir="checkpoints/random_neg",
        epochs=5, batch_size=16,
    )

    print("Generating t4_random.trec …")
    rand_eval_run = encode_and_retrieve(rand_model, corpus_list, all_doc_ids, eval_queries)
    write_trec(rand_eval_run, os.path.join(PRED_DIR, "t4_random.trec"), "ft_random")

    rand_val_run = encode_and_retrieve(rand_model, corpus_list, all_doc_ids, val_queries)
    write_trec(rand_val_run, os.path.join(CACHE_DIR, "val_ft_random.trec"), "ft_random")
    rand_m = evaluate_trec(os.path.join(CACHE_DIR, "val_ft_random.trec"), qrels_val)
    print(f"Random-neg val metrics: {rand_m}")

    # Save run for Task 5
    with open(os.path.join(CACHE_DIR, "ft_random_eval_run.pkl"), "wb") as f:
        pickle.dump(rand_eval_run, f)
    with open(os.path.join(CACHE_DIR, "ft_random_val_run.pkl"), "wb") as f:
        pickle.dump(rand_val_run, f)

    del rand_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 4b — Hard negatives
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("Task 4b: Fine-tuning with HARD negatives")
    print("="*60)

    hard_ds = build_hard_neg_dataset(
        train_claims, corpus, all_doc_ids,
        bm25_train_run, dense_train_run,
        rank_lo=5, rank_hi=50, n_neg=7
    )
    print(f"Hard-neg training examples: {len(hard_ds)}")

    hard_model = train_biencoder(
        BASE_MODEL, hard_ds, evaluator,
        output_dir="checkpoints/hard_neg",
        epochs=5, batch_size=16,
    )

    print("Generating t4_hard.trec …")
    hard_eval_run = encode_and_retrieve(hard_model, corpus_list, all_doc_ids, eval_queries)
    write_trec(hard_eval_run, os.path.join(PRED_DIR, "t4_hard.trec"), "ft_hard")

    hard_val_run = encode_and_retrieve(hard_model, corpus_list, all_doc_ids, val_queries)
    write_trec(hard_val_run, os.path.join(CACHE_DIR, "val_ft_hard.trec"), "ft_hard")
    hard_m = evaluate_trec(os.path.join(CACHE_DIR, "val_ft_hard.trec"), qrels_val)
    print(f"Hard-neg val metrics: {hard_m}")

    with open(os.path.join(CACHE_DIR, "ft_hard_eval_run.pkl"), "wb") as f:
        pickle.dump(hard_eval_run, f)
    with open(os.path.join(CACHE_DIR, "ft_hard_val_run.pkl"), "wb") as f:
        pickle.dump(hard_val_run, f)

    stats = {
        "base_model": BASE_MODEL,
        "val_ndcg10_random_negatives": round(rand_m["ndcg_cut_10"], 4),
        "val_ndcg10_hard_negatives":   round(hard_m["ndcg_cut_10"], 4),
    }
    with open(os.path.join(CACHE_DIR, "task4_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print("\nDone. Written: t4_random.trec  t4_hard.trec")
    return stats


if __name__ == "__main__":
    main()
