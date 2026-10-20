"""Task 1: BM25 Lexical Baseline.

Grid-searches k1 and b on validation nDCG@10 then writes:
  predictions/val_bm25.trec   (162 val claims,  top-100)
  predictions/t1_bm25.trec    (300 eval claims, top-100)

Also caches BM25 top-100 for ALL train claims (needed by Task 4 hard-neg mining)
and saves best params to cache/bm25_params.json.

Run from repo root:
  python code/task1_bm25.py
"""

import itertools
import json
import os
import pickle
import sys

import bm25s
import Stemmer

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


def retrieve_bm25(retriever, queries, all_doc_ids, stemmer, k=100):
    """Return {claim_id: [(doc_id, score), ...]} for a list of (id, text) tuples."""
    run = {}
    for cid, q in queries:
        qtoks        = bm25s.tokenize([q], stopwords="english", stemmer=stemmer)
        docs, scores = retriever.retrieve(qtoks, corpus=all_doc_ids, k=k)
        run[cid]     = list(zip(docs[0].tolist(), scores[0].tolist()))
    return run


def main():
    corpus_list, corpus, all_doc_ids, train_claims, val_claims, eval_claims = load_data()

    doc_texts = [doc_text(d) for d in corpus_list]
    stemmer   = Stemmer.Stemmer("english")  # Snowball via PyStemmer

    print("Tokenising corpus …")
    corpus_tokens = bm25s.tokenize(doc_texts, stopwords="english", stemmer=stemmer)

    val_queries  = [(c["id"], c["claim"]) for c in val_claims]
    eval_queries = [(c["id"], c["claim"]) for c in eval_claims]
    qrels_val    = build_qrels(val_claims)

    # ------------------------------------------------------------------
    # Grid search on validation split
    # ------------------------------------------------------------------
    print("Grid-searching k1 × b on val nDCG@10 …")
    best_ndcg, best_params = 0.0, {"k1": 1.2, "b": 0.75}

    for k1, b in itertools.product(
        [0.5, 0.9, 1.2, 1.5, 2.0], [0.5, 0.75, 1.0]
    ):
        retriever = bm25s.BM25(k1=k1, b=b)
        retriever.index(corpus_tokens)

        tmp_run  = retrieve_bm25(retriever, val_queries, all_doc_ids, stemmer)
        tmp_path = os.path.join(CACHE_DIR, "bm25_tmp.trec")
        write_trec(tmp_run, tmp_path, "bm25_tmp")
        ndcg = evaluate_trec(tmp_path, qrels_val)["ndcg_cut_10"]

        print(f"  k1={k1:4.1f}  b={b:.2f}  nDCG@10={ndcg:.4f}")
        if ndcg > best_ndcg:
            best_ndcg   = ndcg
            best_params = {"k1": k1, "b": b}

    print(f"\nBest BM25: {best_params}  val nDCG@10={best_ndcg:.4f}")

    # ------------------------------------------------------------------
    # Final retrieval with best params
    # ------------------------------------------------------------------
    retriever = bm25s.BM25(**best_params)
    retriever.index(corpus_tokens)

    # val_bm25.trec
    val_run = retrieve_bm25(retriever, val_queries, all_doc_ids, stemmer)
    write_trec(val_run, os.path.join(PRED_DIR, "val_bm25.trec"), "bm25")

    # t1_bm25.trec
    eval_run = retrieve_bm25(retriever, eval_queries, all_doc_ids, stemmer)
    write_trec(eval_run, os.path.join(PRED_DIR, "t1_bm25.trec"), "bm25")

    # Train-claim top-100 for hard-neg mining in Task 4
    train_queries = [
        (c["id"], c["claim"]) for c in train_claims if c.get("evidence")
    ]
    train_run = retrieve_bm25(retriever, train_queries, all_doc_ids, stemmer)

    with open(os.path.join(CACHE_DIR, "bm25_train_run.pkl"), "wb") as f:
        pickle.dump(train_run, f)
    with open(os.path.join(CACHE_DIR, "bm25_val_run.pkl"), "wb") as f:
        pickle.dump(val_run, f)

    params_info = {
        "params": best_params,
        "val_ndcg10": round(best_ndcg, 4),
        "stemmer": "snowball_english",
    }
    with open(os.path.join(CACHE_DIR, "bm25_params.json"), "w") as f:
        json.dump(params_info, f, indent=2)

    # Reconfirm on saved file
    final_metrics = evaluate_trec(
        os.path.join(PRED_DIR, "val_bm25.trec"), qrels_val
    )
    print(f"Final val metrics: {final_metrics}")
    print("Done. Written: predictions/val_bm25.trec  predictions/t1_bm25.trec")
    return best_params, best_ndcg


if __name__ == "__main__":
    main()
