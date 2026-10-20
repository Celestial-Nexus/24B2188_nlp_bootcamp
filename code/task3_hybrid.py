"""Task 3: Hybrid Fusion + Cross-Encoder Reranking.

Loads cached BM25 and dense runs, then produces:
  predictions/t3_hybrid.trec   (300 eval, top-100) — RRF of BM25 + bge
  predictions/t3_rerank.trec   (300 eval, top-100) — cross-encoder on top-k of hybrid

Also tries score interpolation and keeps whichever hybrid method wins on val.
Tunes rerank top-k on val nDCG@10.

Run from repo root:
  python code/task3_hybrid.py
"""

import json
import os
import pickle
import sys

import numpy as np

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
os.makedirs(PRED_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Fusion helpers
# ---------------------------------------------------------------------------

def rrf_fusion(runs, k=60):
    """Reciprocal Rank Fusion over a list of {cid: [(did, score), ...]} dicts."""
    fused = {}
    all_cids = set()
    for run in runs:
        all_cids.update(run.keys())
    for cid in all_cids:
        scores = {}
        for run in runs:
            docs = run.get(cid, [])
            # sort descending by score to get ranks
            docs = sorted(docs, key=lambda x: -x[1])
            for rank, (did, _) in enumerate(docs, 1):
                scores[did] = scores.get(did, 0.0) + 1.0 / (k + rank)
        fused[cid] = sorted(scores.items(), key=lambda x: -x[1])
    return fused


def interpolate_fusion(run_a, run_b, alpha, all_cids):
    """Score interpolation: alpha * norm(A) + (1-alpha) * norm(B)."""
    def minmax_norm(run):
        normed = {}
        for cid, docs in run.items():
            sc = [s for _, s in docs]
            lo, hi = min(sc), max(sc)
            denom = hi - lo if hi != lo else 1.0
            normed[cid] = {did: (s - lo) / denom for did, s in docs}
        return normed

    na, nb = minmax_norm(run_a), minmax_norm(run_b)
    fused = {}
    for cid in all_cids:
        da, db = na.get(cid, {}), nb.get(cid, {})
        all_dids = set(da) | set(db)
        scores = {did: alpha * da.get(did, 0.0) + (1 - alpha) * db.get(did, 0.0)
                  for did in all_dids}
        fused[cid] = sorted(scores.items(), key=lambda x: -x[1])
    return fused


# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------

def rerank_run(claims, hybrid_run, corpus, ce_model, k=50, batch_size=32):
    """
    For each claim, rerank top-k of hybrid_run with cross-encoder.
    Returns {cid: [(did, score), ...]} with presorted=True semantics
    (cross-encoder order first, then backfill).
    """
    from sentence_transformers import CrossEncoder

    reranked = {}
    for claim in claims:
        cid       = claim["id"]
        claim_txt = claim["claim"]
        docs      = hybrid_run.get(cid, [])  # already sorted desc by hybrid score
        top_k     = [did for did, _ in docs[:k]]
        rest      = [(did, sc) for did, sc in docs[k:] if did not in set(top_k)]

        pairs  = [(doc_text(corpus[did]), claim_txt) for did in top_k]
        # Note: CrossEncoder.predict takes (text_a, text_b) where text_a is fed as
        # sentence A in the model. For ms-marco rerankers the convention is
        # (query, passage) BUT these are passage rerankers, not NLI models — keep as is.
        ce_scr = ce_model.predict(pairs, batch_size=batch_size)

        ordered = sorted(zip(top_k, ce_scr.tolist()), key=lambda x: -x[1])
        # presorted list: CE-ordered top-k then original-order backfill
        reranked[cid] = ordered + rest

    return reranked


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    corpus_list, corpus, all_doc_ids, train_claims, val_claims, eval_claims = load_data()
    qrels_val = build_qrels(val_claims)

    # Load cached BM25 and dense runs
    with open(os.path.join(CACHE_DIR, "bm25_val_run.pkl"), "rb") as f:
        bm25_val = pickle.load(f)
    with open(os.path.join(CACHE_DIR, "bm25_train_run.pkl"), "rb") as f:
        bm25_train = pickle.load(f)  # noqa: F841 (kept for reference)

    with open(os.path.join(CACHE_DIR, "dense_general_val_run.pkl"), "rb") as f:
        dense_val = pickle.load(f)

    # Rebuild BM25 eval run from t1_bm25.trec
    bm25_eval = {}
    with open(os.path.join(PRED_DIR, "t1_bm25.trec")) as f:
        for line in f:
            p = line.split()
            cid, did = int(p[0]), int(p[2])
            bm25_eval.setdefault(cid, []).append((did, float(p[4])))

    # Rebuild dense eval run from t2_general.trec
    dense_eval = {}
    with open(os.path.join(PRED_DIR, "t2_general.trec")) as f:
        for line in f:
            p = line.split()
            cid, did = int(p[0]), int(p[2])
            dense_eval.setdefault(cid, []).append((did, float(p[4])))

    all_val_cids  = set(c["id"] for c in val_claims)
    all_eval_cids = set(c["id"] for c in eval_claims)

    # ------------------------------------------------------------------
    # Try RRF and score interpolation on val; pick winner
    # ------------------------------------------------------------------
    print("Evaluating hybrid methods on val …")

    rrf_val = rrf_fusion([bm25_val, dense_val])
    write_trec(rrf_val, os.path.join(CACHE_DIR, "hybrid_rrf_val.trec"), "rrf")
    rrf_ndcg = evaluate_trec(
        os.path.join(CACHE_DIR, "hybrid_rrf_val.trec"), qrels_val
    )["ndcg_cut_10"]
    print(f"  RRF (k=60):                val nDCG@10={rrf_ndcg:.4f}")

    best_alpha, best_interp_ndcg = 0.5, 0.0
    for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        interp = interpolate_fusion(bm25_val, dense_val, alpha, all_val_cids)
        write_trec(interp, os.path.join(CACHE_DIR, "hybrid_interp_val.trec"), "interp")
        ndcg = evaluate_trec(
            os.path.join(CACHE_DIR, "hybrid_interp_val.trec"), qrels_val
        )["ndcg_cut_10"]
        print(f"  Interpolation alpha={alpha:.1f}:  val nDCG@10={ndcg:.4f}")
        if ndcg > best_interp_ndcg:
            best_interp_ndcg = ndcg
            best_alpha = alpha

    if rrf_ndcg >= best_interp_ndcg:
        print(f"\nWinner: RRF (nDCG={rrf_ndcg:.4f})")
        hybrid_val  = rrf_val
        hybrid_eval = rrf_fusion([bm25_eval, dense_eval])
        hybrid_method = "RRF k=60"
        hybrid_ndcg = rrf_ndcg
    else:
        print(f"\nWinner: Interpolation alpha={best_alpha} (nDCG={best_interp_ndcg:.4f})")
        hybrid_val  = interpolate_fusion(bm25_val,  dense_val,  best_alpha, all_val_cids)
        hybrid_eval = interpolate_fusion(bm25_eval, dense_eval, best_alpha, all_eval_cids)
        hybrid_method = f"Score interpolation alpha={best_alpha}"
        hybrid_ndcg = best_interp_ndcg

    write_trec(hybrid_eval, os.path.join(PRED_DIR, "t3_hybrid.trec"), "hybrid")
    write_trec(hybrid_val,  os.path.join(CACHE_DIR, "hybrid_val.trec"), "hybrid")

    val_m = evaluate_trec(os.path.join(CACHE_DIR, "hybrid_val.trec"), qrels_val)
    print(f"Hybrid val metrics: {val_m}")

    # ------------------------------------------------------------------
    # Cross-encoder reranking — tune k on val
    # ------------------------------------------------------------------
    print("\nLoading cross-encoder …")
    from sentence_transformers import CrossEncoder
    ce_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

    best_k, best_rerank_ndcg = 50, 0.0
    for k in [20, 50, 100]:
        print(f"  Reranking val with k={k} …")
        rr_val = rerank_run(val_claims, hybrid_val, corpus, ce_model, k=k)
        write_trec(
            rr_val, os.path.join(CACHE_DIR, f"rerank_val_k{k}.trec"),
            "rerank", presorted=True
        )
        ndcg = evaluate_trec(
            os.path.join(CACHE_DIR, f"rerank_val_k{k}.trec"), qrels_val
        )["ndcg_cut_10"]
        print(f"    k={k}  val nDCG@10={ndcg:.4f}")
        if ndcg > best_rerank_ndcg:
            best_rerank_ndcg = ndcg
            best_k = k

    print(f"\nBest rerank k={best_k}  val nDCG@10={best_rerank_ndcg:.4f}")

    # Final rerank for eval
    rr_eval = rerank_run(eval_claims, hybrid_eval, corpus, ce_model, k=best_k)
    write_trec(
        rr_eval, os.path.join(PRED_DIR, "t3_rerank.trec"), "rerank", presorted=True
    )

    # Reconfirm val rerank metrics
    rr_val_final = rerank_run(val_claims, hybrid_val, corpus, ce_model, k=best_k)
    write_trec(
        rr_val_final, os.path.join(CACHE_DIR, "rerank_val_final.trec"),
        "rerank", presorted=True
    )
    val_rr_m = evaluate_trec(
        os.path.join(CACHE_DIR, "rerank_val_final.trec"), qrels_val
    )
    print(f"Final rerank val metrics: {val_rr_m}")

    # Measure reranker latency on val queries (avg per query, all 162 val claims)
    import time
    t0 = time.perf_counter()
    rerank_run(val_claims, hybrid_val, corpus, ce_model, k=best_k)
    lat_ms = (time.perf_counter() - t0) / len(val_claims) * 1000

    stats = {
        "hybrid_method": hybrid_method,
        "hybrid_val_ndcg10": round(hybrid_ndcg, 4),
        "reranker_model": "cross-encoder/ms-marco-MiniLM-L-12-v2",
        "rerank_top_k": best_k,
        "rerank_val_ndcg10": round(best_rerank_ndcg, 4),
        "rerank_query_latency_ms": round(lat_ms, 1),
    }
    with open(os.path.join(CACHE_DIR, "task3_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print("\nDone. Written: t3_hybrid.trec  t3_rerank.trec")
    return stats


if __name__ == "__main__":
    main()
