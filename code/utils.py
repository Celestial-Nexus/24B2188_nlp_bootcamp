"""Shared utilities: data loading, TREC I/O, evaluation."""
import json
import os
import csv

import pytrec_eval


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(data_dir="data"):
    def _read(fname):
        with open(os.path.join(data_dir, fname), encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]

    corpus_list  = _read("corpus.jsonl")
    train_claims = _read("train_claims.jsonl")
    val_claims   = _read("val_claims.jsonl")
    eval_claims  = _read("eval_claims.jsonl")

    corpus      = {d["doc_id"]: d for d in corpus_list}   # int-keyed
    all_doc_ids = [d["doc_id"] for d in corpus_list]

    return corpus_list, corpus, all_doc_ids, train_claims, val_claims, eval_claims


def doc_text(d):
    """Concatenate title + abstract sentences into a single string."""
    return d["title"] + ". " + " ".join(d["abstract"])


# ---------------------------------------------------------------------------
# TREC writer
# ---------------------------------------------------------------------------

def write_trec(results_dict, path, run_name, presorted=False):
    """Write a TREC run file.

    Args:
        results_dict: {claim_id (int): [(doc_id, score), ...]}
        path:         output file path
        run_name:     arbitrary tag placed in column 6
        presorted:    if True, treat the list as already in the correct output
                      order (use for t3_rerank.trec where cross-encoder and
                      first-stage scores are on different scales).
                      Synthetic descending scores (100..1) are assigned in
                      both modes so scores are always non-increasing.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for cid in sorted(results_dict):
            docs = results_dict[cid]
            if not presorted:
                docs = sorted(docs, key=lambda x: -x[1])
            # deduplicate doc_ids while preserving order
            seen, deduped = set(), []
            for did, sc in docs:
                if did not in seen:
                    seen.add(did)
                    deduped.append((did, sc))
            assert len(deduped) >= 100, (
                f"claim {cid}: only {len(deduped)} unique docs (need 100)"
            )
            for rank, (did, _) in enumerate(deduped[:100], 1):
                score = 100 - rank + 1  # 100, 99, ..., 1 — always non-increasing
                f.write(f"{cid} Q0 {did} {rank} {score} {run_name}\n")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def build_qrels(claims):
    """Build pytrec_eval qrels from claims with evidence."""
    return {
        str(c["id"]): {str(e["doc_id"]): 1 for e in c["evidence"]}
        for c in claims
        if c.get("evidence")
    }


def evaluate_trec(trec_path, qrels):
    """Return dict with ndcg_cut_10, recall_100, recip_rank averaged over claims."""
    run = {}
    with open(trec_path, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) < 6:
                continue
            cid, did, score = str(int(p[0])), str(int(p[2])), float(p[4])
            run.setdefault(cid, {})[did] = score

    if not run:
        return {"ndcg_cut_10": 0.0, "recall_100": 0.0, "recip_rank": 0.0}

    ev  = pytrec_eval.RelevanceEvaluator(
        qrels, {"ndcg_cut_10", "recall_100", "recip_rank"}
    )
    res = ev.evaluate(run)
    if not res:
        return {"ndcg_cut_10": 0.0, "recall_100": 0.0, "recip_rank": 0.0}
    return {
        m: sum(r[m] for r in res.values()) / len(res)
        for m in ["ndcg_cut_10", "recall_100", "recip_rank"]
    }


def per_claim_ndcg(trec_path, qrels, k=10):
    """Return {claim_id_str: ndcg@k} for claims that appear in qrels."""
    run = {}
    with open(trec_path, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) < 6:
                continue
            cid, did, score = str(int(p[0])), str(int(p[2])), float(p[4])
            run.setdefault(cid, {})[did] = score

    ev  = pytrec_eval.RelevanceEvaluator(qrels, {f"ndcg_cut_{k}"})
    res = ev.evaluate(run)
    return {cid: r[f"ndcg_cut_{k}"] for cid, r in res.items()}


# ---------------------------------------------------------------------------
# Verification CSV writer
# ---------------------------------------------------------------------------

def write_verification_csv(rows, path):
    """rows: [(claim_id, doc_id, label), ...]"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["claim_id", "doc_id", "label"])
        for cid, did, lab in rows:
            w.writerow([cid, did, lab])


def write_errors_csv(rows, path):
    """rows: [(claim_id, category), ...]"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["claim_id", "category"])
        for cid, cat in rows:
            w.writerow([cid, cat])
