"""Task 5: Claim Verification + Error Analysis.

Uses the best retriever (selected by val nDCG@10 across all tasks) then
applies sentence-level NLI with cross-encoder/nli-deberta-v3-small.

Produces:
  predictions/t5_verification.csv      (eval claims)
  predictions/t5_val_verification.csv  (val claims)
  predictions/t5_val_errors.csv        (val error analysis)

Run from repo root:
  python code/task5_verify.py
"""

import csv
import json
import os
import pickle
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from code.utils import (
    build_qrels,
    doc_text,
    evaluate_trec,
    load_data,
    write_errors_csv,
    write_verification_csv,
)

PRED_DIR  = "predictions"
CACHE_DIR = "cache"
os.makedirs(PRED_DIR, exist_ok=True)

NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
TOP_DOCS_FOR_NLI = 10   # consider top-10 retrieved docs per claim
MAX_CANDIDATES   = 3    # at most 3 (claim, doc, label) rows in output


# ---------------------------------------------------------------------------
# Best-retriever selection
# ---------------------------------------------------------------------------

def pick_best_retriever(qrels_val):
    """Evaluate all cached val runs and return (run_name, val_ndcg10)."""
    candidates = {
        "bm25":       os.path.join(PRED_DIR, "val_bm25.trec"),
        "general":    os.path.join(PRED_DIR, "val_general.trec"),
        "hybrid":     os.path.join(CACHE_DIR, "hybrid_val.trec"),
        "rerank":     os.path.join(CACHE_DIR, "rerank_val_final.trec"),
        "ft_random":  os.path.join(CACHE_DIR, "val_ft_random.trec"),
        "ft_hard":    os.path.join(CACHE_DIR, "val_ft_hard.trec"),
    }
    best_name, best_ndcg = "bm25", 0.0
    for name, path in candidates.items():
        if not os.path.exists(path):
            print(f"  [{name}] file missing, skipping")
            continue
        m = evaluate_trec(path, qrels_val)
        print(f"  [{name}] val nDCG@10 = {m['ndcg_cut_10']:.4f}")
        if m["ndcg_cut_10"] > best_ndcg:
            best_ndcg = m["ndcg_cut_10"]
            best_name = name
    print(f"\nBest retriever: {best_name}  (val nDCG@10={best_ndcg:.4f})")
    return best_name, best_ndcg


def load_trec_run(path):
    """Return {claim_id (int): [(doc_id (int), score), ...]} sorted desc."""
    run = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) < 6:
                continue
            cid, did, score = int(p[0]), int(p[2]), float(p[4])
            run.setdefault(cid, []).append((did, score))
    for cid in run:
        run[cid].sort(key=lambda x: -x[1])
    return run


# ---------------------------------------------------------------------------
# NLI scoring
# ---------------------------------------------------------------------------

def load_nli_model(model_name):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    id2label = model.config.id2label   # read — NEVER hardcode indices
    label2id = {v: k for k, v in id2label.items()}
    support_idx    = label2id["entailment"]
    contradict_idx = label2id["contradiction"]

    return tokenizer, model, support_idx, contradict_idx


def score_claim_vs_abstract(
    claim, abstract_sentences, tokenizer, nli_model, support_idx, contradict_idx,
    device, threshold=0.5
):
    """
    Batch all abstract sentences in one forward pass.
    Premise = each abstract sentence; Hypothesis = claim.
    Returns (label, confidence) or (None, 0.0) for NEI.
    """
    if not abstract_sentences:
        return None, 0.0

    enc = tokenizer(
        abstract_sentences,                       # premises
        [claim] * len(abstract_sentences),        # hypotheses
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = nli_model(**enc).logits   # (n_sents, n_labels)

    probs = F.softmax(logits, dim=-1)   # softmax before thresholding

    best_support    = probs[:, support_idx].max().item()
    best_contradict = probs[:, contradict_idx].max().item()

    if best_support > threshold and best_support > best_contradict:
        return "SUPPORT", best_support
    if best_contradict > threshold:
        return "CONTRADICT", best_contradict
    return None, 0.0


def run_verification(
    claims, retriever_run, corpus, tokenizer, nli_model,
    support_idx, contradict_idx, device, threshold=0.5
):
    """
    For each claim, score top-10 retrieved docs via sentence-level NLI.
    Collect all confident (doc, label) predictions, sort by confidence, keep top-3.
    Returns list of (claim_id, doc_id, label) rows.
    """
    rows = []
    for i, claim in enumerate(claims):
        cid    = claim["id"]
        docs   = retriever_run.get(cid, [])

        candidates = []  # (confidence, doc_id, label)
        for did, _ in docs[:TOP_DOCS_FOR_NLI]:
            doc  = corpus[did]
            sents = doc["abstract"]
            label, conf = score_claim_vs_abstract(
                claim["claim"], sents, tokenizer, nli_model,
                support_idx, contradict_idx, device, threshold
            )
            if label is not None:
                candidates.append((conf, did, label))

        candidates.sort(key=lambda x: -x[0])
        rows.extend(
            (cid, did, lab) for _, did, lab in candidates[:MAX_CANDIDATES]
        )

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(claims)} claims processed …")

    return rows


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------

VALID_CATEGORIES = {
    "retrieval_miss", "negation", "numerical",
    "entity_mismatch", "needs_multiple_docs", "other",
}


def error_category(claim, gold_ev, predicted_triples, retriever_run):
    """
    Assign one error category to a val claim where prediction != gold.
    gold_ev: list of {doc_id, label, sentences}
    predicted_triples: {doc_id: label} or empty (→ NEI prediction)
    retriever_run: {cid: [(did, score), ...]}
    """
    gold_ids = {e["doc_id"] for e in gold_ev}
    retrieved_ids = {did for did, _ in retriever_run.get(claim["id"], [])}

    # Retrieval miss: gold doc not in top-100
    if gold_ids - retrieved_ids:
        return "retrieval_miss"

    # Multi-doc: multiple gold abstracts required
    if len(gold_ev) > 1:
        return "needs_multiple_docs"

    # Check claim text for signals
    claim_lower = claim["claim"].lower()
    neg_words   = ["not ", "no ", "without ", "lack", "does not", "do not",
                   "cannot", "never", "neither", "nor "]
    num_words   = [str(d) for d in range(10)] + ["%", "fold", "times", "ratio"]

    if any(w in claim_lower for w in neg_words):
        return "negation"
    if any(w in claim_lower for w in num_words):
        return "numerical"

    return "other"


def analyse_errors(val_claims, gold_qrels, predicted_triples_by_cid, retriever_run):
    """
    predicted_triples_by_cid: {cid: {doc_id: label}} (empty → NEI prediction)
    Returns [(claim_id, category), ...] for real errors only.
    """
    error_rows = []
    for claim in val_claims:
        cid      = claim["id"]
        gold_ev  = claim.get("evidence", [])

        if not gold_ev:
            # True NEI: only an error if we predicted something
            if predicted_triples_by_cid.get(cid):
                error_rows.append((cid, "other"))
            continue

        # Gold exists: compare predicted (did, label) set against gold
        gold_set = {(e["doc_id"], e["label"]) for e in gold_ev}
        pred_set = {(did, lab) for did, lab in predicted_triples_by_cid.get(cid, {}).items()}

        if pred_set == gold_set:
            continue  # correct prediction — not an error

        cat = error_category(claim, gold_ev, predicted_triples_by_cid.get(cid, {}),
                             retriever_run)
        error_rows.append((cid, cat))

    return error_rows


# ---------------------------------------------------------------------------
# Threshold tuning on val
# ---------------------------------------------------------------------------

def tune_threshold(val_claims, val_run, corpus, tokenizer, nli_model,
                   support_idx, contradict_idx, device):
    """Grid-search threshold in [0.3, 0.9] by val F1 over (cid, did, label) triples."""
    gold_triples = set()
    for c in val_claims:
        for e in c.get("evidence", []):
            gold_triples.add((c["id"], e["doc_id"], e["label"]))

    best_thresh, best_f1 = 0.5, 0.0
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        rows = run_verification(
            val_claims, val_run, corpus, tokenizer, nli_model,
            support_idx, contradict_idx, device, threshold=thresh
        )
        pred_triples = set((cid, did, lab) for cid, did, lab in rows)
        tp = len(gold_triples & pred_triples)
        prec = tp / len(pred_triples) if pred_triples else 0.0
        rec  = tp / len(gold_triples) if gold_triples else 0.0
        f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        print(f"  thresh={thresh:.1f}  P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}")
        if f1 > best_f1:
            best_f1, best_thresh = f1, thresh

    print(f"Best threshold: {best_thresh}  (F1={best_f1:.4f})")
    return best_thresh, best_f1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    corpus_list, corpus, all_doc_ids, train_claims, val_claims, eval_claims = load_data()
    qrels_val = build_qrels(val_claims)

    # ------------------------------------------------------------------
    # Pick best retriever
    # ------------------------------------------------------------------
    print("Selecting best retriever by val nDCG@10 …")
    best_name, best_ndcg = pick_best_retriever(qrels_val)

    run_map = {
        "bm25":      (os.path.join(PRED_DIR, "val_bm25.trec"),
                      os.path.join(PRED_DIR, "t1_bm25.trec")),
        "general":   (os.path.join(PRED_DIR, "val_general.trec"),
                      os.path.join(PRED_DIR, "t2_general.trec")),
        "hybrid":    (os.path.join(CACHE_DIR, "hybrid_val.trec"),
                      os.path.join(PRED_DIR, "t3_hybrid.trec")),
        "rerank":    (os.path.join(CACHE_DIR, "rerank_val_final.trec"),
                      os.path.join(PRED_DIR, "t3_rerank.trec")),
        "ft_random": (os.path.join(CACHE_DIR, "val_ft_random.trec"),
                      os.path.join(PRED_DIR, "t4_random.trec")),
        "ft_hard":   (os.path.join(CACHE_DIR, "val_ft_hard.trec"),
                      os.path.join(PRED_DIR, "t4_hard.trec")),
    }

    val_trec_path, eval_trec_path = run_map[best_name]
    val_run  = load_trec_run(val_trec_path)
    eval_run = load_trec_run(eval_trec_path)

    # ------------------------------------------------------------------
    # Load NLI model
    # ------------------------------------------------------------------
    print(f"\nLoading NLI model: {NLI_MODEL_NAME}")
    tokenizer, nli_model, support_idx, contradict_idx = load_nli_model(NLI_MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nli_model = nli_model.to(device)
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Tune threshold on val
    # ------------------------------------------------------------------
    print("\nTuning threshold on val claims …")
    threshold, val_f1 = tune_threshold(
        val_claims, val_run, corpus, tokenizer, nli_model,
        support_idx, contradict_idx, device
    )

    # ------------------------------------------------------------------
    # Val verification (t5_val_verification.csv)
    # ------------------------------------------------------------------
    print("\nRunning verification on val claims …")
    val_rows = run_verification(
        val_claims, val_run, corpus, tokenizer, nli_model,
        support_idx, contradict_idx, device, threshold
    )
    write_verification_csv(val_rows, os.path.join(PRED_DIR, "t5_val_verification.csv"))

    # Build predicted triples dict for error analysis
    pred_by_cid = {}
    for cid, did, lab in val_rows:
        pred_by_cid.setdefault(cid, {})[did] = lab

    # ------------------------------------------------------------------
    # Error analysis (t5_val_errors.csv)
    # ------------------------------------------------------------------
    print("Running error analysis …")
    error_rows = analyse_errors(val_claims, qrels_val, pred_by_cid, val_run)
    write_errors_csv(error_rows, os.path.join(PRED_DIR, "t5_val_errors.csv"))

    retrieval_miss_count = sum(1 for _, cat in error_rows if cat == "retrieval_miss")
    print(f"Val errors: {len(error_rows)} total, "
          f"{retrieval_miss_count} retrieval_miss "
          f"({100*retrieval_miss_count/max(len(error_rows),1):.0f}%)")

    # ------------------------------------------------------------------
    # Eval verification (t5_verification.csv)
    # ------------------------------------------------------------------
    print("\nRunning verification on eval claims …")
    eval_rows = run_verification(
        eval_claims, eval_run, corpus, tokenizer, nli_model,
        support_idx, contradict_idx, device, threshold
    )
    write_verification_csv(eval_rows, os.path.join(PRED_DIR, "t5_verification.csv"))

    stats = {
        "retriever_used": best_name,
        "retriever_val_ndcg10": round(best_ndcg, 4),
        "nli_model": NLI_MODEL_NAME,
        "zero_shot_or_finetuned": "zero_shot",
        "threshold": threshold,
        "val_f1": round(val_f1, 4),
        "val_errors_total": len(error_rows),
        "val_errors_retrieval_miss_pct": round(
            100 * retrieval_miss_count / max(len(error_rows), 1), 1
        ),
    }
    with open(os.path.join(CACHE_DIR, "task5_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print("\nDone. Written:")
    print("  predictions/t5_verification.csv")
    print("  predictions/t5_val_verification.csv")
    print("  predictions/t5_val_errors.csv")
    return stats


if __name__ == "__main__":
    main()
