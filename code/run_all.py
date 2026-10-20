"""Orchestrator: runs all 5 tasks in order and writes results.json + report.md.

Run from repo root:
  python code/run_all.py

Individual tasks can also be run stand-alone:
  python code/task1_bm25.py
  python code/task2_dense.py
  python code/task3_hybrid.py
  python code/task4_finetune.py
  python code/task5_verify.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from code.utils import build_qrels, evaluate_trec, load_data, per_claim_ndcg

CACHE_DIR = "cache"
PRED_DIR  = "predictions"


def run_task(name, fn):
    print(f"\n{'#'*70}")
    print(f"#  {name}")
    print(f"{'#'*70}\n")
    return fn()


def build_results_json():
    """Assemble results.json from cached stats files."""
    _, _, _, _, val_claims, _ = load_data()
    qrels_val = build_qrels(val_claims)

    def load_json(path):
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return {}

    bm25_stats    = load_json(os.path.join(CACHE_DIR, "bm25_params.json"))
    dense_stats   = load_json(os.path.join(CACHE_DIR, "dense_stats.json"))
    task3_stats   = load_json(os.path.join(CACHE_DIR, "task3_stats.json"))
    task4_stats   = load_json(os.path.join(CACHE_DIR, "task4_stats.json"))
    task5_stats   = load_json(os.path.join(CACHE_DIR, "task5_stats.json"))

    # Q1: find 2 BM25-wins and 2 dense-wins on val (only claims with evidence)
    bm25_ndcg    = per_claim_ndcg(
        os.path.join(PRED_DIR, "val_bm25.trec"), qrels_val
    )
    general_ndcg = per_claim_ndcg(
        os.path.join(PRED_DIR, "val_general.trec"), qrels_val
    )
    common = set(bm25_ndcg) & set(general_ndcg)

    bm25_wins    = sorted(
        [cid for cid in common if bm25_ndcg[cid] > general_ndcg[cid]],
        key=lambda c: bm25_ndcg[c] - general_ndcg[c], reverse=True
    )[:2]
    general_wins = sorted(
        [cid for cid in common if general_ndcg[cid] > bm25_ndcg[cid]],
        key=lambda c: general_ndcg[c] - bm25_ndcg[c], reverse=True
    )[:2]

    bm25_params = bm25_stats.get("params", {})
    bm25_settings = (
        f"k1={bm25_params.get('k1', '?')}, "
        f"b={bm25_params.get('b', '?')}, "
        f"stemmer=snowball_english"
    )

    gen = dense_stats.get("general", {})
    sci = dense_stats.get("scientific", {})

    results = {
        "name":     "Rishabh Srivastava",
        "roll_no":  "24B2188",
        "seed":     42,
        "gpu_used": "CPU" if not _has_gpu() else "GPU",
        "task1": {
            "bm25_settings": bm25_settings,
            "val_ndcg10":    bm25_stats.get("val_ndcg10"),
        },
        "task2": {
            "general_model":         gen.get("model", "BAAI/bge-base-en-v1.5"),
            "scientific_model":      sci.get("model", "pritamdeka/S-PubMedBert-MS-MARCO"),
            "val_ndcg10_general":    gen.get("val_ndcg10"),
            "val_ndcg10_scientific": sci.get("val_ndcg10"),
            "index_time_sec":        {"general": gen.get("index_time_sec"),
                                      "scientific": sci.get("index_time_sec")},
            "query_latency_ms":      {"general": gen.get("query_latency_ms"),
                                      "scientific": sci.get("query_latency_ms")},
            "memory_mb":             {"general": gen.get("memory_mb"),
                                      "scientific": sci.get("memory_mb")},
        },
        "task3": {
            "hybrid_method":            task3_stats.get("hybrid_method", "RRF k=60"),
            "reranker_model":           task3_stats.get("reranker_model"),
            "rerank_top_k":             task3_stats.get("rerank_top_k"),
            "val_ndcg10_hybrid":        task3_stats.get("hybrid_val_ndcg10"),
            "val_ndcg10_rerank":        task3_stats.get("rerank_val_ndcg10"),
            "rerank_query_latency_ms":  task3_stats.get("rerank_query_latency_ms"),
        },
        "task4": {
            "base_model":                    task4_stats.get("base_model", "BAAI/bge-base-en-v1.5"),
            "val_ndcg10_random_negatives":   task4_stats.get("val_ndcg10_random_negatives"),
            "val_ndcg10_hard_negatives":     task4_stats.get("val_ndcg10_hard_negatives"),
        },
        "task5": {
            "retriever_used":        task5_stats.get("retriever_used"),
            "nli_model":             task5_stats.get("nli_model", "cross-encoder/nli-deberta-v3-small"),
            "zero_shot_or_finetuned": task5_stats.get("zero_shot_or_finetuned", "zero_shot"),
            "val_f1":                task5_stats.get("val_f1"),
        },
        "report_q1": {
            "bm25_wins":    [int(c) for c in bm25_wins],
            "general_wins": [int(c) for c in general_wins],
        },
    }

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nresults.json written.")
    print(f"  Q1 bm25_wins:    {results['report_q1']['bm25_wins']}")
    print(f"  Q1 general_wins: {results['report_q1']['general_wins']}")
    return results


def _has_gpu():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def write_report_md(results):
    """Write a stub report.md that can be filled in with the analysis."""
    r = results

    q1_bm25    = r["report_q1"]["bm25_wins"]
    q1_general = r["report_q1"]["general_wins"]

    report = f"""# NLP Bootcamp Report – SciFact
*Roll No: {r["roll_no"]}*

---

## Q1 — BM25 vs General Dense on Validation Claims

**BM25 wins** (claim IDs: {q1_bm25}): BM25 excels when claims contain rare biomedical
terms or exact-match acronyms (e.g. specific gene names) that appear verbatim in the
abstract. Dense models can miss low-frequency tokens not well-represented in the training
distribution of the embedding model.

**General dense wins** (claim IDs: {q1_general}): Dense retrieval handles paraphrases
and synonym-rich claims (e.g. "heart attack" vs "myocardial infarction") where BM25 finds
no lexical overlap with the supporting abstract.

*(Replace this paragraph with your specific analysis of the 4 claim IDs above.)*

---

## Q2 — Task 3: Largest Gain and Latency Cost

| Stage | val nDCG@10 | Δ from previous |
|-------|-------------|-----------------|
| BM25  | {r["task1"]["val_ndcg10"]} | — |
| General dense | {r["task2"]["val_ndcg10_general"]} | — |
| Hybrid | {r["task3"]["val_ndcg10_hybrid"]} | — |
| Rerank | {r["task3"]["val_ndcg10_rerank"]} | — |

The largest single gain was from **[hybrid / reranking — fill in after running]**.
Reranking added ~{r["task3"]["rerank_query_latency_ms"]} ms per query (cross-encoder
over top-{r["task3"]["rerank_top_k"]} docs).

---

## Q3 — Hard Negatives: Did They Help? Any Overfitting?

| System | val nDCG@10 |
|--------|-------------|
| Random negatives | {r["task4"]["val_ndcg10_random_negatives"]} |
| Hard negatives   | {r["task4"]["val_ndcg10_hard_negatives"]} |

Per-epoch val nDCG@10 curves are in `checkpoints/random_neg/eval_results.csv` and
`checkpoints/hard_neg/eval_results.csv`. Hard negatives [improved / did not improve]
over random. Overfitting was detected when val nDCG peaked at epoch N then declined.
*(Fill in with your actual epoch-by-epoch numbers.)*

---

## Q4 — Fraction of Verification Errors that are retrieval_miss

From `predictions/t5_val_errors.csv`:
retrieval\_miss = **[fill in count]** / **[total errors]** = **[X]%**

---

## Q5 — Single Next Change

**SPLADE sparse retrieval** (Formal et al., 2021): SPLADE learns sparse, lexically
interpretable document representations via regularised MLM. It combines exact-match
reliability of BM25 with semantic generalisation of dense models and is proven on
BEIR benchmarks. Fine-tuning SPLADE on the SciFact training claims with hard negatives
would directly target the retrieval\_miss failures that dominate our error distribution.
"""

    with open("report.md", "w", encoding="utf-8") as f:
        f.write(report)
    print("report.md written (fill in [placeholder] sections with your numbers).")


def main():
    import code.task1_bm25    as t1
    import code.task2_dense   as t2
    import code.task3_hybrid  as t3
    import code.task4_finetune as t4
    import code.task5_verify  as t5

    run_task("Task 1 – BM25 Baseline",              t1.main)
    run_task("Task 2 – Dense Retrieval",             t2.main)
    run_task("Task 3 – Hybrid + Rerank",             t3.main)
    run_task("Task 4 – Fine-Tuning",                 t4.main)
    run_task("Task 5 – Verification + Error Analysis", t5.main)

    results = build_results_json()
    write_report_md(results)

    print("\n" + "="*70)
    print("All tasks complete. Run the format checker:")
    print("  python3 check_format.py 24B2188_nlp_bootcamp.zip")
    print("="*70)


if __name__ == "__main__":
    main()
