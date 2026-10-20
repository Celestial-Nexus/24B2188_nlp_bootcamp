# 24B2188 — NLP Bootcamp: SciFact Scientific Claim Retrieval & Verification

**Inter-IIT Tech Meet 15.0 | NLP Bootcamp**
**Roll No:** 24B2188 | **Name:** Rishabh Srivastava | **Institute:** IIT Bombay

---

## Results Summary

| Task | Method | Val nDCG@10 |
|------|--------|-------------|
| T1 — BM25 Baseline | BM25 (k1=0.9, b=0.5, Snowball stemmer) | **0.8641** |
| T2 — General Dense | BAAI/bge-base-en-v1.5 | **0.8906** |
| T2 — Scientific Dense | pritamdeka/S-PubMedBert-MS-MARCO | **0.8438** |
| T3 — Hybrid Fusion | Score interpolation α=0.4 (BM25 + BGE) | **0.9135** |
| T3 — Rerank | cross-encoder/ms-marco-MiniLM-L-12-v2 (top-20) | 0.8703 |
| T4 — Random Neg Fine-tune | BGE-base fine-tuned, MNRL loss | **0.9107** |
| T4 — Hard Neg Fine-tune | BGE-base + BM25/dense hard negatives | **0.9150** ← best |
| T5 — Verification | DeBERTa-v3-small NLI, zero-shot | F1 = 0.1718 |

---

## Repository Structure

```
.
├── README.md                        ← this file
├── results.json                     ← all metrics and settings
├── report.md                        ← Q1–Q5 analysis
├── code/
│   ├── utils.py                     ← shared: data loading, TREC I/O, evaluation
│   ├── task1_bm25.py                ← BM25 grid search + retrieval
│   ├── task2_dense.py               ← dense encoding (BGE + PubMedBERT)
│   ├── task3_hybrid.py              ← RRF / score interpolation + cross-encoder rerank
│   ├── task4_finetune.py            ← bi-encoder fine-tuning (random + hard negatives)
│   ├── task5_verify.py              ← NLI verification + error analysis
│   ├── run_all.py                   ← orchestrator (runs all tasks in sequence)
│   └── README.md                    ← reproduction instructions
└── predictions/
    ├── t1_bm25.trec                 ← Task 1 eval run (300 queries × 100 docs)
    ├── t2_general.trec              ← Task 2 general dense eval run
    ├── t2_scientific.trec           ← Task 2 scientific dense eval run
    ├── t3_hybrid.trec               ← Task 3 hybrid eval run
    ├── t3_rerank.trec               ← Task 3 reranked eval run
    ├── t4_random.trec               ← Task 4 random-neg fine-tuned eval run
    ├── t4_hard.trec                 ← Task 4 hard-neg fine-tuned eval run
    ├── val_bm25.trec                ← Task 1 val run (required)
    ├── val_general.trec             ← Task 2 val run (required)
    ├── t5_verification.csv          ← Task 5 eval claim verification
    ├── t5_val_verification.csv      ← Task 5 val claim verification
    └── t5_val_errors.csv            ← Task 5 error analysis
```

---

## Pipeline Overview

```
Corpus (5183 docs) + Claims (train=647, val=162, eval=300)
        │
        ▼
┌─────────────┐    ┌──────────────────────┐
│  Task 1     │    │  Task 2              │
│  BM25       │    │  Dense Retrieval     │
│  k1=0.9     │    │  BGE-base (general)  │
│  b=0.5      │    │  PubMedBERT (sci.)   │
└──────┬──────┘    └──────────┬───────────┘
       │                      │
       └──────────┬───────────┘
                  ▼
        ┌─────────────────┐
        │   Task 3        │
        │  Hybrid Fusion  │ ← score interp α=0.4 wins over RRF
        │  + CE Rerank    │ ← MiniLM-L12 ms-marco
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │   Task 4        │
        │  Fine-tuning    │ ← MNRL loss, 5 epochs
        │  Random neg     │   best val nDCG@10 = 0.9107
        │  Hard neg       │   best val nDCG@10 = 0.9150 ✓ best retriever
        └────────┬────────┘
                 │
                 ▼
        ┌─────────────────┐
        │   Task 5        │
        │  NLI Verify     │ ← DeBERTa-v3-small, zero-shot
        │  + Error Anal.  │   threshold tuned on val F1
        └─────────────────┘
```

---

## Models Used

| Role | Model | Params |
|------|-------|--------|
| General bi-encoder | [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) | 109M |
| Scientific bi-encoder | [pritamdeka/S-PubMedBert-MS-MARCO](https://huggingface.co/pritamdeka/S-PubMedBert-MS-MARCO) | 109M |
| Cross-encoder reranker | [cross-encoder/ms-marco-MiniLM-L-12-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-12-v2) | 33M |
| NLI verifier | [cross-encoder/nli-deberta-v3-small](https://huggingface.co/cross-encoder/nli-deberta-v3-small) | 142M |

All encoders are ≤400M parameters ✅

---

## Reproduction

### Requirements
```bash
pip install sentence-transformers==6.1.0 faiss-cpu pytrec-eval-terrier \
            datasets bm25s PyStemmer transformers psutil
```

### Run all tasks
```bash
# From the repo root, with data/ folder containing corpus.jsonl etc.
python3 code/task1_bm25.py
python3 code/task2_dense.py
python3 code/task3_hybrid.py
python3 code/task4_finetune.py
python3 code/task5_verify.py
```

Or run everything at once:
```bash
python3 code/run_all.py
```

> **Note:** Tasks 4 (fine-tuning) and 2 (dense encoding) require a GPU. Tasks 1, 3, 5 run on CPU.

### Validate submission
```bash
python3 check_format.py 24B2188_nlp_bootcamp.zip
```

---

## Key Design Decisions

- **BM25**: Grid-searched `k1 ∈ {0.5,0.9,1.2,1.5}` and `b ∈ {0.3,0.5,0.75}` on val nDCG@10. Best: `k1=0.9, b=0.5`.
- **Hybrid**: Score interpolation `0.4×BM25 + 0.6×dense` beat RRF k=60 on val (0.9135 vs 0.9115).
- **Hard negatives**: Mined from BM25 + dense ranks 5–50 on train claims; all gold docs excluded. Improved val nDCG@10 by +0.0043 over random negatives.
- **NLI**: Sentence-level scoring — each abstract sentence is scored as premise against the claim as hypothesis. Best threshold (0.5) selected by val F1. Dominant error type: NLI reasoning failures (63% "other"), not retrieval misses (only 1%).
- **Best retriever for Task 5**: Hard-neg fine-tuned BGE (val nDCG@10 = 0.9150).
