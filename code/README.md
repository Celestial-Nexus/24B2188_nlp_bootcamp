# NLP Bootcamp Code – Reproduction Instructions

## Requirements

```bash
pip install bm25s PyStemmer sentence-transformers faiss-cpu \
    transformers datasets torch scikit-learn tqdm pytrec_eval psutil
```

Python 3.9+. GPU optional but recommended for Tasks 2–5.
All commands must be run from the **repo root** (the folder containing `data/`).

---

## Run Everything at Once

```bash
python code/run_all.py
```

This runs Tasks 1 → 2 → 3 → 4 → 5 in sequence, then writes `results.json` and
a stub `report.md`.

---

## Run Individual Tasks

### Task 1 – BM25 Baseline
```bash
python code/task1_bm25.py
```
Grid-searches `k1` and `b` on val nDCG@10. Writes:
- `predictions/val_bm25.trec`
- `predictions/t1_bm25.trec`
- `cache/bm25_params.json`
- `cache/bm25_train_run.pkl`  ← needed by Task 4

### Task 2 – Dense Retrieval
```bash
python code/task2_dense.py
```
Downloads `BAAI/bge-base-en-v1.5` and `pritamdeka/S-PubMedBert-MS-MARCO` from
Hugging Face (requires internet on first run; cached locally afterwards). Writes:
- `predictions/val_general.trec`
- `predictions/t2_general.trec`
- `predictions/t2_scientific.trec`
- `cache/dense_stats.json`
- `cache/dense_general_train_run.pkl`  ← needed by Task 4

*Requires Task 1 to be run first (uses cached BM25 run for comparison only).*

### Task 3 – Hybrid + Rerank
```bash
python code/task3_hybrid.py
```
Requires Tasks 1 and 2 outputs. Writes:
- `predictions/t3_hybrid.trec`
- `predictions/t3_rerank.trec`
- `cache/task3_stats.json`

### Task 4 – Fine-Tuning
```bash
python code/task4_finetune.py
```
Requires Tasks 1 and 2 outputs (for hard-neg mining). GPU strongly recommended
(~3–4 hours on T4 per run, 10+ hours on CPU). Writes:
- `predictions/t4_random.trec`
- `predictions/t4_hard.trec`
- `checkpoints/random_neg/`  ← per-epoch val nDCG@10 in `eval_results.csv`
- `checkpoints/hard_neg/`

### Task 5 – Verification + Error Analysis
```bash
python code/task5_verify.py
```
Requires Tasks 1–4 outputs (auto-selects best retriever by val nDCG@10). Writes:
- `predictions/t5_verification.csv`
- `predictions/t5_val_verification.csv`
- `predictions/t5_val_errors.csv`

---

## Validate Format
```bash
python check_format.py 24B2188_nlp_bootcamp.zip
```
`data/` must be a sibling of `check_format.py`. Pass only the zip path — no other arguments.

---

## Reproducibility

All randomness is seeded with `seed=42` in `task4_finetune.py`.
BM25 and FAISS retrieval are deterministic given the same model weights.
For exact reproduction of Task 4 fine-tuning results, use the same GPU model listed in `results.json`.

---

## File Sizes (approximate)

| File | Size |
|------|------|
| BAAI/bge-base-en-v1.5 | ~430 MB |
| pritamdeka/S-PubMedBert-MS-MARCO | ~430 MB |
| cross-encoder/ms-marco-MiniLM-L-12-v2 | ~130 MB |
| cross-encoder/nli-deberta-v3-small | ~180 MB |
