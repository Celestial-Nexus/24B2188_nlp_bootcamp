# NLP Bootcamp Report – SciFact
*Roll No: 24B2188*

---

## Q1 — BM25 vs General Dense on Validation Claims

**BM25 wins** (claim IDs: [34, 1261]): Claim 34 contains the specific gene name "HMGA1" and claim 1261 references the exact drug term "metformin" — both appear verbatim in the supporting abstract. BGE-base fails here because these low-frequency biomedical tokens are underrepresented in its general-domain training data, so the embedding similarity is lower than BM25's exact-match score.

**General dense wins** (claim IDs: [463, 766]): Claim 463 uses "heart muscle damage" while the abstract uses "myocardial injury"; claim 766 says "obesity increases cancer risk" while the abstract discusses "adiposity and tumour development". BM25 finds no lexical overlap, but BGE-base captures the semantic equivalence through its embedding space.

---

## Q2 — Task 3: Largest Gain and Latency Cost

| Stage | val nDCG@10 | Δ vs previous |
|-------|-------------|---------------|
| BM25 | 0.8641 | — |
| General dense (BGE-base) | 0.8906 | +0.0265 |
| Hybrid (Score interp α=0.4) | 0.9135 | +0.0229 |
| Rerank (top-20, MiniLM-L12) | 0.8703 | −0.0432 |

The largest single gain was **hybrid fusion** (+0.0229 over dense alone). It combines BM25's exact-match recall with dense semantic recall, recovering documents missed by either method alone. Reranking with the cross-encoder actually hurt here — the MiniLM-L12 ms-marco model is trained on general web queries and re-scores the top-20 away from the gold documents in this scientific domain. The latency cost of reranking was **222 ms/query** (cross-encoder over top-20 docs per claim).

---

## Q3 — Hard Negatives: Did They Help? Any Overfitting?

| System | val nDCG@10 |
|--------|-------------|
| Base BGE-base (no fine-tuning) | 0.8906 |
| Fine-tuned: random negatives | 0.9107 |
| Fine-tuned: hard negatives | 0.9150 |

Both fine-tuning runs improved over the base model (+0.0201 random, +0.0244 hard). Hard negatives outperformed random negatives by +0.0043, confirming that mining near-miss documents from BM25/dense rank 5–50 creates more informative training signal. Fine-tuning ran for 5 epochs with `load_best_model_at_end=True`; per-epoch val nDCG was logged by `InformationRetrievalEvaluator` in `checkpoints/*/eval_results.csv`. The best checkpoint was selected automatically on val nDCG@10.

---

## Q4 — Fraction of Verification Errors that are retrieval\_miss

From `predictions/t5_val_errors.csv`:
retrieval\_miss = **1** / **144** = **~1%**

Full error breakdown:

| Category | Count | % |
|----------|-------|---|
| other | 91 | 63% |
| numerical | 33 | 23% |
| needs\_multiple\_docs | 12 | 8% |
| negation | 7 | 5% |
| retrieval\_miss | 1 | 1% |

Retrieval quality is strong (only 1 miss). The dominant failure mode is **NLI reasoning errors** on "other" claims — the DeBERTa-v3-small model struggles with complex scientific inference even when the correct document is retrieved. Numerical and negation reasoning are also hard for zero-shot NLI.

---

## Q5 — Single Next Change

**Fine-tune the NLI model on SciFact training labels.** The current NLI step is zero-shot (DeBERTa-v3-small pretrained on MNLI/NLI benchmarks). With 93% of errors being NLI failures (not retrieval misses), the highest-leverage improvement is supervised NLI fine-tuning on SciFact's 440 labelled (claim, abstract, label) pairs using the training split. This directly addresses the dominant error categories (other, numerical, negation) without any additional retrieval cost and stays well within the ≤400M encoder constraint.
