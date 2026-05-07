# Classification Experiments

This directory contains the experiments for the paper:

> **Zero-Shot Feature Extraction via LLM-Generated Questions for Low-Resource Text Classification**

The core idea: instead of fine-tuning a language model on labeled data, we ask an LLM to generate yes/no questions about a text collection, then use NLI to score every document against every question. The resulting score vectors (entailment / neutral / contradiction per question) serve as compact, semantically rich features for a downstream classifier. We show this outperforms TF-IDF and converges much faster than BERT fine-tuning in low-data regimes.

---

## Directory layout

```
experiments/classification/
├── experiment_question_diversity.py   # Step 1: generate questions, score documents, measure diversity
├── experiment_classification.py       # Step 2: train classifiers, plot learning curves
├── finetune.py                        # Auxiliary: BERT fine-tuning baseline (HuggingFace Trainer)
└── results/                           # All outputs land here (→ /mnt/data3/askme_classification_results)
```

---

## How to run

### Step 1 — generate NLI feature matrices

```bash
python experiments/classification/experiment_question_diversity.py
```

This script loops over every combination of `(dataset, LLM model, n_sample, n_docs)` and:

1. Loads the **train split** of each dataset (up to `MAX_TRAIN_DOCS`, shuffled with seed 42 for reproducibility).
2. Samples `n_sample` representative documents using a FAISS-based diversity sampler.
3. Prompts the LLM to generate exactly `N_QUESTIONS` yes/no questions about the sample.
4. Scores **every document** in the train split against every question using NLI, producing an `(n_docs × 3·N_QUESTIONS)` feature matrix of (entailment, neutral, contradiction) probabilities.
5. Does the same for the **test split** using the same questions.
6. Measures question diversity via four pairwise metrics (lexical, semantic, logical, functional NMI) and reports HDBSCAN cluster counts to detect redundant question groups.

**Outputs** (all in `results/`):

| File | Description |
|------|-------------|
| `features_train_{dataset}_{model}_{n_sample}_{n_docs}.parquet` | Train NLI feature matrix |
| `features_test_{dataset}_{model}_{n_sample}_{n_docs}.parquet`  | Test NLI feature matrix |
| `experiment_question_diversity.csv` | Diversity metrics table (one row per run) |
| `experiment_question_diversity.log` | Full log including generated questions |

**Key config variables** (top of file):

| Variable | Default | Meaning |
|----------|---------|---------|
| `OLLAMA_MODELS` | `["llama3.1:8b"]` | LLMs to sweep |
| `DATASETS` | ag_news + 3 others | Datasets to sweep |
| `SAMPLE_SIZES` | `[14]` | Docs shown to LLM for question generation |
| `N_QUESTIONS` | `20` | Questions generated per run |
| `TRAIN_FRACS` | `[0.001, 0.01, 0.1, 1.0]` | Pool-size fractions (ablation) |
| `MAX_TRAIN_DOCS` | `120 000` | Upper bound on train docs loaded |

---

### Step 2 — train classifiers and plot learning curves

```bash
python experiments/classification/experiment_classification.py
```

#### How it works internally

**Auto-discovery.** `find_feature_pairs()` scans `results/` for
`features_train_*.parquet` files and matches each one with its corresponding
`features_test_*.parquet`.  Running Step 1 with a new model or dataset
automatically makes Step 2 pick it up — no manual registration needed.

**Feature loading.** `load_feature_matrix()` reads a parquet file and returns
`(X, labels, orig_indices)`.  `orig_indices` are the original dataset row
numbers recorded during Step 1 to undo the training-set shuffle; they are
passed to `load_texts()` to fetch the exact same documents in the correct order
for TF-IDF and fine-tuning.

**Three classifiers, identical splits.** For every parquet pair the script runs:

| Method | How | Note |
|--------|-----|------|
| **NLI + RF** | `RandomForestClassifier` on the `(n_docs × 3·N_QUESTIONS)` matrix | No text seen at classifier-training time |
| **TF-IDF + RF** | `TfidfVectorizer` (max 5 000 features) + `RandomForestClassifier` | Vectorizer fitted only on the training subsample |
| **BERT fine-tune** | `bert-base-uncased` via HuggingFace `Trainer` (see `finetune.py`) | Fresh pretrained weights at every training-size point |

All three use identical train/test indices so results are directly comparable.

**Learning curves.** `compute_learning_curves()` iterates over `TRAIN_FRACS`,
at each step subsampling `n_sub = max(num_labels, int(frac × n_train))` training
examples with stratified sampling (to ensure every class is represented even at
tiny sizes), then training all three classifiers and recording test accuracy.
The minimum-size clamp prevents stratified splitting from failing when a fraction
rounds to fewer examples than there are classes.

**Fine-tuning details** (`finetune.py`).  Each call to `finetune_and_evaluate()`
loads fresh pretrained weights from `bert-base-uncased` so that learning-curve
points are independent.  Training uses early stopping (patience = 3 epochs on
eval accuracy) to prevent overfitting at small training sizes — the key failure
mode we want to illustrate.  After each call the training/eval loss history is
saved as a PNG so you can see where train and eval loss diverge.

**Saved models.** After the full-training-set run (i.e., `frac = 1.0`), all
three trained models are serialised:
- RF models via `joblib.dump` (compact, fast to reload)
- BERT via `model.save_pretrained()` + `tokenizer.save_pretrained()` (native
  HuggingFace format, compatible with `from_pretrained()`)
- TF-IDF bundle: RF + fitted vectorizer in the same joblib dict, so the
  vectorizer is always available for transforming new texts

**Plot.** After all pairs are processed, `save_learning_curve_plot()` draws
all NLI curves as solid coloured lines (one colour per LLM), a single shared
TF-IDF curve in black dashed (TF-IDF is model-agnostic so all curves coincide),
and a single BERT curve in grey dotted.

This script auto-discovers all `features_train_*.parquet` / `features_test_*.parquet` pairs in `results/` and, for each pair:

1. Trains a **Random Forest on NLI features** (train set → test set accuracy).
2. Trains a **Random Forest on TF-IDF features** as the bag-of-words baseline.
3. Optionally fine-tunes **BERT** (`bert-base-uncased`) with early stopping as a strong supervised baseline.
4. Sweeps over `TRAIN_FRACS` fractions of the training set to produce **learning curves** — the central figure of the paper.

**Rationale for three baselines:**

- **TF-IDF + RF**: fast, interpretable, no language model — shows what surface statistics alone can do.
- **NLI features + RF**: our method — semantically grounded, zero-shot w.r.t. labels, compact (3·N_QUESTIONS features).
- **BERT fine-tuning**: upper bound of supervised learning; expected to fail in the very-low-data regime (< 100 examples) and converge slowly.

**Outputs** (all in `results/`):

| File | Description |
|------|-------------|
| `experiment_classification.csv` | Summary table: accuracy per method per run |
| `learning_curves.png` | Main figure: accuracy vs training set size for all methods and models |
| `loss_curves_finetune_{stem}.png` | Train/eval loss per epoch for the full-data BERT run |
| `lc_finetune_{stem}/loss_curves_n{k}.png` | Per-fraction BERT loss curves (diagnose overfitting) |
| `model_nli_{stem}.joblib` | Saved RF model (NLI features) |
| `model_tfidf_{stem}.joblib` | Saved RF + TF-IDF vectorizer |
| `model_finetune_{stem}/` | Saved BERT model + tokenizer (`save_pretrained` format) |

**Key config variables** (top of file):

| Variable | Default | Meaning |
|----------|---------|---------|
| `TRAIN_FRACS` | `[0.001, 0.01, 0.1, 1.0]` | Training set size fractions for learning curves |
| `USE_FINETUNE` | `True` | Whether to run the BERT baseline (slow — set `False` for quick iteration) |
| `N_TREES` | `100` | Random forest size |
| `TFIDF_MAX_FEATURES` | `5 000` | TF-IDF vocabulary ceiling |

---

## Supported datasets

| Name | Source | Classes | Notes |
|------|--------|---------|-------|
| `ag_news` | HuggingFace `ag_news` | 4 | News topic classification |
| `20_newsgroups` | HuggingFace `SetFit/20_newsgroups` | 20 | Newsgroup topic classification |
| `wikipedia` | Local `/mnt/data3/nlp_datasets/wikipedia/` | 15 | Wikipedia supercategory |
| `bills` | Local `/mnt/data3/nlp_datasets/bills/` | 21 | US congressional bill topics |

String labels (wikipedia, bills) are automatically encoded to integers using a sorted, reproducible mapping shared between train and test splits.

---

## Expected figures and tables

### Table 1 — Full-data accuracy
`experiment_classification.csv` — rows: one per (dataset, model, n_docs); columns: `acc_nli`, `acc_tfidf`, `acc_ft`, `delta`.

### Figure 1 — Learning curves (`learning_curves.png`)
X axis: training set size. Y axis: accuracy.
- Solid coloured lines: NLI features, one per LLM model.
- Black dashed: TF-IDF baseline (model-independent).
- Grey dotted: BERT fine-tuning.

**Expected story**: NLI curves start high (~80%) even with very few labels and plateau quickly. TF-IDF starts low (~50%) and converges slowly. BERT is near-random at < 100 examples, eventually surpasses TF-IDF but stays below NLI for most of the low-data regime.

### Figure 2 — Pool-size ablation
Plot `acc_nli` from `experiment_classification.csv` against `n_docs` (the question-generation pool size), holding `n_sample` fixed. Shows whether the FAISS diversity sampler's access to more documents improves question quality.

### Figure 3 — Fine-tuning loss curves (`loss_curves_finetune_*.png`)
Train vs eval loss per epoch. Diagnoses overfitting in the BERT baseline, especially at small training sizes.

### Appendix — Diversity metrics (`experiment_question_diversity.csv`)
Lexical / semantic / logical / functional similarity mean ± std and HDBSCAN cluster count (`*_k`) per run. Low values indicate diverse, non-redundant questions. The `fun_k` column (functional cluster count) is the best predictor of downstream classification quality.

---

## Dependencies

```
pip install transformers torch datasets scikit-learn sentence-transformers pandas pyarrow joblib matplotlib
```

Ollama must be running locally with the required models pulled (`ollama pull llama3.1:8b`, etc.).
All results are written to `/mnt/data3/askme_classification_results/` to avoid filling the main disk.
