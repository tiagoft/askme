"""
Experiment: Question Diversity Assessment and NLI Feature Matrix Generation

Objective
---------
Given a text collection, we want to know whether an LLM can generate a small set
of yes/no questions that collectively act as a semantic fingerprint of every document.
Each question is answered by an NLI model for every document, yielding a
(n_docs × 3·N_QUESTIONS) feature matrix of (entailment, neutral, contradiction)
probabilities. This matrix is the input to the downstream classification experiment.

Before saving the feature matrices, we also measure how *diverse* the generated
questions are, using four pairwise similarity metrics:
  - Lexical   : Jaccard n-gram overlap — detects surface-level duplicates
  - Semantic  : cosine similarity of sentence embeddings — detects paraphrase
  - Logical   : NLI-based entailment similarity — detects equivalent meaning
  - Functional: NMI on binary NLI answer patterns — detects questions that
                partition the collection identically (the most informative metric)

HDBSCAN cluster counts (*_k) complement the means: k=0 means no tight redundant
groups exist; k>0 reveals how many clusters of near-duplicate questions were found.

Low similarity scores + low k = diverse, informative questions → better features.

Ablation design
---------------
TRAIN_FRACS controls the question-generation pool size (n_docs).  At each fraction
we run the full pipeline independently, generating different questions from a
differently-sized pool.  This lets us answer: "does access to more documents during
question generation improve downstream classification?"

The same TRAIN_FRACS list is used in experiment_classification.py when subsampling
the training set for the learning curves, so pool size and classifier training size
stay aligned across the two experiments.

Outputs (→ RESULTS_DIR)
-----------------------
  features_train_{dataset}_{model}_{n_sample}_{n_docs}.parquet
  features_test_{dataset}_{model}_{n_sample}_{n_docs}.parquet
  experiment_question_diversity.csv   (diversity metrics table)
  experiment_question_diversity.log   (full run log with generated questions)

Usage
-----
  python experiments/classification/experiment_question_diversity.py
"""

import csv
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import numpy as np
import pandas as pd
from datasets import load_dataset

from askme.manyquestions import ManyQuestions
from askme.config.config import config_factory, MakeQuestionsConfig
from evalsim.similarities import SimilarityCalculator

# All outputs go to an external disk to avoid filling the main drive.
RESULTS_DIR = Path("/mnt/data3/askme_classification_results")
RESULTS_DIR.mkdir(exist_ok=True)

# Log to both console and file so long overnight runs are fully recoverable.
_formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s — %(message)s", datefmt="%H:%M:%S")
_file_handler = logging.FileHandler(RESULTS_DIR / "experiment_question_diversity.log", encoding="utf-8")
_file_handler.setFormatter(_formatter)
_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(_formatter)
logging.root.setLevel(logging.INFO)
logging.root.addHandler(_file_handler)
logging.root.addHandler(_stream_handler)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Experiment configuration — edit these variables to customise the sweep
# ---------------------------------------------------------------------------

# LLMs available via Ollama.  Uncomment models to include them in the sweep.
OLLAMA_MODELS = [
    "llama3.1:8b",
    #"qwen3:8b",
    #"qwen3:14b",
    #"gpt-oss:20b",
]

# Each entry describes one dataset.  HuggingFace datasets use hf_name; local
# JSONL datasets (wikipedia, bills) supply data_files paths instead.
DATASETS = [
    {"name": "ag_news",         "hf_name": "ag_news",              "train_split": "train", "test_split": "test", "text_field": "text",    "label_field": "label"},
    {"name": "20_newsgroups",   "hf_name": "SetFit/20_newsgroups", "train_split": "train", "test_split": "test", "text_field": "text",    "label_field": "label"},
    {"name": "wikipedia",       "hf_name": "json",                 "train_split": "train", "test_split": "test", "text_field": "text",    "label_field": "supercategory",
     "data_files": {"train": "/mnt/data3/nlp_datasets/wikipedia/train.metadata.jsonl",
                    "test":  "/mnt/data3/nlp_datasets/wikipedia/test.metadata.jsonl"}},
    {"name": "bills",           "hf_name": "json",                 "train_split": "train", "test_split": "test", "text_field": "summary", "label_field": "topic",
     "data_files": {"train": "/mnt/data3/nlp_datasets/bills/train.metadata.jsonl",
                    "test":  "/mnt/data3/nlp_datasets/bills/test.metadata.jsonl"}},
    #{"name": "rotten_tomatoes", "hf_name": "rotten_tomatoes",     "train_split": "train", "test_split": "test", "text_field": "text",    "label_field": "label"},
]

# Number of documents shown to the LLM for question generation.
# Kept small so the prompt fits in context and the LLM can reason about all texts.
SAMPLE_SIZES = [14]

N_QUESTIONS = 20        # questions generated per run (LLM is instructed to produce exactly this many)
N_TEST_DOCS = None      # None = use the entire test split
MAX_TRAIN_DOCS = 120_000  # ceiling on train docs loaded; the dataset is shuffled before slicing

# Pool-size fractions for the ablation.  Each fraction f yields
# n_docs = int(f * total_train_size) documents available for question generation.
# Using the same list in experiment_classification.py keeps the two experiments aligned.
TRAIN_FRACS = [0.001, 0.01, 0.1, 1.00]

OUTPUT_CSV = RESULTS_DIR / "experiment_question_diversity.csv"
OUTPUT_LOG = RESULTS_DIR / "experiment_question_diversity.log"

# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_texts(dataset_cfg: dict, split: str, n_docs: int, label_map: dict | None = None):
    """Load texts and integer labels from a dataset config.

    Handles both HuggingFace hub datasets and local JSONL files transparently.
    For datasets with string labels (wikipedia: supercategory, bills: topic),
    encodes them via a sorted mapping so labels are integers and reproducible.
    Pass the label_map returned from the train split when loading the test split
    to guarantee identical encoding across both splits.

    Returns (texts, int_labels, label_map).
    """
    if "data_files" in dataset_cfg:
        # Local JSONL: load_dataset('json') always uses split='train' internally.
        ds = load_dataset("json", data_files=[dataset_cfg["data_files"][split]], split="train")
    else:
        ds = load_dataset(dataset_cfg["hf_name"], split=split)

    n = min(n_docs, len(ds))
    text_field  = dataset_cfg["text_field"]
    label_field = dataset_cfg.get("label_field")

    texts = [ds[i][text_field] for i in range(n)]

    if not label_field:
        return texts, [None] * n, None

    raw_labels = [ds[i][label_field] for i in range(n)]
    if raw_labels and isinstance(raw_labels[0], str):
        # Build a deterministic int encoding from sorted unique label strings.
        if label_map is None:
            label_map = {lbl: idx for idx, lbl in enumerate(sorted(set(raw_labels)))}
        labels = [label_map[l] for l in raw_labels]
    else:
        labels = [int(l) for l in raw_labels]

    return texts, labels, label_map


# ---------------------------------------------------------------------------
# Feature matrix export
# ---------------------------------------------------------------------------

def export_feature_parquet(question_answers, labels, output_path, orig_idx: list[int] | None = None):
    """Write NLI feature matrix to parquet.

    Each row is one document.  Columns:
      doc_index  — position in the (shuffled) collection passed to NLI
      orig_idx   — original row index in the dataset before any shuffling;
                   used by experiment_classification.py to reload the correct text
      label      — integer class label
      q{i}_ent, q{i}_neu, q{i}_con — NLI ternary probabilities for question i

    orig_idx is None for the test split (which is never shuffled), so doc_index
    and orig_idx coincide and we store doc_index in both columns for consistency.
    """
    rows = {}
    for qa in question_answers:
        for doc_ans in qa.answers:
            j = doc_ans.document_index
            rows.setdefault(j, {
                "doc_index": j,
                "orig_idx":  orig_idx[j] if orig_idx is not None else j,
                "label":     labels[j],
            })
    for qi, qa in enumerate(question_answers):
        for doc_ans in qa.answers:
            j = doc_ans.document_index
            rows[j][f"q{qi}_ent"] = round(doc_ans.entailment_score, 6)
            rows[j][f"q{qi}_neu"] = round(doc_ans.neutral_score, 6)
            rows[j][f"q{qi}_con"] = round(doc_ans.contradiction_score, 6)
    pd.DataFrame(rows[i] for i in sorted(rows)).to_parquet(output_path, index=False)


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

COLS = ["dataset", "model", "n_sample", "n_docs", "n_questions",
        "lexical", "lex_k", "semantic", "sem_k", "logical", "log_k", "functional", "fun_k"]
COL_W = [18, 16, 8, 8, 11, 8, 6, 9, 6, 8, 6, 11, 6]


def _header():
    print()
    print("  ".join(c.ljust(w) for c, w in zip(COLS, COL_W)))
    print("  ".join("-" * w for w in COL_W))


def _row(rec: dict):
    vals = [str(rec.get(c, "")).ljust(w) for c, w in zip(COLS, COL_W)]
    print("  ".join(vals))


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_experiment():
    print("=" * 70)
    print("Question Diversity Experiment")
    print("=" * 70)
    print(f"Models      : {OLLAMA_MODELS}")
    print(f"Datasets    : {[d['name'] for d in DATASETS]}")
    print(f"Sample sizes: {SAMPLE_SIZES}")
    print(f"N questions : {N_QUESTIONS}  |  Train fracs: {TRAIN_FRACS}  |  Max train docs: {MAX_TRAIN_DOCS}")

    # Load similarity models once — reused across all runs.
    # NLI and sentence embedder are expensive to load; keeping them alive
    # avoids re-initialisation overhead at every (model, dataset, n_docs) step.
    print("\nLoading similarity models (NLI + sentence embedder)…")
    sc = SimilarityCalculator(
        use_lexical=True,
        use_semantic=True,
        use_logical=True,
        use_functional=True,
    )

    results = []
    _header()

    for dataset_cfg in DATASETS:
        # Load the full train split once and derive all n_docs slices from it.
        # This avoids repeated HuggingFace downloads across pool-size fractions.
        logger.info("Loading '%s' train split (up to %d docs)…", dataset_cfg["name"], MAX_TRAIN_DOCS)
        full_collection, full_labels, label_map = load_texts(
            dataset_cfg, dataset_cfg["train_split"], MAX_TRAIN_DOCS
        )
        n_total = len(full_collection)

        # Derive absolute pool sizes from TRAIN_FRACS applied to the actual dataset size.
        N_DOCS_LIST = sorted(set(max(1, int(f * n_total)) for f in TRAIN_FRACS))

        # Shuffle with a fixed seed so slices are random subsets rather than the
        # first N documents (which may be ordered by class in the original dataset).
        # _orig_idx[j] records which original dataset row ended up at shuffled position j,
        # so we can recover the correct text later in the classification experiment.
        _rng = np.random.default_rng(42)
        _orig_idx = _rng.permutation(n_total).tolist()
        full_collection = [full_collection[i] for i in _orig_idx]
        full_labels     = [full_labels[i]     for i in _orig_idx]

        # Test split is loaded once and reused for all (model, n_docs) combinations.
        # The same questions generated from the train pool are applied to the test split,
        # ensuring that test evaluation is always on genuinely unseen documents.
        logger.info("Loading '%s' test split…", dataset_cfg["name"])
        test_collection, test_labels, _ = load_texts(
            dataset_cfg, dataset_cfg["test_split"], N_TEST_DOCS or 999_999, label_map=label_map
        )

        for model_name in OLLAMA_MODELS:
            llm_config = config_factory(MakeQuestionsConfig)
            llm_config.model_name = model_name

            for n_sample in SAMPLE_SIZES:
                for n_docs in N_DOCS_LIST:
                    if n_docs < n_sample:
                        # Cannot sample more documents than are available in the pool.
                        continue

                    collection = full_collection[:n_docs]
                    labels     = full_labels[:n_docs]

                    label = f"{dataset_cfg['name']} / {model_name} / n_sample={n_sample} / n_docs={n_docs}"
                    logger.info("Running: %s", label)

                    try:
                        # ManyQuestions pipeline:
                        #   1. Embeds collection and builds a FAISS index.
                        #   2. Samples n_sample diverse documents via k-means.
                        #   3. Prompts the LLM to generate N_QUESTIONS yes/no questions.
                        #   4. Runs NLI on the full collection for each question.
                        pipeline = ManyQuestions(
                            n_sample=n_sample,
                            n_questions=N_QUESTIONS,
                            use_gpu=True,
                            llm_config=llm_config,
                        )
                        result = pipeline(collection)
                    except Exception as exc:
                        logger.error("FAILED [%s]: %s", label, exc)
                        rec = {
                            "dataset": dataset_cfg["name"],
                            "model": model_name,
                            "n_sample": n_sample,
                            "n_docs": n_docs,
                            "n_questions": N_QUESTIONS,
                            "lexical": "ERR", "lex_k": "",
                            "semantic": "ERR", "sem_k": "",
                            "logical": "ERR", "log_k": "",
                            "functional": "ERR", "fun_k": "",
                        }
                        results.append(rec)
                        _row(rec)
                        continue

                    logger.info(
                        "Questions [%s]:\n%s",
                        label,
                        "\n".join(f"  {i + 1}. {q}" for i, q in enumerate(result.questions)),
                    )

                    # Functional similarity uses binary NLI answers (P_entailment ≥ 0.5)
                    # to compute pairwise NMI between questions.  Two questions with
                    # identical binary answer patterns carry no additional information
                    # about the collection beyond what the other already provides.
                    functional_scores = np.array([
                        [a.P_entailment_binary for a in qa.answers]
                        for qa in result.question_answers
                    ])
                    sim = sc.calculate_similarity(result.questions, functional_scores=functional_scores)

                    rec = {
                        "dataset": dataset_cfg["name"],
                        "model": model_name,
                        "n_sample": n_sample,
                        "n_docs": n_docs,
                        "n_questions": len(result.questions),
                        # mean pairwise similarity (lower = more diverse)
                        "lexical":    round(sim.lexical.mean,    4) if sim.lexical    else "",
                        "lex_k":      sim.lexical.n_clusters          if sim.lexical    else "",
                        "semantic":   round(sim.semantic.mean,   4) if sim.semantic   else "",
                        "sem_k":      sim.semantic.n_clusters         if sim.semantic   else "",
                        "logical":    round(sim.logical.mean,    4) if sim.logical    else "",
                        "log_k":      sim.logical.n_clusters          if sim.logical    else "",
                        "functional": round(sim.functional.mean, 4) if sim.functional else "",
                        "fun_k":      sim.functional.n_clusters       if sim.functional else "",
                    }
                    results.append(rec)
                    _row(rec)

                    # Save train feature matrix.  orig_idx records the original dataset
                    # row for each shuffled position so texts can be reloaded correctly.
                    stem = f"{dataset_cfg['name']}_{model_name}_{n_sample}_{n_docs}"
                    train_csv = OUTPUT_CSV.parent / f"features_train_{stem}.parquet"
                    export_feature_parquet(result.question_answers, labels, train_csv,
                                           orig_idx=_orig_idx[:n_docs])
                    logger.info("Train feature matrix saved to %s", train_csv)

                    # Score the test split with the same questions generated above.
                    # The test split is never used during question generation, ensuring
                    # a fair evaluation with no information leakage.
                    logger.info("Running NLI on test split (%d docs)…", len(test_collection))
                    test_qa = pipeline.collection_answerer(test_collection, result.questions)
                    test_csv = OUTPUT_CSV.parent / f"features_test_{stem}.parquet"
                    export_feature_parquet(test_qa, test_labels, test_csv)  # no shuffle on test
                    logger.info("Test feature matrix saved to %s", test_csv)

                    logger.info(
                        "Similarity [%s]: lexical=%.4f±%.4f(k=%d)  semantic=%.4f±%.4f(k=%d)  "
                        "logical=%.4f±%.4f(k=%d)  functional=%.4f±%.4f(k=%d)",
                        label,
                        sim.lexical.mean,    sim.lexical.std,    sim.lexical.n_clusters,
                        sim.semantic.mean,   sim.semantic.std,   sim.semantic.n_clusters,
                        sim.logical.mean,    sim.logical.std,    sim.logical.n_clusters,
                        sim.functional.mean, sim.functional.std, sim.functional.n_clusters,
                    )

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLS)
        writer.writeheader()
        writer.writerows(results)

    logger.info("Results saved to %s", OUTPUT_CSV)
    print("\nDone.")


if __name__ == "__main__":
    run_experiment()
