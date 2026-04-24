"""
Experiment: Question Diversity Assessment

For each combination of (LLM model, dataset, sample size), this script:
  1. Loads N_DOCS documents from a HuggingFace dataset.
  2. Generates N_QUESTIONS yes/no questions via ManyQuestions (Ollama LLM).
  3. Answers every question across every document using NLI (inside ManyQuestions).
  4. Measures question diversity with four metrics:
       - Lexical  : average pairwise Jaccard n-gram similarity (mean ± std)
       - Semantic : average pairwise cosine similarity (mean ± std)
       - Logical  : average pairwise NLI entailment similarity (mean ± std)
       - Functional: average pairwise NMI on binary NLI answer patterns (mean ± std)

Low scores on all four metrics = diverse, non-redundant questions.
High functional similarity = questions that partition the collection the same way.

Results are printed as a table and saved to OUTPUT_CSV.

Usage:
    python examples/experiment_question_diversity.py
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

RESULTS_DIR = Path("/mnt/data3/askme_classification_results")
RESULTS_DIR.mkdir(exist_ok=True)

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

OLLAMA_MODELS = [
    "llama3.1:8b",
    #"qwen3:8b",
    #"qwen3:14b",
    #"gpt-oss:20b",
]

DATASETS = [
    {"name": "ag_news",         "hf_name": "ag_news",         "train_split": "train", "test_split": "test", "text_field": "text", "label_field": "label"},
    #{"name": "rotten_tomatoes", "hf_name": "rotten_tomatoes", "train_split": "train", "test_split": "test", "text_field": "text", "label_field": "label"},
]

SAMPLE_SIZES = [14]  # n_sample: texts fed to the LLM per run

N_QUESTIONS = 20     # yes/no questions to generate per run
N_TEST_DOCS = None   # documents loaded from the test split (None = all)
MAX_TRAIN_DOCS = 120_000  # ceiling on the train split to load

# Same fractions as experiment_classification.py — N_DOCS_LIST is derived
# from these after the train split is loaded.
TRAIN_FRACS = [0.001, 0.01, 0.1, 1.00]
OUTPUT_CSV = RESULTS_DIR / "experiment_question_diversity.csv"
OUTPUT_LOG = RESULTS_DIR / "experiment_question_diversity.log"

# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_texts(hf_name: str, split: str, text_field: str, n_docs: int, label_field: str | None = None):
    ds = load_dataset(hf_name, split=split)
    n = min(n_docs, len(ds))
    texts = [ds[i][text_field] for i in range(n)]
    labels = [ds[i][label_field] for i in range(n)] if label_field else [None] * n
    return texts, labels


def export_feature_parquet(question_answers, labels, output_path):
    rows = {}
    for qa in question_answers:
        for doc_ans in qa.answers:
            rows.setdefault(doc_ans.document_index, {
                "doc_index": doc_ans.document_index,
                "label": labels[doc_ans.document_index],
            })
    for qi, qa in enumerate(question_answers):
        for doc_ans in qa.answers:
            rows[doc_ans.document_index][f"q{qi}_ent"] = round(doc_ans.entailment_score, 6)
            rows[doc_ans.document_index][f"q{qi}_neu"] = round(doc_ans.neutral_score, 6)
            rows[doc_ans.document_index][f"q{qi}_con"] = round(doc_ans.contradiction_score, 6)
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
        logger.info("Loading '%s' train split (up to %d docs)…", dataset_cfg["name"], MAX_TRAIN_DOCS)
        full_collection, full_labels = load_texts(
            dataset_cfg["hf_name"],
            dataset_cfg["train_split"],
            dataset_cfg["text_field"],
            MAX_TRAIN_DOCS,
            label_field=dataset_cfg.get("label_field"),
        )
        n_total = len(full_collection)
        N_DOCS_LIST = sorted(set(max(1, int(f * n_total)) for f in TRAIN_FRACS))
        # Shuffle so each slice[:n_docs] is a random subset, not the first N
        _rng = np.random.default_rng(42)
        _idx = _rng.permutation(len(full_collection))
        full_collection = [full_collection[i] for i in _idx]
        full_labels     = [full_labels[i]     for i in _idx]

        logger.info("Loading '%s' test split…", dataset_cfg["name"])
        test_collection, test_labels = load_texts(
            dataset_cfg["hf_name"],
            dataset_cfg["test_split"],
            dataset_cfg["text_field"],
            N_TEST_DOCS or 999_999,
            label_field=dataset_cfg.get("label_field"),
        )

        for model_name in OLLAMA_MODELS:
            llm_config = config_factory(MakeQuestionsConfig)
            llm_config.model_name = model_name

            for n_sample in SAMPLE_SIZES:
                for n_docs in N_DOCS_LIST:
                    if n_docs < n_sample:
                        continue
                    collection = full_collection[:n_docs]
                    labels     = full_labels[:n_docs]

                    label = f"{dataset_cfg['name']} / {model_name} / n_sample={n_sample} / n_docs={n_docs}"
                    logger.info("Running: %s", label)

                    try:
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
                        "lexical": round(sim.lexical.mean, 4) if sim.lexical else "",
                        "lex_k": sim.lexical.n_clusters if sim.lexical else "",
                        "semantic": round(sim.semantic.mean, 4) if sim.semantic else "",
                        "sem_k": sim.semantic.n_clusters if sim.semantic else "",
                        "logical": round(sim.logical.mean, 4) if sim.logical else "",
                        "log_k": sim.logical.n_clusters if sim.logical else "",
                        "functional": round(sim.functional.mean, 4) if sim.functional else "",
                        "fun_k": sim.functional.n_clusters if sim.functional else "",
                    }
                    results.append(rec)
                    _row(rec)

                    stem = f"{dataset_cfg['name']}_{model_name}_{n_sample}_{n_docs}"
                    train_csv = OUTPUT_CSV.parent / f"features_train_{stem}.parquet"
                    export_feature_parquet(result.question_answers, labels, train_csv)
                    logger.info("Train feature matrix saved to %s", train_csv)

                    logger.info("Running NLI on test split (%d docs)…", len(test_collection))
                    test_qa = pipeline.collection_answerer(test_collection, result.questions)
                    test_csv = OUTPUT_CSV.parent / f"features_test_{stem}.parquet"
                    export_feature_parquet(test_qa, test_labels, test_csv)
                    logger.info("Test feature matrix saved to %s", test_csv)

                    logger.info(
                        "Similarity [%s]: lexical=%.4f±%.4f(k=%d)  semantic=%.4f±%.4f(k=%d)  "
                        "logical=%.4f±%.4f(k=%d)  functional=%.4f±%.4f(k=%d)",
                        label,
                        sim.lexical.mean, sim.lexical.std, sim.lexical.n_clusters,
                        sim.semantic.mean, sim.semantic.std, sim.semantic.n_clusters,
                        sim.logical.mean, sim.logical.std, sim.logical.n_clusters,
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
