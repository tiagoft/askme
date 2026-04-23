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

import logging
import sys
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from datasets import load_dataset

from askme.manyquestions import ManyQuestions
from askme.config.config import config_factory, MakeQuestionsConfig
from evalsim.similarities import SimilarityCalculator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Experiment configuration — edit these variables to customise the sweep
# ---------------------------------------------------------------------------

OLLAMA_MODELS = [
    #"llama3.1:8b",
    "qwen3:14b",
    #"gpt-oss:20b",
]

DATASETS = [
    {"name": "ag_news",          "hf_name": "ag_news",          "split": "test",  "text_field": "text"},
    #{"name": "rotten_tomatoes",  "hf_name": "rotten_tomatoes",  "split": "test",  "text_field": "text"},
]

SAMPLE_SIZES = [5, 10, 20, 30,]   # n_sample: texts fed to the LLM per run

N_QUESTIONS = 50             # yes/no questions to generate per run
N_DOCS = 10000               # documents loaded from each dataset
OUTPUT_CSV = Path(__file__).parent / "experiment_question_diversity.csv"

# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_texts(hf_name: str, split: str, text_field: str, n_docs: int) -> list[str]:
    ds = load_dataset(hf_name, split=split)
    n = min(n_docs, len(ds))
    return [ds[i][text_field] for i in range(n)]

# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

COLS = ["dataset", "model", "n_sample", "n_questions",
        "lexical", "lex_k", "semantic", "sem_k", "logical", "log_k", "functional", "fun_k"]
COL_W = [18, 16, 8, 11, 8, 6, 9, 6, 8, 6, 11, 6]


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
    print(f"N questions : {N_QUESTIONS}  |  N docs: {N_DOCS}")

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
        logger.info("Loading dataset '%s' (%d docs)…", dataset_cfg["name"], N_DOCS)
        collection = load_texts(
            dataset_cfg["hf_name"],
            dataset_cfg["split"],
            dataset_cfg["text_field"],
            N_DOCS,
        )

        for model_name in OLLAMA_MODELS:
            llm_config = config_factory(MakeQuestionsConfig)
            llm_config.model_name = model_name

            for n_sample in SAMPLE_SIZES:
                label = f"{dataset_cfg['name']} / {model_name} / n_sample={n_sample}"
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
