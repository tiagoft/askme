"""
Experiment: Question Diversity Assessment

For each combination of (LLM model, dataset, sample size), this script:
  1. Loads N_DOCS documents from a HuggingFace dataset.
  2. Generates N_QUESTIONS yes/no questions via ManyQuestions (Ollama LLM).
  3. Answers every question across every document using NLI (inside ManyQuestions).
  4. Measures question diversity with four metrics:
       - Lexical  : average pairwise Jaccard n-gram similarity
       - Semantic : average pairwise cosine similarity (sentence embeddings)
       - Logical  : average pairwise NLI entailment similarity
       - Functional: average pairwise NMI on binary NLI answer patterns

Low scores on all four metrics = diverse, non-redundant questions.
High functional similarity = questions that partition the collection the same way.

Results are printed as a table and saved to OUTPUT_CSV.

Usage:
    python examples/experiment_question_diversity.py
"""

import sys
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from datasets import load_dataset

from askme.manyquestions import ManyQuestions
from askme.config.config import config_factory, MakeQuestionsConfig
from evalsim.similarities import SimilarityCalculator
from evalsim.functional_similarity import pairwise_functional_similarity

# ---------------------------------------------------------------------------
# Experiment configuration — edit these variables to customise the sweep
# ---------------------------------------------------------------------------

OLLAMA_MODELS = [
    "llama3.2",
    "gemma3:4b",
    "mistral",
]

DATASETS = [
    {"name": "ag_news",          "hf_name": "ag_news",          "split": "test",  "text_field": "text"},
    {"name": "rotten_tomatoes",  "hf_name": "rotten_tomatoes",  "split": "test",  "text_field": "text"},
]

SAMPLE_SIZES = [5, 10, 20]   # n_sample: texts fed to the LLM per run

N_QUESTIONS = 20             # yes/no questions to generate per run
N_DOCS = 200                 # documents loaded from each dataset
OUTPUT_CSV = Path(__file__).parent / "experiment_question_diversity.csv"

# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_texts(hf_name: str, split: str, text_field: str, n_docs: int) -> list[str]:
    ds = load_dataset(hf_name, split=split)
    n = min(n_docs, len(ds))
    return [ds[i][text_field] for i in range(n)]

# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_question_similarity(
    questions: list[str],
    sc: SimilarityCalculator,
) -> dict[str, float]:
    """Lexical, semantic, and logical similarity of the question list."""
    sim = sc(questions)
    return {
        "lexical": round(float(sim.lexical), 4),
        "semantic": round(float(sim.semantic), 4),
        "logical": round(float(sim.logical), 4),
    }


def compute_functional_similarity(question_answers) -> float:
    """NMI-based functional similarity reusing pre-computed NLI answers."""
    scores = np.array([
        [a.P_entailment_binary for a in qa.answers]
        for qa in question_answers
    ])
    matrix = pairwise_functional_similarity(scores)
    upper = matrix[np.triu_indices(len(matrix), k=1)]
    return round(float(upper.mean()) if len(upper) > 0 else 0.0, 4)

# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

COLS = ["dataset", "model", "n_sample", "n_questions",
        "lexical", "semantic", "logical", "functional"]
COL_W = [18, 16, 8, 11, 8, 9, 8, 11]


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

    # Build the question-similarity calculator once (loads NLI + embedder).
    print("\nLoading similarity models (NLI + sentence embedder)…")
    sc = SimilarityCalculator(
        use_lexical=True,
        use_semantic=True,
        use_logical=True,
        use_functional=False,  # functional is computed from pre-computed NLI scores
    )

    results = []
    _header()

    for dataset_cfg in DATASETS:
        print(f"\nLoading dataset '{dataset_cfg['name']}' ({N_DOCS} docs)…")
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
                print(f"\n  Running: {label}")

                try:
                    pipeline = ManyQuestions(
                        n_sample=n_sample,
                        n_questions=N_QUESTIONS,
                        use_gpu=False,
                        llm_config=llm_config,
                    )
                    result = pipeline(collection)
                except Exception as exc:
                    print(f"  ERROR: {exc}")
                    rec = {
                        "dataset": dataset_cfg["name"],
                        "model": model_name,
                        "n_sample": n_sample,
                        "n_questions": N_QUESTIONS,
                        "lexical": "ERR",
                        "semantic": "ERR",
                        "logical": "ERR",
                        "functional": "ERR",
                    }
                    results.append(rec)
                    _row(rec)
                    continue

                q_sim = compute_question_similarity(result.questions, sc)
                func_sim = compute_functional_similarity(result.question_answers)

                rec = {
                    "dataset": dataset_cfg["name"],
                    "model": model_name,
                    "n_sample": n_sample,
                    "n_questions": len(result.questions),
                    **q_sim,
                    "functional": func_sim,
                }
                results.append(rec)
                _row(rec)

    # Save to CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLS)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {OUTPUT_CSV}")
    print("\nDone.")


if __name__ == "__main__":
    run_experiment()
