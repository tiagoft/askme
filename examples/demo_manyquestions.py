"""
Demo: ManyQuestions on AG News (HuggingFace)

Loads a slice of the AG News dataset, generates yes/no questions that
characterise the collection, then answers every question across every
document using NLI.

Usage:
    python examples/demo_manyquestions.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datasets import load_dataset

from askme.manyquestions import ManyQuestions

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_NAME = "ag_news"
N_DOCS = 100                 # number of documents to load
N_SAMPLE = 12                # texts sampled for the LLM prompt
N_QUESTIONS = 6              # yes/no questions to generate
TOP_K_DOCS = 5               # documents to show per question in the summary


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_ag_news(n_docs: int) -> list[str]:
    """Return the first n_docs texts from the AG News test split."""
    dataset = load_dataset(DATASET_NAME, split="test")
    return [item["text"] for item in dataset.select(range(n_docs))]


# ---------------------------------------------------------------------------
# Pretty printing helpers
# ---------------------------------------------------------------------------

def print_section(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def print_question_summary(question_answers, texts, top_k):
    for i, qa in enumerate(question_answers, 1):
        print(f"\nQ{i}: {qa.question}")
        print("-" * 60)

        sorted_answers = sorted(
            qa.answers, key=lambda a: a.P_entailment_binary, reverse=True
        )
        print(f"  Top-{top_k} most entailed documents:")
        for rank, answer in enumerate(sorted_answers[:top_k], 1):
            idx = answer.document_index
            snippet = texts[idx][:90].replace("\n", " ")
            print(
                f"    {rank}. [score={answer.P_entailment_binary:.2f}] "
                f"[doc {idx:3d}] {snippet}…"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print_section("ManyQuestions Demo — AG News")

    # 1. Load dataset
    print(f"\nLoading '{DATASET_NAME}' ({N_DOCS} documents)…")
    texts = load_ag_news(N_DOCS)
    print(f"Loaded {len(texts)} documents.")

    # 2. Run ManyQuestions pipeline
    print_section("Running ManyQuestions pipeline")
    print(f"  n_sample    = {N_SAMPLE}  (texts fed to the LLM)")
    print(f"  n_questions = {N_QUESTIONS}  (yes/no questions to generate)")
    print()

    pipeline = ManyQuestions(
        n_sample=N_SAMPLE,
        n_questions=N_QUESTIONS,
        use_gpu=False,
    )
    result = pipeline(texts)

    # 3. Show generated questions
    print_section("Generated questions")
    for i, q in enumerate(result.questions, 1):
        print(f"  {i}. {q}")

    # 4. Show top-K most entailed documents per question
    print_section(f"Top-{TOP_K_DOCS} most entailed documents per question")
    print_question_summary(result.question_answers, texts, TOP_K_DOCS)

    # 5. Show which texts were sampled for question generation
    print_section("Texts used to generate questions (sampled indices)")
    for idx in result.sampled_indices:
        snippet = texts[idx][:80].replace("\n", " ")
        print(f"  [{idx:3d}] {snippet}…")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
