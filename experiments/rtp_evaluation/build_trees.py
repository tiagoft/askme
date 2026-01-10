import sys
import os
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from askme.rtp import (
    RTPBuilder,
    RTPRecursion,
    tree_to_pdf,
)

from datasets import load_dataset
import numpy as np
from typing import List, Tuple


def load_agnews_sample(n_samples: int | None = 500,
                       seed: int = 42) -> Tuple[List[str], List[int]]:
    """
    Load a sample of n documents from the AG News dataset.
    
    Args:
        n_samples: Number of samples to load
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (texts, labels)
    """
    print(f"Loading {n_samples} samples from AG News dataset...")
    dataset = load_dataset('fancyzhx/ag_news', split='train')

    # Set random seed for reproducibility
    np.random.seed(seed)

    # Sample n_samples indices
    total_size = len(dataset)
    if n_samples is None:
        n_samples = total_size
    indices = np.random.choice(total_size,
                               size=min(n_samples, total_size),
                               replace=False)

    # Extract texts and labels
    texts = []
    labels = []
    for idx in indices:
        item = dataset[int(idx)]
        # Combine title and description for richer text
        text = item['text']
        texts.append(text)
        labels.append(item['label'])

    print(f"Loaded {len(texts)} documents")
    print(f"Label distribution: {np.bincount(labels)}")
    print(f"Classes: 0=World, 1=Sports, 2=Business, 3=Sci/Tech")

    return texts, labels


def run_rtp_evaluation(
    texts: List[str],
    labels: List[int],
    llm_model_name: str = 'qwen3:14b',
    n_documents_to_answer: int | float = 0.1,
    max_depth: int = 4,
    selection_strategy: str = 'kmeans',
):
    """
    Run RTP Recursion on the dataset and evaluate.
    
    Args:
        texts: List of text documents
        labels: Ground truth labels
    """
    print("\n" + "=" * 80)
    print("RTP RECURSION EVALUATION WITH MODEL:", llm_model_name)
    print("=" * 80)

    # Initialize RTPBuilder
    print("\nInitializing RTPBuilder...")
    builder = RTPBuilder(
        use_gpu=True,
        n_medoids=15,
        n_documents_to_answer=n_documents_to_answer,
        llm_model_name=llm_model_name,
        max_retries=10,
        min_split_ratio=0.1,
        max_split_ratio=0.9,
        alpha=1e-2,
        verbose=True,
        cache_dir='~/.askme_cache',
        selection_strategy=selection_strategy,
    )
    print("RTPBuilder initialized!")

    # Initialize RTPRecursion with stopping criteria
    print("\nInitializing RTPRecursion...")
    recursion = RTPRecursion(
        builder=builder,
        min_node_size=30,  # Don't split nodes with fewer than 5 documents
        min_split_ratio=0.1,  # Split should have at least 10% in smaller child
        max_split_ratio=0.9,  # Split should have at most 90% in larger child
        max_depth=max_depth,  # Maximum tree depth
    )
    print("RTPRecursion initialized!")

    # Build the recursive tree
    print("\nBuilding RTP recursive tree...")
    tree_root, global_metrics = recursion(texts)
    print("Tree building complete!")

    # Display global metrics
    print("\n=== RTP Global Metrics ===")
    print(f"LLM model: {llm_model_name}")
    print(f"Total Documents: {len(texts)}")
    print(f"Total LLM Input Tokens: {global_metrics.llm_input_tokens}")
    print(f"Total LLM Output Tokens: {global_metrics.llm_output_tokens}")
    print(f"Total NLI Calls: {global_metrics.nli_calls}")
    print(
        f"Total FAISS Search Time: {global_metrics.faiss_search_time_ms:.2f} ms"
    )
    print(
        f"Total Label Propagation Time: {global_metrics.label_propagation_time_ms:.2f} ms"
    )
    print(f"Total Time: {global_metrics.total_time_ms:.2f} ms")
    print(f"Total LLM Request Time: {global_metrics.llm_request_time:.2f} ms")
    print(f"Total NLI Time: {global_metrics.nli_time:.2f} ms")
    print(f"Number of Nodes: {global_metrics.num_nodes}")

    # Compute averages for metrics that should be averaged
    if global_metrics.num_nodes > 0:
        avg_split_ratio = global_metrics.split_ratio / global_metrics.num_nodes
        avg_nli_confidence = global_metrics.medoid_nli_confidence_avg / global_metrics.num_nodes
        print(f"Average Split Ratio: {avg_split_ratio:.2f}")
        print(f"Average Medoid NLI Confidence: {avg_nli_confidence:.2f}")

    # print("\nLeaf Purities (by leaf):")
    # for leaf_id, purity in results['leaf_purities'].items():
    #     print(f"  Leaf {leaf_id}: {purity:.4f}")

    # print("\nLeaf Entropies (by leaf):")
    # for leaf_id, entropy in results['leaf_entropies'].items():
    #     print(f"  Leaf {leaf_id}: {entropy:.4f}")

    return tree_root


def main(model, strategy, depth, frac):
    """Main function to run the complete evaluation."""
    print("=" * 80)
    print("AG NEWS DATASET EVALUATION: RTP RECURSION")
    print(
        f"Model: {model}, Strategy: {strategy}, Max Depth: {depth}, Fraction to Answer: {frac}"
    )
    print("=" * 80)

    # Load dataset
    texts, labels = load_agnews_sample(n_samples=None, seed=42)
    n_samples = len(texts)
    print(f"\nTotal samples to evaluate: {n_samples}")

    print(f"Running evaluation with LLM model: {model}")
    rtp_tree = run_rtp_evaluation(
        texts,
        labels,
        llm_model_name=model,
        n_documents_to_answer=frac,
        max_depth=depth,
        selection_strategy=strategy,
    )

    # Save the rtp_tree for further analysis if needed
    json_string = rtp_tree.model_dump_json()

    with open(f"rtp_tree_on_small_agnews_{model}_{strategy}.json", 'w') as f:
        f.write(json_string)

    pdf_path = tree_to_pdf.tree_to_pdf(
        rtp_tree, output_path=f"tree_agnews_rtp_{model}_{strategy}")
    print(f"PDF saved to: {pdf_path}")
    # run ollama stop on terminal using exec
    #exec(f"ollama stop {model}")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


def read_input_arguments():
    import argparse
    parser = argparse.ArgumentParser(description="Build trees")

    # Define arguments
    parser.add_argument("--model",
                        type=str,
                        required=True,
                        help="LLM model name")
    parser.add_argument("--strategy",
                        type=str,
                        required=True,
                        help="Selection strategy")
    parser.add_argument("--depth",
                        type=int,
                        default=4,
                        help="Maximum tree depth")
    parser.add_argument("--frac",
                        type=float,
                        default=0.25,
                        help="Fraction of documents to answer")

    # Parse arguments
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = read_input_arguments()
    main(
        model=args.model,
        strategy=args.strategy,
        depth=args.depth,
        frac=args.frac,
    )
