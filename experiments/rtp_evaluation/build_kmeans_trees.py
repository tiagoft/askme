import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


from typing import List, Tuple

import numpy as np
from datasets import load_dataset
from load_dataset import load_dataset_sample

from askme.rtp import KMeansTreeBuilder, KMeansTreeRecursion, tree_to_pdf


def run_kmeans_evaluation(
    texts: List[str],
    labels: List[int],
    llm_model_name: str = 'qwen3:14b',
    
    min_node_size=3,
    min_split_ratio=0.2,
    max_split_ratio=0.8,
    max_depth: int = 4,
    n_documents_to_answer: int | float = 0.1,
    selection_strategy: str = 'kmeans',
    nli_selection_strategy: str = 'kmeans',
    nli_overrides_kmeans: bool = False,
):
    """
    Run KMeans Recursion on the dataset and evaluate.
    
    Args:
        texts: List of text documents
        labels: Ground truth labels
    """
    print("\n" + "=" * 80)
    print("KMEANS RECURSION EVALUATION WITH MODEL")
    print(f"LLM model: {llm_model_name}")
    print(f"Selection strategy: {selection_strategy}")
    print(f"NLI selection strategy: {nli_selection_strategy}")
    print(f"NLI overrides KMeans: {nli_overrides_kmeans}")
    print("=" * 80)

    # Initialize RTPBuilder
    print("\nInitializing RTPBuilder...")
    builder = KMeansTreeBuilder(
        use_gpu=True,
        n_medoids_per_cluster=7,
        n_documents_to_answer=n_documents_to_answer,
        llm_model_name=llm_model_name,
        max_retries=10,
        min_split_ratio=0.1,
        max_split_ratio=0.9,
        nli_batch_size=1,
        chunk_size=150,
        overlap=50,
        alpha=1e-2,
        verbose=True,
        cache_dir='~/.askme_cache',
        nli_overrides_kmeans=nli_overrides_kmeans,

    )
    print("KMeans Builder initialized!")

    # Initialize KMeansTreeRecursion with stopping criteria
    print("\nInitializing KMeansTreeRecursion...")
    recursion = KMeansTreeRecursion(
        builder=builder,
        min_node_size=200,  # Don't split nodes with fewer than 200 documents
        min_split_ratio=0.1,  # Split should have at least 10% in smaller child
        max_split_ratio=0.9,  # Split should have at most 90% in larger child
        max_depth=max_depth,  # Maximum tree depth
    )
    print("RTPRecursion initialized!")

    # Build the recursive tree
    print("\nBuilding KMeans recursive tree...")
    tree_root, global_metrics = recursion(texts)
    print("Tree building complete!")

    # Display global metrics
    print("\n=== KMeans Global Metrics ===")
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
    print(
        f"Total LLM Request Time: {global_metrics.llm_request_time_ms:.2f} ms")
    print(f"Total NLI Time: {global_metrics.nli_time_ms:.2f} ms")
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


def main(model,
         strategy,
         nli_selection_strategy,
         depth,
         frac,
         dataset_name='fancyzhx/ag_news',
         nli_overrides_kmeans=False):
    """Main function to run the complete evaluation."""
    print("=" * 80)
    print(f"{dataset_name.upper()} DATASET EVALUATION: RTP RECURSION")
    print(
        f"Model: {model}, Strategy: {strategy}, NLI Selection Strategy: {nli_selection_strategy}, Max Depth: {depth}, Fraction to Answer: {frac}"
    )
    print("=" * 80)

    # Load dataset
    texts, labels = load_dataset_sample(n_samples=None,
                                        seed=42,
                                        dataset_name=dataset_name)
    n_samples = len(texts)
    print(f"\nTotal samples to evaluate: {n_samples}")

    print(f"Running evaluation with LLM model: {model}")
    kmeans_tree = run_kmeans_evaluation(
        texts,
        labels,
        llm_model_name=model,
        max_depth=depth,
        n_documents_to_answer=frac,
        selection_strategy=strategy,
        nli_selection_strategy=nli_selection_strategy,
        nli_overrides_kmeans=nli_overrides_kmeans,
    )
    # Save the rtp_tree for further analysis if needed
    json_string = kmeans_tree.model_dump_json()
    dataset_tag = dataset_name.replace('/', '_')
    
    if nli_overrides_kmeans == True:
        filename_suffix = "nli-overrides"
    else:
        filename_suffix = "kmeans-only"

    with open(
            f"kmeans_tree_{filename_suffix}_{dataset_tag}_{model}_{strategy}_{nli_selection_strategy}.json",
            'w') as f:
        f.write(json_string)

    pdf_path = tree_to_pdf.tree_to_pdf(
        kmeans_tree,
        output_path=
        f"kmeans_tree_{filename_suffix}_{dataset_tag}_{model}_{strategy}_{nli_selection_strategy}")
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
    parser.add_argument("--nli_selection_strategy",
                        type=str,
                        required=True,
                        help="NLI Selection strategy")
    parser.add_argument("--depth",
                        type=int,
                        default=4,
                        help="Maximum tree depth")
    parser.add_argument("--frac",
                        type=float,
                        default=0.25,
                        help="Fraction of documents to answer")
    parser.add_argument("--dataset_name",
                        type=str,
                        default="fancyzhx/ag_news",
                        help="Dataset name")
    parser.add_argument("--nli_overrides_kmeans",
                        action='store_true',
                        default=False,
                        help="Whether NLI labels override KMeans labels")

    # Parse arguments
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = read_input_arguments()
    main(
        model=args.model,
        strategy=args.strategy,
        nli_selection_strategy=args.nli_selection_strategy,
        depth=args.depth,
        frac=args.frac,
        dataset_name=args.dataset_name,
        nli_overrides_kmeans=args.nli_overrides_kmeans,
    )
