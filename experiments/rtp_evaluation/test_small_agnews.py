"""
Test script using a sample (n=100) from the AG News dataset.

This script:
1. Loads a sample of 100 documents from the fancyzhx/ag_news dataset
2. Runs RTPRecursion tree building procedure and calculates metrics
3. Runs HDBSCAN baseline and calculates metrics
4. Compares the results between both methods
"""

import sys
import os
import pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from askme.rtp import (
    RTPBuilder,
    RTPRecursion,
    run_hdbscan_baseline,
    run_bertopic_baseline,
    evaluate_exploratory_power,
    calculate_tree_depth,
    tree_to_pdf,
)

from datasets import load_dataset
import numpy as np
from typing import List, Tuple


def load_agnews_sample(n_samples: int | None = 500, seed: int = 42) -> Tuple[List[str], List[int]]:
    """
    Load a sample of n documents from the AG News dataset.
    
    Args:
        n_samples: Number of samples to load
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (texts, labels)
    """
    print(f"Loading {n_samples} samples from AG News dataset...")
    dataset = load_dataset('fancyzhx/ag_news', split='test')
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Sample n_samples indices
    total_size = len(dataset)
    if n_samples is None:
        n_samples = total_size
    indices = np.random.choice(total_size, size=min(n_samples, total_size), replace=False)
    
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


def print_tree_structure(node, texts, depth=0, max_samples=3):
    """Print tree structure with sample documents."""
    indent = "  " * depth
    print(f"{indent}Node with {len(node.documents)} documents")
    
    if node.question:
        print(f"{indent}  Question: {node.question}")
    
    # Print sample documents
    sample_docs = [texts[i] for i in node.documents[:max_samples]]
    for doc in sample_docs:
        print(f"{indent}  - {doc[:80]}...")
    if len(node.documents) > max_samples:
        print(f"{indent}  ... and {len(node.documents) - max_samples} more")
    
    # Print metrics if available
    if node.metrics:
        print(f"{indent}  Split ratio: {node.metrics.split_ratio:.2f}")
        print(f"{indent}  NLI calls: {node.metrics.nli_calls}")
    
    # Recurse into children
    if node.left:
        print(f"{indent}  LEFT (YES):")
        print_tree_structure(node.left, texts, depth + 1, max_samples)
    
    if node.right:
        print(f"{indent}  RIGHT (NO):")
        print_tree_structure(node.right, texts, depth + 1, max_samples)


def run_rtp_evaluation(texts: List[str], labels: List[int]):
    """
    Run RTP Recursion on the dataset and evaluate.
    
    Args:
        texts: List of text documents
        labels: Ground truth labels
    """
    print("\n" + "=" * 80)
    print("RTP RECURSION EVALUATION")
    print("=" * 80)
    
    # Initialize RTPBuilder
    print("\nInitializing RTPBuilder...")
    builder = RTPBuilder(
        use_gpu=True,
        n_medoids=15,
        n_documents_to_answer=200,
        llm_model_name='gpt-oss:20b',
        max_retries=10,
        min_split_ratio=0.1,
        max_split_ratio=0.9,
        alpha=1e-2,
        verbose=True,
    )
    print("RTPBuilder initialized!")
    
    # Initialize RTPRecursion with stopping criteria
    print("\nInitializing RTPRecursion...")
    recursion = RTPRecursion(
        builder=builder,
        min_node_size=30,       # Don't split nodes with fewer than 5 documents
        min_split_ratio=0.1,   # Split should have at least 10% in smaller child
        max_split_ratio=0.9,   # Split should have at most 90% in larger child
        max_depth=4,           # Maximum tree depth
    )
    print("RTPRecursion initialized!")
    
    # Build the recursive tree
    print("\nBuilding RTP recursive tree...")
    tree_root, global_metrics = recursion(texts)
    print("Tree building complete!")
    
    # Display tree structure
    print("\n=== RTP Tree Structure ===")
    print_tree_structure(tree_root, texts)
    
    # Display global metrics
    print("\n=== RTP Global Metrics ===")
    print(f"Total LLM Input Tokens: {global_metrics.llm_input_tokens}")
    print(f"Total LLM Output Tokens: {global_metrics.llm_output_tokens}")
    print(f"Total NLI Calls: {global_metrics.nli_calls}")
    print(f"Total FAISS Search Time: {global_metrics.faiss_search_time_ms:.2f} ms")
    print(f"Total Label Propagation Time: {global_metrics.label_propagation_time_ms:.2f} ms")
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
    
    # Calculate tree depth
    tree_depth = calculate_tree_depth(tree_root)
    print(f"Tree Depth: {tree_depth}")
    
    # Evaluate exploratory power
    print("\n=== RTP Exploratory Power Evaluation ===")
    results = evaluate_exploratory_power(tree_root, labels)
    
    print(f"Number of leaf nodes: {results['num_leaves']}")
    print(f"Average leaf purity: {results['average_leaf_purity']:.4f}")
    print(f"Average leaf entropy: {results['average_leaf_entropy']:.4f}")
    
    # print("\nLeaf Purities (by leaf):")
    # for leaf_id, purity in results['leaf_purities'].items():
    #     print(f"  Leaf {leaf_id}: {purity:.4f}")
    
    # print("\nLeaf Entropies (by leaf):")
    # for leaf_id, entropy in results['leaf_entropies'].items():
    #     print(f"  Leaf {leaf_id}: {entropy:.4f}")
    
    print("\nIsolation Depths (by class):")
    class_names = ["World", "Sports", "Business", "Sci/Tech"]
    for class_label, iso_depth in results['isolation_depths'].items():
        class_name = class_names[class_label] if class_label < len(class_names) else f"Class {class_label}"
        if iso_depth is not None:
            print(f"  {class_name}: isolated at depth {iso_depth}")
        else:
            print(f"  {class_name}: never fully isolated")

    return tree_root, results, global_metrics


def run_hdbscan_evaluation(texts: List[str], labels: List[int]):
    """
    Run HDBSCAN baseline on the dataset and evaluate.
    
    Args:
        texts: List of text documents
        labels: Ground truth labels
    """
    print("\n" + "=" * 80)
    print("HDBSCAN BASELINE EVALUATION")
    print("=" * 80)
    
    # Run HDBSCAN baseline
    print("\nRunning HDBSCAN clustering...")
    print("This may take a moment while the model processes...")
    tree, embeddings = run_hdbscan_baseline(
        texts,
        model_name="all-MiniLM-L6-v2",
        min_leaf_size=30,
        max_tree_depth=4,
        device="cpu",
    )
    
    print("\nClustering complete!")
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Calculate tree depth
    depth = calculate_tree_depth(tree)
    print(f"Tree depth: {depth}")
    
    # Evaluate exploratory power
    print("\n=== HDBSCAN Exploratory Power Evaluation ===")
    results = evaluate_exploratory_power(tree, labels)
    
    print(f"Number of leaf nodes: {results['num_leaves']}")
    print(f"Average leaf purity: {results['average_leaf_purity']:.4f}")
    print(f"Average leaf entropy: {results['average_leaf_entropy']:.4f}")
    
    print("\nLeaf Purities (by leaf):")
    for leaf_id, purity in results['leaf_purities'].items():
        print(f"  Leaf {leaf_id}: {purity:.4f}")
    
    print("\nLeaf Entropies (by leaf):")
    for leaf_id, entropy in results['leaf_entropies'].items():
        print(f"  Leaf {leaf_id}: {entropy:.4f}")
    
    print("\nIsolation Depths (by class):")
    class_names = ["World", "Sports", "Business", "Sci/Tech"]
    for class_label, iso_depth in results['isolation_depths'].items():
        class_name = class_names[class_label] if class_label < len(class_names) else f"Class {class_label}"
        if iso_depth is not None:
            print(f"  {class_name}: isolated at depth {iso_depth}")
        else:
            print(f"  {class_name}: never fully isolated")
    
    return tree, results, embeddings


def run_bertopic_evaluation(texts: List[str], labels: List[int]):
    """
    Run BERTopic baseline on the dataset and evaluate.
    
    Args:
        texts: List of text documents
        labels: Ground truth labels
    """
    print("\n" + "=" * 80)
    print("BERTOPIC BASELINE EVALUATION")
    print("=" * 80)
    
    # Run BERTopic baseline
    print("\nRunning BERTopic clustering...")
    print("This may take a moment while the model processes...")
    tree, embeddings, topic_model = run_bertopic_baseline(
        texts,
        model_name="all-MiniLM-L6-v2",
        nr_topics="auto",
        min_leaf_size=30,
        max_tree_depth=4,
        device="cpu",
    )
    
    print("\nClustering complete!")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Number of topics found: {len(set(topic_model.topics_)) - (1 if -1 in topic_model.topics_ else 0)}")
    
    # Calculate tree depth
    depth = calculate_tree_depth(tree)
    print(f"Tree depth: {depth}")
    
    # Evaluate exploratory power
    print("\n=== BERTopic Exploratory Power Evaluation ===")
    results = evaluate_exploratory_power(tree, labels)
    
    print(f"Number of leaf nodes: {results['num_leaves']}")
    print(f"Average leaf purity: {results['average_leaf_purity']:.4f}")
    print(f"Average leaf entropy: {results['average_leaf_entropy']:.4f}")
    
    print("\nLeaf Purities (by leaf):")
    for leaf_id, purity in results['leaf_purities'].items():
        print(f"  Leaf {leaf_id}: {purity:.4f}")
    
    print("\nLeaf Entropies (by leaf):")
    for leaf_id, entropy in results['leaf_entropies'].items():
        print(f"  Leaf {leaf_id}: {entropy:.4f}")
    
    print("\nIsolation Depths (by class):")
    class_names = ["World", "Sports", "Business", "Sci/Tech"]
    for class_label, iso_depth in results['isolation_depths'].items():
        class_name = class_names[class_label] if class_label < len(class_names) else f"Class {class_label}"
        if iso_depth is not None:
            print(f"  {class_name}: isolated at depth {iso_depth}")
        else:
            print(f"  {class_name}: never fully isolated")
    
    return tree, results, embeddings, topic_model


def compare_results(rtp_results: dict, hdbscan_results: dict, bertopic_results: dict, 
                   rtp_metrics, rtp_tree, hdbscan_tree, bertopic_tree):
    """
    Compare results between RTP, HDBSCAN, and BERTopic methods.
    
    Args:
        rtp_results: RTP exploratory power results
        hdbscan_results: HDBSCAN exploratory power results
        bertopic_results: BERTopic exploratory power results
        rtp_metrics: RTP global metrics
        rtp_tree: RTP tree root
        hdbscan_tree: HDBSCAN tree root
        bertopic_tree: BERTopic tree root
    """
    print("\n" + "=" * 80)
    print("COMPARISON: RTP vs HDBSCAN vs BERTopic")
    print("=" * 80)
    
    print("\n=== Tree Structure ===")
    rtp_depth = calculate_tree_depth(rtp_tree) if rtp_tree else None
    hdbscan_depth = calculate_tree_depth(hdbscan_tree)
    bertopic_depth = calculate_tree_depth(bertopic_tree)
    print(f"RTP Tree Depth:      {rtp_depth}")
    print(f"HDBSCAN Tree Depth:  {hdbscan_depth}")
    print(f"BERTopic Tree Depth: {bertopic_depth}")
    print(f"RTP Number of Leaves:      {rtp_results['num_leaves'] if rtp_results else 'N/A'}")
    print(f"HDBSCAN Number of Leaves:  {hdbscan_results['num_leaves']}")
    print(f"BERTopic Number of Leaves: {bertopic_results['num_leaves']}")
    
    print("\n=== Purity Metrics ===")
    print(f"RTP Average Leaf Purity:      {rtp_results['average_leaf_purity']:.4f}" if rtp_results else "RTP Average Leaf Purity: N/A")
    print(f"HDBSCAN Average Leaf Purity:  {hdbscan_results['average_leaf_purity']:.4f}")
    print(f"BERTopic Average Leaf Purity: {bertopic_results['average_leaf_purity']:.4f}")
    
    best_purity = max(
        rtp_results['average_leaf_purity'] if rtp_results else float('-inf'),
        hdbscan_results['average_leaf_purity'],
        bertopic_results['average_leaf_purity']
    )
    if rtp_results and rtp_results['average_leaf_purity'] == best_purity:
        print("→ RTP has the best purity")
    elif hdbscan_results['average_leaf_purity'] == best_purity:
        print("→ HDBSCAN has the best purity")
    else:
        print("→ BERTopic has the best purity")
    
    print("\n=== Entropy Metrics ===")
    print(f"RTP Average Leaf Entropy:      {rtp_results['average_leaf_entropy']:.4f}" if rtp_results else "RTP Average Leaf Entropy: N/A")
    print(f"HDBSCAN Average Leaf Entropy:  {hdbscan_results['average_leaf_entropy']:.4f}")
    print(f"BERTopic Average Leaf Entropy: {bertopic_results['average_leaf_entropy']:.4f}")
    
    best_entropy = min(
        rtp_results['average_leaf_entropy'] if rtp_results else float('inf'),
        hdbscan_results['average_leaf_entropy'],
        bertopic_results['average_leaf_entropy']
    )
    if rtp_results and rtp_results['average_leaf_entropy'] == best_entropy:
        print("→ RTP has the lowest entropy (best)")
    elif hdbscan_results['average_leaf_entropy'] == best_entropy:
        print("→ HDBSCAN has the lowest entropy (best)")
    else:
        print("→ BERTopic has the lowest entropy (best)")
    
    print("\n=== RTP-Specific Metrics ===")
    if rtp_metrics:
        print(f"Total LLM Input Tokens: {rtp_metrics.llm_input_tokens}")
        print(f"Total LLM Output Tokens: {rtp_metrics.llm_output_tokens}")
        print(f"Total NLI Calls: {rtp_metrics.nli_calls}")
        print(f"Total Time: {rtp_metrics.total_time_ms:.2f} ms")
        print(f"Number of Split Nodes: {rtp_metrics.num_nodes}")
    
    print("\n" + "=" * 80)
    print("Interpretation:")
    print("=" * 80)
    print("""
- Higher purity indicates better separation of document categories
- Lower entropy indicates less mixing of classes within leaf nodes
- RTP uses LLM-generated questions for splitting (interpretable but costly)
- HDBSCAN uses density-based clustering (fast, purely algorithmic)
- BERTopic uses topic modeling with hierarchical clustering (topics are interpretable)
""")


def main():
    """Main function to run the complete evaluation."""
    print("=" * 80)
    print("AG NEWS DATASET EVALUATION: RTP vs HDBSCAN vs BERTopic")
    print("=" * 80)
    
    # Load dataset
    texts, labels = load_agnews_sample(n_samples=None, seed=42)
    
    # # Run RTP evaluation
    rtp_tree, rtp_results, rtp_metrics = run_rtp_evaluation(texts, labels)
    
    # Save the rtp_tree for further analysis if needed
    json_string = rtp_tree.model_dump_json()
    
    with open("rtp_tree_on_small_agnews.json", 'w') as f:
        f.write(json_string)
    
    pdf_path = tree_to_pdf.tree_to_pdf(rtp_tree, output_path="tree_agnews_rtp.pdf")
    print(f"PDF saved to: {pdf_path}")
    
    # Run HDBSCAN evaluation
    hdbscan_tree, hdbscan_results, hdbscan_embeddings = run_hdbscan_evaluation(texts, labels)
    
    # Run BERTopic evaluation
    bertopic_tree, bertopic_results, bertopic_embeddings, topic_model = run_bertopic_evaluation(texts, labels)
    
    # # Compare results
    compare_results(rtp_results, hdbscan_results, bertopic_results, 
                   rtp_metrics, rtp_tree, hdbscan_tree, bertopic_tree)
    
    
    # Compare results
    # compare_results(None, hdbscan_results, bertopic_results, 
    #                None, None, hdbscan_tree, bertopic_tree)
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
