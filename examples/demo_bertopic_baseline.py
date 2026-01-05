"""
Demo script for BERTopic baseline clustering with hierarchical structure.

This script demonstrates how to use the BERTopic baseline to cluster documents
and evaluate the resulting tree structure using exploratory power metrics.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from askme.rtp import (
    run_bertopic_baseline,
    evaluate_exploratory_power,
    calculate_tree_depth,
)


def create_sample_dataset():
    """Create a sample text dataset with known categories."""
    # Category 0: Animals
    animals = [
        "The cat sat on the mat and purred contentedly.",
        "Dogs are known for their loyalty and companionship.",
        "The elephant is the largest land animal on Earth.",
        "Cats are independent animals that make great pets.",
        "Dogs love to play fetch and go for walks.",
        "Elephants have excellent memory and strong social bonds.",
    ]
    
    # Category 1: Technology
    technology = [
        "Artificial intelligence is transforming modern computing.",
        "Machine learning algorithms can recognize patterns in data.",
        "Neural networks are inspired by the human brain.",
        "Deep learning has revolutionized computer vision tasks.",
        "Natural language processing helps computers understand text.",
        "AI systems can now generate realistic images and text.",
    ]
    
    # Category 2: Sports
    sports = [
        "Football is the most popular sport in the world.",
        "Basketball requires excellent coordination and teamwork.",
        "Tennis players need great agility and quick reflexes.",
        "Soccer matches can be thrilling to watch and play.",
        "Basketball games often have exciting buzzer-beater moments.",
        "Professional tennis tournaments attract millions of viewers.",
    ]
    
    texts = animals + technology + sports
    labels = [0] * len(animals) + [1] * len(technology) + [2] * len(sports)
    
    return texts, labels


def main():
    """Run BERTopic baseline demo."""
    print("=" * 80)
    print("BERTopic Baseline Demo")
    print("=" * 80)
    
    # Create sample dataset
    print("\nCreating sample dataset...")
    texts, labels = create_sample_dataset()
    print(f"Dataset size: {len(texts)} documents")
    print(f"Categories: 0 (Animals), 1 (Technology), 2 (Sports)")
    print(f"Documents per category: {len(texts) // 3}")
    
    # Run BERTopic baseline
    print("\nRunning BERTopic clustering...")
    print("This may take a moment while the model downloads and processes...")
    tree, embeddings, topic_model = run_bertopic_baseline(
        texts,
        model_name="all-MiniLM-L6-v2",
        nr_topics="auto",
        device="cpu",
    )
    
    print("\nClustering complete!")
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Number of topics found: {len(set(topic_model.topics_)) - (1 if -1 in topic_model.topics_ else 0)}")
    
    # Calculate tree depth
    depth = calculate_tree_depth(tree)
    print(f"Tree depth: {depth}")
    
    # Evaluate exploratory power
    print("\nEvaluating exploratory power...")
    results = evaluate_exploratory_power(tree, labels)
    
    print("\n" + "=" * 80)
    print("Evaluation Results")
    print("=" * 80)
    
    print(f"\nTree Metrics:")
    print(f"  Tree depth: {depth}")
    print(f"  Number of leaf nodes: {results['num_leaves']}")
    
    print(f"\nPurity Metrics:")
    print(f"  Average leaf purity: {results['average_leaf_purity']:.4f}")
    print("  (1.0 = perfectly pure, 0.0 = completely mixed)")
    
    print(f"\nEntropy Metrics:")
    print(f"  Average leaf entropy: {results['average_leaf_entropy']:.4f}")
    print("  (0.0 = perfectly pure, higher = more mixed)")
    
    print(f"\nLeaf Purities (by leaf):")
    for leaf_id, purity in results['leaf_purities'].items():
        print(f"  Leaf {leaf_id}: {purity:.4f}")
    
    print(f"\nLeaf Entropies (by leaf):")
    for leaf_id, entropy in results['leaf_entropies'].items():
        print(f"  Leaf {leaf_id}: {entropy:.4f}")
    
    print(f"\nIsolation Depths (by class):")
    for class_label, iso_depth in results['isolation_depths'].items():
        category_name = ["Animals", "Technology", "Sports"][class_label]
        if iso_depth is not None:
            print(f"  Class {class_label} ({category_name}): isolated at depth {iso_depth}")
        else:
            print(f"  Class {class_label} ({category_name}): never fully isolated")
    
    print("\n" + "=" * 80)
    print("Topic Information")
    print("=" * 80)
    
    # Display topic information
    topic_info = topic_model.get_topic_info()
    print("\nTopics discovered by BERTopic:")
    print(topic_info[['Topic', 'Count', 'Name']].head(10))
    
    print("\n" + "=" * 80)
    print("Interpretation")
    print("=" * 80)
    print("""
The BERTopic baseline creates a hierarchical clustering of documents using
topic modeling with hierarchical structure. BERTopic:
- Generates topics using c-TF-IDF representation
- Creates hierarchical relationships between topics
- Can identify outliers as separate topics

- **Purity** measures how well each leaf node contains documents of a single class.
- **Entropy** measures the disorder/mixing of classes within each leaf node.
- **Isolation depth** shows how deep in the tree each class becomes separated.
- **Tree depth** shows the maximum depth of the hierarchical structure.

Higher purity and lower entropy indicate better separation of document categories.
Lower isolation depths indicate that categories are separated earlier in the tree.
""")


if __name__ == "__main__":
    main()
