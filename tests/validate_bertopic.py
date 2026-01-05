"""
Validation script for BERTopic baseline functionality.

This script tests the BERTopic baseline with mock data to ensure
the core logic works correctly without requiring heavy dependencies.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from askme.rtp.bertopic_baseline import (
    build_tree_from_bertopic_hierarchy,
    calculate_tree_depth,
)
from askme.rtp import TreeNode, evaluate_exploratory_power


def test_two_clear_clusters():
    """Test with two distinct clusters."""
    print("\nTest 1: Two clear clusters")
    topics = [0, 0, 0, 1, 1, 1]
    n_samples = 6
    
    class MockBERTopic:
        pass
    
    topic_model = MockBERTopic()
    tree = build_tree_from_bertopic_hierarchy(topic_model, topics, n_samples)
    
    print(f"  Root documents: {tree.documents}")
    print(f"  Tree depth: {calculate_tree_depth(tree)}")
    
    # Evaluate with labels
    labels = [0, 0, 0, 1, 1, 1]  # Perfect match with topics
    results = evaluate_exploratory_power(tree, labels)
    print(f"  Average leaf purity: {results['average_leaf_purity']:.4f}")
    print(f"  Average leaf entropy: {results['average_leaf_entropy']:.4f}")
    print(f"  Number of leaves: {results['num_leaves']}")


def test_four_clusters():
    """Test with four distinct clusters."""
    print("\nTest 2: Four distinct clusters")
    topics = [0, 0, 1, 1, 2, 2, 3, 3]
    n_samples = 8
    
    class MockBERTopic:
        pass
    
    topic_model = MockBERTopic()
    tree = build_tree_from_bertopic_hierarchy(topic_model, topics, n_samples)
    
    print(f"  Root documents: {tree.documents}")
    print(f"  Tree depth: {calculate_tree_depth(tree)}")
    
    # Evaluate with labels
    labels = [0, 0, 1, 1, 2, 2, 3, 3]  # Perfect match with topics
    results = evaluate_exploratory_power(tree, labels)
    print(f"  Average leaf purity: {results['average_leaf_purity']:.4f}")
    print(f"  Average leaf entropy: {results['average_leaf_entropy']:.4f}")
    print(f"  Number of leaves: {results['num_leaves']}")


def test_with_outliers():
    """Test with outlier points (topic -1)."""
    print("\nTest 3: Clusters with outlier points")
    topics = [0, 0, 1, 1, -1, -1]
    n_samples = 6
    
    class MockBERTopic:
        pass
    
    topic_model = MockBERTopic()
    tree = build_tree_from_bertopic_hierarchy(topic_model, topics, n_samples)
    
    print(f"  Root documents: {tree.documents}")
    print(f"  Tree depth: {calculate_tree_depth(tree)}")
    
    # Evaluate with labels
    labels = [0, 0, 1, 1, 2, 2]  # Outliers are labeled as different class
    results = evaluate_exploratory_power(tree, labels)
    print(f"  Average leaf purity: {results['average_leaf_purity']:.4f}")
    print(f"  Average leaf entropy: {results['average_leaf_entropy']:.4f}")
    print(f"  Number of leaves: {results['num_leaves']}")


def test_single_topic():
    """Test with a single topic (no split needed)."""
    print("\nTest 4: Single topic (no split needed)")
    topics = [0, 0, 0, 0]
    n_samples = 4
    
    class MockBERTopic:
        pass
    
    topic_model = MockBERTopic()
    tree = build_tree_from_bertopic_hierarchy(topic_model, topics, n_samples)
    
    print(f"  Root documents: {tree.documents}")
    print(f"  Tree depth: {calculate_tree_depth(tree)}")
    print(f"  Tree is leaf: {tree.left is None and tree.right is None}")
    
    # Evaluate with labels
    labels = [0, 0, 0, 0]  # All same class
    results = evaluate_exploratory_power(tree, labels)
    print(f"  Average leaf purity: {results['average_leaf_purity']:.4f}")
    print(f"  Average leaf entropy: {results['average_leaf_entropy']:.4f}")
    print(f"  Number of leaves: {results['num_leaves']}")


def test_all_outliers():
    """Test when all points are outliers."""
    print("\nTest 5: All points are outliers")
    topics = [-1, -1, -1, -1]
    n_samples = 4
    
    class MockBERTopic:
        pass
    
    topic_model = MockBERTopic()
    tree = build_tree_from_bertopic_hierarchy(topic_model, topics, n_samples)
    
    print(f"  Root documents: {tree.documents}")
    print(f"  Tree depth: {calculate_tree_depth(tree)}")
    print(f"  Tree is leaf: {tree.left is None and tree.right is None}")
    
    # Evaluate with labels
    labels = [0, 1, 0, 1]  # Mixed classes
    results = evaluate_exploratory_power(tree, labels)
    print(f"  Average leaf purity: {results['average_leaf_purity']:.4f}")
    print(f"  Average leaf entropy: {results['average_leaf_entropy']:.4f}")
    print(f"  Number of leaves: {results['num_leaves']}")


def main():
    """Run all validation tests."""
    print("=" * 80)
    print("BERTopic Baseline Validation")
    print("=" * 80)
    
    test_two_clear_clusters()
    test_four_clusters()
    test_with_outliers()
    test_single_topic()
    test_all_outliers()
    
    print("\n" + "=" * 80)
    print("All validation tests completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
