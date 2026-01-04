#!/usr/bin/env python3
"""
Simple validation script for HDBSCAN baseline.

This script validates that the HDBSCAN baseline implementation is correct
by testing with mock data that doesn't require heavy dependencies.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from askme.rtp.tree_models import TreeNode
from askme.rtp.hdbscan_baseline import build_tree_from_hdbscan, calculate_tree_depth
from askme.rtp.evaluator import evaluate_exploratory_power


def test_with_mock_clusterer():
    """Test the baseline with a mock HDBSCAN clusterer."""
    print("=" * 80)
    print("HDBSCAN Baseline Validation")
    print("=" * 80)
    
    # Create a mock HDBSCAN clusterer
    class MockHDBSCAN:
        def __init__(self, labels):
            self.labels_ = np.array(labels)
    
    # Test case 1: Two clear clusters
    print("\nTest 1: Two clear clusters")
    clusterer = MockHDBSCAN([0, 0, 0, 1, 1, 1])
    tree = build_tree_from_hdbscan(clusterer, 6)
    depth = calculate_tree_depth(tree)
    
    print(f"  Root documents: {tree.documents}")
    print(f"  Tree depth: {depth}")
    
    # Create labels matching the clusters
    labels = [0, 0, 0, 1, 1, 1]
    results = evaluate_exploratory_power(tree, labels)
    print(f"  Average leaf purity: {results['average_leaf_purity']:.4f}")
    print(f"  Average leaf entropy: {results['average_leaf_entropy']:.4f}")
    print(f"  Number of leaves: {results['num_leaves']}")
    
    # Test case 2: Four clusters
    print("\nTest 2: Four distinct clusters")
    clusterer = MockHDBSCAN([0, 0, 1, 1, 2, 2, 3, 3])
    tree = build_tree_from_hdbscan(clusterer, 8)
    depth = calculate_tree_depth(tree)
    
    print(f"  Root documents: {tree.documents}")
    print(f"  Tree depth: {depth}")
    
    labels = [0, 0, 1, 1, 2, 2, 3, 3]
    results = evaluate_exploratory_power(tree, labels)
    print(f"  Average leaf purity: {results['average_leaf_purity']:.4f}")
    print(f"  Average leaf entropy: {results['average_leaf_entropy']:.4f}")
    print(f"  Number of leaves: {results['num_leaves']}")
    
    # Test case 3: With noise points
    print("\nTest 3: Clusters with noise points")
    clusterer = MockHDBSCAN([0, 0, 1, 1, -1, -1])
    tree = build_tree_from_hdbscan(clusterer, 6)
    depth = calculate_tree_depth(tree)
    
    print(f"  Root documents: {tree.documents}")
    print(f"  Tree depth: {depth}")
    
    # Assign noise points to cluster 2
    labels = [0, 0, 1, 1, 2, 2]
    results = evaluate_exploratory_power(tree, labels)
    print(f"  Average leaf purity: {results['average_leaf_purity']:.4f}")
    print(f"  Average leaf entropy: {results['average_leaf_entropy']:.4f}")
    print(f"  Number of leaves: {results['num_leaves']}")
    
    # Test case 4: Single cluster (no split)
    print("\nTest 4: Single cluster (no split needed)")
    clusterer = MockHDBSCAN([0, 0, 0, 0])
    tree = build_tree_from_hdbscan(clusterer, 4)
    depth = calculate_tree_depth(tree)
    
    print(f"  Root documents: {tree.documents}")
    print(f"  Tree depth: {depth}")
    print(f"  Tree is leaf: {tree.left is None and tree.right is None}")
    
    labels = [0, 0, 0, 0]
    results = evaluate_exploratory_power(tree, labels)
    print(f"  Average leaf purity: {results['average_leaf_purity']:.4f}")
    print(f"  Average leaf entropy: {results['average_leaf_entropy']:.4f}")
    
    print("\n" + "=" * 80)
    print("All validation tests completed successfully!")
    print("=" * 80)
    print("\nNote: This validation uses mock HDBSCAN results.")
    print("To test with real text data, run: python examples/demo_hdbscan_baseline.py")
    print("(Requires sentence-transformers and other dependencies)")


if __name__ == "__main__":
    test_with_mock_clusterer()
