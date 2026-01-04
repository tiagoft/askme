"""Lightweight tests for HDBSCAN baseline functionality that don't require heavy dependencies."""

import pytest
import sys
import os
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from askme.rtp.tree_models import TreeNode
from askme.rtp.hdbscan_baseline import (
    calculate_tree_depth,
    build_tree_from_hdbscan,
)


def test_calculate_tree_depth_single_node():
    """Test tree depth calculation for a single node (leaf)."""
    root = TreeNode(documents=[0, 1, 2])
    depth = calculate_tree_depth(root)
    
    assert depth == 0


def test_calculate_tree_depth_two_levels():
    """Test tree depth calculation for a tree with two levels."""
    root = TreeNode(documents=[0, 1, 2, 3])
    root.left = TreeNode(documents=[0, 1])
    root.right = TreeNode(documents=[2, 3])
    
    depth = calculate_tree_depth(root)
    
    assert depth == 1


def test_calculate_tree_depth_three_levels():
    """Test tree depth calculation for a tree with three levels."""
    root = TreeNode(documents=[0, 1, 2, 3, 4])
    root.left = TreeNode(documents=[0, 1, 2])
    root.right = TreeNode(documents=[3, 4])
    
    # Add another level to left child
    root.left.left = TreeNode(documents=[0, 1])
    root.left.right = TreeNode(documents=[2])
    
    depth = calculate_tree_depth(root)
    
    assert depth == 2


def test_calculate_tree_depth_unbalanced():
    """Test tree depth calculation for an unbalanced tree."""
    root = TreeNode(documents=[0, 1, 2, 3])
    root.left = TreeNode(documents=[0])
    root.right = TreeNode(documents=[1, 2, 3])
    
    # Right side goes deeper
    root.right.left = TreeNode(documents=[1, 2])
    root.right.right = TreeNode(documents=[3])
    
    depth = calculate_tree_depth(root)
    
    assert depth == 2


def test_calculate_tree_depth_none():
    """Test tree depth calculation with None input."""
    depth = calculate_tree_depth(None)
    
    assert depth == -1


def test_calculate_tree_depth_deep_tree():
    """Test tree depth calculation for a deep tree."""
    root = TreeNode(documents=[0, 1, 2, 3, 4, 5, 6, 7])
    
    # Build a deep left branch
    current = root
    for i in range(5):
        current.left = TreeNode(documents=[i])
        current.right = TreeNode(documents=[i + 1])
        current = current.left
    
    depth = calculate_tree_depth(root)
    
    assert depth == 5


def test_build_tree_from_hdbscan_mock_single_cluster():
    """Test tree building with a mock HDBSCAN clusterer with single cluster."""
    # Create a mock HDBSCAN object
    class MockHDBSCAN:
        def __init__(self):
            self.labels_ = np.array([0, 0, 0, 0, 0])
    
    clusterer = MockHDBSCAN()
    tree = build_tree_from_hdbscan(clusterer, 5)
    
    # Should create a root node with all documents
    assert tree.documents == list(range(5))
    # With only one cluster, should not split
    assert tree.left is None and tree.right is None


def test_build_tree_from_hdbscan_mock_two_clusters():
    """Test tree building with a mock HDBSCAN clusterer with two clusters."""
    # Create a mock HDBSCAN object with two clusters
    class MockHDBSCAN:
        def __init__(self):
            self.labels_ = np.array([0, 0, 1, 1])
    
    clusterer = MockHDBSCAN()
    tree = build_tree_from_hdbscan(clusterer, 4)
    
    # Should create a root node with all documents
    assert tree.documents == list(range(4))
    # With two clusters, should have split
    assert tree.left is not None or tree.right is not None


def test_build_tree_from_hdbscan_mock_all_noise():
    """Test tree building when all points are noise."""
    # Create a mock HDBSCAN object where all points are noise (-1)
    class MockHDBSCAN:
        def __init__(self):
            self.labels_ = np.array([-1, -1, -1, -1])
    
    clusterer = MockHDBSCAN()
    tree = build_tree_from_hdbscan(clusterer, 4)
    
    # Should create a root node with all documents
    assert tree.documents == list(range(4))
    # All noise, so should remain as single node
    assert tree.left is None and tree.right is None


def test_build_tree_from_hdbscan_mock_with_noise():
    """Test tree building with a mix of clusters and noise."""
    # Create a mock HDBSCAN object with clusters and noise
    class MockHDBSCAN:
        def __init__(self):
            # Cluster 0, 1, 2, and noise
            self.labels_ = np.array([0, 0, 1, 1, 2, 2, -1, -1])
    
    clusterer = MockHDBSCAN()
    tree = build_tree_from_hdbscan(clusterer, 8)
    
    # Should create a root node with all documents
    assert tree.documents == list(range(8))
    # With multiple clusters, should have structure
    assert tree.left is not None or tree.right is not None


def test_build_tree_from_hdbscan_mock_four_clusters():
    """Test tree building with four distinct clusters."""
    # Create a mock HDBSCAN object with four clusters
    class MockHDBSCAN:
        def __init__(self):
            self.labels_ = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    
    clusterer = MockHDBSCAN()
    tree = build_tree_from_hdbscan(clusterer, 8)
    
    # Should create a tree structure
    assert tree.documents == list(range(8))
    
    # Calculate depth - with 4 clusters, we should have at least depth 1
    depth = calculate_tree_depth(tree)
    assert depth >= 1
