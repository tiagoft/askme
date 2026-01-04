"""Tests for HDBSCAN baseline functionality."""

import pytest
import sys
import os
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from askme.rtp.tree_models import TreeNode
from askme.rtp.hdbscan_baseline import (
    vectorize_texts,
    build_tree_from_hdbscan,
    calculate_tree_depth,
    run_hdbscan_baseline,
)
from sklearn.cluster import HDBSCAN


def test_vectorize_texts_basic():
    """Test basic text vectorization."""
    texts = ["Hello world", "Machine learning is great"]
    embeddings = vectorize_texts(texts, model_name="all-MiniLM-L6-v2", device="cpu")
    
    # Check output shape
    assert embeddings.shape[0] == len(texts)
    assert embeddings.shape[1] > 0  # Should have some embedding dimension
    
    # Check that embeddings are different for different texts
    assert not np.allclose(embeddings[0], embeddings[1])


def test_vectorize_texts_single_text():
    """Test vectorization of a single text."""
    texts = ["Single text document"]
    embeddings = vectorize_texts(texts, device="cpu")
    
    assert embeddings.shape[0] == 1
    assert embeddings.shape[1] > 0


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


def test_build_tree_from_hdbscan_single_cluster():
    """Test tree building when HDBSCAN finds a single cluster."""
    # Create simple 2D data in one cluster
    X = np.random.randn(10, 2)
    
    clusterer = HDBSCAN(min_cluster_size=2, min_samples=1)
    clusterer.fit(X)
    
    tree = build_tree_from_hdbscan(clusterer, len(X))
    
    # Should create a root node with all documents
    assert tree.documents == list(range(len(X)))


def test_build_tree_from_hdbscan_multiple_clusters():
    """Test tree building when HDBSCAN finds multiple clusters."""
    # Create data with two clear clusters
    X1 = np.random.randn(10, 2) + np.array([5, 5])
    X2 = np.random.randn(10, 2) + np.array([-5, -5])
    X = np.vstack([X1, X2])
    
    clusterer = HDBSCAN(min_cluster_size=2, min_samples=1)
    clusterer.fit(X)
    
    tree = build_tree_from_hdbscan(clusterer, len(X))
    
    # Should create a tree with root and children
    assert tree.documents == list(range(len(X)))
    # With two clusters, we should have some tree structure
    # (exact structure depends on clustering result)


def test_run_hdbscan_baseline_basic():
    """Test running the full HDBSCAN baseline pipeline."""
    texts = [
        "The cat sat on the mat.",
        "Dogs are loyal companions.",
        "Cats love to sleep.",
        "Dogs enjoy playing fetch.",
    ]
    
    tree, embeddings = run_hdbscan_baseline(
        texts,
        model_name="all-MiniLM-L6-v2",
        min_cluster_size=2,
        min_samples=1,
        device="cpu",
    )
    
    # Check that we get a tree and embeddings
    assert tree is not None
    assert tree.documents == list(range(len(texts)))
    assert embeddings.shape[0] == len(texts)
    assert embeddings.shape[1] > 0


def test_run_hdbscan_baseline_tree_structure():
    """Test that the baseline creates a valid tree structure."""
    texts = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks.",
        "The cat climbed the tree.",
        "Dogs are great pets.",
    ]
    
    tree, _ = run_hdbscan_baseline(
        texts,
        min_cluster_size=2,
        min_samples=1,
        device="cpu",
    )
    
    # Tree should have all documents at root
    assert len(tree.documents) == len(texts)
    
    # Calculate depth (should be >= 0)
    depth = calculate_tree_depth(tree)
    assert depth >= 0


def test_run_hdbscan_baseline_with_similar_texts():
    """Test baseline with very similar texts."""
    texts = [
        "Hello world",
        "Hello world",
        "Hello world",
    ]
    
    tree, embeddings = run_hdbscan_baseline(
        texts,
        min_cluster_size=2,
        device="cpu",
    )
    
    # Should still create a valid tree
    assert tree is not None
    assert len(tree.documents) == len(texts)
    
    # Embeddings should be nearly identical
    assert np.allclose(embeddings[0], embeddings[1])
    assert np.allclose(embeddings[1], embeddings[2])
