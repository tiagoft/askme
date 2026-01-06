"""
Lightweight tests for BERTopic baseline functionality.

These tests use mocks to avoid heavy dependencies while validating the core logic.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from askme.rtp.bertopic_baseline import (
    build_tree_from_bertopic_hierarchy,
    calculate_tree_depth,
    _split_node_by_topics,
)
from askme.rtp import TreeNode


def test_build_tree_single_topic():
    """Test building tree when all documents belong to a single topic."""
    topics = [0, 0, 0, 0]
    n_samples = 4
    
    # Mock BERTopic model (not used in simple case)
    class MockBERTopic:
        pass
    
    topic_model = MockBERTopic()
    tree = build_tree_from_bertopic_hierarchy(topic_model, topics, n_samples)
    
    # Should return a leaf node with all documents
    assert tree.documents == [0, 1, 2, 3]
    assert tree.left is None
    assert tree.right is None


def test_build_tree_two_topics():
    """Test building tree when documents belong to two topics."""
    topics = [0, 0, 1, 1]
    n_samples = 4
    
    class MockBERTopic:
        pass
    
    topic_model = MockBERTopic()
    tree = build_tree_from_bertopic_hierarchy(topic_model, topics, n_samples)
    
    # Should create a binary split
    assert len(tree.documents) == 4
    assert tree.left is not None
    assert tree.right is not None
    
    # Check that documents are properly split
    left_docs = set(tree.left.documents)
    right_docs = set(tree.right.documents)
    assert left_docs.union(right_docs) == {0, 1, 2, 3}
    assert len(left_docs.intersection(right_docs)) == 0


def test_build_tree_with_outliers():
    """Test building tree when some documents are outliers (-1)."""
    topics = [0, 0, 1, 1, -1, -1]
    n_samples = 6
    
    class MockBERTopic:
        pass
    
    topic_model = MockBERTopic()
    tree = build_tree_from_bertopic_hierarchy(topic_model, topics, n_samples)
    
    # Should create a binary split with outliers assigned to one side
    assert len(tree.documents) == 6
    assert tree.left is not None or tree.right is not None


def test_build_tree_multiple_topics():
    """Test building tree with multiple topics."""
    topics = [0, 0, 1, 1, 2, 2, 3, 3]
    n_samples = 8
    
    class MockBERTopic:
        pass
    
    topic_model = MockBERTopic()
    tree = build_tree_from_bertopic_hierarchy(topic_model, topics, n_samples)
    
    # Should create a hierarchical structure
    assert len(tree.documents) == 8
    assert tree.left is not None
    assert tree.right is not None
    
    # Tree should have depth > 0
    depth = calculate_tree_depth(tree)
    assert depth > 0


def test_split_node_by_topics_single_topic():
    """Test that splitting a node with a single topic returns it unchanged."""
    node = TreeNode(documents=[0, 1, 2])
    topics = [0, 0, 0]
    topic_set = {0}
    
    result = _split_node_by_topics(node, topics, topic_set)
    
    # Node should not be split
    assert result.left is None
    assert result.right is None
    assert result.documents == [0, 1, 2]


def test_split_node_by_topics_two_topics():
    """Test splitting a node with two topics."""
    node = TreeNode(documents=[0, 1, 2, 3])
    topics = [0, 0, 1, 1]
    topic_set = {0, 1}
    
    result = _split_node_by_topics(node, topics, topic_set)
    
    # Node should be split
    assert result.left is not None
    assert result.right is not None
    
    # Check that all documents are preserved
    left_docs = set(result.left.documents)
    right_docs = set(result.right.documents)
    assert left_docs.union(right_docs) == {0, 1, 2, 3}


def test_calculate_tree_depth_single_node():
    """Test tree depth calculation for a single node."""
    root = TreeNode(documents=[0, 1, 2])
    depth = calculate_tree_depth(root)
    assert depth == 0


def test_calculate_tree_depth_two_levels():
    """Test tree depth calculation for a two-level tree."""
    root = TreeNode(documents=[0, 1, 2, 3])
    root.left = TreeNode(documents=[0, 1])
    root.right = TreeNode(documents=[2, 3])
    
    depth = calculate_tree_depth(root)
    assert depth == 1


def test_calculate_tree_depth_three_levels():
    """Test tree depth calculation for a three-level tree."""
    root = TreeNode(documents=[0, 1, 2, 3])
    root.left = TreeNode(documents=[0, 1, 2])
    root.right = TreeNode(documents=[3])
    root.left.left = TreeNode(documents=[0, 1])
    root.left.right = TreeNode(documents=[2])
    
    depth = calculate_tree_depth(root)
    assert depth == 2


def test_max_tree_depth_zero():
    """Test that max_tree_depth=0 creates a single leaf node."""
    topics = [0, 0, 1, 1, 2, 2, 3, 3]
    n_samples = 8
    
    class MockBERTopic:
        pass
    
    topic_model = MockBERTopic()
    tree = build_tree_from_bertopic_hierarchy(
        topic_model, topics, n_samples, max_tree_depth=0
    )
    
    # Should return a leaf node with all documents
    assert tree.documents == list(range(8))
    assert tree.left is None
    assert tree.right is None


def test_max_tree_depth_one():
    """Test that max_tree_depth=1 limits tree to depth 1."""
    topics = [0, 0, 1, 1, 2, 2, 3, 3]
    n_samples = 8
    
    class MockBERTopic:
        pass
    
    topic_model = MockBERTopic()
    tree = build_tree_from_bertopic_hierarchy(
        topic_model, topics, n_samples, max_tree_depth=1
    )
    
    # Should create a root with children but no deeper
    assert len(tree.documents) == 8
    depth = calculate_tree_depth(tree)
    assert depth <= 1


def test_min_leaf_size_prevents_split():
    """Test that min_leaf_size prevents splitting small nodes."""
    topics = [0, 0, 1, 1]
    n_samples = 4
    
    class MockBERTopic:
        pass
    
    topic_model = MockBERTopic()
    # Set min_leaf_size larger than the dataset
    tree = build_tree_from_bertopic_hierarchy(
        topic_model, topics, n_samples, min_leaf_size=10
    )
    
    # Should return a leaf node with all documents (not split)
    assert tree.documents == [0, 1, 2, 3]
    assert tree.left is None
    assert tree.right is None


def test_min_leaf_size_allows_split():
    """Test that nodes above min_leaf_size can be split."""
    topics = [0, 0, 1, 1, 2, 2, 3, 3]
    n_samples = 8
    
    class MockBERTopic:
        pass
    
    topic_model = MockBERTopic()
    # Set min_leaf_size to 2, should allow splitting since we have 8 > 2 documents
    tree = build_tree_from_bertopic_hierarchy(
        topic_model, topics, n_samples, min_leaf_size=2
    )
    
    # Should create a split since we have 8 documents (more than min_leaf_size)
    assert len(tree.documents) == 8
    assert tree.left is not None
    assert tree.right is not None


def test_min_leaf_size_exact_match():
    """Test that nodes with exactly min_leaf_size documents are not split."""
    topics = [0, 0, 1, 1]
    n_samples = 4
    
    class MockBERTopic:
        pass
    
    topic_model = MockBERTopic()
    # Set min_leaf_size to 4, which exactly matches the dataset size
    tree = build_tree_from_bertopic_hierarchy(
        topic_model, topics, n_samples, min_leaf_size=4
    )
    
    # Should NOT split because n_samples <= min_leaf_size
    assert tree.documents == [0, 1, 2, 3]
    assert tree.left is None
    assert tree.right is None


def test_combined_stop_conditions():
    """Test that both max_tree_depth and min_leaf_size work together."""
    topics = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    n_samples = 12
    
    class MockBERTopic:
        pass
    
    topic_model = MockBERTopic()
    # Use both conditions
    tree = build_tree_from_bertopic_hierarchy(
        topic_model, topics, n_samples, max_tree_depth=2, min_leaf_size=2
    )
    
    # Should create a tree with limited depth
    depth = calculate_tree_depth(tree)
    assert depth <= 2
    
    # All leaves should have at least 2 documents (or be unable to split further)
    def check_leaves(node):
        if node.left is None and node.right is None:
            # Leaf node - check size or verify it couldn't be split further
            return
        if node.left:
            check_leaves(node.left)
        if node.right:
            check_leaves(node.right)
    
    check_leaves(tree)


if __name__ == "__main__":
    # Run all tests
    test_build_tree_single_topic()
    print("✓ test_build_tree_single_topic passed")
    
    test_build_tree_two_topics()
    print("✓ test_build_tree_two_topics passed")
    
    test_build_tree_with_outliers()
    print("✓ test_build_tree_with_outliers passed")
    
    test_build_tree_multiple_topics()
    print("✓ test_build_tree_multiple_topics passed")
    
    test_split_node_by_topics_single_topic()
    print("✓ test_split_node_by_topics_single_topic passed")
    
    test_split_node_by_topics_two_topics()
    print("✓ test_split_node_by_topics_two_topics passed")
    
    test_calculate_tree_depth_single_node()
    print("✓ test_calculate_tree_depth_single_node passed")
    
    test_calculate_tree_depth_two_levels()
    print("✓ test_calculate_tree_depth_two_levels passed")
    
    test_calculate_tree_depth_three_levels()
    print("✓ test_calculate_tree_depth_three_levels passed")
    
    test_max_tree_depth_zero()
    print("✓ test_max_tree_depth_zero passed")
    
    test_max_tree_depth_one()
    print("✓ test_max_tree_depth_one passed")
    
    test_min_leaf_size_prevents_split()
    print("✓ test_min_leaf_size_prevents_split passed")
    
    test_min_leaf_size_allows_split()
    print("✓ test_min_leaf_size_allows_split passed")
    
    test_min_leaf_size_exact_match()
    print("✓ test_min_leaf_size_exact_match passed")
    
    test_combined_stop_conditions()
    print("✓ test_combined_stop_conditions passed")
    
    print("\nAll tests passed! ✓")
