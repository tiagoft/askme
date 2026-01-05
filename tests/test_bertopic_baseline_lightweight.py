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
    
    print("\nAll tests passed! ✓")
