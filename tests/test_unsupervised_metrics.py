"""Tests for the unsupervised_metrics module."""

import pytest
import sys
import os

# Add src directory to path to allow direct imports without full package installation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from askme.rtp.tree_models import TreeNode
from askme.rtp.unsupervised_metrics import (
    UnsupervisedMetric,
    NumberOfNodes,
    TreeHeight,
    NumberOfLeafNodes,
    TreeNodeUnbalance,
    DocumentsPerLeaf,
    TreeDocumentUnbalance,
    _count_nodes,
    _calculate_tree_height,
    _count_leaf_nodes,
    _tree_node_unbalance,
    _documents_per_leaf,
    _tree_document_unbalance,
)


def test_base_class_not_implemented():
    """Test that the base UnsupervisedMetric class raises NotImplementedError."""
    metric = UnsupervisedMetric()
    root = TreeNode(documents=[0, 1, 2])
    
    with pytest.raises(NotImplementedError):
        metric(root)


def test_number_of_nodes_single_node():
    """Test counting nodes for a single node tree."""
    root = TreeNode(documents=[0, 1, 2])
    
    metric = NumberOfNodes()
    assert metric(root) == 1


def test_number_of_nodes_with_children():
    """Test counting nodes for a tree with children."""
    root = TreeNode(documents=[0, 1, 2, 3])
    root.left = TreeNode(documents=[0, 1])
    root.right = TreeNode(documents=[2, 3])
    
    metric = NumberOfNodes()
    assert metric(root) == 3


def test_number_of_nodes_complex_tree():
    """Test counting nodes for a more complex tree."""
    root = TreeNode(documents=[0, 1, 2, 3, 4, 5])
    root.left = TreeNode(documents=[0, 1, 2])
    root.right = TreeNode(documents=[3, 4, 5])
    root.left.left = TreeNode(documents=[0])
    root.left.right = TreeNode(documents=[1, 2])
    
    metric = NumberOfNodes()
    assert metric(root) == 5  # root + 2 children + 2 grandchildren


def test_tree_height_single_node():
    """Test tree height for a single node (leaf)."""
    root = TreeNode(documents=[0, 1, 2])
    
    metric = TreeHeight()
    assert metric(root) == 0


def test_tree_height_with_children():
    """Test tree height for a tree with one level of children."""
    root = TreeNode(documents=[0, 1, 2, 3])
    root.left = TreeNode(documents=[0, 1])
    root.right = TreeNode(documents=[2, 3])
    
    metric = TreeHeight()
    assert metric(root) == 1


def test_tree_height_unbalanced():
    """Test tree height for an unbalanced tree."""
    root = TreeNode(documents=[0, 1, 2, 3, 4])
    root.left = TreeNode(documents=[0, 1, 2])
    root.right = TreeNode(documents=[3, 4])
    root.left.left = TreeNode(documents=[0])
    root.left.right = TreeNode(documents=[1, 2])
    root.left.left.left = TreeNode(documents=[0])
    
    metric = TreeHeight()
    assert metric(root) == 3


def test_number_of_leaf_nodes_single_node():
    """Test counting leaf nodes for a single node tree."""
    root = TreeNode(documents=[0, 1, 2])
    
    metric = NumberOfLeafNodes()
    assert metric(root) == 1


def test_number_of_leaf_nodes_with_children():
    """Test counting leaf nodes for a tree with children."""
    root = TreeNode(documents=[0, 1, 2, 3])
    root.left = TreeNode(documents=[0, 1])
    root.right = TreeNode(documents=[2, 3])
    
    metric = NumberOfLeafNodes()
    assert metric(root) == 2


def test_number_of_leaf_nodes_complex_tree():
    """Test counting leaf nodes for a more complex tree."""
    root = TreeNode(documents=[0, 1, 2, 3, 4, 5])
    root.left = TreeNode(documents=[0, 1, 2])
    root.right = TreeNode(documents=[3, 4, 5])
    root.left.left = TreeNode(documents=[0])
    root.left.right = TreeNode(documents=[1, 2])
    root.right.left = TreeNode(documents=[3])
    
    metric = NumberOfLeafNodes()
    assert metric(root) == 3  # Three leaf nodes


def test_tree_node_unbalance_balanced():
    """Test node unbalance for a perfectly balanced tree."""
    root = TreeNode(documents=[0, 1, 2, 3])
    root.left = TreeNode(documents=[0, 1])
    root.right = TreeNode(documents=[2, 3])
    
    metric = TreeNodeUnbalance()
    assert metric(root) == 0  # Both children are leaves (height 0)


def test_tree_node_unbalance_unbalanced():
    """Test node unbalance for an unbalanced tree."""
    root = TreeNode(documents=[0, 1, 2, 3])
    root.left = TreeNode(documents=[0, 1, 2])
    root.right = TreeNode(documents=[3])
    root.left.left = TreeNode(documents=[0])
    root.left.right = TreeNode(documents=[1, 2])
    
    metric = TreeNodeUnbalance()
    # Left subtree has height 1, right subtree has height 0
    assert metric(root) == 1


def test_tree_node_unbalance_very_unbalanced():
    """Test node unbalance for a very unbalanced tree."""
    root = TreeNode(documents=[0, 1, 2, 3])
    root.left = TreeNode(documents=[0, 1, 2])
    root.left.left = TreeNode(documents=[0])
    root.left.left.left = TreeNode(documents=[0])
    # No right child, so right height is -1
    
    metric = TreeNodeUnbalance()
    # Left subtree has height 2, right subtree has height -1 (no child)
    assert metric(root) == 3


def test_documents_per_leaf_single_node():
    """Test getting documents per leaf for a single node tree."""
    root = TreeNode(documents=[0, 1, 2])
    
    metric = DocumentsPerLeaf()
    result = metric(root)
    assert result == [3]


def test_documents_per_leaf_with_children():
    """Test getting documents per leaf for a tree with children."""
    root = TreeNode(documents=[0, 1, 2, 3])
    root.left = TreeNode(documents=[0, 1])
    root.right = TreeNode(documents=[2, 3])
    
    metric = DocumentsPerLeaf()
    result = metric(root)
    assert sorted(result) == [2, 2]


def test_documents_per_leaf_complex_tree():
    """Test getting documents per leaf for a more complex tree."""
    root = TreeNode(documents=[0, 1, 2, 3, 4, 5])
    root.left = TreeNode(documents=[0, 1, 2])
    root.right = TreeNode(documents=[3, 4, 5])
    root.left.left = TreeNode(documents=[0])
    root.left.right = TreeNode(documents=[1, 2])
    root.right.left = TreeNode(documents=[3])
    
    metric = DocumentsPerLeaf()
    result = metric(root)
    # Should have 3 leaves with 1, 2, and 1 documents (or some permutation)
    assert sorted(result) == [1, 1, 2]


def test_tree_document_unbalance_balanced():
    """Test document unbalance for a balanced tree."""
    root = TreeNode(documents=[0, 1, 2, 3])
    root.left = TreeNode(documents=[0, 1])
    root.right = TreeNode(documents=[2, 3])
    
    metric = TreeDocumentUnbalance()
    assert metric(root) == 0  # 2 documents in left, 2 in right


def test_tree_document_unbalance_unbalanced():
    """Test document unbalance for an unbalanced tree."""
    root = TreeNode(documents=[0, 1, 2, 3, 4])
    root.left = TreeNode(documents=[0, 1, 2])
    root.right = TreeNode(documents=[3, 4])
    
    metric = TreeDocumentUnbalance()
    assert metric(root) == 1  # 3 documents in left, 2 in right


def test_tree_document_unbalance_very_unbalanced():
    """Test document unbalance for a very unbalanced tree."""
    root = TreeNode(documents=[0, 1, 2, 3, 4, 5])
    root.left = TreeNode(documents=[0, 1, 2, 3])
    root.right = TreeNode(documents=[4, 5])
    
    metric = TreeDocumentUnbalance()
    assert metric(root) == 2  # 4 documents in left, 2 in right


def test_tree_document_unbalance_no_right_child():
    """Test document unbalance when there's no right child."""
    root = TreeNode(documents=[0, 1, 2])
    root.left = TreeNode(documents=[0, 1, 2])
    
    metric = TreeDocumentUnbalance()
    assert metric(root) == 3  # 3 documents in left, 0 in right


def test_helper_functions():
    """Test that helper functions work correctly."""
    root = TreeNode(documents=[0, 1, 2, 3])
    root.left = TreeNode(documents=[0, 1])
    root.right = TreeNode(documents=[2, 3])
    
    # Test helper functions directly
    assert _count_nodes(root) == 3
    assert _calculate_tree_height(root) == 1
    assert _count_leaf_nodes(root) == 2
    assert _tree_node_unbalance(root) == 0
    assert sorted(_documents_per_leaf(root)) == [2, 2]
    assert _tree_document_unbalance(root) == 0


def test_empty_tree_edge_cases():
    """Test edge cases with None nodes."""
    # Test helper functions with None
    assert _count_nodes(None) == 0
    assert _calculate_tree_height(None) == -1
    assert _count_leaf_nodes(None) == 0
    assert _tree_node_unbalance(None) == 0
    assert _documents_per_leaf(None) == []
    assert _tree_document_unbalance(None) == 0
