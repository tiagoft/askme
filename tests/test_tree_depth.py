"""Standalone tests for tree depth calculation that don't require HDBSCAN imports."""

import pytest
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from askme.rtp.tree_models import TreeNode


def calculate_tree_depth(root):
    """
    Calculate the maximum depth of a tree.
    
    Args:
        root: Root TreeNode of the tree
        
    Returns:
        Maximum depth of the tree (root has depth 0)
    """
    if root is None:
        return -1
    
    if root.left is None and root.right is None:
        return 0
    
    left_depth = calculate_tree_depth(root.left) if root.left else -1
    right_depth = calculate_tree_depth(root.right) if root.right else -1
    
    return 1 + max(left_depth, right_depth)


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


def test_calculate_tree_depth_only_left_children():
    """Test tree depth with only left children (like a linked list)."""
    root = TreeNode(documents=[0, 1, 2, 3])
    root.left = TreeNode(documents=[0, 1])
    root.left.left = TreeNode(documents=[0])
    
    depth = calculate_tree_depth(root)
    
    assert depth == 2


def test_calculate_tree_depth_only_right_children():
    """Test tree depth with only right children."""
    root = TreeNode(documents=[0, 1, 2, 3])
    root.right = TreeNode(documents=[1, 2, 3])
    root.right.right = TreeNode(documents=[2, 3])
    root.right.right.right = TreeNode(documents=[3])
    
    depth = calculate_tree_depth(root)
    
    assert depth == 3
