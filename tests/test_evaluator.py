"""Tests for the evaluator module."""

import pytest
import sys
import os

# Add src directory to path to allow direct imports without full package installation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import directly from module files to avoid __init__.py dependencies
from askme.rtp.tree_models import TreeNode
from askme.rtp.evaluator import (
    calculate_node_purity,
    calculate_node_entropy,
    get_all_leaves,
    calculate_all_leaf_purities,
    calculate_all_leaf_entropies,
    calculate_isolation_depth,
    calculate_all_isolation_depths,
    evaluate_exploratory_power,
)


def test_calculate_node_purity_pure_node():
    """Test node purity calculation for a perfectly pure node (all same class)."""
    node = TreeNode(documents=[0, 1, 2])
    labels = [0, 0, 0]  # All documents have the same label
    
    purity = calculate_node_purity(node, labels)
    
    # Perfect purity should be 1.0
    assert purity == 1.0


def test_calculate_node_purity_impure_node():
    """Test node purity calculation for a completely impure node (evenly distributed)."""
    node = TreeNode(documents=[0, 1, 2, 3])
    labels = [0, 1, 0, 1]  # Two classes, evenly distributed
    
    purity = calculate_node_purity(node, labels)
    
    # With 2 classes evenly distributed (2 of class 0, 2 of class 1)
    # Most common class has proportion 2/4 = 0.5
    # Purity = proportion of most common class = 0.5
    assert purity == 0.5


def test_calculate_node_purity_mixed_node():
    """Test node purity calculation for a mixed node."""
    node = TreeNode(documents=[0, 1, 2, 3, 4])
    labels = [0, 0, 0, 1, 1]  # 3 of class 0, 2 of class 1
    
    purity = calculate_node_purity(node, labels)
    
    # Most common class (0) has proportion 3/5 = 0.6
    # Purity = 0.6
    assert abs(purity - 0.6) < 0.001


def test_calculate_node_purity_three_classes():
    """Test node purity calculation with three classes."""
    node = TreeNode(documents=[0, 1, 2, 3, 4, 5])
    labels = [0, 0, 1, 1, 2, 2]  # Three classes, evenly distributed
    
    purity = calculate_node_purity(node, labels)
    
    # All classes equally common, each with proportion 2/6 = 1/3
    # Purity = 1/3 ≈ 0.3333
    assert abs(purity - (1/3)) < 0.001


def test_calculate_node_purity_empty_node_raises_error():
    """Test that calculating purity on an empty node raises an error."""
    node = TreeNode(documents=[])
    labels = []
    
    with pytest.raises(ValueError, match="Node has no documents"):
        calculate_node_purity(node, labels)


def test_calculate_node_purity_invalid_index_raises_error():
    """Test that invalid document indices raise an error."""
    node = TreeNode(documents=[0, 1, 5])  # Index 5 doesn't exist in labels
    labels = [0, 0, 1]  # Only 3 labels (indices 0, 1, 2)
    
    with pytest.raises(ValueError, match="Document index out of range"):
        calculate_node_purity(node, labels)


def test_get_all_leaves_single_node():
    """Test getting leaves from a tree with only root node."""
    root = TreeNode(documents=[0, 1, 2])
    
    leaves = get_all_leaves(root)
    
    assert len(leaves) == 1
    assert leaves[0] == root


def test_get_all_leaves_with_children():
    """Test getting leaves from a tree with multiple levels."""
    root = TreeNode(documents=[0, 1, 2, 3])
    left_child = TreeNode(documents=[0, 1])
    right_child = TreeNode(documents=[2, 3])
    
    root.left = left_child
    root.right = right_child
    
    leaves = get_all_leaves(root)
    
    assert len(leaves) == 2
    assert left_child in leaves
    assert right_child in leaves
    assert root not in leaves


def test_get_all_leaves_unbalanced_tree():
    """Test getting leaves from an unbalanced tree."""
    # Root with left child having more structure
    root = TreeNode(documents=[0, 1, 2, 3, 4])
    left_child = TreeNode(documents=[0, 1, 2])
    right_child = TreeNode(documents=[3, 4])
    
    # Left child has children
    left_left = TreeNode(documents=[0, 1])
    left_right = TreeNode(documents=[2])
    left_child.left = left_left
    left_child.right = left_right
    
    root.left = left_child
    root.right = right_child
    
    leaves = get_all_leaves(root)
    
    # Should have 3 leaves: left_left, left_right, right_child
    assert len(leaves) == 3
    assert left_left in leaves
    assert left_right in leaves
    assert right_child in leaves
    assert root not in leaves
    assert left_child not in leaves


def test_calculate_all_leaf_purities():
    """Test calculating purities for all leaves in a tree."""
    root = TreeNode(documents=[0, 1, 2, 3, 4, 5])
    left_child = TreeNode(documents=[0, 1, 2])
    right_child = TreeNode(documents=[3, 4, 5])
    
    root.left = left_child
    root.right = right_child
    
    # Left child has pure class 0, right child has pure class 1
    labels = [0, 0, 0, 1, 1, 1]
    
    purities = calculate_all_leaf_purities(root, labels)
    
    # Should have 2 leaves
    assert len(purities) == 2
    # Both leaves should be perfectly pure
    assert all(purity == 1.0 for purity in purities.values())


def test_calculate_isolation_depth_root_level():
    """Test isolation depth when a class is isolated at the root level."""
    # Root node contains only one class
    root = TreeNode(documents=[0, 1, 2])
    labels = [0, 0, 0]
    
    depth = calculate_isolation_depth(root, labels, target_class=0)
    
    assert depth == 0  # Isolated at root level


def test_calculate_isolation_depth_first_level():
    """Test isolation depth when a class is isolated at first level."""
    root = TreeNode(documents=[0, 1, 2, 3])
    left_child = TreeNode(documents=[0, 1])
    right_child = TreeNode(documents=[2, 3])
    
    root.left = left_child
    root.right = right_child
    
    # Left child has only class 0, right child has only class 1
    labels = [0, 0, 1, 1]
    
    depth_0 = calculate_isolation_depth(root, labels, target_class=0)
    depth_1 = calculate_isolation_depth(root, labels, target_class=1)
    
    assert depth_0 == 1
    assert depth_1 == 1


def test_calculate_isolation_depth_never_isolated():
    """Test isolation depth when a class is never fully isolated."""
    root = TreeNode(documents=[0, 1, 2, 3])
    left_child = TreeNode(documents=[0, 1])
    right_child = TreeNode(documents=[2, 3])
    
    root.left = left_child
    root.right = right_child
    
    # Class 0 appears in both left and right children (never isolated)
    labels = [0, 1, 0, 1]
    
    depth = calculate_isolation_depth(root, labels, target_class=0)
    
    assert depth is None  # Never isolated


def test_calculate_isolation_depth_deeper_tree():
    """Test isolation depth in a deeper tree."""
    root = TreeNode(documents=[0, 1, 2, 3, 4, 5])
    left_child = TreeNode(documents=[0, 1, 2])
    right_child = TreeNode(documents=[3, 4, 5])
    
    # Left child splits further
    left_left = TreeNode(documents=[0, 1])
    left_right = TreeNode(documents=[2])
    left_child.left = left_left
    left_child.right = left_right
    
    root.left = left_child
    root.right = right_child
    
    # Class 0 only in left_left (depth 2)
    # Class 1 only in left_right (depth 2)
    # Class 2 only in right_child (depth 1)
    labels = [0, 0, 1, 2, 2, 2]
    
    depth_0 = calculate_isolation_depth(root, labels, target_class=0)
    depth_1 = calculate_isolation_depth(root, labels, target_class=1)
    depth_2 = calculate_isolation_depth(root, labels, target_class=2)
    
    assert depth_0 == 2
    assert depth_1 == 2
    assert depth_2 == 1


def test_calculate_all_isolation_depths():
    """Test calculating isolation depths for all classes."""
    root = TreeNode(documents=[0, 1, 2, 3, 4, 5])
    left_child = TreeNode(documents=[0, 1, 2])
    right_child = TreeNode(documents=[3, 4, 5])
    
    root.left = left_child
    root.right = right_child
    
    # Three classes: 0, 1, 2
    labels = [0, 0, 1, 1, 2, 2]
    
    isolation_depths = calculate_all_isolation_depths(root, labels)
    
    # Should have entries for all three classes
    assert len(isolation_depths) == 3
    assert 0 in isolation_depths
    assert 1 in isolation_depths
    assert 2 in isolation_depths
    # None of them are isolated (all mixed in both children)
    assert isolation_depths[0] is None
    assert isolation_depths[1] is None
    assert isolation_depths[2] is None


def test_evaluate_exploratory_power():
    """Test the main evaluation function."""
    root = TreeNode(documents=[0, 1, 2, 3, 4, 5])
    left_child = TreeNode(documents=[0, 1, 2])
    right_child = TreeNode(documents=[3, 4, 5])
    
    root.left = left_child
    root.right = right_child
    
    # Left child: mostly class 0, right child: mostly class 1
    labels = [0, 0, 0, 1, 1, 1]
    
    results = evaluate_exploratory_power(root, labels)
    
    # Check structure
    assert 'leaf_purities' in results
    assert 'isolation_depths' in results
    assert 'average_leaf_purity' in results
    assert 'num_leaves' in results
    
    # Check values
    assert len(results['leaf_purities']) == 2
    assert results['num_leaves'] == 2
    assert results['average_leaf_purity'] == 1.0  # Both leaves are pure
    assert len(results['isolation_depths']) == 2  # Two classes
    assert results['isolation_depths'][0] == 1  # Class 0 isolated at depth 1
    assert results['isolation_depths'][1] == 1  # Class 1 isolated at depth 1


def test_evaluate_exploratory_power_complex_tree():
    """Test evaluation on a more complex tree with mixed purities."""
    root = TreeNode(documents=[0, 1, 2, 3, 4, 5, 6, 7])
    left_child = TreeNode(documents=[0, 1, 2, 3])
    right_child = TreeNode(documents=[4, 5, 6, 7])
    
    # Further split left child
    left_left = TreeNode(documents=[0, 1])
    left_right = TreeNode(documents=[2, 3])
    left_child.left = left_left
    left_child.right = left_right
    
    root.left = left_child
    root.right = right_child
    
    # Mix of classes with varying purities
    labels = [0, 0, 0, 1, 1, 1, 0, 1]
    
    results = evaluate_exploratory_power(root, labels)
    
    # Should have 3 leaves: left_left, left_right, right_child
    assert results['num_leaves'] == 3
    assert len(results['leaf_purities']) == 3
    
    # left_left: [0, 0] - pure
    # left_right: [0, 1] - impure (purity = 0.5)
    # right_child: [1, 1, 0, 1] - impure
    
    # Average purity should be less than 1.0
    assert 0.0 < results['average_leaf_purity'] < 1.0
    
    # Should have isolation depths for classes 0 and 1
    assert len(results['isolation_depths']) == 2


def test_evaluate_exploratory_power_single_leaf():
    """Test evaluation on a tree that's just a single leaf."""
    root = TreeNode(documents=[0, 1, 2])
    labels = [0, 0, 1]
    
    results = evaluate_exploratory_power(root, labels)
    
    assert results['num_leaves'] == 1
    assert len(results['leaf_purities']) == 1
    
    # Most common class (0) has proportion 2/3
    # Purity = 2/3 ≈ 0.667
    assert abs(results['average_leaf_purity'] - (2/3)) < 0.001
    
    # No class is isolated (both appear in the same leaf, which is impure)
    assert results['isolation_depths'][0] is None
    assert results['isolation_depths'][1] is None


def test_node_purity_single_document():
    """Test node purity with a single document (should be perfectly pure)."""
    node = TreeNode(documents=[5])
    labels = [0, 1, 2, 3, 4, 0, 1]  # Only document 5 (class 0) is in this node
    
    purity = calculate_node_purity(node, labels)
    
    assert purity == 1.0  # Single document is perfectly pure


def test_calculate_node_entropy_pure_node():
    """Test entropy calculation for a perfectly pure node (all same class)."""
    node = TreeNode(documents=[0, 1, 2])
    labels = [0, 0, 0]  # All documents have the same label
    
    entropy = calculate_node_entropy(node, labels)
    
    # Perfect purity means zero entropy
    assert entropy == 0.0


def test_calculate_node_entropy_two_classes_even():
    """Test entropy calculation for two evenly distributed classes."""
    node = TreeNode(documents=[0, 1, 2, 3])
    labels = [0, 1, 0, 1]  # Two classes, evenly distributed
    
    entropy = calculate_node_entropy(node, labels)
    
    # With 2 classes evenly distributed: entropy = -0.5*log2(0.5) - 0.5*log2(0.5) = 1.0
    assert abs(entropy - 1.0) < 0.001


def test_calculate_node_entropy_mixed_node():
    """Test entropy calculation for a mixed node."""
    import math
    node = TreeNode(documents=[0, 1, 2, 3, 4])
    labels = [0, 0, 0, 1, 1]  # 3 of class 0, 2 of class 1
    
    entropy = calculate_node_entropy(node, labels)
    
    # Entropy = -(3/5)*log2(3/5) - (2/5)*log2(2/5)
    p0 = 3/5
    p1 = 2/5
    expected = -(p0 * math.log2(p0) + p1 * math.log2(p1))
    assert abs(entropy - expected) < 0.001


def test_calculate_node_entropy_three_classes():
    """Test entropy calculation with three evenly distributed classes."""
    import math
    node = TreeNode(documents=[0, 1, 2, 3, 4, 5])
    labels = [0, 0, 1, 1, 2, 2]  # Three classes, evenly distributed
    
    entropy = calculate_node_entropy(node, labels)
    
    # Entropy = -3 * (1/3)*log2(1/3) = log2(3)
    expected = math.log2(3)
    assert abs(entropy - expected) < 0.001


def test_calculate_node_entropy_empty_node_raises_error():
    """Test that calculating entropy on an empty node raises an error."""
    node = TreeNode(documents=[])
    labels = []
    
    with pytest.raises(ValueError, match="Node has no documents"):
        calculate_node_entropy(node, labels)


def test_evaluate_exploratory_power_includes_entropy():
    """Test that evaluate_exploratory_power includes entropy metrics."""
    root = TreeNode(documents=[0, 1, 2, 3, 4, 5])
    left_child = TreeNode(documents=[0, 1, 2])
    right_child = TreeNode(documents=[3, 4, 5])
    
    root.left = left_child
    root.right = right_child
    
    # Left child: all class 0, right child: all class 1
    labels = [0, 0, 0, 1, 1, 1]
    
    results = evaluate_exploratory_power(root, labels)
    
    # Check structure includes entropy
    assert 'leaf_entropies' in results
    assert 'average_leaf_entropy' in results
    
    # Both leaves should be pure, so entropy should be 0
    assert len(results['leaf_entropies']) == 2
    assert results['average_leaf_entropy'] == 0.0
    assert all(entropy == 0.0 for entropy in results['leaf_entropies'].values())
