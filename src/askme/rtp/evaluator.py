"""
Evaluator for Exploratory Power metrics of RTP trees.

This module provides functions to evaluate the quality of RTP trees based on
their ability to separate documents with different labels.
"""

from typing import Dict, List, Optional, Tuple
from collections import Counter
from .tree_models import TreeNode


def calculate_node_purity(node: TreeNode, labels: List[int]) -> float:
    """
    Calculate the purity of a node using Gini impurity.
    
    Node purity is measured as 1 - Gini impurity, where:
    - Gini impurity = 1 - sum(p_i^2) for each class i
    - p_i is the proportion of documents of class i in the node
    - Purity = 1 means all documents belong to the same class (pure)
    - Purity = 0 means documents are evenly distributed across classes (impure)
    
    Args:
        node: A TreeNode containing document indices
        labels: List of labels where labels[i] is the label for document i
        
    Returns:
        Node purity score between 0 and 1
        
    Raises:
        ValueError: If node has no documents or if document indices are out of range
    """
    if not node.documents:
        raise ValueError("Node has no documents")
    
    # Get labels for documents in this node
    try:
        node_labels = [labels[doc_idx] for doc_idx in node.documents]
    except IndexError as e:
        raise ValueError(f"Document index out of range: {e}")
    
    # Count occurrences of each label
    label_counts = Counter(node_labels)
    total_docs = len(node_labels)
    
    # Calculate Gini impurity
    gini_impurity = 1.0
    for count in label_counts.values():
        proportion = count / total_docs
        gini_impurity -= proportion ** 2
    
    # Purity is 1 - Gini impurity
    purity = 1.0 - gini_impurity
    
    return purity


def get_all_leaves(root: TreeNode) -> List[TreeNode]:
    """
    Get all leaf nodes from a tree.
    
    Args:
        root: The root TreeNode of the tree
        
    Returns:
        List of all leaf nodes (nodes with no children)
    """
    leaves = []
    
    def traverse(node: TreeNode):
        if node.left is None and node.right is None:
            # This is a leaf node
            leaves.append(node)
        else:
            # Traverse children
            if node.left is not None:
                traverse(node.left)
            if node.right is not None:
                traverse(node.right)
    
    traverse(root)
    return leaves


def calculate_all_leaf_purities(root: TreeNode, labels: List[int]) -> Dict[int, float]:
    """
    Calculate node purity for all leaf nodes in the tree.
    
    Args:
        root: The root TreeNode of the tree
        labels: List of labels where labels[i] is the label for document i
        
    Returns:
        Dictionary mapping leaf node id (using first document index as identifier)
        to its purity score
    """
    leaves = get_all_leaves(root)
    purities = {}
    
    for leaf in leaves:
        if leaf.documents:
            # Use the first document index as a unique identifier for this leaf
            leaf_id = leaf.documents[0]
            purities[leaf_id] = calculate_node_purity(leaf, labels)
    
    return purities


def calculate_isolation_depth(root: TreeNode, labels: List[int], target_class: int) -> Optional[int]:
    """
    Calculate the isolation depth for a specific class.
    
    Isolation depth is the minimum depth at which a node contains only documents
    of the target class (i.e., the class becomes isolated from all other classes).
    
    Args:
        root: The root TreeNode of the tree
        labels: List of labels where labels[i] is the label for document i
        target_class: The class label to find isolation depth for
        
    Returns:
        The minimum depth at which the target class is isolated, or None if the
        class is never fully isolated in the tree
    """
    min_isolation_depth = None
    
    def traverse(node: TreeNode, depth: int):
        nonlocal min_isolation_depth
        
        if not node.documents:
            return
        
        # Get labels for documents in this node
        node_labels = [labels[doc_idx] for doc_idx in node.documents]
        
        # Check if this node contains only the target class
        unique_labels = set(node_labels)
        
        # If this node only contains the target class and has at least one document
        if unique_labels == {target_class}:
            # Update minimum isolation depth
            if min_isolation_depth is None or depth < min_isolation_depth:
                min_isolation_depth = depth
            # No need to traverse children as they will be at greater depth
            return
        
        # Traverse children
        if node.left is not None:
            traverse(node.left, depth + 1)
        if node.right is not None:
            traverse(node.right, depth + 1)
    
    traverse(root, depth=0)
    return min_isolation_depth


def calculate_all_isolation_depths(root: TreeNode, labels: List[int]) -> Dict[int, Optional[int]]:
    """
    Calculate isolation depths for all classes present in the labels.
    
    Args:
        root: The root TreeNode of the tree
        labels: List of labels where labels[i] is the label for document i
        
    Returns:
        Dictionary mapping class label to its isolation depth (or None if never isolated)
    """
    # Find all unique classes in the labels
    unique_classes = set(labels)
    
    # Calculate isolation depth for each class
    isolation_depths = {}
    for class_label in unique_classes:
        isolation_depths[class_label] = calculate_isolation_depth(root, labels, class_label)
    
    return isolation_depths


def evaluate_exploratory_power(root: TreeNode, labels: List[int]) -> Dict:
    """
    Evaluate the exploratory power of an RTP tree.
    
    This is the main evaluation function that calculates both node purity
    for all leaves and isolation depths for all classes.
    
    Args:
        root: The root TreeNode of the tree
        labels: List of labels where labels[i] is the label for document i
        
    Returns:
        Dictionary containing:
            - 'leaf_purities': Dict mapping leaf identifiers to purity scores
            - 'isolation_depths': Dict mapping class labels to isolation depths
            - 'average_leaf_purity': Average purity across all leaves
            - 'num_leaves': Total number of leaf nodes
    """
    # Calculate leaf purities
    leaf_purities = calculate_all_leaf_purities(root, labels)
    
    # Calculate isolation depths
    isolation_depths = calculate_all_isolation_depths(root, labels)
    
    # Calculate average leaf purity
    avg_purity = sum(leaf_purities.values()) / len(leaf_purities) if leaf_purities else 0.0
    
    return {
        'leaf_purities': leaf_purities,
        'isolation_depths': isolation_depths,
        'average_leaf_purity': avg_purity,
        'num_leaves': len(leaf_purities),
    }
