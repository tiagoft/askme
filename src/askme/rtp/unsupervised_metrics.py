"""
Unsupervised (no-label) metrics for RTP trees.

This module provides metrics that can be calculated on trees without requiring
document labels. These metrics help evaluate tree structure and balance.
"""

from .tree_models import TreeNode
from numpy import mean

class UnsupervisedMetric:
    """Base class for unsupervised tree metrics."""
    
    def __call__(self, root: TreeNode, **kwargs):
        """
        Calculate the metric for the given tree.
        
        Args:
            root: The root TreeNode of the tree
            **kwargs: Additional arguments specific to the metric
            
        Returns:
            The calculated metric value
            
        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError


def _count_nodes(root: TreeNode) -> int:
    """
    Count the total number of nodes in the tree.
    
    Args:
        root: The root TreeNode of the tree
        
    Returns:
        Total number of nodes in the tree
    """
    if root is None:
        return 0
    
    count = 1  # Count the root node
    
    if root.left is not None:
        count += _count_nodes(root.left)
    if root.right is not None:
        count += _count_nodes(root.right)
    
    return count


class NumberOfNodes(UnsupervisedMetric):
    """Metric that counts the total number of nodes in the tree."""
    
    def __call__(self, root: TreeNode, **kwargs) -> int:
        """
        Count the total number of nodes in the tree.
        
        Args:
            root: The root TreeNode of the tree
            **kwargs: Additional arguments (unused)
            
        Returns:
            Total number of nodes in the tree
        """
        return _count_nodes(root)


def _calculate_tree_height(root: TreeNode) -> int:
    """
    Calculate the height of the tree.
    
    Height is defined as the maximum number of edges from root to any leaf.
    A tree with only a root node has height 0.
    
    Args:
        root: The root TreeNode of the tree
        
    Returns:
        Height of the tree
    """
    if root is None:
        return -1  # Empty tree has height -1
    
    # If this is a leaf node, height is 0
    if root.left is None and root.right is None:
        return 0
    
    # Calculate height of left and right subtrees
    left_height = _calculate_tree_height(root.left) if root.left is not None else -1
    right_height = _calculate_tree_height(root.right) if root.right is not None else -1
    
    # Height is 1 + maximum of left and right heights
    return 1 + max(left_height, right_height)


class TreeHeight(UnsupervisedMetric):
    """Metric that calculates the height of the tree."""
    
    def __call__(self, root: TreeNode, **kwargs) -> int:
        """
        Calculate the height of the tree.
        
        Height is defined as the maximum number of edges from root to any leaf.
        A tree with only a root node has height 0.
        
        Args:
            root: The root TreeNode of the tree
            **kwargs: Additional arguments (unused)
            
        Returns:
            Height of the tree
        """
        return _calculate_tree_height(root)


def _count_leaf_nodes(root: TreeNode) -> int:
    """
    Count the number of leaf nodes in the tree.
    
    A leaf node is a node with no children.
    
    Args:
        root: The root TreeNode of the tree
        
    Returns:
        Number of leaf nodes in the tree
    """
    if root is None:
        return 0
    
    # If this is a leaf node
    if root.left is None and root.right is None:
        return 1
    
    # Count leaves in left and right subtrees
    left_leaves = _count_leaf_nodes(root.left) if root.left is not None else 0
    right_leaves = _count_leaf_nodes(root.right) if root.right is not None else 0
    
    return left_leaves + right_leaves


class NumberOfLeafNodes(UnsupervisedMetric):
    """Metric that counts the number of leaf nodes in the tree."""
    
    def __call__(self, root: TreeNode, **kwargs) -> int:
        """
        Count the number of leaf nodes in the tree.
        
        A leaf node is a node with no children.
        
        Args:
            root: The root TreeNode of the tree
            **kwargs: Additional arguments (unused)
            
        Returns:
            Number of leaf nodes in the tree
        """
        return _count_leaf_nodes(root)


def _tree_node_unbalance(root: TreeNode) -> int:
    """
    Calculate the tree node unbalance.
    
    Node unbalance is defined as the absolute difference between the height
    of the left child and the height of the right child.
    
    Args:
        root: The root TreeNode of the tree
        
    Returns:
        Absolute difference between left and right subtree heights
    """
    if root is None:
        return 0
    
    # Calculate height of left and right subtrees
    left_height = _calculate_tree_height(root.left) if root.left is not None else -1
    right_height = _calculate_tree_height(root.right) if root.right is not None else -1
    
    return abs(left_height - right_height)


class TreeNodeUnbalance(UnsupervisedMetric):
    """Metric that calculates tree node unbalance based on subtree heights."""
    
    def __call__(self, root: TreeNode, **kwargs) -> int:
        """
        Calculate the tree node unbalance.
        
        Node unbalance is defined as the absolute difference between the height
        of the left child and the height of the right child.
        
        Args:
            root: The root TreeNode of the tree
            **kwargs: Additional arguments (unused)
            
        Returns:
            Absolute difference between left and right subtree heights
        """
        return _tree_node_unbalance(root)


def _documents_per_leaf(root: TreeNode) -> list[int]:
    """
    Get the number of documents in each leaf node.
    
    Returns a list where each element is the count of documents in a leaf node.
    This can be used to calculate mean and variance of document distribution.
    
    Args:
        root: The root TreeNode of the tree
        
    Returns:
        List of document counts for each leaf node
    """
    if root is None:
        return []
    
    # If this is a leaf node
    if root.left is None and root.right is None:
        return [len(root.documents)]
    
    # Collect documents from left and right subtrees
    result = []
    if root.left is not None:
        result.extend(_documents_per_leaf(root.left))
    if root.right is not None:
        result.extend(_documents_per_leaf(root.right))
    
    return result


class DocumentsPerLeaf(UnsupervisedMetric):
    """Metric that returns the number of documents in each leaf node."""
    def __init__(self, pool_fn=mean,) -> None:
        super().__init__()
        self.pool_fn = pool_fn
        
    def __call__(self, root: TreeNode,  **kwargs) -> list[int]:
        """
        Get the number of documents in each leaf node.
        
        Returns a list where each element is the count of documents in a leaf node.
        This can be used to calculate mean and variance of document distribution.
        
        Args:
            root: The root TreeNode of the tree
            **kwargs: Additional arguments (unused)
            
        Returns:
            List of document counts for each leaf node
        """
        return self.pool_fn(_documents_per_leaf(root))


def _count_documents_in_node(node: TreeNode) -> int:
    """
    Count the number of documents in a node.
    
    Args:
        node: A TreeNode (can be None)
        
    Returns:
        Number of documents in the node, or 0 if node is None
    """
    if node is None:
        return 0
    return len(node.documents)


def _tree_document_unbalance(root: TreeNode) -> int:
    """
    Calculate the tree document unbalance.
    
    Document unbalance is defined as the absolute difference between the number
    of documents in the left child and the number of documents in the right child.
    
    Args:
        root: The root TreeNode of the tree
        
    Returns:
        Absolute difference between document counts in left and right children
    """
    if root is None:
        return 0
    
    left_docs = _count_documents_in_node(root.left)
    right_docs = _count_documents_in_node(root.right)
    
    return abs(left_docs - right_docs)


class TreeDocumentUnbalance(UnsupervisedMetric):
    """Metric that calculates tree document unbalance based on document counts."""
    
    def __call__(self, root: TreeNode, **kwargs) -> int:
        """
        Calculate the tree document unbalance.
        
        Document unbalance is defined as the absolute difference between the number
        of documents in the left child and the number of documents in the right child.
        
        Args:
            root: The root TreeNode of the tree
            **kwargs: Additional arguments (unused)
            
        Returns:
            Absolute difference between document counts in left and right children
        """
        return _tree_document_unbalance(root)
