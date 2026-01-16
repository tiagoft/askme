"""
Supervised (label-aware) metrics for evaluating RTP trees.

This module provides metrics that use ground truth labels to evaluate
tree quality, including clustering metrics like NMI, ARI, and classification
metrics like accuracy and F1-score.
"""

from typing import List, Dict, Optional, Union, Tuple
from collections import Counter
import numpy as np
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    homogeneity_completeness_v_measure,
    accuracy_score,
    f1_score,
    confusion_matrix,
)
from .tree_models import TreeNode


def get_all_nodes(root: TreeNode) -> List[TreeNode]:
    """
    Get all nodes from a tree (internal nodes and leaves).
    
    Args:
        root: The root TreeNode of the tree
        
    Returns:
        List of all nodes in the tree
    """
    nodes = []
    
    def traverse(node: TreeNode):
        nodes.append(node)
        if node.left is not None:
            traverse(node.left)
        if node.right is not None:
            traverse(node.right)
    
    traverse(root)
    return nodes


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
            leaves.append(node)
        else:
            if node.left is not None:
                traverse(node.left)
            if node.right is not None:
                traverse(node.right)
    
    traverse(root)
    return leaves


def _get_cluster_assignments(root: TreeNode, labels: List[int], use_leaves_only: bool = True) -> Tuple[List[int], List[int]]:
    """
    Get cluster assignments for documents based on tree structure.
    
    For each document, assigns it to a cluster ID based on which node contains it.
    Also returns the true labels for these documents.
    
    Args:
        root: The root TreeNode of the tree
        labels: List of true labels where labels[i] is the label for document i
        use_leaves_only: If True, only consider leaf nodes; if False, consider all nodes
        
    Returns:
        Tuple of (predicted_clusters, true_labels) where both are lists aligned by document index
    """
    # Get nodes to consider
    nodes = get_all_leaves(root) if use_leaves_only else get_all_nodes(root)
    
    # Build mapping from document to cluster ID
    doc_to_cluster = {}
    for cluster_id, node in enumerate(nodes):
        for doc_idx in node.documents:
            # If a document appears in multiple nodes (shouldn't happen in leaves),
            # use the first assignment
            if doc_idx not in doc_to_cluster:
                doc_to_cluster[doc_idx] = cluster_id
    
    # Create aligned lists of predictions and true labels
    doc_indices = sorted(doc_to_cluster.keys())
    predicted_clusters = [doc_to_cluster[doc_idx] for doc_idx in doc_indices]
    true_labels = [labels[doc_idx] for doc_idx in doc_indices]
    
    return predicted_clusters, true_labels


def _get_predicted_labels(root: TreeNode, labels: List[int], use_leaves_only: bool = True) -> Tuple[List[int], List[int]]:
    """
    Get predicted class labels for documents based on majority class in each node.
    
    Each document is assigned the majority class of the node it belongs to.
    
    Args:
        root: The root TreeNode of the tree
        labels: List of true labels where labels[i] is the label for document i
        use_leaves_only: If True, only consider leaf nodes; if False, consider all nodes
        
    Returns:
        Tuple of (predicted_labels, true_labels) where both are lists aligned by document index
    """
    # Get nodes to consider
    nodes = get_all_leaves(root) if use_leaves_only else get_all_nodes(root)
    
    # Build mapping from document to predicted label (majority class in node)
    doc_to_predicted = {}
    for node in nodes:
        if not node.documents:
            continue
        
        # Get majority class in this node
        node_labels = [labels[doc_idx] for doc_idx in node.documents]
        majority_class = Counter(node_labels).most_common(1)[0][0]
        
        # Assign all documents in this node to the majority class
        for doc_idx in node.documents:
            if doc_idx not in doc_to_predicted:
                doc_to_predicted[doc_idx] = majority_class
    
    # Create aligned lists of predictions and true labels
    doc_indices = sorted(doc_to_predicted.keys())
    predicted_labels = [doc_to_predicted[doc_idx] for doc_idx in doc_indices]
    true_labels = [labels[doc_idx] for doc_idx in doc_indices]
    
    return predicted_labels, true_labels


class SupervisedMetric:
    """
    Base class for supervised (label-aware) tree metrics.
    
    All supervised metrics should inherit from this class and implement
    the call method.
    """
    
    def call(self, root: TreeNode, labels: List[int], **kwargs):
        """
        Calculate the metric for the given tree and labels.
        
        Args:
            root: The root TreeNode of the tree
            labels: List of true labels where labels[i] is the label for document i
            **kwargs: Additional metric-specific parameters
            
        Returns:
            The computed metric value
            
        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement the call method")


def _normalized_mutual_info(root: TreeNode, labels: List[int], use_leaves_only: bool = True, average_method: str = 'arithmetic') -> float:
    """
    Calculate Normalized Mutual Information between tree structure and true labels.
    
    NMI measures the mutual information between cluster assignments (tree nodes)
    and true labels, normalized to be between 0 and 1.
    
    Args:
        root: The root TreeNode of the tree
        labels: List of true labels where labels[i] is the label for document i
        use_leaves_only: If True, only consider leaf nodes; if False, consider all nodes
        average_method: How to average mutual information. Options: 'min', 'geometric', 'arithmetic', 'max'
        
    Returns:
        NMI score between 0 and 1 (higher is better)
    """
    predicted_clusters, true_labels = _get_cluster_assignments(root, labels, use_leaves_only)
    
    if len(predicted_clusters) == 0:
        return 0.0
    
    return normalized_mutual_info_score(true_labels, predicted_clusters, average_method=average_method)


class NormalizedMutualInformation(SupervisedMetric):
    """
    Normalized Mutual Information (NMI) metric.
    
    Measures the mutual information between cluster assignments (tree nodes)
    and true labels, normalized to be between 0 and 1.
    """
    
    def call(self, root: TreeNode, labels: List[int], use_leaves_only: bool = True, average_method: str = 'arithmetic', **kwargs) -> float:
        """
        Calculate NMI for the tree.
        
        Args:
            root: The root TreeNode of the tree
            labels: List of true labels where labels[i] is the label for document i
            use_leaves_only: If True, only consider leaf nodes; if False, consider all nodes
            average_method: How to average mutual information. Options: 'min', 'geometric', 'arithmetic', 'max'
            **kwargs: Additional parameters (ignored)
            
        Returns:
            NMI score between 0 and 1 (higher is better)
        """
        return _normalized_mutual_info(root, labels, use_leaves_only, average_method)


def _adjusted_rand_index(root: TreeNode, labels: List[int], use_leaves_only: bool = True) -> float:
    """
    Calculate Adjusted Rand Index between tree structure and true labels.
    
    ARI measures the similarity between cluster assignments (tree nodes)
    and true labels, adjusted for chance. Score is between -1 and 1, where
    1 means perfect agreement, 0 means random labeling, and negative values
    mean worse than random.
    
    Args:
        root: The root TreeNode of the tree
        labels: List of true labels where labels[i] is the label for document i
        use_leaves_only: If True, only consider leaf nodes; if False, consider all nodes
        
    Returns:
        ARI score between -1 and 1 (higher is better)
    """
    predicted_clusters, true_labels = _get_cluster_assignments(root, labels, use_leaves_only)
    
    if len(predicted_clusters) == 0:
        return 0.0
    
    return adjusted_rand_score(true_labels, predicted_clusters)


class AdjustedRandIndex(SupervisedMetric):
    """
    Adjusted Rand Index (ARI) metric.
    
    Measures the similarity between cluster assignments (tree nodes)
    and true labels, adjusted for chance.
    """
    
    def call(self, root: TreeNode, labels: List[int], use_leaves_only: bool = True, **kwargs) -> float:
        """
        Calculate ARI for the tree.
        
        Args:
            root: The root TreeNode of the tree
            labels: List of true labels where labels[i] is the label for document i
            use_leaves_only: If True, only consider leaf nodes; if False, consider all nodes
            **kwargs: Additional parameters (ignored)
            
        Returns:
            ARI score between -1 and 1 (higher is better)
        """
        return _adjusted_rand_index(root, labels, use_leaves_only)


def _homogeneity_completeness_vmeasure(root: TreeNode, labels: List[int], use_leaves_only: bool = True, beta: float = 1.0) -> Dict[str, float]:
    """
    Calculate Homogeneity, Completeness, and V-measure.
    
    - Homogeneity: Each cluster contains only members of a single class
    - Completeness: All members of a given class are assigned to the same cluster
    - V-measure: Harmonic mean of homogeneity and completeness
    
    Args:
        root: The root TreeNode of the tree
        labels: List of true labels where labels[i] is the label for document i
        use_leaves_only: If True, only consider leaf nodes; if False, consider all nodes
        beta: Weight of homogeneity vs completeness in V-measure (default 1.0 for harmonic mean)
        
    Returns:
        Dictionary with keys 'homogeneity', 'completeness', 'v_measure'
    """
    predicted_clusters, true_labels = _get_cluster_assignments(root, labels, use_leaves_only)
    
    if len(predicted_clusters) == 0:
        return {'homogeneity': 0.0, 'completeness': 0.0, 'v_measure': 0.0}
    
    h, c, v = homogeneity_completeness_v_measure(true_labels, predicted_clusters, beta=beta)
    
    return {
        'homogeneity': h,
        'completeness': c,
        'v_measure': v,
    }


class HomogeneityCompletenessVMeasure(SupervisedMetric):
    """
    Homogeneity, Completeness, and V-measure metrics.
    
    Returns all three metrics as a dictionary.
    """
    
    def call(self, root: TreeNode, labels: List[int], use_leaves_only: bool = True, beta: float = 1.0, **kwargs) -> Dict[str, float]:
        """
        Calculate Homogeneity, Completeness, and V-measure for the tree.
        
        Args:
            root: The root TreeNode of the tree
            labels: List of true labels where labels[i] is the label for document i
            use_leaves_only: If True, only consider leaf nodes; if False, consider all nodes
            beta: Weight of homogeneity vs completeness in V-measure (default 1.0)
            **kwargs: Additional parameters (ignored)
            
        Returns:
            Dictionary with keys 'homogeneity', 'completeness', 'v_measure'
        """
        return _homogeneity_completeness_vmeasure(root, labels, use_leaves_only, beta)


def _accuracy(root: TreeNode, labels: List[int], use_leaves_only: bool = True) -> float:
    """
    Calculate classification accuracy.
    
    Each document is assigned the majority class of the node it belongs to,
    then accuracy is computed against true labels.
    
    Args:
        root: The root TreeNode of the tree
        labels: List of true labels where labels[i] is the label for document i
        use_leaves_only: If True, only consider leaf nodes; if False, consider all nodes
        
    Returns:
        Accuracy score between 0 and 1 (higher is better)
    """
    predicted_labels, true_labels = _get_predicted_labels(root, labels, use_leaves_only)
    
    if len(predicted_labels) == 0:
        return 0.0
    
    return accuracy_score(true_labels, predicted_labels)


class Accuracy(SupervisedMetric):
    """
    Classification accuracy metric.
    
    Each document is assigned the majority class of the node it belongs to.
    """
    
    def call(self, root: TreeNode, labels: List[int], use_leaves_only: bool = True, **kwargs) -> float:
        """
        Calculate accuracy for the tree.
        
        Args:
            root: The root TreeNode of the tree
            labels: List of true labels where labels[i] is the label for document i
            use_leaves_only: If True, only consider leaf nodes; if False, consider all nodes
            **kwargs: Additional parameters (ignored)
            
        Returns:
            Accuracy score between 0 and 1 (higher is better)
        """
        return _accuracy(root, labels, use_leaves_only)


def _f1_score(root: TreeNode, labels: List[int], use_leaves_only: bool = True, average: str = 'weighted') -> float:
    """
    Calculate F1-score.
    
    Each document is assigned the majority class of the node it belongs to,
    then F1-score is computed against true labels.
    
    Args:
        root: The root TreeNode of the tree
        labels: List of true labels where labels[i] is the label for document i
        use_leaves_only: If True, only consider leaf nodes; if False, consider all nodes
        average: Averaging method for multiclass. Options: 'micro', 'macro', 'weighted', 'samples', or None
        
    Returns:
        F1-score (higher is better)
    """
    predicted_labels, true_labels = _get_predicted_labels(root, labels, use_leaves_only)
    
    if len(predicted_labels) == 0:
        return 0.0
    
    return f1_score(true_labels, predicted_labels, average=average, zero_division=0)


class F1Score(SupervisedMetric):
    """
    F1-score metric.
    
    Each document is assigned the majority class of the node it belongs to.
    """
    
    def call(self, root: TreeNode, labels: List[int], use_leaves_only: bool = True, average: str = 'weighted', **kwargs) -> float:
        """
        Calculate F1-score for the tree.
        
        Args:
            root: The root TreeNode of the tree
            labels: List of true labels where labels[i] is the label for document i
            use_leaves_only: If True, only consider leaf nodes; if False, consider all nodes
            average: Averaging method for multiclass. Options: 'micro', 'macro', 'weighted', 'samples', or None
            **kwargs: Additional parameters (ignored)
            
        Returns:
            F1-score (higher is better)
        """
        return _f1_score(root, labels, use_leaves_only, average)


def _confusion_matrix(root: TreeNode, labels: List[int], use_leaves_only: bool = True) -> np.ndarray:
    """
    Calculate confusion matrix.
    
    Each document is assigned the majority class of the node it belongs to,
    then a confusion matrix is computed against true labels.
    
    Args:
        root: The root TreeNode of the tree
        labels: List of true labels where labels[i] is the label for document i
        use_leaves_only: If True, only consider leaf nodes; if False, consider all nodes
        
    Returns:
        Confusion matrix as a numpy array where entry [i, j] is the number of
        documents with true label i predicted as label j
    """
    predicted_labels, true_labels = _get_predicted_labels(root, labels, use_leaves_only)
    
    if len(predicted_labels) == 0:
        # Return empty 1D array for empty inputs
        return np.array([])
    
    return confusion_matrix(true_labels, predicted_labels)


class ConfusionMatrix(SupervisedMetric):
    """
    Confusion matrix metric.
    
    Each document is assigned the majority class of the node it belongs to.
    """
    
    def call(self, root: TreeNode, labels: List[int], use_leaves_only: bool = True, **kwargs) -> np.ndarray:
        """
        Calculate confusion matrix for the tree.
        
        Args:
            root: The root TreeNode of the tree
            labels: List of true labels where labels[i] is the label for document i
            use_leaves_only: If True, only consider leaf nodes; if False, consider all nodes
            **kwargs: Additional parameters (ignored)
            
        Returns:
            Confusion matrix as a numpy array
        """
        return _confusion_matrix(root, labels, use_leaves_only)
