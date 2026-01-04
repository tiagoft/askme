"""
HDBSCAN baseline for comparison with RTP methods.

This module provides functionality to cluster documents using HDBSCAN and
convert the resulting hierarchy into a TreeNode structure for evaluation.
"""

from typing import List, Tuple
import numpy as np
from sklearn.cluster import HDBSCAN
from sentence_transformers import SentenceTransformer

from .tree_models import TreeNode


def vectorize_texts(
    texts: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    device: str = "cpu",
) -> np.ndarray:
    """
    Vectorize a list of texts using a sentence transformer model.
    
    Args:
        texts: List of text strings to vectorize
        model_name: Name of the sentence transformer model to use
        device: Device to run the model on ('cpu' or 'cuda')
        
    Returns:
        Array of embeddings with shape (n_texts, embedding_dim)
    """
    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings


def build_tree_from_hdbscan(
    clusterer: HDBSCAN,
    n_samples: int,
) -> TreeNode:
    """
    Build a binary TreeNode structure from HDBSCAN clustering results.
    
    HDBSCAN creates a hierarchy, but it's not necessarily binary. This function
    converts the HDBSCAN hierarchy into a binary tree structure by iteratively
    splitting clusters based on the condensed tree.
    
    Args:
        clusterer: Fitted HDBSCAN clusterer
        n_samples: Number of samples in the dataset
        
    Returns:
        Root TreeNode of the constructed tree
    """
    # Get cluster labels
    labels = clusterer.labels_
    
    # Create a simple binary tree based on cluster membership
    # Root contains all documents
    root = TreeNode(documents=list(range(n_samples)))
    
    # Get unique cluster labels (excluding noise points labeled as -1)
    unique_clusters = sorted(set(labels))
    
    # If there's only one cluster (or all noise), return root as leaf
    if len(unique_clusters) <= 1 or (len(unique_clusters) == 2 and -1 in unique_clusters):
        return root
    
    # Split into clusters
    # Group noise points (-1) with the first valid cluster for simplicity
    valid_clusters = [c for c in unique_clusters if c != -1]
    
    if len(valid_clusters) == 0:
        # All points are noise
        return root
    
    # Create a simple binary split
    # Left: first half of clusters, Right: second half of clusters
    mid_point = len(valid_clusters) // 2
    left_clusters = set(valid_clusters[:mid_point])
    right_clusters = set(valid_clusters[mid_point:])
    
    # Add noise points to the smaller group
    if -1 in unique_clusters:
        noise_docs = [i for i, l in enumerate(labels) if l == -1]
        if len(left_clusters) <= len(right_clusters):
            left_docs = [i for i, l in enumerate(labels) if l in left_clusters or l == -1]
            right_docs = [i for i, l in enumerate(labels) if l in right_clusters]
        else:
            left_docs = [i for i, l in enumerate(labels) if l in left_clusters]
            right_docs = [i for i, l in enumerate(labels) if l in right_clusters or l == -1]
    else:
        left_docs = [i for i, l in enumerate(labels) if l in left_clusters]
        right_docs = [i for i, l in enumerate(labels) if l in right_clusters]
    
    if left_docs:
        root.left = TreeNode(documents=left_docs)
    if right_docs:
        root.right = TreeNode(documents=right_docs)
    
    # Recursively split children if they have multiple clusters
    if root.left and len(left_clusters) > 1:
        root.left = _split_node_by_clusters(root.left, labels, left_clusters)
    if root.right and len(right_clusters) > 1:
        root.right = _split_node_by_clusters(root.right, labels, right_clusters)
    
    return root


def _split_node_by_clusters(
    node: TreeNode,
    labels: np.ndarray,
    clusters: set,
) -> TreeNode:
    """
    Helper function to recursively split a node based on cluster membership.
    
    Args:
        node: TreeNode to split
        labels: Cluster labels for all documents
        clusters: Set of cluster IDs in this node
        
    Returns:
        Updated TreeNode with children
    """
    if len(clusters) <= 1:
        return node
    
    # Split clusters into two groups
    cluster_list = sorted(clusters)
    mid_point = len(cluster_list) // 2
    left_clusters = set(cluster_list[:mid_point])
    right_clusters = set(cluster_list[mid_point:])
    
    # Split documents
    left_docs = [i for i in node.documents if labels[i] in left_clusters]
    right_docs = [i for i in node.documents if labels[i] in right_clusters]
    
    if left_docs:
        node.left = TreeNode(documents=left_docs)
        if len(left_clusters) > 1:
            node.left = _split_node_by_clusters(node.left, labels, left_clusters)
    
    if right_docs:
        node.right = TreeNode(documents=right_docs)
        if len(right_clusters) > 1:
            node.right = _split_node_by_clusters(node.right, labels, right_clusters)
    
    return node


def calculate_tree_depth(root: TreeNode) -> int:
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


def run_hdbscan_baseline(
    texts: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    min_cluster_size: int = 5,
    min_samples: int = 1,
    device: str = "cpu",
) -> Tuple[TreeNode, np.ndarray]:
    """
    Run HDBSCAN clustering on a text dataset and return a TreeNode structure.
    
    Args:
        texts: List of text strings to cluster
        model_name: Name of the sentence transformer model to use
        min_cluster_size: Minimum cluster size for HDBSCAN
        min_samples: Minimum samples parameter for HDBSCAN
        device: Device to run the model on ('cpu' or 'cuda')
        
    Returns:
        Tuple of (root TreeNode, embeddings array)
    """
    # Vectorize texts
    embeddings = vectorize_texts(texts, model_name=model_name, device=device)
    
    # Run HDBSCAN
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
    )
    clusterer.fit(embeddings)
    
    # Build tree from HDBSCAN results
    tree = build_tree_from_hdbscan(clusterer, len(texts))
    
    return tree, embeddings
