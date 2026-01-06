"""
BERTopic baseline with hierarchical clustering for comparison with RTP methods.

This module provides functionality to cluster documents using BERTopic with
hierarchical clustering and convert the resulting hierarchy into a TreeNode
structure for evaluation.
"""

from typing import List, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer

from .tree_models import TreeNode


def vectorize_texts(
    texts: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    device: str = "cpu",
):
    """
    Vectorize a list of texts using a sentence transformer model.
    
    Args:
        texts: List of text strings to vectorize
        model_name: Name of the sentence transformer model to use
        device: Device to run the model on ('cpu' or 'cuda')
        
    Returns:
        Array of embeddings with shape (n_texts, embedding_dim)
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is required for text vectorization. "
            "Install it with: pip install sentence-transformers"
        )
    
    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings


def build_tree_from_bertopic_hierarchy(
    topic_model,
    topics: List[int],
    n_samples: int,
    max_tree_depth: int = 10,
    min_leaf_size: int = 1,
) -> TreeNode:
    """
    Build a binary TreeNode structure from BERTopic hierarchical clustering results.
    
    BERTopic can create a hierarchy using hierarchical clustering. This function
    converts the BERTopic hierarchy into a binary tree structure for evaluation.
    
    Args:
        topic_model: Fitted BERTopic model with hierarchical topics
        topics: Topic assignments for each document
        n_samples: Number of samples in the dataset
        max_tree_depth: Maximum depth of the tree (default: 10). Root has depth 0.
        min_leaf_size: Minimum number of documents for a node to be split (default: 1)
        
    Returns:
        Root TreeNode of the constructed tree
    """
    # Create root containing all documents
    root = TreeNode(documents=list(range(n_samples)))
    
    # Check stopping criteria: max depth reached or node too small
    if max_tree_depth <= 0 or n_samples < min_leaf_size:
        return root
    
    # Get unique topics (excluding outliers labeled as -1)
    unique_topics = sorted(set(topics))
    
    # If there's only one topic (or all outliers), return root as leaf
    if len(unique_topics) <= 1 or (len(unique_topics) == 2 and -1 in unique_topics):
        return root
    
    # Split into topics
    # Group outlier points (-1) with the first valid topic for simplicity
    valid_topics = [t for t in unique_topics if t != -1]
    
    if len(valid_topics) == 0:
        # All points are outliers
        return root
    
    # Try to use hierarchical topics if available
    try:
        hierarchical_topics = topic_model.hierarchical_topics(topics)
        # Build tree based on hierarchical structure
        return _build_tree_from_hierarchical_structure(
            hierarchical_topics, topics, n_samples, max_tree_depth, min_leaf_size
        )
    except (AttributeError, ValueError, TypeError) as e:
        # Fall back to simple binary split if hierarchical topics not available
        # AttributeError: method doesn't exist
        # ValueError: invalid input data
        # TypeError: wrong argument types
        pass
    
    # Create a simple binary split based on topics
    mid_point = len(valid_topics) // 2
    left_topics = set(valid_topics[:mid_point])
    right_topics = set(valid_topics[mid_point:])
    
    # Add outlier points to the smaller group
    if -1 in unique_topics:
        if len(left_topics) <= len(right_topics):
            left_docs = [i for i, t in enumerate(topics) if t in left_topics or t == -1]
            right_docs = [i for i, t in enumerate(topics) if t in right_topics]
        else:
            left_docs = [i for i, t in enumerate(topics) if t in left_topics]
            right_docs = [i for i, t in enumerate(topics) if t in right_topics or t == -1]
    else:
        left_docs = [i for i, t in enumerate(topics) if t in left_topics]
        right_docs = [i for i, t in enumerate(topics) if t in right_topics]
    
    if left_docs:
        root.left = TreeNode(documents=left_docs)
    if right_docs:
        root.right = TreeNode(documents=right_docs)
    
    # Recursively split children if they have multiple topics
    if root.left and len(left_topics) > 1:
        root.left = _split_node_by_topics(root.left, topics, left_topics, max_tree_depth - 1, min_leaf_size)
    if root.right and len(right_topics) > 1:
        root.right = _split_node_by_topics(root.right, topics, right_topics, max_tree_depth - 1, min_leaf_size)
    
    return root


def _build_tree_from_hierarchical_structure(
    hierarchical_topics,
    topics: List[int],
    n_samples: int,
    max_tree_depth: int = 10,
    min_leaf_size: int = 1,
) -> TreeNode:
    """
    Build a tree from BERTopic's hierarchical topics structure.
    
    Args:
        hierarchical_topics: DataFrame with hierarchical topic structure
        topics: Topic assignments for each document
        n_samples: Number of samples
        max_tree_depth: Maximum depth of the tree (default: 10)
        min_leaf_size: Minimum number of documents for a node to be split (default: 1)
        
    Returns:
        Root TreeNode of the constructed tree
    """
    # This is a more sophisticated approach that uses BERTopic's hierarchical structure
    # For now, we'll use a simple binary split approach
    # Future enhancement: fully leverage BERTopic's dendrogram structure
    
    root = TreeNode(documents=list(range(n_samples)))
    
    # Check stopping criteria
    if max_tree_depth <= 0 or n_samples < min_leaf_size:
        return root
    
    unique_topics = sorted(set(t for t in topics if t != -1))
    
    if len(unique_topics) <= 1:
        return root
    
    # Create binary splits based on topic IDs
    mid_point = len(unique_topics) // 2
    left_topics = set(unique_topics[:mid_point])
    right_topics = set(unique_topics[mid_point:])
    
    left_docs = [i for i, t in enumerate(topics) if t in left_topics or (t == -1 and len(left_topics) <= len(right_topics))]
    right_docs = [i for i, t in enumerate(topics) if t in right_topics or (t == -1 and len(left_topics) > len(right_topics))]
    
    if left_docs:
        root.left = TreeNode(documents=left_docs)
        if len(left_topics) > 1:
            root.left = _split_node_by_topics(root.left, topics, left_topics, max_tree_depth - 1, min_leaf_size)
    
    if right_docs:
        root.right = TreeNode(documents=right_docs)
        if len(right_topics) > 1:
            root.right = _split_node_by_topics(root.right, topics, right_topics, max_tree_depth - 1, min_leaf_size)
    
    return root


def _split_node_by_topics(
    node: TreeNode,
    topics,
    topic_set: set,
    max_tree_depth: int = 10,
    min_leaf_size: int = 1,
) -> TreeNode:
    """
    Helper function to recursively split a node based on topic membership.
    
    Args:
        node: TreeNode to split
        topics: Topic assignments for all documents (list or array-like)
        topic_set: Set of topic IDs in this node
        max_tree_depth: Maximum remaining depth for recursion (default: 10)
        min_leaf_size: Minimum number of documents for a node to be split (default: 1)
        
    Returns:
        Updated TreeNode with children
    """
    # Check stopping criteria
    if max_tree_depth <= 0 or len(node.documents) < min_leaf_size or len(topic_set) <= 1:
        return node
    
    # Split topics into two groups
    topic_list = sorted(topic_set)
    mid_point = len(topic_list) // 2
    left_topics = set(topic_list[:mid_point])
    right_topics = set(topic_list[mid_point:])
    
    # Split documents
    left_docs = [i for i in node.documents if topics[i] in left_topics]
    right_docs = [i for i in node.documents if topics[i] in right_topics]
    
    if left_docs:
        node.left = TreeNode(documents=left_docs)
        if len(left_topics) > 1:
            node.left = _split_node_by_topics(node.left, topics, left_topics, max_tree_depth - 1, min_leaf_size)
    
    if right_docs:
        node.right = TreeNode(documents=right_docs)
        if len(right_topics) > 1:
            node.right = _split_node_by_topics(node.right, topics, right_topics, max_tree_depth - 1, min_leaf_size)
    
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


def run_bertopic_baseline(
    texts: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    nr_topics: str = "auto",
    device: str = "cpu",
    calculate_probabilities: bool = False,
    max_tree_depth: int = 10,
    min_leaf_size: int = 1,
):
    """
    Run BERTopic clustering on a text dataset and return a TreeNode structure.
    
    BERTopic uses hierarchical clustering to create topics and can generate
    a hierarchical structure of topics for comparison with RTP methods.
    
    Args:
        texts: List of text strings to cluster
        model_name: Name of the sentence transformer model to use
        nr_topics: Number of topics to create ("auto" for automatic, or an integer)
        device: Device to run the model on ('cpu' or 'cuda')
        calculate_probabilities: Whether to calculate topic probabilities
        max_tree_depth: Maximum depth of the tree (default: 10). Root has depth 0.
        min_leaf_size: Minimum number of documents for a node to be split (default: 1)
        
    Returns:
        Tuple of (root TreeNode, embeddings array, topic_model)
    """
    try:
        from bertopic import BERTopic
        from sklearn.cluster import AgglomerativeClustering
    except ImportError:
        raise ImportError(
            "bertopic and scikit-learn are required for BERTopic clustering. "
            "Install them with: pip install bertopic scikit-learn"
        )
    
    # Vectorize texts
    embeddings = vectorize_texts(texts, model_name=model_name, device=device)
    
    # Create BERTopic model with hierarchical clustering
    # Use AgglomerativeClustering for hierarchical structure
    hdbscan_model = AgglomerativeClustering(n_clusters=None, distance_threshold=1.0)
    
    topic_model = BERTopic(
        hdbscan_model=hdbscan_model,
        nr_topics=nr_topics,
        calculate_probabilities=calculate_probabilities,
    )
    
    # Fit the model
    topics, _ = topic_model.fit_transform(texts, embeddings)
    
    # Build tree from BERTopic results
    tree = build_tree_from_bertopic_hierarchy(
        topic_model, topics, len(texts), max_tree_depth, min_leaf_size
    )
    
    return tree, embeddings, topic_model
