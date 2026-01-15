"""
Self-supervised tree metrics for evaluating RTP trees without labels.

This module provides metrics to evaluate tree quality using only document embeddings
and tree structure, without requiring ground truth labels.
"""

from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod
import numpy as np
from .tree_models import TreeNode


class SelfSupervisedMetric(ABC):
    """
    Base class for self-supervised tree metrics.
    
    All metrics should inherit from this class and implement the call method.
    """
    
    @abstractmethod
    def call(self, root: TreeNode, embeddings: np.ndarray, **kwargs) -> Any:
        """
        Calculate the metric for a given tree and embeddings.
        
        Args:
            root: Root node of the tree
            embeddings: Document embeddings array of shape (n_documents, embedding_dim)
            **kwargs: Additional metric-specific parameters
            
        Returns:
            Metric value(s)
        """
        raise NotImplementedError


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


def _silhouette_score_metric(root: TreeNode, embeddings: np.ndarray, **kwargs) -> float:
    """
    Calculate silhouette score for leaf node clusters.
    
    Treats each leaf node as a cluster and calculates how well documents
    are clustered within their leaf nodes.
    
    Args:
        root: Root node of the tree
        embeddings: Document embeddings array of shape (n_documents, embedding_dim)
        **kwargs: Additional parameters (unused)
        
    Returns:
        Silhouette score (-1 to 1, higher is better)
        
    Raises:
        ValueError: If there are fewer than 2 clusters or all documents in one cluster
    """
    from sklearn.metrics import silhouette_score
    
    leaves = get_all_leaves(root)
    
    # Need at least 2 clusters
    if len(leaves) < 2:
        raise ValueError("Need at least 2 leaf nodes (clusters) to calculate silhouette score")
    
    # Create cluster labels based on leaf membership
    n_documents = embeddings.shape[0]
    labels = np.full(n_documents, -1, dtype=int)
    
    for cluster_id, leaf in enumerate(leaves):
        for doc_idx in leaf.documents:
            labels[doc_idx] = cluster_id
    
    # Check that all documents are assigned
    if np.any(labels == -1):
        raise ValueError("Not all documents are assigned to leaf nodes")
    
    # Check that there are at least 2 clusters with documents
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        raise ValueError("Need at least 2 non-empty clusters")
    
    return silhouette_score(embeddings, labels)


def _davies_bouldin_score_metric(root: TreeNode, embeddings: np.ndarray, **kwargs) -> float:
    """
    Calculate Davies-Bouldin score for leaf node clusters.
    
    Measures the average similarity ratio of each cluster with its most similar cluster.
    Lower values indicate better clustering.
    
    Args:
        root: Root node of the tree
        embeddings: Document embeddings array of shape (n_documents, embedding_dim)
        **kwargs: Additional parameters (unused)
        
    Returns:
        Davies-Bouldin score (>= 0, lower is better)
        
    Raises:
        ValueError: If there are fewer than 2 clusters
    """
    from sklearn.metrics import davies_bouldin_score
    
    leaves = get_all_leaves(root)
    
    # Need at least 2 clusters
    if len(leaves) < 2:
        raise ValueError("Need at least 2 leaf nodes (clusters) to calculate Davies-Bouldin score")
    
    # Create cluster labels based on leaf membership
    n_documents = embeddings.shape[0]
    labels = np.full(n_documents, -1, dtype=int)
    
    for cluster_id, leaf in enumerate(leaves):
        for doc_idx in leaf.documents:
            labels[doc_idx] = cluster_id
    
    # Check that all documents are assigned
    if np.any(labels == -1):
        raise ValueError("Not all documents are assigned to leaf nodes")
    
    return davies_bouldin_score(embeddings, labels)


def _calinski_harabasz_score_metric(root: TreeNode, embeddings: np.ndarray, **kwargs) -> float:
    """
    Calculate Calinski-Harabasz score (variance ratio) for leaf node clusters.
    
    Measures the ratio of between-cluster dispersion to within-cluster dispersion.
    Higher values indicate better defined clusters.
    
    Args:
        root: Root node of the tree
        embeddings: Document embeddings array of shape (n_documents, embedding_dim)
        **kwargs: Additional parameters (unused)
        
    Returns:
        Calinski-Harabasz score (>= 0, higher is better)
        
    Raises:
        ValueError: If there are fewer than 2 clusters
    """
    from sklearn.metrics import calinski_harabasz_score
    
    leaves = get_all_leaves(root)
    
    # Need at least 2 clusters
    if len(leaves) < 2:
        raise ValueError("Need at least 2 leaf nodes (clusters) to calculate Calinski-Harabasz score")
    
    # Create cluster labels based on leaf membership
    n_documents = embeddings.shape[0]
    labels = np.full(n_documents, -1, dtype=int)
    
    for cluster_id, leaf in enumerate(leaves):
        for doc_idx in leaf.documents:
            labels[doc_idx] = cluster_id
    
    # Check that all documents are assigned
    if np.any(labels == -1):
        raise ValueError("Not all documents are assigned to leaf nodes")
    
    return calinski_harabasz_score(embeddings, labels)


def _topic_diversity_metric(
    root: TreeNode,
    embeddings: np.ndarray,
    topk: int = 10,
    mode: str = "full_tree",
    embedding_model: Optional[Any] = None,
    device: str = "cpu",
    **kwargs
) -> float:
    """
    Calculate topic diversity based on node questions/hypotheses.
    
    Topic diversity measures the proportion of unique words across all top-k words
    of all topics (questions). Higher values indicate more diverse topics with
    less redundancy.
    
    Args:
        root: Root node of the tree
        embeddings: Document embeddings array of shape (n_documents, embedding_dim)
        topk: Number of top words to consider per topic (default: 10)
        mode: "full_tree" to calculate for all nodes, or "leaf_paths" to calculate
              per-leaf and average (default: "full_tree")
        embedding_model: Optional embedding model for extracting top words from questions
        device: Device to use for embedding model (default: "cpu")
        **kwargs: Additional parameters (unused)
        
    Returns:
        Topic diversity score (0 to 1, higher is better)
        
    Raises:
        ValueError: If no questions/hypotheses are found in the tree
    """
    def get_top_words(text: str, k: int) -> List[str]:
        """Extract top k words from text (simple tokenization)."""
        if not text:
            return []
        # Simple word extraction - split on whitespace and punctuation
        import re
        words = re.findall(r'\b\w+\b', text.lower())
        # Remove common stop words
        stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with', 'to', 'for', 'of', 'as', 'by'}
        words = [w for w in words if w not in stop_words and len(w) > 2]
        return words[:k]
    
    def collect_questions(node: TreeNode) -> List[str]:
        """Collect all questions from the tree."""
        questions = []
        if node.question:
            questions.append(node.question)
        if node.left is not None:
            questions.extend(collect_questions(node.left))
        if node.right is not None:
            questions.extend(collect_questions(node.right))
        return questions
    
    def collect_leaf_paths(node: TreeNode, path: List[str]) -> List[List[str]]:
        """Collect questions along each path from root to leaf."""
        paths = []
        current_path = path + ([node.question] if node.question else [])
        
        if node.left is None and node.right is None:
            # Leaf node
            paths.append(current_path)
        else:
            if node.left is not None:
                paths.extend(collect_leaf_paths(node.left, current_path))
            if node.right is not None:
                paths.extend(collect_leaf_paths(node.right, current_path))
        return paths
    
    if mode == "full_tree":
        # Collect all questions in the tree
        questions = collect_questions(root)
        
        if not questions:
            raise ValueError("No questions found in the tree")
        
        # Extract top words from each question
        all_topics = [get_top_words(q, topk) for q in questions]
        
        # Calculate diversity
        unique_words = set()
        for topic in all_topics:
            unique_words.update(topic)
        
        total_words = sum(len(topic) for topic in all_topics)
        if total_words == 0:
            return 0.0
        
        diversity = len(unique_words) / total_words
        return diversity
        
    elif mode == "leaf_paths":
        # Calculate diversity for each leaf's path and average
        leaf_paths = collect_leaf_paths(root, [])
        
        if not leaf_paths:
            raise ValueError("No leaf paths found in the tree")
        
        diversities = []
        for path in leaf_paths:
            if not path:
                continue
            
            # Extract top words from each question in this path
            all_topics = [get_top_words(q, topk) for q in path]
            
            # Calculate diversity for this path
            unique_words = set()
            for topic in all_topics:
                unique_words.update(topic)
            
            total_words = sum(len(topic) for topic in all_topics)
            if total_words > 0:
                diversity = len(unique_words) / total_words
                diversities.append(diversity)
        
        if not diversities:
            raise ValueError("No valid paths with questions found")
        
        return np.mean(diversities)
    
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'full_tree' or 'leaf_paths'")


def _child_parent_uniqueness_metric(
    root: TreeNode,
    embeddings: np.ndarray,
    embedding_model: Optional[Any] = None,
    device: str = "cpu",
    **kwargs
) -> Dict[str, float]:
    """
    Calculate uniqueness of child nodes relative to their parents.
    
    Measures how distinct child node embeddings are from their parent node embeddings.
    Uses cosine similarity between average embeddings of parent and child nodes.
    Lower similarity indicates higher uniqueness.
    
    Args:
        root: Root node of the tree
        embeddings: Document embeddings array of shape (n_documents, embedding_dim)
        embedding_model: Optional embedding model for computing question embeddings
        device: Device to use for embedding model (default: "cpu")
        **kwargs: Additional parameters (unused)
        
    Returns:
        Dictionary containing:
            - 'avg_cosine_similarity': Average cosine similarity (0 to 1, lower is better)
            - 'avg_uniqueness': Average uniqueness (1 - similarity, higher is better)
            - 'num_parent_child_pairs': Number of parent-child pairs evaluated
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    def get_node_embedding(node: TreeNode) -> np.ndarray:
        """Get average embedding for documents in a node."""
        if not node.documents:
            return None
        doc_embeddings = embeddings[node.documents]
        return np.mean(doc_embeddings, axis=0)
    
    def traverse_and_compare(node: TreeNode, similarities: List[float]):
        """Traverse tree and collect parent-child similarities."""
        parent_emb = get_node_embedding(node)
        
        if parent_emb is None:
            return
        
        for child in [node.left, node.right]:
            if child is not None:
                child_emb = get_node_embedding(child)
                
                if child_emb is not None:
                    # Compute cosine similarity
                    similarity = cosine_similarity(
                        parent_emb.reshape(1, -1),
                        child_emb.reshape(1, -1)
                    )[0, 0]
                    similarities.append(similarity)
                
                # Recurse
                traverse_and_compare(child, similarities)
    
    similarities = []
    traverse_and_compare(root, similarities)
    
    if not similarities:
        return {
            'avg_cosine_similarity': 0.0,
            'avg_uniqueness': 0.0,
            'num_parent_child_pairs': 0
        }
    
    avg_similarity = np.mean(similarities)
    
    return {
        'avg_cosine_similarity': float(avg_similarity),
        'avg_uniqueness': float(1.0 - avg_similarity),
        'num_parent_child_pairs': len(similarities)
    }


class SilhouetteScoreMetric(SelfSupervisedMetric):
    """Silhouette score metric for leaf node clustering quality."""
    
    def call(self, root: TreeNode, embeddings: np.ndarray, **kwargs) -> float:
        """
        Calculate silhouette score for the tree.
        
        Args:
            root: Root node of the tree
            embeddings: Document embeddings array
            **kwargs: Additional parameters
            
        Returns:
            Silhouette score (-1 to 1, higher is better)
        """
        return _silhouette_score_metric(root, embeddings, **kwargs)


class DaviesBouldinScoreMetric(SelfSupervisedMetric):
    """Davies-Bouldin score metric for clustering quality."""
    
    def call(self, root: TreeNode, embeddings: np.ndarray, **kwargs) -> float:
        """
        Calculate Davies-Bouldin score for the tree.
        
        Args:
            root: Root node of the tree
            embeddings: Document embeddings array
            **kwargs: Additional parameters
            
        Returns:
            Davies-Bouldin score (>= 0, lower is better)
        """
        return _davies_bouldin_score_metric(root, embeddings, **kwargs)


class CalinskiHarabaszScoreMetric(SelfSupervisedMetric):
    """Calinski-Harabasz score metric for clustering quality."""
    
    def call(self, root: TreeNode, embeddings: np.ndarray, **kwargs) -> float:
        """
        Calculate Calinski-Harabasz score for the tree.
        
        Args:
            root: Root node of the tree
            embeddings: Document embeddings array
            **kwargs: Additional parameters
            
        Returns:
            Calinski-Harabasz score (>= 0, higher is better)
        """
        return _calinski_harabasz_score_metric(root, embeddings, **kwargs)


class TopicDiversityMetric(SelfSupervisedMetric):
    """Topic diversity metric based on question uniqueness."""
    
    def call(
        self,
        root: TreeNode,
        embeddings: np.ndarray,
        topk: int = 10,
        mode: str = "full_tree",
        **kwargs
    ) -> float:
        """
        Calculate topic diversity for the tree.
        
        Args:
            root: Root node of the tree
            embeddings: Document embeddings array
            topk: Number of top words per topic (default: 10)
            mode: "full_tree" or "leaf_paths" (default: "full_tree")
            **kwargs: Additional parameters
            
        Returns:
            Topic diversity score (0 to 1, higher is better)
        """
        return _topic_diversity_metric(root, embeddings, topk=topk, mode=mode, **kwargs)


class ChildParentUniquenessMetric(SelfSupervisedMetric):
    """Child-parent uniqueness metric based on embedding similarity."""
    
    def call(self, root: TreeNode, embeddings: np.ndarray, **kwargs) -> Dict[str, float]:
        """
        Calculate child-parent uniqueness for the tree.
        
        Args:
            root: Root node of the tree
            embeddings: Document embeddings array
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with uniqueness metrics
        """
        return _child_parent_uniqueness_metric(root, embeddings, **kwargs)
