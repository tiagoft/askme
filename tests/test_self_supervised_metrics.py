"""Tests for self-supervised metrics module."""

import pytest
import sys
import os
import numpy as np

# Add src directory to path to allow direct imports without full package installation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from askme.rtp.tree_models import TreeNode
from askme.rtp.self_supervised_metrics import (
    SelfSupervisedMetric,
    SilhouetteScoreMetric,
    DaviesBouldinScoreMetric,
    CalinskiHarabaszScoreMetric,
    TopicDiversityMetric,
    ChildParentUniquenessMetric,
    get_all_leaves,
    _silhouette_score_metric,
    _davies_bouldin_score_metric,
    _calinski_harabasz_score_metric,
    _topic_diversity_metric,
    _child_parent_uniqueness_metric,
)


@pytest.fixture
def simple_embeddings():
    """Create simple 2D embeddings for testing."""
    # Create 6 documents with 2D embeddings
    # First 3 documents are clustered around (0, 0)
    # Last 3 documents are clustered around (10, 10)
    embeddings = np.array([
        [0.0, 0.0],
        [0.5, 0.5],
        [0.2, 0.3],
        [10.0, 10.0],
        [10.5, 10.5],
        [10.2, 10.3],
    ])
    return embeddings


@pytest.fixture
def simple_tree():
    """Create a simple binary tree with two leaf nodes."""
    root = TreeNode(documents=[0, 1, 2, 3, 4, 5], question="Is it about topic A?")
    left_child = TreeNode(documents=[0, 1, 2], question="Is it about subtopic A1?")
    right_child = TreeNode(documents=[3, 4, 5], question="Is it about subtopic B1?")
    
    root.left = left_child
    root.right = right_child
    
    return root


@pytest.fixture
def deeper_tree():
    """Create a deeper tree with multiple levels."""
    root = TreeNode(documents=[0, 1, 2, 3, 4, 5], question="Main topic?")
    left_child = TreeNode(documents=[0, 1, 2], question="Left topic?")
    right_child = TreeNode(documents=[3, 4, 5], question="Right topic?")
    
    # Further split left child
    left_left = TreeNode(documents=[0, 1], question="Left-left topic?")
    left_right = TreeNode(documents=[2], question="Left-right topic?")
    left_child.left = left_left
    left_child.right = left_right
    
    root.left = left_child
    root.right = right_child
    
    return root


def test_get_all_leaves_simple_tree(simple_tree):
    """Test getting leaves from a simple tree."""
    leaves = get_all_leaves(simple_tree)
    
    assert len(leaves) == 2
    assert simple_tree.left in leaves
    assert simple_tree.right in leaves


def test_get_all_leaves_deeper_tree(deeper_tree):
    """Test getting leaves from a deeper tree."""
    leaves = get_all_leaves(deeper_tree)
    
    # Should have 3 leaves: left_left, left_right, right_child
    assert len(leaves) == 3


def test_get_all_leaves_single_node():
    """Test getting leaves from a single node tree."""
    root = TreeNode(documents=[0, 1, 2])
    leaves = get_all_leaves(root)
    
    assert len(leaves) == 1
    assert leaves[0] == root


def test_silhouette_score_metric(simple_tree, simple_embeddings):
    """Test silhouette score calculation."""
    score = _silhouette_score_metric(simple_tree, simple_embeddings)
    
    # Score should be high because we have well-separated clusters
    assert 0.5 < score <= 1.0


def test_silhouette_score_metric_single_leaf_raises_error():
    """Test that silhouette score raises error with single leaf."""
    root = TreeNode(documents=[0, 1, 2])
    embeddings = np.array([[0, 0], [1, 1], [2, 2]])
    
    with pytest.raises(ValueError, match="at least 2 leaf nodes"):
        _silhouette_score_metric(root, embeddings)


def test_silhouette_score_class(simple_tree, simple_embeddings):
    """Test SilhouetteScoreMetric class."""
    metric = SilhouetteScoreMetric()
    score = metric.call(simple_tree, simple_embeddings)
    
    assert isinstance(score, float)
    assert 0.5 < score <= 1.0


def test_davies_bouldin_score_metric(simple_tree, simple_embeddings):
    """Test Davies-Bouldin score calculation."""
    score = _davies_bouldin_score_metric(simple_tree, simple_embeddings)
    
    # Lower is better; should be relatively low for well-separated clusters
    assert score >= 0.0
    assert score < 1.0  # Should be quite low for our well-separated clusters


def test_davies_bouldin_score_class(simple_tree, simple_embeddings):
    """Test DaviesBouldinScoreMetric class."""
    metric = DaviesBouldinScoreMetric()
    score = metric.call(simple_tree, simple_embeddings)
    
    assert isinstance(score, float)
    assert score >= 0.0


def test_calinski_harabasz_score_metric(simple_tree, simple_embeddings):
    """Test Calinski-Harabasz score calculation."""
    score = _calinski_harabasz_score_metric(simple_tree, simple_embeddings)
    
    # Higher is better; should be high for well-separated clusters
    assert score > 0.0


def test_calinski_harabasz_score_class(simple_tree, simple_embeddings):
    """Test CalinskiHarabaszScoreMetric class."""
    metric = CalinskiHarabaszScoreMetric()
    score = metric.call(simple_tree, simple_embeddings)
    
    assert isinstance(score, float)
    assert score > 0.0


def test_topic_diversity_full_tree(simple_tree, simple_embeddings):
    """Test topic diversity in full tree mode."""
    score = _topic_diversity_metric(simple_tree, simple_embeddings, topk=10, mode="full_tree")
    
    # Should return a value between 0 and 1
    assert 0.0 <= score <= 1.0


def test_topic_diversity_leaf_paths(deeper_tree, simple_embeddings):
    """Test topic diversity in leaf paths mode."""
    score = _topic_diversity_metric(deeper_tree, simple_embeddings, topk=10, mode="leaf_paths")
    
    # Should return a value between 0 and 1
    assert 0.0 <= score <= 1.0


def test_topic_diversity_no_questions_raises_error(simple_embeddings):
    """Test that topic diversity raises error when no questions exist."""
    root = TreeNode(documents=[0, 1, 2])
    left = TreeNode(documents=[0, 1])
    right = TreeNode(documents=[2])
    root.left = left
    root.right = right
    
    with pytest.raises(ValueError, match="No questions found"):
        _topic_diversity_metric(root, simple_embeddings)


def test_topic_diversity_class(simple_tree, simple_embeddings):
    """Test TopicDiversityMetric class."""
    metric = TopicDiversityMetric()
    score = metric.call(simple_tree, simple_embeddings, topk=10)
    
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_topic_diversity_with_diverse_questions(simple_embeddings):
    """Test topic diversity with diverse questions."""
    root = TreeNode(documents=[0, 1, 2, 3, 4, 5], question="Is this about technology?")
    left = TreeNode(documents=[0, 1, 2], question="Is this about computers?")
    right = TreeNode(documents=[3, 4, 5], question="Is this about biology?")
    root.left = left
    root.right = right
    
    score = _topic_diversity_metric(root, simple_embeddings, topk=5, mode="full_tree")
    
    # Should have high diversity as questions use different words
    assert score > 0.3


def test_topic_diversity_with_similar_questions(simple_embeddings):
    """Test topic diversity with similar/repeated questions."""
    root = TreeNode(documents=[0, 1, 2, 3, 4, 5], question="Is this about science?")
    left = TreeNode(documents=[0, 1, 2], question="Is this about science?")
    right = TreeNode(documents=[3, 4, 5], question="Is this about science?")
    root.left = left
    root.right = right
    
    score = _topic_diversity_metric(root, simple_embeddings, topk=5, mode="full_tree")
    
    # Should have lower diversity as questions repeat the same words
    assert 0.0 <= score <= 1.0


def test_child_parent_uniqueness_metric(simple_tree, simple_embeddings):
    """Test child-parent uniqueness calculation."""
    result = _child_parent_uniqueness_metric(simple_tree, simple_embeddings)
    
    # Check structure
    assert 'avg_cosine_similarity' in result
    assert 'avg_uniqueness' in result
    assert 'num_parent_child_pairs' in result
    
    # Check values
    assert 0.0 <= result['avg_cosine_similarity'] <= 1.0
    assert 0.0 <= result['avg_uniqueness'] <= 1.0
    assert result['num_parent_child_pairs'] == 2  # root->left and root->right


def test_child_parent_uniqueness_class(simple_tree, simple_embeddings):
    """Test ChildParentUniquenessMetric class."""
    metric = ChildParentUniquenessMetric()
    result = metric.call(simple_tree, simple_embeddings)
    
    assert isinstance(result, dict)
    assert 'avg_cosine_similarity' in result
    assert 'avg_uniqueness' in result


def test_child_parent_uniqueness_deeper_tree(deeper_tree, simple_embeddings):
    """Test child-parent uniqueness on deeper tree."""
    result = _child_parent_uniqueness_metric(deeper_tree, simple_embeddings)
    
    # Should have 4 parent-child pairs:
    # root->left, root->right, left->left_left, left->left_right
    assert result['num_parent_child_pairs'] == 4


def test_child_parent_uniqueness_well_separated_clusters():
    """Test uniqueness with well-separated clusters (should have low similarity)."""
    # Create embeddings where children are very different from parent average
    embeddings = np.array([
        [0.0, 0.0],   # doc 0 (left child)
        [0.1, 0.1],   # doc 1 (left child)
        [10.0, 10.0], # doc 2 (right child)
        [10.1, 10.1], # doc 3 (right child)
    ])
    
    root = TreeNode(documents=[0, 1, 2, 3])
    left = TreeNode(documents=[0, 1])
    right = TreeNode(documents=[2, 3])
    root.left = left
    root.right = right
    
    result = _child_parent_uniqueness_metric(root, embeddings)
    
    # Children are very different from parent average (which is around [5, 5])
    # So similarity should be relatively low, uniqueness high
    assert result['avg_uniqueness'] > 0.0


def test_child_parent_uniqueness_single_leaf():
    """Test uniqueness with single leaf (no parent-child pairs)."""
    root = TreeNode(documents=[0, 1, 2])
    embeddings = np.array([[0, 0], [1, 1], [2, 2]])
    
    result = _child_parent_uniqueness_metric(root, embeddings)
    
    assert result['num_parent_child_pairs'] == 0
    assert result['avg_cosine_similarity'] == 0.0
    assert result['avg_uniqueness'] == 0.0


def test_clustering_metrics_with_poor_clustering():
    """Test clustering metrics with poorly separated clusters."""
    # Create overlapping clusters
    embeddings = np.array([
        [0.0, 0.0],
        [0.5, 0.5],
        [1.0, 1.0],
        [1.5, 1.5],
        [2.0, 2.0],
        [2.5, 2.5],
    ])
    
    root = TreeNode(documents=[0, 1, 2, 3, 4, 5])
    left = TreeNode(documents=[0, 1, 2])
    right = TreeNode(documents=[3, 4, 5])
    root.left = left
    root.right = right
    
    # Silhouette score should be lower for overlapping clusters
    silhouette = _silhouette_score_metric(root, embeddings)
    assert -1.0 <= silhouette <= 1.0
    
    # Davies-Bouldin should be higher (worse) for overlapping clusters
    davies_bouldin = _davies_bouldin_score_metric(root, embeddings)
    assert davies_bouldin >= 0.0


def test_clustering_metrics_with_many_clusters():
    """Test clustering metrics with many small clusters."""
    # Create 4 well-separated clusters
    embeddings = np.array([
        [0.0, 0.0],
        [0.1, 0.1],
        [10.0, 0.0],
        [10.1, 0.1],
        [0.0, 10.0],
        [0.1, 10.1],
        [10.0, 10.0],
        [10.1, 10.1],
    ])
    
    root = TreeNode(documents=[0, 1, 2, 3, 4, 5, 6, 7])
    left = TreeNode(documents=[0, 1, 2, 3])
    right = TreeNode(documents=[4, 5, 6, 7])
    
    left_left = TreeNode(documents=[0, 1])
    left_right = TreeNode(documents=[2, 3])
    left.left = left_left
    left.right = left_right
    
    right_left = TreeNode(documents=[4, 5])
    right_right = TreeNode(documents=[6, 7])
    right.left = right_left
    right.right = right_right
    
    root.left = left
    root.right = right
    
    # All metrics should work with 4 clusters
    silhouette = _silhouette_score_metric(root, embeddings)
    assert 0.0 < silhouette <= 1.0
    
    davies_bouldin = _davies_bouldin_score_metric(root, embeddings)
    assert davies_bouldin >= 0.0
    
    calinski = _calinski_harabasz_score_metric(root, embeddings)
    assert calinski > 0.0


def test_base_class_not_implemented():
    """Test that base class cannot be instantiated without implementing call."""
    class DummyMetric(SelfSupervisedMetric):
        pass
    
    # Should not be able to instantiate abstract class without implementing call
    with pytest.raises(TypeError, match="abstract"):
        metric = DummyMetric()


def test_topic_diversity_invalid_mode(simple_tree, simple_embeddings):
    """Test topic diversity with invalid mode."""
    with pytest.raises(ValueError, match="Unknown mode"):
        _topic_diversity_metric(simple_tree, simple_embeddings, mode="invalid_mode")


def test_clustering_metrics_with_high_dimensional_embeddings():
    """Test clustering metrics with higher dimensional embeddings."""
    # Create 100-dimensional embeddings
    np.random.seed(42)
    embeddings = np.random.randn(6, 100)
    
    # Make two clusters clearly separated in first dimension
    embeddings[0:3, 0] = -5.0
    embeddings[3:6, 0] = 5.0
    
    root = TreeNode(documents=[0, 1, 2, 3, 4, 5])
    left = TreeNode(documents=[0, 1, 2])
    right = TreeNode(documents=[3, 4, 5])
    root.left = left
    root.right = right
    
    silhouette = _silhouette_score_metric(root, embeddings)
    davies_bouldin = _davies_bouldin_score_metric(root, embeddings)
    calinski = _calinski_harabasz_score_metric(root, embeddings)
    
    # All metrics should return valid values
    assert -1.0 <= silhouette <= 1.0
    assert davies_bouldin >= 0.0
    assert calinski >= 0.0


def test_all_metrics_together(simple_tree, simple_embeddings):
    """Test that all metrics can be computed together."""
    # Instantiate all metric classes
    silhouette_metric = SilhouetteScoreMetric()
    davies_bouldin_metric = DaviesBouldinScoreMetric()
    calinski_metric = CalinskiHarabaszScoreMetric()
    diversity_metric = TopicDiversityMetric()
    uniqueness_metric = ChildParentUniquenessMetric()
    
    # Compute all metrics
    silhouette = silhouette_metric.call(simple_tree, simple_embeddings)
    davies_bouldin = davies_bouldin_metric.call(simple_tree, simple_embeddings)
    calinski = calinski_metric.call(simple_tree, simple_embeddings)
    diversity = diversity_metric.call(simple_tree, simple_embeddings)
    uniqueness = uniqueness_metric.call(simple_tree, simple_embeddings)
    
    # Verify all metrics return valid values
    assert isinstance(silhouette, float)
    assert isinstance(davies_bouldin, float)
    assert isinstance(calinski, float)
    assert isinstance(diversity, float)
    assert isinstance(uniqueness, dict)
    
    assert -1.0 <= silhouette <= 1.0
    assert davies_bouldin >= 0.0
    assert calinski >= 0.0
    assert 0.0 <= diversity <= 1.0
    assert 'avg_uniqueness' in uniqueness
