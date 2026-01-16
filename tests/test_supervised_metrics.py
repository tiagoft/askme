"""Tests for supervised metrics module."""

import pytest
import sys
import os
import numpy as np

# Add src directory to path to allow direct imports without full package installation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from askme.rtp.tree_models import TreeNode
from askme.rtp.supervised_metrics import (
    SupervisedMetric,
    NormalizedMutualInformation,
    AdjustedRandIndex,
    HomogeneityCompletenessVMeasure,
    Accuracy,
    F1Score,
    ConfusionMatrix,
    get_all_nodes,
    get_all_leaves,
    _get_cluster_assignments,
    _get_predicted_labels,
)


def test_supervised_metric_base_class():
    """Test that SupervisedMetric base class raises NotImplementedError."""
    metric = SupervisedMetric()
    root = TreeNode(documents=[0, 1, 2])
    labels = [0, 0, 1]
    
    with pytest.raises(NotImplementedError):
        metric.call(root, labels)


def test_get_all_nodes_single_node():
    """Test getting all nodes from a tree with only root node."""
    root = TreeNode(documents=[0, 1, 2])
    
    nodes = get_all_nodes(root)
    
    assert len(nodes) == 1
    assert nodes[0] == root


def test_get_all_nodes_with_children():
    """Test getting all nodes from a tree with multiple levels."""
    root = TreeNode(documents=[0, 1, 2, 3])
    left_child = TreeNode(documents=[0, 1])
    right_child = TreeNode(documents=[2, 3])
    
    root.left = left_child
    root.right = right_child
    
    nodes = get_all_nodes(root)
    
    assert len(nodes) == 3
    assert root in nodes
    assert left_child in nodes
    assert right_child in nodes


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


def test_get_cluster_assignments_leaves_only():
    """Test getting cluster assignments for leaf nodes only."""
    root = TreeNode(documents=[0, 1, 2, 3])
    left_child = TreeNode(documents=[0, 1])
    right_child = TreeNode(documents=[2, 3])
    
    root.left = left_child
    root.right = right_child
    
    labels = [0, 0, 1, 1]
    
    predicted_clusters, true_labels = _get_cluster_assignments(root, labels, use_leaves_only=True)
    
    # Should have assignments for all 4 documents
    assert len(predicted_clusters) == 4
    assert len(true_labels) == 4
    
    # Documents 0 and 1 should be in one cluster (left_child)
    assert predicted_clusters[0] == predicted_clusters[1]
    
    # Documents 2 and 3 should be in another cluster (right_child)
    assert predicted_clusters[2] == predicted_clusters[3]
    
    # The two clusters should be different
    assert predicted_clusters[0] != predicted_clusters[2]


def test_get_cluster_assignments_all_nodes():
    """Test getting cluster assignments for all nodes."""
    root = TreeNode(documents=[0, 1, 2, 3])
    left_child = TreeNode(documents=[0, 1])
    right_child = TreeNode(documents=[2, 3])
    
    root.left = left_child
    root.right = right_child
    
    labels = [0, 0, 1, 1]
    
    predicted_clusters, true_labels = _get_cluster_assignments(root, labels, use_leaves_only=False)
    
    # Should have assignments for all 4 documents (from first assignment)
    assert len(predicted_clusters) == 4
    assert len(true_labels) == 4


def test_get_predicted_labels_leaf_nodes():
    """Test getting predicted labels based on majority class in leaf nodes."""
    root = TreeNode(documents=[0, 1, 2, 3, 4, 5])
    left_child = TreeNode(documents=[0, 1, 2])
    right_child = TreeNode(documents=[3, 4, 5])
    
    root.left = left_child
    root.right = right_child
    
    # Left child: 2 of class 0, 1 of class 1 -> majority is 0
    # Right child: all class 1
    labels = [0, 0, 1, 1, 1, 1]
    
    predicted_labels, true_labels = _get_predicted_labels(root, labels, use_leaves_only=True)
    
    assert len(predicted_labels) == 6
    assert len(true_labels) == 6
    
    # Documents 0, 1, 2 should all be predicted as class 0 (majority in left_child)
    assert predicted_labels[0] == 0
    assert predicted_labels[1] == 0
    assert predicted_labels[2] == 0
    
    # Documents 3, 4, 5 should all be predicted as class 1 (majority in right_child)
    assert predicted_labels[3] == 1
    assert predicted_labels[4] == 1
    assert predicted_labels[5] == 1


def test_nmi_perfect_clustering():
    """Test NMI with perfect clustering (each class in separate leaf)."""
    root = TreeNode(documents=[0, 1, 2, 3])
    left_child = TreeNode(documents=[0, 1])
    right_child = TreeNode(documents=[2, 3])
    
    root.left = left_child
    root.right = right_child
    
    # Perfect clustering: left has class 0, right has class 1
    labels = [0, 0, 1, 1]
    
    metric = NormalizedMutualInformation()
    nmi = metric.call(root, labels, use_leaves_only=True)
    
    # Perfect clustering should have NMI = 1.0
    assert abs(nmi - 1.0) < 0.001


def test_nmi_random_clustering():
    """Test NMI with random-like clustering."""
    root = TreeNode(documents=[0, 1, 2, 3])
    left_child = TreeNode(documents=[0, 2])
    right_child = TreeNode(documents=[1, 3])
    
    root.left = left_child
    root.right = right_child
    
    # Mixed clustering: both leaves have both classes
    labels = [0, 1, 0, 1]
    
    metric = NormalizedMutualInformation()
    nmi = metric.call(root, labels, use_leaves_only=True)
    
    # Mixed clustering should have low NMI
    assert 0.0 <= nmi <= 1.0


def test_ari_perfect_clustering():
    """Test ARI with perfect clustering."""
    root = TreeNode(documents=[0, 1, 2, 3])
    left_child = TreeNode(documents=[0, 1])
    right_child = TreeNode(documents=[2, 3])
    
    root.left = left_child
    root.right = right_child
    
    # Perfect clustering: left has class 0, right has class 1
    labels = [0, 0, 1, 1]
    
    metric = AdjustedRandIndex()
    ari = metric.call(root, labels, use_leaves_only=True)
    
    # Perfect clustering should have ARI = 1.0
    assert abs(ari - 1.0) < 0.001


def test_ari_random_clustering():
    """Test ARI with random-like clustering."""
    root = TreeNode(documents=[0, 1, 2, 3])
    left_child = TreeNode(documents=[0, 2])
    right_child = TreeNode(documents=[1, 3])
    
    root.left = left_child
    root.right = right_child
    
    # Mixed clustering: both leaves have both classes evenly
    labels = [0, 1, 0, 1]
    
    metric = AdjustedRandIndex()
    ari = metric.call(root, labels, use_leaves_only=True)
    
    # Random clustering should have ARI around 0
    assert -1.0 <= ari <= 1.0


def test_homogeneity_completeness_vmeasure():
    """Test Homogeneity, Completeness, and V-measure."""
    root = TreeNode(documents=[0, 1, 2, 3])
    left_child = TreeNode(documents=[0, 1])
    right_child = TreeNode(documents=[2, 3])
    
    root.left = left_child
    root.right = right_child
    
    # Perfect clustering: left has class 0, right has class 1
    labels = [0, 0, 1, 1]
    
    metric = HomogeneityCompletenessVMeasure()
    result = metric.call(root, labels, use_leaves_only=True)
    
    # Check structure
    assert 'homogeneity' in result
    assert 'completeness' in result
    assert 'v_measure' in result
    
    # Perfect clustering should have all scores = 1.0
    assert abs(result['homogeneity'] - 1.0) < 0.001
    assert abs(result['completeness'] - 1.0) < 0.001
    assert abs(result['v_measure'] - 1.0) < 0.001


def test_homogeneity_completeness_vmeasure_imperfect():
    """Test metrics with imperfect clustering."""
    root = TreeNode(documents=[0, 1, 2, 3, 4, 5])
    left_child = TreeNode(documents=[0, 1, 2])
    right_child = TreeNode(documents=[3, 4, 5])
    
    root.left = left_child
    root.right = right_child
    
    # Imperfect clustering
    labels = [0, 0, 1, 0, 1, 1]
    
    metric = HomogeneityCompletenessVMeasure()
    result = metric.call(root, labels, use_leaves_only=True)
    
    # All scores should be between 0 and 1
    assert 0.0 <= result['homogeneity'] <= 1.0
    assert 0.0 <= result['completeness'] <= 1.0
    assert 0.0 <= result['v_measure'] <= 1.0


def test_accuracy_perfect():
    """Test accuracy with perfect classification."""
    root = TreeNode(documents=[0, 1, 2, 3])
    left_child = TreeNode(documents=[0, 1])
    right_child = TreeNode(documents=[2, 3])
    
    root.left = left_child
    root.right = right_child
    
    # Perfect: left has class 0, right has class 1
    labels = [0, 0, 1, 1]
    
    metric = Accuracy()
    acc = metric.call(root, labels, use_leaves_only=True)
    
    # Perfect classification should have accuracy = 1.0
    assert abs(acc - 1.0) < 0.001


def test_accuracy_imperfect():
    """Test accuracy with imperfect classification."""
    root = TreeNode(documents=[0, 1, 2, 3, 4, 5])
    left_child = TreeNode(documents=[0, 1, 2])
    right_child = TreeNode(documents=[3, 4, 5])
    
    root.left = left_child
    root.right = right_child
    
    # Left child: 2 of class 0, 1 of class 1 -> majority is 0, accuracy = 2/3
    # Right child: all class 1 -> accuracy = 1.0
    # Overall: 5/6 correct
    labels = [0, 0, 1, 1, 1, 1]
    
    metric = Accuracy()
    acc = metric.call(root, labels, use_leaves_only=True)
    
    # Expected: (2 + 3) / 6 = 5/6 ≈ 0.833
    assert abs(acc - 5/6) < 0.001


def test_f1_score_perfect():
    """Test F1-score with perfect classification."""
    root = TreeNode(documents=[0, 1, 2, 3])
    left_child = TreeNode(documents=[0, 1])
    right_child = TreeNode(documents=[2, 3])
    
    root.left = left_child
    root.right = right_child
    
    # Perfect: left has class 0, right has class 1
    labels = [0, 0, 1, 1]
    
    metric = F1Score()
    f1 = metric.call(root, labels, use_leaves_only=True, average='weighted')
    
    # Perfect classification should have F1-score = 1.0
    assert abs(f1 - 1.0) < 0.001


def test_f1_score_imperfect():
    """Test F1-score with imperfect classification."""
    root = TreeNode(documents=[0, 1, 2, 3, 4, 5])
    left_child = TreeNode(documents=[0, 1, 2])
    right_child = TreeNode(documents=[3, 4, 5])
    
    root.left = left_child
    root.right = right_child
    
    # Imperfect classification
    labels = [0, 0, 1, 1, 1, 1]
    
    metric = F1Score()
    f1 = metric.call(root, labels, use_leaves_only=True, average='weighted')
    
    # F1-score should be between 0 and 1
    assert 0.0 <= f1 <= 1.0


def test_f1_score_macro_average():
    """Test F1-score with macro averaging."""
    root = TreeNode(documents=[0, 1, 2, 3])
    left_child = TreeNode(documents=[0, 1])
    right_child = TreeNode(documents=[2, 3])
    
    root.left = left_child
    root.right = right_child
    
    labels = [0, 0, 1, 1]
    
    metric = F1Score()
    f1 = metric.call(root, labels, use_leaves_only=True, average='macro')
    
    assert abs(f1 - 1.0) < 0.001


def test_confusion_matrix_binary():
    """Test confusion matrix with binary classification."""
    root = TreeNode(documents=[0, 1, 2, 3])
    left_child = TreeNode(documents=[0, 1])
    right_child = TreeNode(documents=[2, 3])
    
    root.left = left_child
    root.right = right_child
    
    # Perfect: left has class 0, right has class 1
    labels = [0, 0, 1, 1]
    
    metric = ConfusionMatrix()
    cm = metric.call(root, labels, use_leaves_only=True)
    
    # Expected confusion matrix:
    # [[2, 0],   <- 2 true class 0 predicted as 0, 0 predicted as 1
    #  [0, 2]]   <- 0 true class 1 predicted as 0, 2 predicted as 1
    assert cm.shape == (2, 2)
    assert cm[0, 0] == 2  # True negatives
    assert cm[0, 1] == 0  # False positives
    assert cm[1, 0] == 0  # False negatives
    assert cm[1, 1] == 2  # True positives


def test_confusion_matrix_multiclass():
    """Test confusion matrix with multiclass classification."""
    root = TreeNode(documents=[0, 1, 2, 3, 4, 5])
    left_child = TreeNode(documents=[0, 1])
    middle_child = TreeNode(documents=[2, 3])
    right_child = TreeNode(documents=[4, 5])
    
    # Create a tree with 3 leaves
    internal = TreeNode(documents=[2, 3, 4, 5])
    internal.left = middle_child
    internal.right = right_child
    
    root.left = left_child
    root.right = internal
    
    # Three classes: 0, 1, 2
    labels = [0, 0, 1, 1, 2, 2]
    
    metric = ConfusionMatrix()
    cm = metric.call(root, labels, use_leaves_only=True)
    
    # Should be a 3x3 matrix
    assert cm.shape == (3, 3)
    
    # Diagonal should have perfect predictions for each class
    assert cm[0, 0] == 2  # Class 0 correctly classified
    assert cm[1, 1] == 2  # Class 1 correctly classified
    assert cm[2, 2] == 2  # Class 2 correctly classified


def test_confusion_matrix_imperfect():
    """Test confusion matrix with imperfect classification."""
    root = TreeNode(documents=[0, 1, 2, 3, 4, 5])
    left_child = TreeNode(documents=[0, 1, 2])
    right_child = TreeNode(documents=[3, 4, 5])
    
    root.left = left_child
    root.right = right_child
    
    # Left child: 2 of class 0, 1 of class 1 -> majority is 0
    # Right child: all class 1
    labels = [0, 0, 1, 1, 1, 1]
    
    metric = ConfusionMatrix()
    cm = metric.call(root, labels, use_leaves_only=True)
    
    # Expected:
    # - Doc 0 (true 0) -> predicted 0 ✓
    # - Doc 1 (true 0) -> predicted 0 ✓
    # - Doc 2 (true 1) -> predicted 0 ✗
    # - Doc 3 (true 1) -> predicted 1 ✓
    # - Doc 4 (true 1) -> predicted 1 ✓
    # - Doc 5 (true 1) -> predicted 1 ✓
    
    # CM[i, j] = count of true label i predicted as j
    assert cm[0, 0] == 2  # 2 true 0s predicted as 0
    assert cm[1, 0] == 1  # 1 true 1 predicted as 0
    assert cm[1, 1] == 3  # 3 true 1s predicted as 1


def test_metrics_with_all_nodes():
    """Test that metrics work with use_leaves_only=False."""
    root = TreeNode(documents=[0, 1, 2, 3])
    left_child = TreeNode(documents=[0, 1])
    right_child = TreeNode(documents=[2, 3])
    
    root.left = left_child
    root.right = right_child
    
    labels = [0, 0, 1, 1]
    
    # Test NMI with all nodes
    nmi_metric = NormalizedMutualInformation()
    nmi = nmi_metric.call(root, labels, use_leaves_only=False)
    assert 0.0 <= nmi <= 1.0
    
    # Test ARI with all nodes
    ari_metric = AdjustedRandIndex()
    ari = ari_metric.call(root, labels, use_leaves_only=False)
    assert -1.0 <= ari <= 1.0
    
    # Test Accuracy with all nodes
    acc_metric = Accuracy()
    acc = acc_metric.call(root, labels, use_leaves_only=False)
    assert 0.0 <= acc <= 1.0


def test_single_leaf_tree():
    """Test metrics on a tree with only one node (root is a leaf)."""
    root = TreeNode(documents=[0, 1, 2, 3])
    labels = [0, 0, 1, 1]
    
    # NMI
    nmi_metric = NormalizedMutualInformation()
    nmi = nmi_metric.call(root, labels, use_leaves_only=True)
    # Single cluster should have NMI = 0
    assert abs(nmi - 0.0) < 0.001
    
    # ARI
    ari_metric = AdjustedRandIndex()
    ari = ari_metric.call(root, labels, use_leaves_only=True)
    # Single cluster should have ARI = 0
    assert abs(ari - 0.0) < 0.001
    
    # Accuracy
    acc_metric = Accuracy()
    acc = acc_metric.call(root, labels, use_leaves_only=True)
    # Majority class (2 docs) out of 4 total = 0.5
    assert abs(acc - 0.5) < 0.001


def test_empty_tree():
    """Test metrics on an empty tree (no documents)."""
    root = TreeNode(documents=[])
    labels = []
    
    # NMI
    nmi_metric = NormalizedMutualInformation()
    nmi = nmi_metric.call(root, labels, use_leaves_only=True)
    assert nmi == 0.0
    
    # ARI
    ari_metric = AdjustedRandIndex()
    ari = ari_metric.call(root, labels, use_leaves_only=True)
    assert ari == 0.0
    
    # Accuracy
    acc_metric = Accuracy()
    acc = acc_metric.call(root, labels, use_leaves_only=True)
    assert acc == 0.0
    
    # F1-score
    f1_metric = F1Score()
    f1 = f1_metric.call(root, labels, use_leaves_only=True)
    assert f1 == 0.0
    
    # Confusion matrix
    cm_metric = ConfusionMatrix()
    cm = cm_metric.call(root, labels, use_leaves_only=True)
    # Empty tree results in shape (1, 0) from sklearn - just check it's a valid array
    assert isinstance(cm, np.ndarray)
    assert cm.size == 0  # Should have no elements


def test_nmi_with_different_average_methods():
    """Test NMI with different averaging methods."""
    root = TreeNode(documents=[0, 1, 2, 3, 4, 5])
    left_child = TreeNode(documents=[0, 1, 2])
    right_child = TreeNode(documents=[3, 4, 5])
    
    root.left = left_child
    root.right = right_child
    
    labels = [0, 0, 1, 1, 2, 2]
    
    metric = NormalizedMutualInformation()
    
    # Test different average methods
    for method in ['arithmetic', 'geometric', 'min', 'max']:
        nmi = metric.call(root, labels, use_leaves_only=True, average_method=method)
        assert 0.0 <= nmi <= 1.0, f"NMI with {method} averaging out of range"


def test_metrics_three_class_problem():
    """Test all metrics on a three-class classification problem."""
    root = TreeNode(documents=[0, 1, 2, 3, 4, 5, 6, 7, 8])
    
    # Create a more complex tree structure
    left_branch = TreeNode(documents=[0, 1, 2, 3])
    right_branch = TreeNode(documents=[4, 5, 6, 7, 8])
    
    left_left = TreeNode(documents=[0, 1])
    left_right = TreeNode(documents=[2, 3])
    left_branch.left = left_left
    left_branch.right = left_right
    
    right_left = TreeNode(documents=[4, 5, 6])
    right_right = TreeNode(documents=[7, 8])
    right_branch.left = right_left
    right_branch.right = right_right
    
    root.left = left_branch
    root.right = right_branch
    
    # Three classes
    labels = [0, 0, 1, 1, 1, 2, 2, 2, 2]
    
    # Test all metrics
    nmi_metric = NormalizedMutualInformation()
    nmi = nmi_metric.call(root, labels)
    assert 0.0 <= nmi <= 1.0
    
    ari_metric = AdjustedRandIndex()
    ari = ari_metric.call(root, labels)
    assert -1.0 <= ari <= 1.0
    
    hcv_metric = HomogeneityCompletenessVMeasure()
    hcv = hcv_metric.call(root, labels)
    assert 0.0 <= hcv['homogeneity'] <= 1.0
    assert 0.0 <= hcv['completeness'] <= 1.0
    assert 0.0 <= hcv['v_measure'] <= 1.0
    
    acc_metric = Accuracy()
    acc = acc_metric.call(root, labels)
    assert 0.0 <= acc <= 1.0
    
    f1_metric = F1Score()
    f1 = f1_metric.call(root, labels, average='weighted')
    assert 0.0 <= f1 <= 1.0
    
    cm_metric = ConfusionMatrix()
    cm = cm_metric.call(root, labels)
    assert cm.shape == (3, 3)
    assert cm.sum() == 9  # Total number of documents
