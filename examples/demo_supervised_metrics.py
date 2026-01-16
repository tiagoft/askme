"""
Example usage of supervised metrics for evaluating RTP trees.

This example demonstrates how to use the supervised metrics to evaluate
the quality of a tree structure when ground truth labels are available.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from askme.rtp.tree_models import TreeNode
from askme.rtp.supervised_metrics import (
    NormalizedMutualInformation,
    AdjustedRandIndex,
    HomogeneityCompletenessVMeasure,
    Accuracy,
    F1Score,
    ConfusionMatrix,
)


def create_example_tree():
    """Create an example tree for demonstration."""
    # Create a simple tree with 3 leaf nodes
    root = TreeNode(documents=[0, 1, 2, 3, 4, 5, 6, 7, 8])
    
    # First level split
    left_branch = TreeNode(documents=[0, 1, 2, 3])
    right_branch = TreeNode(documents=[4, 5, 6, 7, 8])
    
    # Second level splits (left side)
    left_left = TreeNode(documents=[0, 1])
    left_right = TreeNode(documents=[2, 3])
    left_branch.left = left_left
    left_branch.right = left_right
    
    # Second level splits (right side)
    right_left = TreeNode(documents=[4, 5, 6])
    right_right = TreeNode(documents=[7, 8])
    right_branch.left = right_left
    right_branch.right = right_right
    
    # Connect to root
    root.left = left_branch
    root.right = right_branch
    
    return root


def main():
    """Run example demonstrating supervised metrics."""
    # Create example tree
    tree_root = create_example_tree()
    
    # Example ground truth labels
    # Let's say we have 3 classes: 0, 1, and 2
    labels = [0, 0, 0, 0, 1, 1, 1, 2, 2]
    
    print("=" * 70)
    print("Supervised Metrics Example")
    print("=" * 70)
    print(f"\nTree structure:")
    print(f"  - Root: documents {tree_root.documents}")
    print(f"  - 4 leaf nodes with documents:")
    print(f"    - Leaf 1: {tree_root.left.left.documents}")
    print(f"    - Leaf 2: {tree_root.left.right.documents}")
    print(f"    - Leaf 3: {tree_root.right.left.documents}")
    print(f"    - Leaf 4: {tree_root.right.right.documents}")
    
    print(f"\nGround truth labels: {labels}")
    print(f"  - Documents 0-3: Class 0")
    print(f"  - Documents 4-6: Class 1")
    print(f"  - Documents 7-8: Class 2")
    
    print("\n" + "=" * 70)
    print("Clustering Metrics (using leaf nodes only)")
    print("=" * 70)
    
    # Normalized Mutual Information
    nmi_metric = NormalizedMutualInformation()
    nmi = nmi_metric.call(tree_root, labels, use_leaves_only=True)
    print(f"\nNormalized Mutual Information (NMI): {nmi:.4f}")
    print("  - Measures mutual information between clusters and true labels")
    print("  - Range: [0, 1], higher is better")
    print("  - 1.0 = perfect agreement, 0.0 = no mutual information")
    
    # Adjusted Rand Index
    ari_metric = AdjustedRandIndex()
    ari = ari_metric.call(tree_root, labels, use_leaves_only=True)
    print(f"\nAdjusted Rand Index (ARI): {ari:.4f}")
    print("  - Measures similarity adjusted for chance")
    print("  - Range: [-1, 1], higher is better")
    print("  - 1.0 = perfect agreement, 0.0 = random labeling")
    
    # Homogeneity, Completeness, V-measure
    hcv_metric = HomogeneityCompletenessVMeasure()
    hcv = hcv_metric.call(tree_root, labels, use_leaves_only=True)
    print(f"\nHomogeneity: {hcv['homogeneity']:.4f}")
    print("  - Each cluster contains only members of a single class")
    print(f"Completeness: {hcv['completeness']:.4f}")
    print("  - All members of a class are in the same cluster")
    print(f"V-measure: {hcv['v_measure']:.4f}")
    print("  - Harmonic mean of homogeneity and completeness")
    
    print("\n" + "=" * 70)
    print("Classification Metrics (using leaf nodes only)")
    print("=" * 70)
    print("(Each document is assigned the majority class of its leaf node)")
    
    # Accuracy
    acc_metric = Accuracy()
    acc = acc_metric.call(tree_root, labels, use_leaves_only=True)
    print(f"\nAccuracy: {acc:.4f}")
    print("  - Proportion of correctly classified documents")
    print("  - Range: [0, 1], higher is better")
    
    # F1-Score
    f1_metric = F1Score()
    f1_weighted = f1_metric.call(tree_root, labels, use_leaves_only=True, average='weighted')
    f1_macro = f1_metric.call(tree_root, labels, use_leaves_only=True, average='macro')
    print(f"\nF1-Score (weighted): {f1_weighted:.4f}")
    print("  - Weighted average F1 across all classes")
    print(f"F1-Score (macro): {f1_macro:.4f}")
    print("  - Unweighted average F1 across all classes")
    
    # Confusion Matrix
    cm_metric = ConfusionMatrix()
    cm = cm_metric.call(tree_root, labels, use_leaves_only=True)
    print(f"\nConfusion Matrix:")
    print("  (rows = true labels, columns = predicted labels)")
    for i, row in enumerate(cm):
        print(f"  Class {i}: {row}")
    
    print("\n" + "=" * 70)
    print("Using All Nodes (not just leaves)")
    print("=" * 70)
    
    # Example with all nodes
    nmi_all = nmi_metric.call(tree_root, labels, use_leaves_only=False)
    ari_all = ari_metric.call(tree_root, labels, use_leaves_only=False)
    acc_all = acc_metric.call(tree_root, labels, use_leaves_only=False)
    
    print(f"\nNMI (all nodes): {nmi_all:.4f}")
    print(f"ARI (all nodes): {ari_all:.4f}")
    print(f"Accuracy (all nodes): {acc_all:.4f}")
    print("\nNote: Using all nodes includes internal nodes, so documents")
    print("may be counted multiple times in different clusters.")
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("\nThe supervised metrics provide quantitative measures of:")
    print("  1. How well the tree structure aligns with true labels")
    print("  2. Classification performance when using majority voting")
    print("  3. Both clustering quality (NMI, ARI, H/C/V) and")
    print("     classification quality (Accuracy, F1, Confusion Matrix)")
    print("\nUse these metrics to:")
    print("  - Compare different tree building strategies")
    print("  - Evaluate tree quality during development")
    print("  - Understand which classes are well-separated")
    print("=" * 70)


if __name__ == "__main__":
    main()
