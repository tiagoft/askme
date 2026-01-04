"""
Example usage of the Exploratory Power evaluator.

This script demonstrates how to use the evaluator functions to assess
the quality of an RTP tree based on node purity and isolation depth.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from askme.rtp import TreeNode, evaluate_exploratory_power


def create_sample_tree():
    """Create a sample tree for demonstration."""
    # Create a tree with the following structure:
    #        Root [0,1,2,3,4,5]
    #       /                  \
    #  Left [0,1,2]        Right [3,4,5]
    #   /        \
    # LL[0,1]   LR[2]
    
    root = TreeNode(documents=[0, 1, 2, 3, 4, 5], question="Is this about animals?")
    
    left_child = TreeNode(documents=[0, 1, 2], question="Is it about cats?")
    right_child = TreeNode(documents=[3, 4, 5])
    
    left_left = TreeNode(documents=[0, 1])
    left_right = TreeNode(documents=[2])
    
    left_child.left = left_left
    left_child.right = left_right
    
    root.left = left_child
    root.right = right_child
    
    return root


def main():
    """Demonstrate the evaluator functions."""
    
    # Create a sample tree
    tree = create_sample_tree()
    
    # Define labels for the documents
    # Documents: 0,1 = class 0 (cats)
    # Document: 2 = class 1 (dogs) 
    # Documents: 3,4,5 = class 2 (other animals)
    labels = [0, 0, 1, 2, 2, 2]
    
    print("=" * 60)
    print("Exploratory Power Evaluation Demo")
    print("=" * 60)
    
    print("\nTree Structure:")
    print("  Root: documents [0,1,2,3,4,5]")
    print("    Left: documents [0,1,2]")
    print("      Left-Left (leaf): documents [0,1]")
    print("      Left-Right (leaf): documents [2]")
    print("    Right (leaf): documents [3,4,5]")
    
    print("\nLabels:")
    print(f"  {labels}")
    print("  Class 0: documents 0, 1 (cats)")
    print("  Class 1: document 2 (dogs)")
    print("  Class 2: documents 3, 4, 5 (other animals)")
    
    # Evaluate the tree
    results = evaluate_exploratory_power(tree, labels)
    
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    
    print(f"\nNumber of leaf nodes: {results['num_leaves']}")
    
    print("\nLeaf Purities:")
    for leaf_id, purity in results['leaf_purities'].items():
        print(f"  Leaf {leaf_id}: {purity:.4f}")
    
    print(f"\nAverage Leaf Purity: {results['average_leaf_purity']:.4f}")
    print("  (1.0 = perfectly pure, 0.0 = completely impure)")
    
    print("\nIsolation Depths:")
    for class_label, depth in results['isolation_depths'].items():
        if depth is not None:
            print(f"  Class {class_label}: isolated at depth {depth}")
        else:
            print(f"  Class {class_label}: never fully isolated")
    
    print("\n" + "=" * 60)
    print("Interpretation:")
    print("=" * 60)
    print("""
The tree successfully separates the documents by class:
- Class 0 (cats) is isolated in the left-left leaf with perfect purity (1.0)
- Class 1 (dogs) is isolated in the left-right leaf with perfect purity (1.0)
- Class 2 (other animals) is isolated in the right leaf with perfect purity (1.0)

All classes are isolated at depth ≤ 2, indicating that the tree structure
effectively organizes the documents by their labels.
""")


if __name__ == "__main__":
    main()
