"""
Demo script to test entropy and information gain calculations with mock data.

This script demonstrates the metrics functionality without requiring
actual NLI models, using synthetic data for testing.
"""

from askme.rtp.metrics import calculate_entropy, calculate_information_gain


def demo_entropy():
    """Demonstrate entropy calculations."""
    print("=== Entropy Calculation Demos ===\n")
    
    # Pure set
    labels1 = [0, 0, 0, 0]
    entropy1 = calculate_entropy(labels1)
    print(f"Pure set (all same class): {labels1}")
    print(f"Entropy: {entropy1:.4f} (should be 0.0)\n")
    
    # Perfect binary split
    labels2 = [0, 1, 0, 1]
    entropy2 = calculate_entropy(labels2)
    print(f"Balanced binary: {labels2}")
    print(f"Entropy: {entropy2:.4f} (should be 1.0)\n")
    
    # Unbalanced binary
    labels3 = [0, 0, 0, 1]
    entropy3 = calculate_entropy(labels3)
    print(f"Unbalanced binary: {labels3}")
    print(f"Entropy: {entropy3:.4f} (should be ~0.811)\n")
    
    # Three classes
    labels4 = ['cat', 'dog', 'bird', 'cat', 'dog', 'bird']
    entropy4 = calculate_entropy(labels4)
    print(f"Three balanced classes: {labels4}")
    print(f"Entropy: {entropy4:.4f} (should be ~1.585)\n")


def demo_information_gain():
    """Demonstrate information gain calculations."""
    print("=== Information Gain Calculation Demos ===\n")
    
    # Perfect split
    parent = [0, 0, 1, 1]
    left = [0, 0]
    right = [1, 1]
    ig = calculate_information_gain(parent, left, right)
    print(f"Perfect split:")
    print(f"  Parent: {parent}")
    print(f"  Left:   {left}")
    print(f"  Right:  {right}")
    print(f"  Information Gain: {ig:.4f} (should be 1.0)\n")
    
    # No improvement
    parent2 = [0, 1, 0, 1]
    left2 = [0, 1]
    right2 = [0, 1]
    ig2 = calculate_information_gain(parent2, left2, right2)
    print(f"No improvement (split doesn't separate classes):")
    print(f"  Parent: {parent2}")
    print(f"  Left:   {left2}")
    print(f"  Right:  {right2}")
    print(f"  Information Gain: {ig2:.4f} (should be 0.0)\n")
    
    # Partial improvement
    parent3 = [0, 0, 1, 1, 1, 1]
    left3 = [0, 0, 1]
    right3 = [1, 1, 1]
    ig3 = calculate_information_gain(parent3, left3, right3)
    parent_entropy = calculate_entropy(parent3)
    print(f"Partial improvement:")
    print(f"  Parent: {parent3} (entropy={parent_entropy:.4f})")
    print(f"  Left:   {left3} (entropy={calculate_entropy(left3):.4f})")
    print(f"  Right:  {right3} (entropy={calculate_entropy(right3):.4f})")
    print(f"  Information Gain: {ig3:.4f} (should be positive but < parent entropy)\n")
    
    # Multi-class example
    parent4 = ['A', 'A', 'B', 'B', 'C', 'C']
    left4 = ['A', 'A', 'B']
    right4 = ['B', 'C', 'C']
    ig4 = calculate_information_gain(parent4, left4, right4)
    parent_entropy4 = calculate_entropy(parent4)
    print(f"Multi-class split:")
    print(f"  Parent: {parent4} (entropy={parent_entropy4:.4f})")
    print(f"  Left:   {left4} (entropy={calculate_entropy(left4):.4f})")
    print(f"  Right:  {right4} (entropy={calculate_entropy(right4):.4f})")
    print(f"  Information Gain: {ig4:.4f}\n")


def demo_tree_node_with_metrics():
    """Demonstrate TreeNode with entropy and IG."""
    from askme.rtp.tree_models import TreeNode
    
    print("=== TreeNode with Metrics Demo ===\n")
    
    # Create a simple tree with metrics
    root = TreeNode(
        documents=[0, 1, 2, 3],
        question="Is this about machine learning?",
        entropy=1.0,
        information_gain=0.5
    )
    
    left_child = TreeNode(
        documents=[0, 1],
        entropy=0.0
    )
    
    right_child = TreeNode(
        documents=[2, 3],
        entropy=0.0
    )
    
    root.left = left_child
    root.right = right_child
    
    print(f"Root Node:")
    print(f"  Documents: {root.documents}")
    print(f"  Question: {root.question}")
    print(f"  Entropy: {root.entropy:.4f}")
    print(f"  Information Gain: {root.information_gain:.4f}")
    print(f"\nLeft Child:")
    print(f"  Documents: {left_child.documents}")
    print(f"  Entropy: {left_child.entropy:.4f}")
    print(f"\nRight Child:")
    print(f"  Documents: {right_child.documents}")
    print(f"  Entropy: {right_child.entropy:.4f}")


if __name__ == "__main__":
    demo_entropy()
    print("\n" + "="*60 + "\n")
    demo_information_gain()
    print("\n" + "="*60 + "\n")
    demo_tree_node_with_metrics()
