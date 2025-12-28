"""
Example demonstrating RTP with entropy and information gain metrics.

This example shows a mock implementation without actual ML models to demonstrate
the logging and metrics functionality.
"""

from askme.rtp.tree_models import TreeNode
from askme.rtp.metrics import calculate_entropy, calculate_information_gain


def simulate_rtp_with_metrics():
    """
    Simulate an RTP tree with entropy and information gain calculations.
    This demonstrates the expected behavior when used with actual models.
    """
    print("=" * 70)
    print("RTP with Entropy and Information Gain - Simulated Example")
    print("=" * 70)
    print()
    
    # Simulated dataset: documents with labels
    documents = [
        "The cat sat on the mat.",
        "The cat is in the box.",
        "The dog barked loudly.",
        "I like cats.",
        "I like dogs.",
        "The dog is in the yard.",
    ]
    
    # Ground-truth labels (0 = cat-related, 1 = dog-related)
    labels = [0, 0, 1, 0, 1, 1]
    
    print("Dataset:")
    for i, (doc, label) in enumerate(zip(documents, labels)):
        print(f"  [{i}] (Label={label}): {doc}")
    print()
    
    # Calculate initial entropy
    initial_entropy = calculate_entropy(labels)
    print(f"Initial Entropy: {initial_entropy:.4f}")
    print()
    
    # Simulate a split at depth 0
    # Question: "Is this about cats?"
    # Let's say documents 0, 1, 3 go left (about cats)
    # Documents 2, 4, 5 go right (not about cats)
    
    left_indices = [0, 1, 3]
    right_indices = [2, 4, 5]
    
    left_labels = [labels[i] for i in left_indices]
    right_labels = [labels[i] for i in right_indices]
    
    left_entropy = calculate_entropy(left_labels)
    right_entropy = calculate_entropy(right_labels)
    ig = calculate_information_gain(labels, left_labels, right_labels)
    
    print("Simulated Split at Depth 0:")
    print(f"  Question: 'Is this about cats?'")
    print(f"  Left documents (answered YES): {left_indices}")
    print(f"    Labels: {left_labels}")
    print(f"    Entropy: {left_entropy:.4f}")
    print(f"  Right documents (answered NO): {right_indices}")
    print(f"    Labels: {right_labels}")
    print(f"    Entropy: {right_entropy:.4f}")
    print(f"  Information Gain: {ig:.4f}")
    print()
    
    # Create tree structure
    root = TreeNode(
        documents=list(range(len(documents))),
        question="Is this about cats?",
        entropy=initial_entropy,
        information_gain=ig
    )
    
    left_child = TreeNode(
        documents=left_indices,
        entropy=left_entropy
    )
    
    right_child = TreeNode(
        documents=right_indices,
        entropy=right_entropy
    )
    
    root.left = left_child
    root.right = right_child
    
    # Display tree structure
    print("Tree Structure:")
    print(f"Root Node:")
    print(f"  Documents: {root.documents}")
    print(f"  Question: {root.question}")
    print(f"  Entropy: {root.entropy:.4f}")
    print(f"  Information Gain: {root.information_gain:.4f}")
    print()
    print(f"  Left Child:")
    print(f"    Documents: {left_child.documents}")
    print(f"    Entropy: {left_child.entropy:.4f}")
    print()
    print(f"  Right Child:")
    print(f"    Documents: {right_child.documents}")
    print(f"    Entropy: {right_child.entropy:.4f}")
    print()
    
    # Show what would be logged
    print("=" * 70)
    print("Expected Log Output:")
    print("=" * 70)
    print(f"Depth 0: Information Gain = {ig:.4f} "
          f"(Entropy before: {initial_entropy:.4f}, "
          f"Left: {len(left_labels)}/{len(labels)}, "
          f"Right: {len(right_labels)}/{len(labels)})")
    print()
    
    # Demonstrate log file format
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("This would be written to log file:")
    print(f"{timestamp} - Depth 0: Information Gain = {ig:.4f} "
          f"(Entropy before: {initial_entropy:.4f}, "
          f"Left: {len(left_labels)}/{len(labels)}, "
          f"Right: {len(right_labels)}/{len(labels)})")


if __name__ == "__main__":
    simulate_rtp_with_metrics()
