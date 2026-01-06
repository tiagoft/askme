"""
Example usage of tree_to_pdf functionality.

This script demonstrates how to:
1. Build an RTP tree from text documents
2. Save the tree to JSON
3. Load the tree from JSON
4. Convert the tree to PDF with various customization options
"""

from askme.rtp import RTPBuilder, load_tree_from_json
from askme.rtp.tree_to_pdf import tree_to_pdf
import pickle 

# Sample text collection
text_collection = [
    "The cat sat on the mat.",
    "The cat is in the box.",
    "The dog barked loudly.",
    "I like cats",
    "I like dogs",
    "The dog is in the yard.",
    "Birds can fly high in the sky.",
    "Fish swim in the ocean.",
    "Elephants are the largest land animals.",
    "Lions are known as the kings of the jungle.",
]


def create_and_save_tree():
    """Create an RTP tree and save it to JSON."""
    print("=" * 60)
    print("Step 1: Creating RTP tree...")
    print("=" * 60)
    
    # Initialize RTPBuilder
    builder = RTPBuilder(
        use_gpu=False,
        n_medoids=4,
        n_documents_to_answer=6,
    )
    
    # Build tree with metrics
    tree_root, metrics = builder(text_collection, return_metrics=True)
    
    print(f"Tree created successfully!")
    print(f"Question: {tree_root.question}")
    print(f"Total documents: {len(tree_root.documents)}")
    print(f"Split ratio: {metrics.split_ratio:.3f}")
    print(f"NLI calls: {metrics.nli_calls}")
    print(f"Total time: {metrics.total_time_ms:.1f}ms")
    
    # Save to JSON
    json_string = tree_root.model_dump_json()
    with open('example_tree.json', 'w') as f:
        f.write(json_string)
    print("\nTree saved to example_tree.json")
    
    return tree_root


def example_basic_pdf(tree):
    """Example 1: Basic PDF with default settings."""
    print("\n" + "=" * 60)
    print("Example 1: Basic PDF with default settings")
    print("=" * 60)
    
    pdf_path = tree_to_pdf(tree, output_path="tree_basic")
    print(f"PDF saved to: {pdf_path}")


def example_with_metrics(tree):
    """Example 2: PDF with split metrics displayed."""
    print("\n" + "=" * 60)
    print("Example 2: PDF with split metrics")
    print("=" * 60)
    
    pdf_path = tree_to_pdf(
        tree,
        output_path="tree_with_metrics",
        metrics_to_display=['split_ratio', 'nli_calls', 'total_time_ms'],
        font_size=11,
    )
    print(f"PDF saved to: {pdf_path}")


def example_landscape_orientation(tree):
    """Example 3: PDF with landscape orientation (left-to-right layout)."""
    print("\n" + "=" * 60)
    print("Example 3: PDF with landscape orientation")
    print("=" * 60)
    
    pdf_path = tree_to_pdf(
        tree,
        output_path="tree_landscape",
        graph_attr={'rankdir': 'LR', 'size': '10,6'},  # Left-to-right, wide aspect
        font_size=10,
    )
    print(f"PDF saved to: {pdf_path}")


def example_custom_size_for_article(tree):
    """Example 4: PDF sized for article publication."""
    print("\n" + "=" * 60)
    print("Example 4: PDF sized for article publication")
    print("=" * 60)
    
    pdf_path = tree_to_pdf(
        tree,
        output_path="tree_article",
        metrics_to_display=['split_ratio', 'nli_calls'],
        font_size=12,
        graph_attr={
            'rankdir': 'TB',
            'size': '6,8',  # Standard article width
            'dpi': '300',   # High resolution for publication
        },
    )
    print(f"PDF saved to: {pdf_path}")


def example_limited_nodes(tree):
    """Example 5: PDF with limited number of nodes (for large trees)."""
    print("\n" + "=" * 60)
    print("Example 5: PDF with max nodes limit")
    print("=" * 60)
    
    pdf_path = tree_to_pdf(
        tree,
        output_path="tree_limited",
        max_nodes=5,  # Only show first 5 nodes
        font_size=10,
    )
    print(f"PDF saved to: {pdf_path}")


def example_custom_styling(tree):
    """Example 6: PDF with custom node and edge styling."""
    print("\n" + "=" * 60)
    print("Example 6: PDF with custom styling")
    print("=" * 60)
    
    pdf_path = tree_to_pdf(
        tree,
        output_path="tree_custom_style",
        font_size=11,
        node_attr={
            'shape': 'ellipse',
            'style': 'filled',
            'fillcolor': 'lightyellow',
            'fontname': 'Arial',
        },
        edge_attr={
            'color': 'darkgreen',
            'penwidth': '2',
        },
    )
    print(f"PDF saved to: {pdf_path}")


def example_load_from_json():
    """Example 7: Load tree from JSON and convert to PDF."""
    print("\n" + "=" * 60)
    print("Example 7: Load from JSON and convert to PDF")
    print("=" * 60)
    
    # Load the tree from JSON
    tree = load_tree_from_json("example_tree.json")
    print("Tree loaded successfully from example_tree.json")
    print(f"Question: {tree.question}")
    print(f"Total documents: {len(tree.documents)}")
    
    # Convert to PDF
    pdf_path = tree_to_pdf(
        tree,
        output_path="tree_from_json",
        metrics_to_display=['split_ratio'],
    )
    print(f"PDF saved to: {pdf_path}")


def main():
    """Run all examples."""
    print("\n" + "█" * 60)
    print("█" + " " * 58 + "█")
    print("█" + "  Tree to PDF Examples".center(58) + "█")
    print("█" + " " * 58 + "█")
    print("█" * 60 + "\n")
    
    # Create and save tree
    #tree = create_and_save_tree()
    with open('rtp_tree_on_small_agnews.pkl', 'rb') as f:
        tree = pickle.load(f)
         
    # Run all examples
    example_basic_pdf(tree)
    example_with_metrics(tree)
    example_landscape_orientation(tree)
    example_custom_size_for_article(tree)
    example_limited_nodes(tree)
    example_custom_styling(tree)
    example_load_from_json()
    
    print("\n" + "█" * 60)
    print("█" + " " * 58 + "█")
    print("█" + "  All examples completed successfully!".center(58) + "█")
    print("█" + " " * 58 + "█")
    print("█" * 60 + "\n")
    
    print("Generated files:")
    print("  - example_tree.json")
    print("  - tree_basic.pdf")
    print("  - tree_with_metrics.pdf")
    print("  - tree_landscape.pdf")
    print("  - tree_article.pdf")
    print("  - tree_limited.pdf")
    print("  - tree_custom_style.pdf")
    print("  - tree_from_json.pdf")


if __name__ == "__main__":
    main()
