"""
Simple demonstration of tree_to_pdf functionality.

This script demonstrates the tree_to_pdf feature by:
1. Creating a simple tree structure manually
2. Saving it to JSON
3. Loading it from JSON
4. Converting it to PDF with different configurations
"""

import json
from pathlib import Path
from askme.rtp import TreeNode, SplitMetrics
from askme.rtp.tree_to_pdf import load_tree_from_json, tree_to_pdf

def create_demo_tree():
    """Create a demonstration tree structure."""
    # Create metrics for the root node
    metrics = SplitMetrics(
        split_ratio=0.5,
        nli_calls=10,
        total_time_ms=150.5,
        llm_input_tokens=100,
        llm_output_tokens=50,
    )
    
    # Create left subtree (cats)
    left_leaf1 = TreeNode(documents=[0, 1])
    left_leaf2 = TreeNode(documents=[2, 3])
    left_child = TreeNode(
        documents=[0, 1, 2, 3],
        question="Is it about domestic cats?",
        left=left_leaf1,
        right=left_leaf2,
        metrics=SplitMetrics(split_ratio=0.5, nli_calls=5, total_time_ms=75.0)
    )
    
    # Create right subtree (dogs)
    right_leaf1 = TreeNode(documents=[4, 5])
    right_leaf2 = TreeNode(documents=[6, 7])
    right_child = TreeNode(
        documents=[4, 5, 6, 7],
        question="Is it about small dogs?",
        left=right_leaf1,
        right=right_leaf2,
        metrics=SplitMetrics(split_ratio=0.5, nli_calls=5, total_time_ms=70.0)
    )
    
    # Create root
    root = TreeNode(
        documents=[0, 1, 2, 3, 4, 5, 6, 7],
        question="Is this document about cats?",
        left=left_child,
        right=right_child,
        metrics=metrics,
    )
    
    return root


def main():
    print("=" * 70)
    print(" Tree to PDF Demonstration ".center(70))
    print("=" * 70)
    
    # Step 1: Create a demo tree
    print("\n1. Creating demonstration tree...")
    tree = create_demo_tree()
    print(f"   ✓ Created tree with question: '{tree.question}'")
    print(f"   ✓ Total documents: {len(tree.documents)}")
    
    # Step 2: Save tree to JSON
    print("\n2. Saving tree to JSON...")
    json_path = Path("demo_tree.json")
    with open(json_path, 'w') as f:
        json_string = tree.model_dump_json(indent=2)
        f.write(json_string)
    print(f"   ✓ Saved to: {json_path}")
    
    # Step 3: Load tree from JSON
    print("\n3. Loading tree from JSON...")
    loaded_tree = load_tree_from_json(json_path)
    print(f"   ✓ Loaded tree with {len(loaded_tree.documents)} documents")
    
    # Step 4: Create basic PDF
    print("\n4. Creating basic PDF...")
    pdf_path = tree_to_pdf(
        loaded_tree,
        output_path="demo_tree_basic",
    )
    print(f"   ✓ Created: {pdf_path}")
    
    # Step 5: Create PDF with metrics
    print("\n5. Creating PDF with metrics...")
    pdf_path_metrics = tree_to_pdf(
        loaded_tree,
        output_path="demo_tree_with_metrics",
        metrics_to_display=['split_ratio', 'nli_calls', 'total_time_ms'],
        font_size=11,
    )
    print(f"   ✓ Created: {pdf_path_metrics}")
    
    # Step 6: Create PDF with custom layout
    print("\n6. Creating PDF with landscape layout...")
    pdf_path_landscape = tree_to_pdf(
        loaded_tree,
        output_path="demo_tree_landscape",
        graph_attr={'rankdir': 'LR', 'size': '10,6'},
        font_size=10,
    )
    print(f"   ✓ Created: {pdf_path_landscape}")
    
    # Step 7: Create PDF for article publication
    print("\n7. Creating PDF sized for articles...")
    pdf_path_article = tree_to_pdf(
        loaded_tree,
        output_path="demo_tree_article",
        metrics_to_display=['split_ratio'],
        font_size=12,
        graph_attr={
            'rankdir': 'TB',
            'size': '6,8',
            'dpi': '300',
        },
    )
    print(f"   ✓ Created: {pdf_path_article}")
    
    # Summary
    print("\n" + "=" * 70)
    print(" Summary ".center(70))
    print("=" * 70)
    print("\nGenerated files:")
    print("  • demo_tree.json              - Tree structure in JSON format")
    print("  • demo_tree_basic.pdf         - Basic tree visualization")
    print("  • demo_tree_with_metrics.pdf  - With split metrics displayed")
    print("  • demo_tree_landscape.pdf     - Landscape orientation")
    print("  • demo_tree_article.pdf       - Formatted for article publication")
    print("\nAll files created successfully! ✓")
    print("=" * 70)


if __name__ == "__main__":
    main()
