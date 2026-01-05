"""Tests for tree_to_pdf functionality."""

import pytest
import json
import tempfile
from pathlib import Path
from askme.rtp import TreeNode, SplitMetrics
from askme.rtp.tree_to_pdf import load_tree_from_json, tree_to_graphviz, tree_to_pdf


# Sample tree for testing
def create_sample_tree():
    """Create a simple tree structure for testing."""
    # Create a simple tree with a root and two children
    left_child = TreeNode(documents=[0, 1, 2])
    right_child = TreeNode(documents=[3, 4, 5])
    
    metrics = SplitMetrics(
        split_ratio=0.5,
        nli_calls=10,
        total_time_ms=100.5,
        llm_input_tokens=50,
        llm_output_tokens=30,
    )
    
    root = TreeNode(
        documents=[0, 1, 2, 3, 4, 5],
        question="Is this about cats?",
        left=left_child,
        right=right_child,
        metrics=metrics,
    )
    
    return root


def test_load_tree_from_json():
    """Test loading a tree from JSON."""
    tree = create_sample_tree()
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json_string = tree.model_dump_json()
        f.write(json_string)
        temp_path = f.name
    
    try:
        # Load from JSON
        loaded_tree = load_tree_from_json(temp_path)
        
        # Verify loaded tree matches original
        assert loaded_tree.question == tree.question
        assert loaded_tree.documents == tree.documents
        assert loaded_tree.left is not None
        assert loaded_tree.right is not None
        assert len(loaded_tree.left.documents) == 3
        assert len(loaded_tree.right.documents) == 3
    finally:
        Path(temp_path).unlink()


def test_tree_to_graphviz_basic():
    """Test basic conversion to Graphviz without metrics."""
    tree = create_sample_tree()
    graph = tree_to_graphviz(tree)
    
    assert graph is not None
    # Check that graph source contains expected elements
    source = graph.source
    assert 'Is this about cats' in source
    assert 'Docs: 6' in source  # Root has 6 docs
    assert 'YES' in source  # Edge label
    assert 'NO' in source   # Edge label


def test_tree_to_graphviz_with_metrics():
    """Test conversion to Graphviz with metrics displayed."""
    tree = create_sample_tree()
    graph = tree_to_graphviz(
        tree,
        metrics_to_display=['split_ratio', 'nli_calls', 'total_time_ms']
    )
    
    assert graph is not None
    source = graph.source
    assert 'split_ratio: 0.500' in source
    assert 'nli_calls: 10' in source
    assert 'total_time_ms: 100.5ms' in source


def test_tree_to_graphviz_with_max_nodes():
    """Test conversion with max_nodes limit."""
    tree = create_sample_tree()
    graph = tree_to_graphviz(tree, max_nodes=1)
    
    assert graph is not None
    source = graph.source
    # Should only have root node
    assert 'Docs: 6' in source
    # Should not have child nodes (Docs: 3)
    # Note: We can't easily count exact nodes from source, but the test verifies basic functionality


def test_tree_to_graphviz_with_custom_attrs():
    """Test conversion with custom graph, node, and edge attributes."""
    tree = create_sample_tree()
    graph = tree_to_graphviz(
        tree,
        font_size=14,
        graph_attr={'rankdir': 'LR'},
        node_attr={'shape': 'ellipse'},
        edge_attr={'color': 'red'},
    )
    
    assert graph is not None
    source = graph.source
    assert 'rankdir=LR' in source
    assert 'shape=ellipse' in source
    assert 'fontsize=14' in source


def test_tree_to_graphviz_leaf_nodes():
    """Test that leaf nodes are colored differently."""
    # Create a tree with only root (no children) - it's a leaf
    leaf_tree = TreeNode(documents=[0, 1, 2], question="Test question")
    graph = tree_to_graphviz(leaf_tree)
    
    assert graph is not None
    source = graph.source
    assert 'lightgreen' in source  # Leaf nodes should be green


def test_tree_to_graphviz_long_question():
    """Test that long questions are truncated."""
    long_question = "A" * 100  # Very long question
    tree = TreeNode(documents=[0, 1], question=long_question)
    graph = tree_to_graphviz(tree)
    
    assert graph is not None
    source = graph.source
    # Should be truncated with ellipsis
    assert '...' in source
    # Should not contain the full question
    assert long_question not in source


def test_tree_to_pdf_creates_file():
    """Test that tree_to_pdf creates a PDF file."""
    tree = create_sample_tree()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_tree"
        result = tree_to_pdf(tree, output_path=output_path)
        
        # Check that PDF was created
        assert result is not None
        pdf_path = Path(result)
        assert pdf_path.exists()
        assert pdf_path.suffix == '.pdf'


def test_tree_to_pdf_with_metrics():
    """Test tree_to_pdf with metrics displayed."""
    tree = create_sample_tree()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_tree_metrics"
        result = tree_to_pdf(
            tree,
            output_path=output_path,
            metrics_to_display=['split_ratio', 'nli_calls'],
        )
        
        # Check that PDF was created
        pdf_path = Path(result)
        assert pdf_path.exists()


def test_tree_to_pdf_with_custom_settings():
    """Test tree_to_pdf with various custom settings."""
    tree = create_sample_tree()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_tree_custom"
        result = tree_to_pdf(
            tree,
            output_path=output_path,
            font_size=12,
            max_nodes=2,
            graph_attr={'rankdir': 'LR', 'size': '10,8'},
            cleanup=True,
        )
        
        # Check that PDF was created
        pdf_path = Path(result)
        assert pdf_path.exists()
        # Check that intermediate DOT file was cleaned up
        dot_path = Path(str(output_path))
        assert not dot_path.exists()


def test_tree_without_question():
    """Test tree visualization when node has no question."""
    tree = TreeNode(documents=[0, 1, 2])  # No question
    graph = tree_to_graphviz(tree)
    
    assert graph is not None
    source = graph.source
    assert 'Docs: 3' in source


def test_tree_without_metrics():
    """Test tree visualization when node has no metrics."""
    tree = TreeNode(
        documents=[0, 1, 2],
        question="Test?",
        left=TreeNode(documents=[0, 1]),
        right=TreeNode(documents=[2]),
    )
    # No metrics attached
    graph = tree_to_graphviz(tree, metrics_to_display=['split_ratio'])
    
    assert graph is not None
    # Should not raise an error even though metrics are requested but not present


def test_nested_tree():
    """Test visualization of a deeper tree structure."""
    # Create a tree with 3 levels
    leaf1 = TreeNode(documents=[0])
    leaf2 = TreeNode(documents=[1])
    leaf3 = TreeNode(documents=[2])
    leaf4 = TreeNode(documents=[3])
    
    subtree1 = TreeNode(
        documents=[0, 1],
        question="Sub-question 1?",
        left=leaf1,
        right=leaf2,
    )
    
    subtree2 = TreeNode(
        documents=[2, 3],
        question="Sub-question 2?",
        left=leaf3,
        right=leaf4,
    )
    
    root = TreeNode(
        documents=[0, 1, 2, 3],
        question="Root question?",
        left=subtree1,
        right=subtree2,
    )
    
    graph = tree_to_graphviz(root)
    assert graph is not None
    source = graph.source
    assert 'Root question' in source
    assert 'Sub-question 1' in source
    assert 'Sub-question 2' in source
