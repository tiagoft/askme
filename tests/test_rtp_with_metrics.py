"""
Tests for RTP with entropy and information gain calculations.
"""

import pytest
from askme.rtp.tree_models import TreeNode


def test_tree_node_with_entropy():
    """Test that TreeNode can store entropy values."""
    node = TreeNode(
        documents=[0, 1, 2],
        question="Is this about science?",
        entropy=0.918
    )
    assert node.documents == [0, 1, 2]
    assert node.question == "Is this about science?"
    assert abs(node.entropy - 0.918) < 1e-3


def test_tree_node_with_information_gain():
    """Test that TreeNode can store information gain values."""
    node = TreeNode(
        documents=[0, 1, 2, 3],
        question="Is this about AI?",
        entropy=1.0,
        information_gain=0.5
    )
    assert node.entropy == 1.0
    assert node.information_gain == 0.5


def test_tree_node_hierarchy_with_metrics():
    """Test tree hierarchy with entropy and IG values."""
    parent_node = TreeNode(
        documents=[0, 1, 2, 3],
        question="Is this about science?",
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
    
    parent_node.left = left_child
    parent_node.right = right_child
    
    assert parent_node.left == left_child
    assert parent_node.right == right_child
    assert parent_node.entropy == 1.0
    assert parent_node.information_gain == 0.5
    assert left_child.entropy == 0.0
    assert right_child.entropy == 0.0


def test_save_tree_node_with_metrics():
    """Test saving and loading TreeNode with entropy and IG."""
    node = TreeNode(
        documents=[0, 1, 2],
        question="Is this about science?",
        entropy=0.918,
        information_gain=0.311
    )
    file_path = "/tmp/tree_node_metrics.json"
    json_string = node.model_dump_json()
    
    with open(file_path, 'w') as f:
        f.write(json_string)
    
    with open(file_path, 'r') as f:
        json_data = f.read()
        
    loaded_node = TreeNode.model_validate_json(json_data)
    assert loaded_node == node
    assert abs(loaded_node.entropy - 0.918) < 1e-3
    assert abs(loaded_node.information_gain - 0.311) < 1e-3
