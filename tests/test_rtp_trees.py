from askme.rtp.tree_models import TreeNode
import pytest

def test_tree_node_creation():
    node = TreeNode(documents=[0, 1, 2], question="Is this about science?")
    assert node.documents == [0, 1, 2]
    assert node.question == "Is this about science?"
    assert node.left is None
    assert node.right is None

def test_tree_node_hierarchy():
    parent_node = TreeNode(documents=[0, 1, 2], question="Is this about science?")
    left_child = TreeNode(documents=[0, 1],)
    right_child = TreeNode(documents=[2],)
    
    parent_node.left = left_child
    parent_node.right = right_child
    
    assert parent_node.left == left_child
    assert parent_node.right == right_child
    
def test_save_tree_node():
    node = TreeNode(documents=[0, 1, 2], question="Is this about science?")
    file_path = "tree_node.json"
    json_string = node.model_dump_json()
    
    with open(file_path, 'w') as f:
        f.write(json_string)
    
    with open(file_path, 'r') as f:
        json_data = f.read()
        
    loaded_node = TreeNode.model_validate_json(json_data)
    assert loaded_node == node

def test_save_tree_node_with_children():
    parent_node = TreeNode(documents=[0, 1, 2], question="Is this about science?")
    left_child = TreeNode(documents=[0, 1])
    right_child = TreeNode(documents=[2])
    
    parent_node.left = left_child
    parent_node.right = right_child
    
    file_path = "tree_node_with_children.json"
    json_string = parent_node.model_dump_json()
    
    with open(file_path, 'w') as f:
        f.write(json_string)
    
    with open(file_path, 'r') as f:
        json_data = f.read()
        
    loaded_node = TreeNode.model_validate_json(json_data)
    assert loaded_node == parent_node
    assert loaded_node.left == left_child
    assert loaded_node.right == right_child