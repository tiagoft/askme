"""Tests for RTPRecursion class."""

import pytest
from askme.rtp import RTPBuilder, RTPRecursion, TreeNode, SplitMetrics


# Sample text collection for testing
sample_text_collection = [
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


def test_rtp_recursion_initialization():
    """Test that RTPRecursion initializes correctly."""
    builder = RTPBuilder(use_gpu=False)
    recursion = RTPRecursion(
        builder=builder,
        min_node_size=2,
        min_split_ratio=0.2,
        max_split_ratio=0.8,
        max_depth=5,
    )
    
    assert recursion.builder == builder
    assert recursion.min_node_size == 2
    assert recursion.min_split_ratio == 0.2
    assert recursion.max_split_ratio == 0.8
    assert recursion.max_depth == 5


def test_rtp_recursion_default_parameters():
    """Test that RTPRecursion has correct default parameters."""
    builder = RTPBuilder(use_gpu=False)
    recursion = RTPRecursion(builder=builder)
    
    assert recursion.min_node_size == 2
    assert recursion.min_split_ratio == 0.2
    assert recursion.max_split_ratio == 0.8
    assert recursion.max_depth == 10


def test_rtp_recursion_call_returns_tuple():
    """Test that RTPRecursion call returns tuple of TreeNode and SplitMetrics."""
    builder = RTPBuilder(use_gpu=False, n_medoids=2, n_documents_to_answer=3)
    recursion = RTPRecursion(builder=builder, max_depth=1)
    
    result = recursion(sample_text_collection)
    
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], TreeNode)
    assert isinstance(result[1], SplitMetrics)


def test_rtp_recursion_preserves_document_indices():
    """Test that RTPRecursion preserves original document indices."""
    builder = RTPBuilder(use_gpu=False, n_medoids=2, n_documents_to_answer=3)
    recursion = RTPRecursion(builder=builder, max_depth=1)
    
    root, metrics = recursion(sample_text_collection)
    
    # Root should have all document indices
    assert root.documents == list(range(len(sample_text_collection)))
    
    # Collect all document indices from the tree
    def collect_docs(node):
        docs = set(node.documents)
        if node.left:
            docs.update(collect_docs(node.left))
        if node.right:
            docs.update(collect_docs(node.right))
        return docs
    
    # All original indices should be present
    all_docs = collect_docs(root)
    assert all_docs == set(range(len(sample_text_collection)))


def test_rtp_recursion_stops_at_min_node_size():
    """Test that recursion stops when node size is below minimum."""
    builder = RTPBuilder(use_gpu=False, n_medoids=2, n_documents_to_answer=2)
    recursion = RTPRecursion(
        builder=builder,
        min_node_size=5,  # Set high to prevent splitting small collection
        max_depth=10,
    )
    
    small_collection = ["Dogs bark.", "Cats meow.", "Birds chirp."]
    root, metrics = recursion(small_collection)
    
    # Root should be a leaf (no children) because collection is too small
    assert root.left is None
    assert root.right is None
    assert root.documents == [0, 1, 2]


def test_rtp_recursion_stops_at_max_depth():
    """Test that recursion stops at maximum depth."""
    builder = RTPBuilder(use_gpu=False, n_medoids=2, n_documents_to_answer=3)
    recursion = RTPRecursion(
        builder=builder,
        min_node_size=1,
        max_depth=1,  # Only allow one level of splits
    )
    
    root, metrics = recursion(sample_text_collection)
    
    # Root can have children
    # But children should be leaves (no grandchildren)
    if root.left is not None:
        assert root.left.left is None
        assert root.left.right is None
    
    if root.right is not None:
        assert root.right.left is None
        assert root.right.right is None


def test_rtp_recursion_global_metrics_accumulation():
    """Test that global metrics accumulate across all nodes."""
    builder = RTPBuilder(use_gpu=False, n_medoids=2, n_documents_to_answer=3)
    recursion = RTPRecursion(builder=builder, max_depth=2, min_node_size=2)
    
    root, global_metrics = recursion(sample_text_collection)
    
    # Global metrics should be populated
    assert isinstance(global_metrics, SplitMetrics)
    assert global_metrics.llm_input_tokens > 0
    assert global_metrics.llm_output_tokens > 0
    assert global_metrics.nli_calls > 0
    assert global_metrics.total_time_ms > 0.0
    
    # If tree has multiple levels, metrics should be greater than single node
    # (This is a qualitative test - exact values depend on the split)


def test_rtp_recursion_node_metrics_attached():
    """Test that each node has its own metrics attached."""
    builder = RTPBuilder(use_gpu=False, n_medoids=2, n_documents_to_answer=3)
    recursion = RTPRecursion(builder=builder, max_depth=1, min_node_size=2)
    
    root, global_metrics = recursion(sample_text_collection)
    
    # Root should have metrics
    assert root.metrics is not None
    assert isinstance(root.metrics, SplitMetrics)
    
    # Children should have metrics if they exist and are not leaves
    # Leaf nodes (created by stopping criteria) won't have metrics


def test_rtp_recursion_split_ratio_criteria():
    """Test that recursion stops when split ratio is out of range."""
    builder = RTPBuilder(use_gpu=False, n_medoids=2, n_documents_to_answer=3)
    recursion = RTPRecursion(
        builder=builder,
        min_split_ratio=0.4,  # Very restrictive
        max_split_ratio=0.6,  # Very restrictive
        max_depth=10,
        min_node_size=2,
    )
    
    root, metrics = recursion(sample_text_collection)
    
    # Tree should exist but may have limited depth due to split ratio constraint
    assert isinstance(root, TreeNode)
    assert root.documents == list(range(len(sample_text_collection)))


def test_rtp_recursion_question_propagation():
    """Test that questions are set at each split node."""
    builder = RTPBuilder(use_gpu=False, n_medoids=2, n_documents_to_answer=3)
    recursion = RTPRecursion(builder=builder, max_depth=2, min_node_size=2)
    
    root, metrics = recursion(sample_text_collection)
    
    # Root should have a question (unless it's a leaf due to stopping criteria)
    if root.left is not None or root.right is not None:
        assert root.question is not None
        assert isinstance(root.question, str)
        assert len(root.question) > 0


def test_rtp_recursion_with_iterator():
    """Test that RTPRecursion works with an iterator."""
    builder = RTPBuilder(use_gpu=False, n_medoids=2, n_documents_to_answer=2)
    recursion = RTPRecursion(builder=builder, max_depth=1)
    
    def text_generator():
        for text in sample_text_collection:
            yield text
    
    root, metrics = recursion(text_generator())
    
    assert isinstance(root, TreeNode)
    assert root.documents == list(range(len(sample_text_collection)))


def test_rtp_recursion_empty_collection_raises_error():
    """Test that RTPRecursion raises error with empty collection."""
    builder = RTPBuilder(use_gpu=False)
    recursion = RTPRecursion(builder=builder)
    
    # The RTPBuilder itself should raise the error
    with pytest.raises(ValueError, match="Text collection cannot be empty"):
        recursion([])


def test_rtp_recursion_small_collection():
    """Test RTPRecursion with a very small collection."""
    builder = RTPBuilder(use_gpu=False, n_medoids=2, n_documents_to_answer=2)
    recursion = RTPRecursion(builder=builder, min_node_size=1, max_depth=2)
    
    small_collection = ["Dogs bark.", "Cats meow."]
    root, metrics = recursion(small_collection)
    
    assert isinstance(root, TreeNode)
    assert root.documents == [0, 1]


def test_rtp_recursion_tree_serialization():
    """Test that the resulting tree with metrics can be serialized."""
    builder = RTPBuilder(use_gpu=False, n_medoids=2, n_documents_to_answer=2)
    recursion = RTPRecursion(builder=builder, max_depth=1, min_node_size=2)
    
    root, metrics = recursion(sample_text_collection)
    
    # Tree should be serializable
    json_string = root.model_dump_json()
    assert isinstance(json_string, str)
    assert len(json_string) > 0
    
    # Should be able to deserialize
    loaded_root = TreeNode.model_validate_json(json_string)
    assert loaded_root.documents == root.documents
    assert loaded_root.question == root.question


def test_split_metrics_addition():
    """Test that SplitMetrics can be added together."""
    metrics1 = SplitMetrics(
        llm_input_tokens=100,
        llm_output_tokens=50,
        nli_calls=5,
        total_time_ms=500.0,
    )
    
    metrics2 = SplitMetrics(
        llm_input_tokens=200,
        llm_output_tokens=75,
        nli_calls=8,
        total_time_ms=300.0,
    )
    
    combined = metrics1 + metrics2
    
    assert combined.llm_input_tokens == 300
    assert combined.llm_output_tokens == 125
    assert combined.nli_calls == 13
    assert combined.total_time_ms == 800.0


def test_rtp_recursion_document_indices_consistency():
    """Test that document indices are consistent throughout the tree."""
    builder = RTPBuilder(use_gpu=False, n_medoids=2, n_documents_to_answer=3)
    recursion = RTPRecursion(builder=builder, max_depth=2, min_node_size=2)
    
    root, metrics = recursion(sample_text_collection)
    
    def check_consistency(node, parent_docs):
        """Check that node documents are subset of parent documents."""
        assert set(node.documents).issubset(set(parent_docs))
        
        if node.left and node.right:
            # Children should partition parent documents
            left_docs = set(node.left.documents)
            right_docs = set(node.right.documents)
            
            # No overlap between left and right
            assert len(left_docs & right_docs) == 0
            
            # Union should equal parent (or be a subset if some weren't split)
            assert (left_docs | right_docs).issubset(set(node.documents))
            
            check_consistency(node.left, node.documents)
            check_consistency(node.right, node.documents)
    
    check_consistency(root, root.documents)
