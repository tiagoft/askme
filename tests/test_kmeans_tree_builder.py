"""Tests for KMeansTreeBuilder class."""

import pytest
from askme.rtp import KMeansTreeBuilder, KMeansTreeRecursion, TreeNode, SplitMetrics


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


@pytest.mark.llm
def test_kmeans_tree_builder_initialization_cpu():
    """Test that KMeansTreeBuilder initializes correctly with CPU."""
    builder = KMeansTreeBuilder(use_gpu=False)
    
    assert builder.use_gpu is False
    assert builder.embedding_model is not None
    assert builder.nli_model is not None
    assert builder.nli_tokenizer is not None
    assert builder.llm_model is not None
    assert builder.gpu_resources is None
    assert builder.embedding_model_name == builder.embedding_model_name  # Verify it was set
    assert builder.nli_model_name == builder.nli_model_name  # Verify it was set
    assert builder.llm_model_name == builder.llm_model_name  # Verify it was set


@pytest.mark.llm
def test_kmeans_tree_builder_initialization_with_custom_params():
    """Test that KMeansTreeBuilder accepts custom parameters."""
    builder = KMeansTreeBuilder(
        use_gpu=False,
        chunk_size=100,
        overlap=20,
        n_medoids_per_cluster=3,
        n_documents_to_answer=4,
        knn_neighbors=3,
        alpha=0.95,
        max_iter=50,
        tol=1e-4,
    )
    
    assert builder.chunk_size == 100
    assert builder.overlap == 20
    assert builder.n_medoids_per_cluster == 3
    assert builder.n_documents_to_answer == 4
    assert builder.knn_neighbors == 3
    assert builder.alpha == 0.95
    assert builder.max_iter == 50
    assert builder.tol == 1e-4


@pytest.mark.llm
def test_kmeans_tree_builder_call_with_list():
    """Test that KMeansTreeBuilder can be called with a list of strings."""
    builder = KMeansTreeBuilder(use_gpu=False)
    
    # Call with the sample collection
    result = builder(sample_text_collection)
    
    # Verify the result is a TreeNode
    assert isinstance(result, TreeNode)
    assert result.documents == list(range(len(sample_text_collection)))
    assert result.question is not None
    assert isinstance(result.question, str)
    assert len(result.question) > 0


@pytest.mark.llm
def test_kmeans_tree_builder_call_creates_tree_structure():
    """Test that KMeansTreeBuilder creates a tree with children nodes."""
    builder = KMeansTreeBuilder(use_gpu=False)
    
    result = builder(sample_text_collection)
    
    # The tree should have a question at the root
    assert result.question is not None
    
    # If there's a meaningful split, there should be children
    if result.left is not None:
        assert isinstance(result.left, TreeNode)
        assert len(result.left.documents) > 0
    
    if result.right is not None:
        assert isinstance(result.right, TreeNode)
        assert len(result.right.documents) > 0
    
    # If both children exist, they should partition the documents
    if result.left is not None and result.right is not None:
        all_docs = set(result.left.documents + result.right.documents)
        # All documents should be in one child or the other
        assert len(all_docs) <= len(sample_text_collection)


@pytest.mark.llm
def test_kmeans_tree_builder_call_with_iterator():
    """Test that KMeansTreeBuilder can be called with an iterator."""
    builder = KMeansTreeBuilder(use_gpu=False)
    
    # Create an iterator
    def text_generator():
        for text in sample_text_collection:
            yield text
    
    result = builder(text_generator())
    
    # Verify the result is a TreeNode
    assert isinstance(result, TreeNode)
    assert result.documents == list(range(len(sample_text_collection)))
    assert result.question is not None


@pytest.mark.llm
def test_kmeans_tree_builder_call_with_small_collection():
    """Test that KMeansTreeBuilder works with a very small collection."""
    builder = KMeansTreeBuilder(use_gpu=False, n_medoids_per_cluster=1, n_documents_to_answer=2)
    
    small_collection = ["Dogs are pets.", "Cats are pets.", "Birds fly."]
    
    result = builder(small_collection)
    
    assert isinstance(result, TreeNode)
    assert result.documents == [0, 1, 2]
    assert result.question is not None


@pytest.mark.llm
def test_kmeans_tree_builder_call_empty_collection_raises_error():
    """Test that KMeansTreeBuilder raises an error with empty collection."""
    builder = KMeansTreeBuilder(use_gpu=False)
    
    with pytest.raises(ValueError, match="Text collection cannot be empty"):
        builder([])


@pytest.mark.llm
def test_kmeans_tree_builder_multiple_calls():
    """Test that KMeansTreeBuilder can be called multiple times."""
    builder = KMeansTreeBuilder(use_gpu=False, n_medoids_per_cluster=1, n_documents_to_answer=3)
    
    collection1 = ["Dogs bark.", "Cats meow.", "Birds chirp."]
    collection2 = ["The sun is hot.", "The moon is bright.", "Stars twinkle."]
    
    result1 = builder(collection1)
    result2 = builder(collection2)
    
    # Both should produce valid tree nodes
    assert isinstance(result1, TreeNode)
    assert isinstance(result2, TreeNode)
    
    # They should have different questions (most likely)
    assert result1.question is not None
    assert result2.question is not None
    
    # Document indices should match collection sizes
    assert result1.documents == [0, 1, 2]
    assert result2.documents == [0, 1, 2]


@pytest.mark.llm
def test_kmeans_tree_builder_result_can_be_serialized():
    """Test that the TreeNode result can be serialized to JSON."""
    builder = KMeansTreeBuilder(use_gpu=False, n_medoids_per_cluster=1, n_documents_to_answer=2)
    
    small_collection = ["Dogs are pets.", "Cats are pets.", "Birds fly."]
    result = builder(small_collection)
    
    # Should be able to serialize to JSON
    json_string = result.model_dump_json()
    assert isinstance(json_string, str)
    assert len(json_string) > 0
    
    # Should be able to deserialize back
    loaded_result = TreeNode.model_validate_json(json_string)
    assert loaded_result == result


@pytest.mark.llm
def test_kmeans_tree_builder_returns_metrics_when_requested():
    """Test that KMeansTreeBuilder returns metrics when return_metrics=True."""
    builder = KMeansTreeBuilder(use_gpu=False, n_medoids_per_cluster=1, n_documents_to_answer=3)
    
    result, metrics = builder(sample_text_collection, return_metrics=True)
    
    # Verify result is still a TreeNode
    assert isinstance(result, TreeNode)
    assert result.documents == list(range(len(sample_text_collection)))
    assert result.question is not None
    
    # Verify metrics is a SplitMetrics instance
    assert isinstance(metrics, SplitMetrics)
    
    # Verify all metrics fields are populated
    assert metrics.llm_input_tokens > 0
    assert metrics.llm_output_tokens > 0
    assert metrics.nli_calls > 0
    assert metrics.faiss_search_time_ms >= 0.0
    assert metrics.label_propagation_time_ms >= 0.0
    assert metrics.total_time_ms > 0.0
    assert 0.0 <= metrics.split_ratio <= 1.0
    assert 0.0 <= metrics.medoid_nli_confidence_avg <= 1.0
    assert metrics.llm_request_time_ms > 0.0
    assert metrics.nli_time_ms > 0.0


@pytest.mark.llm
def test_kmeans_tree_builder_default_no_metrics():
    """Test that KMeansTreeBuilder returns only TreeNode by default."""
    builder = KMeansTreeBuilder(use_gpu=False, n_medoids_per_cluster=1, n_documents_to_answer=2)
    
    result = builder(sample_text_collection)
    
    # Should return just TreeNode, not a tuple
    assert isinstance(result, TreeNode)
    assert not isinstance(result, tuple)


@pytest.mark.llm
def test_kmeans_tree_builder_with_retry_parameters():
    """Test that KMeansTreeBuilder accepts retry parameters."""
    builder = KMeansTreeBuilder(
        use_gpu=False,
        max_retries=5,
        min_split_ratio=0.3,
        max_split_ratio=0.7,
    )
    
    assert builder.max_retries == 5
    assert builder.min_split_ratio == 0.3
    assert builder.max_split_ratio == 0.7


@pytest.mark.llm
def test_kmeans_tree_builder_default_retry_parameters():
    """Test that KMeansTreeBuilder has correct default retry parameters."""
    builder = KMeansTreeBuilder(use_gpu=False)
    
    assert builder.max_retries == 3
    assert builder.min_split_ratio is None
    assert builder.max_split_ratio is None


@pytest.mark.llm
def test_kmeans_tree_recursion_initialization():
    """Test that KMeansTreeRecursion initializes correctly."""
    builder = KMeansTreeBuilder(use_gpu=False)
    recursion = KMeansTreeRecursion(
        builder=builder,
        min_node_size=3,
        min_split_ratio=0.2,
        max_split_ratio=0.8,
        max_depth=5,
    )
    
    assert recursion.builder == builder
    assert recursion.min_node_size == 3
    assert recursion.min_split_ratio == 0.2
    assert recursion.max_split_ratio == 0.8
    assert recursion.max_depth == 5


@pytest.mark.llm
def test_kmeans_tree_recursion_call():
    """Test that KMeansTreeRecursion can build a recursive tree."""
    builder = KMeansTreeBuilder(use_gpu=False, n_medoids_per_cluster=1, n_documents_to_answer=3)
    recursion = KMeansTreeRecursion(
        builder=builder,
        min_node_size=2,
        min_split_ratio=0.1,
        max_split_ratio=0.9,
        max_depth=2,
    )
    
    result, metrics = recursion(sample_text_collection)
    
    # Verify result is a TreeNode
    assert isinstance(result, TreeNode)
    assert result.documents == list(range(len(sample_text_collection)))
    assert result.question is not None
    
    # Verify metrics is a SplitMetrics instance
    assert isinstance(metrics, SplitMetrics)
    assert metrics.total_time_ms > 0.0


@pytest.mark.llm
def test_kmeans_tree_recursion_respects_min_node_size():
    """Test that KMeansTreeRecursion respects min_node_size stopping criterion."""
    builder = KMeansTreeBuilder(use_gpu=False, n_medoids_per_cluster=1)
    recursion = KMeansTreeRecursion(
        builder=builder,
        min_node_size=100,  # Larger than our collection
        max_depth=10,
    )
    
    result, metrics = recursion(sample_text_collection)
    
    # Should create a leaf node without splitting
    assert result.left is None
    assert result.right is None


@pytest.mark.llm
def test_kmeans_tree_recursion_respects_max_depth():
    """Test that KMeansTreeRecursion respects max_depth stopping criterion."""
    builder = KMeansTreeBuilder(use_gpu=False, n_medoids_per_cluster=1)
    recursion = KMeansTreeRecursion(
        builder=builder,
        min_node_size=1,
        max_depth=0,  # No recursion
    )
    
    result, metrics = recursion(sample_text_collection)
    
    # Should create a leaf node without splitting
    assert result.left is None
    assert result.right is None
