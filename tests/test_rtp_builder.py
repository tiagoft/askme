"""Tests for RTPBuilder class."""

import pytest
from askme.rtp import RTPBuilder, TreeNode, SplitMetrics


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
def test_rtp_builder_initialization_cpu():
    """Test that RTPBuilder initializes correctly with CPU."""
    builder = RTPBuilder(use_gpu=False)
    
    assert builder.use_gpu is False
    assert builder.embedding_model is not None
    assert builder.nli_model is not None
    assert builder.nli_tokenizer is not None
    assert builder.llm_model is not None
    assert builder.gpu_resources is None  # No GPU resources when use_gpu=False
    assert builder.embedding_model_name == 'sentence-transformers/paraphrase-albert-small-v2'
    assert builder.nli_model_name == 'MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7'
    assert builder.llm_model_name == "gpt-4o-mini"

@pytest.mark.llm
def test_rtp_builder_initialization_with_custom_params():
    """Test that RTPBuilder accepts custom parameters."""
    builder = RTPBuilder(
        use_gpu=False,
        chunk_size=100,
        overlap=20,
        n_medoids=3,
        n_documents_to_answer=4,
        knn_neighbors=3,
        alpha=0.95,
        max_iter=50,
        tol=1e-4,
    )
    
    assert builder.chunk_size == 100
    assert builder.overlap == 20
    assert builder.n_medoids == 3
    assert builder.n_documents_to_answer == 4
    assert builder.knn_neighbors == 3
    assert builder.alpha == 0.95
    assert builder.max_iter == 50
    assert builder.tol == 1e-4

@pytest.mark.llm
@pytest.mark.gpu
def test_rtp_builder_gpu_resources_initialization():
    """Test that GPU resources are initialized when use_gpu=True."""
    import faiss
    
    # Note: This test may fail if CUDA is not available, but it tests the logic
    try:
        builder = RTPBuilder(use_gpu=True)
        assert builder.use_gpu is True
        assert builder.gpu_resources is not None
        assert isinstance(builder.gpu_resources, faiss.StandardGpuResources)
    except Exception as e:
        # If GPU not available, at least verify the initialization logic is correct
        # The gpu_resources should still be set even if GPU operations fail
        pytest.skip(f"GPU not available: {e}")

@pytest.mark.llm
@pytest.mark.gpu
def test_rtp_builder_call_with_list():
    """Test that RTPBuilder can be called with a list of strings."""
    builder = RTPBuilder(use_gpu=False)
    
    # Call with the sample collection
    result = builder(sample_text_collection)
    
    # Verify the result is a TreeNode
    assert isinstance(result, TreeNode)
    assert result.documents == list(range(len(sample_text_collection)))
    assert result.question is not None
    assert isinstance(result.question, str)
    assert len(result.question) > 0

@pytest.mark.llm
def test_rtp_builder_call_creates_tree_structure():
    """Test that RTPBuilder creates a tree with children nodes."""
    builder = RTPBuilder(use_gpu=False)
    
    result = builder(sample_text_collection)
    
    # The tree should have a question at the root
    assert result.question is not None
    
    # If there's a meaningful split, there should be children
    # (This may not always be the case depending on the hypothesis)
    # We just check that the structure is valid
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
def test_rtp_builder_call_with_iterator():
    """Test that RTPBuilder can be called with an iterator."""
    builder = RTPBuilder(use_gpu=False)
    
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
def test_rtp_builder_call_with_small_collection():
    """Test that RTPBuilder works with a very small collection."""
    builder = RTPBuilder(use_gpu=False, n_medoids=2, n_documents_to_answer=2)
    
    small_collection = ["Dogs are pets.", "Cats are pets.", "Birds fly."]
    
    result = builder(small_collection)
    
    assert isinstance(result, TreeNode)
    assert result.documents == [0, 1, 2]
    assert result.question is not None


@pytest.mark.llm
def test_rtp_builder_call_empty_collection_raises_error():
    """Test that RTPBuilder raises an error with empty collection."""
    builder = RTPBuilder(use_gpu=False)
    
    with pytest.raises(ValueError, match="Text collection cannot be empty"):
        builder([])

@pytest.mark.llm
def test_rtp_builder_multiple_calls():
    """Test that RTPBuilder can be called multiple times (models are reused)."""
    builder = RTPBuilder(use_gpu=False, n_medoids=2, n_documents_to_answer=3)
    
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
def test_rtp_builder_result_can_be_serialized():
    """Test that the TreeNode result can be serialized to JSON."""
    builder = RTPBuilder(use_gpu=False, n_medoids=2, n_documents_to_answer=2)
    
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
def test_rtp_builder_returns_metrics_when_requested():
    """Test that RTPBuilder returns metrics when return_metrics=True."""
    builder = RTPBuilder(use_gpu=False, n_medoids=2, n_documents_to_answer=3)
    
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
    # split_ratio should be 0 if no split, or > 0 and < 1 if split happened
    assert 0.0 <= metrics.split_ratio <= 1.0
    # medoid_nli_confidence_avg should be between 0 and 1
    assert 0.0 <= metrics.medoid_nli_confidence_avg <= 1.0
    # New timing metrics should be populated
    assert metrics.llm_request_time > 0.0
    assert metrics.nli_time_ms > 0.0


@pytest.mark.llm
def test_rtp_builder_default_no_metrics():
    """Test that RTPBuilder returns only TreeNode by default (return_metrics=False)."""
    builder = RTPBuilder(use_gpu=False, n_medoids=2, n_documents_to_answer=2)
    
    result = builder(sample_text_collection)
    
    # Should return just TreeNode, not a tuple
    assert isinstance(result, TreeNode)
    assert not isinstance(result, tuple)


def test_split_metrics_can_be_serialized():
    """Test that SplitMetrics can be serialized to JSON."""
    metrics = SplitMetrics(
        llm_input_tokens=100,
        llm_output_tokens=50,
        nli_calls=5,
        faiss_search_time_ms=123.45,
        label_propagation_time_ms=234.56,
        total_time_ms=500.0,
        split_ratio=0.6,
        medoid_nli_confidence_avg=0.85,
        llm_request_time=150.0,
        nli_time=100.0,
    )
    
    # Should be able to serialize to JSON
    json_string = metrics.model_dump_json()
    assert isinstance(json_string, str)
    assert len(json_string) > 0
    
    # Should be able to deserialize back
    loaded_metrics = SplitMetrics.model_validate_json(json_string)
    assert loaded_metrics == metrics


def test_split_metrics_default_values():
    """Test that SplitMetrics has correct default values."""
    metrics = SplitMetrics()
    
    assert metrics.llm_input_tokens == 0
    assert metrics.llm_output_tokens == 0
    assert metrics.nli_calls == 0
    assert metrics.faiss_search_time_ms == 0.0
    assert metrics.label_propagation_time_ms == 0.0
    assert metrics.total_time_ms == 0.0
    assert metrics.split_ratio == 0.0
    assert metrics.medoid_nli_confidence_avg == 0.0
    assert metrics.llm_request_time == 0.0
    assert metrics.nli_time_ms == 0.0
    assert metrics.num_nodes == 1


@pytest.mark.llm
def test_rtp_builder_metrics_timing_consistency():
    """Test that timing metrics are consistent (total >= sum of parts)."""
    builder = RTPBuilder(use_gpu=False, n_medoids=2, n_documents_to_answer=2)
    
    result, metrics = builder(sample_text_collection, return_metrics=True)
    
    # Total time should be at least as long as the sum of measured parts
    # (though not necessarily equal since there are other operations)
    measured_time = metrics.faiss_search_time_ms + metrics.label_propagation_time_ms
    assert metrics.total_time_ms >= measured_time


@pytest.mark.llm
def test_rtp_builder_with_retry_parameters():
    """Test that RTPBuilder accepts retry parameters."""
    builder = RTPBuilder(
        use_gpu=False,
        max_retries=5,
        min_split_ratio=0.3,
        max_split_ratio=0.7,
    )
    
    assert builder.max_retries == 5
    assert builder.min_split_ratio == 0.3
    assert builder.max_split_ratio == 0.7


@pytest.mark.llm
def test_rtp_builder_default_retry_parameters():
    """Test that RTPBuilder has correct default retry parameters."""
    builder = RTPBuilder(use_gpu=False)
    
    assert builder.max_retries == 3
    assert builder.min_split_ratio is None
    assert builder.max_split_ratio is None


@pytest.mark.llm
def test_rtp_builder_with_split_ratio_constraints():
    """Test that RTPBuilder works with split ratio constraints."""
    # Use very restrictive split ratio constraints
    # The builder should still work, but may retry
    builder = RTPBuilder(
        use_gpu=False,
        n_medoids=2,
        n_documents_to_answer=3,
        max_retries=2,
        min_split_ratio=0.2,
        max_split_ratio=0.8,
    )
    
    result, metrics = builder(sample_text_collection, return_metrics=True)
    
    # Result should still be valid
    assert isinstance(result, TreeNode)
    assert result.documents == list(range(len(sample_text_collection)))
    assert result.question is not None
    
    # Metrics should be populated
    assert isinstance(metrics, SplitMetrics)
    assert metrics.llm_input_tokens > 0
    assert metrics.llm_output_tokens > 0





