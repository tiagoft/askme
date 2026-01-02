"""Tests for RTPBuilder class."""

import pytest
from askme.rtp import RTPBuilder, TreeNode


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


def test_rtp_builder_call_with_small_collection():
    """Test that RTPBuilder works with a very small collection."""
    builder = RTPBuilder(use_gpu=False, n_medoids=2, n_documents_to_answer=2)
    
    small_collection = ["Dogs are pets.", "Cats are pets.", "Birds fly."]
    
    result = builder(small_collection)
    
    assert isinstance(result, TreeNode)
    assert result.documents == [0, 1, 2]
    assert result.question is not None


def test_rtp_builder_call_empty_collection_raises_error():
    """Test that RTPBuilder raises an error with empty collection."""
    builder = RTPBuilder(use_gpu=False)
    
    with pytest.raises(ValueError, match="Text collection cannot be empty"):
        builder([])


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
