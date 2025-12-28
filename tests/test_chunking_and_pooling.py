from askme.askquestions import check_entailment, models
import pytest


def test_chunk_text_small_document():
    """Test that documents smaller than chunk_size are not split."""
    text = " ".join(["word"] * 100)  # 100 words
    chunks = check_entailment.chunk_text(text, chunk_size=350, overlap=50)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_large_document():
    """Test that documents larger than chunk_size are split correctly."""
    # Create a document with 500 words
    words = [f"word{i}" for i in range(500)]
    text = " ".join(words)
    
    chunks = check_entailment.chunk_text(text, chunk_size=350, overlap=50)
    
    # Should have 2 chunks: first 350 words, then starting at word 300 (350-50) for 200 words
    assert len(chunks) == 2
    
    # First chunk should have 350 words
    first_chunk_words = chunks[0].split()
    assert len(first_chunk_words) == 350
    assert first_chunk_words[0] == "word0"
    assert first_chunk_words[-1] == "word349"
    
    # Second chunk should start at word 300 (350 - 50 overlap)
    second_chunk_words = chunks[1].split()
    assert len(second_chunk_words) == 200  # remaining words from 300 to 499
    assert second_chunk_words[0] == "word300"
    assert second_chunk_words[-1] == "word499"


def test_chunk_text_with_overlap():
    """Test that overlap works correctly."""
    words = [f"word{i}" for i in range(400)]
    text = " ".join(words)
    
    chunks = check_entailment.chunk_text(text, chunk_size=350, overlap=50)
    
    # Verify overlap: last 50 words of first chunk should be first 50 words of second chunk
    first_chunk_words = chunks[0].split()
    second_chunk_words = chunks[1].split()
    
    # Words 300-349 should appear in both chunks
    assert first_chunk_words[300] == "word300"
    assert second_chunk_words[0] == "word300"


def test_pool_nli_scores_single_chunk():
    """Test max-pooling with a single chunk (document < 350 words)."""
    
    # Mock check function that returns predictable results
    def mock_check_fn(premise, hypothesis, **kwargs):
        return (True, 0.9, 0.1, 0.9)
    
    text = " ".join(["word"] * 100)
    result = check_entailment.pool_nli_scores(
        mock_check_fn,
        premise=text,
        hypothesis="test hypothesis"
    )
    
    assert result == (True, 0.9, 0.1, 0.9)


def test_pool_nli_scores_max_pooling():
    """Test that max-pooling selects the chunk with highest entailment score."""
    
    call_count = [0]
    
    # Mock check function that returns different scores for different chunks
    def mock_check_fn(premise, hypothesis, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            # First chunk: low entailment
            return (False, 0.3, 0.7, 0.3)
        else:
            # Second chunk: high entailment
            return (True, 0.9, 0.1, 0.9)
    
    # Create a document with 500 words to force chunking
    text = " ".join([f"word{i}" for i in range(500)])
    
    result = check_entailment.pool_nli_scores(
        mock_check_fn,
        premise=text,
        hypothesis="test hypothesis",
        chunk_size=350,
        overlap=50
    )
    
    # Should return the result from the second chunk (higher entailment)
    assert result == (True, 0.9, 0.1, 0.9)
    assert call_count[0] == 2  # Should have called mock_check_fn twice


def test_pool_nli_scores_document_labeled_yes_with_one_entailing_chunk():
    """
    Test that a document is correctly labeled 'Yes' if only one chunk entails the hypothesis.
    This is the key acceptance criterion from the issue.
    """
    
    chunks_seen = []
    
    def mock_check_fn(premise, hypothesis, **kwargs):
        chunks_seen.append(premise)
        # First chunk: no entailment
        if len(chunks_seen) == 1:
            return (False, 0.2, 0.8, 0.2)
        # Second chunk: strong entailment
        elif len(chunks_seen) == 2:
            return (True, 0.95, 0.05, 0.95)
        # Third chunk: no entailment
        else:
            return (False, 0.1, 0.9, 0.1)
    
    # Create a document with 800 words to create 3 chunks
    text = " ".join([f"word{i}" for i in range(800)])
    
    result = check_entailment.pool_nli_scores(
        mock_check_fn,
        premise=text,
        hypothesis="test hypothesis",
        chunk_size=350,
        overlap=50
    )
    
    # Should be labeled as entailed (True) because at least one chunk entails
    is_entailed, entailment_score, contradiction_score, p_entailment = result
    assert is_entailed == True
    assert entailment_score == 0.95  # Max entailment score across all chunks
    assert len(chunks_seen) == 3  # Should have processed 3 chunks


def test_pool_nli_scores_with_real_nli_model():
    """Test pool_nli_scores with actual NLI model (integration test)."""
    model_name = 'MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7'
    model, tokenizer = models.make_nli_model(model_name=model_name)
    
    # Create a long document where only one part entails the hypothesis
    entailing_text = "The sky is blue and clear. The weather is sunny."
    non_entailing_text = "The cat sat on the mat. The dog ran in the park."
    
    # Build a document with 400 words, with entailing text in the middle
    filler = " ".join(["filler"] * 180)
    long_document = f"{non_entailing_text} {filler} {entailing_text} {filler}"
    
    # Use pool_nli_scores with the real check_entailment_nli function
    from functools import partial
    check_fn = partial(
        check_entailment.check_entailment_nli,
        model=model,
        tokenizer=tokenizer,
        device='cpu'
    )
    
    result = check_entailment.pool_nli_scores(
        check_fn,
        premise=long_document,
        hypothesis="The sky is blue.",
        chunk_size=350,
        overlap=50
    )
    
    is_entailed, entailment_score, contradiction_score, p_entailment = result
    
    # The document should be labeled as entailed because one chunk contains entailing text
    # Note: This is a weaker assertion since we can't guarantee exact scores
    assert entailment_score > contradiction_score
