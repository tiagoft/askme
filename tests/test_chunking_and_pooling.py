from askme.utils import TextEmbeddingWithChunker
try:
    import torch
except Exception:
    torch = None
import numpy as np
import pytest


def test_chunk_text():
    text = "This is a sample text that will be chunked into smaller pieces. " * 100
    large_text_embedding = TextEmbeddingWithChunker(
        model_name='all-MiniLM-L6-v2',
        chunk_size=20,
        overlap=5,
    )(text)
    assert large_text_embedding is not None

def test_chunk_text_gpu():
    text = "This is a sample text that will be chunked into smaller pieces. " * 100
    large_text_embedding = TextEmbeddingWithChunker(
        model_name='all-MiniLM-L6-v2',
        chunk_size=20,
        overlap=5,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )(text)
    assert large_text_embedding is not None
    assert isinstance(large_text_embedding, np.ndarray)