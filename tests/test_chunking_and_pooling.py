from askme.config.config import TextEmbeddingConfig, config_factory
from askme.utils import TextEmbeddingWithChunker
try:
    import torch
except Exception:
    torch = None
import numpy as np
from pathlib import Path
import os


def test_chunk_text():
    text = "This is a sample text that will be chunked into smaller pieces. " * 100
    config = config_factory(TextEmbeddingConfig)
    large_text_embedding = TextEmbeddingWithChunker(config=config)
    embeddings = large_text_embedding(text)
    assert embeddings is not None
    assert isinstance(embeddings, np.ndarray)
    assert os.path.exists(Path(large_text_embedding.cache_fn).expanduser())
