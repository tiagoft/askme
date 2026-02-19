import numpy as np
from sentence_transformers import SentenceTransformer

from .commons import cosine_similarity


def pairwise_cosine_similarity(
    texts: list[str],
    model: SentenceTransformer,
) -> np.ndarray:
    """Calculate the pairwise semantic similarity between two texts using Cosine similarity."""
    embeddings = model.encode(texts)
    similarities = np.zeros((len(embeddings), len(embeddings)))
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            similarities[i, j] = cosine_similarity(embeddings[i], embeddings[j])
            similarities[j, i] = similarities[i, j]  # Symmetric matrix
    return similarities
