import numpy as np

from .commons import jaccard_similarity


def get_ngrams_up_to_n(text, n):
    """Get all n-grams up to n for a given text."""
    tokens = text.split()
    ngrams = set()
    for i in range(1, n + 1):
        for j in range(len(tokens) - i + 1):
            ngrams.add(tuple(tokens[j:j + i]))
    return ngrams

def pairwise_jaccard_ngram_similarity(texts : list[str], n=3) -> np.ndarray:
    """Calculate the pairwise lexical similarity between two texts using Jaccard similarity."""
    ngrams_list = [get_ngrams_up_to_n(text, n) for text in texts]
    similarities = np.zeros((len(ngrams_list), len(ngrams_list)))
    for i in range(len(ngrams_list)):
        for j in range(i + 1, len(ngrams_list)):
            similarities[i, j] = jaccard_similarity(ngrams_list[i], ngrams_list[j])
            similarities[j, i] = similarities[i, j]  # Symmetric matrix
    return similarities



