"""Common similarity utility functions for evalsim."""


def jaccard_similarity(set1, set2):
    """Calculate the Jaccard similarity between two sets."""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    if union == 0:
        return 0.0
    return intersection / union


def cosine_similarity(vec1, vec2):
    """Calculate the Cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude_vec1 = sum(a ** 2 for a in vec1) ** 0.5
    magnitude_vec2 = sum(b ** 2 for b in vec2) ** 0.5
    if magnitude_vec1 == 0 or magnitude_vec2 == 0:
        return 0.0
    return dot_product / (magnitude_vec1 * magnitude_vec2)


def nmi_binary_similarity(a, b, threshold: float = 0.5) -> float:
    """Normalized Mutual Information between two entailment score vectors.

    Binarizes both vectors at `threshold`, then computes NMI using the
    geometric-mean normalization: MI(x,y) / sqrt(H(x) * H(y)).

    Returns 0 when either vector is constant (no information to share).
    Returns values in [0, 1], where 1 means the two binary patterns are
    perfectly predictable from each other (correlated or anti-correlated).
    """
    import numpy as np

    x = (np.asarray(a) >= threshold).astype(int)
    y = (np.asarray(b) >= threshold).astype(int)
    n = len(x)

    def entropy(v):
        _, counts = np.unique(v, return_counts=True)
        p = counts / n
        return float(-np.sum(p * np.log2(p + 1e-15)))

    hx = entropy(x)
    hy = entropy(y)
    denom = (hx * hy) ** 0.5
    if denom < 1e-12:
        return 0.0

    hxy = entropy(x * 2 + y)  # joint distribution via unique int per pair
    mi = hx + hy - hxy
    return float(max(0.0, mi) / denom)  # clamp for floating-point noise
