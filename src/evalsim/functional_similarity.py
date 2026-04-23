"""Functional similarity between hypotheses measured via NMI on NLI answers."""

import numpy as np

from askme.rtp.nli import NLIWithChunkingAndPooling

from .commons import nmi_binary_similarity


def all_entailment_scores(
    hypotheses: list[str],
    premises: list[str],
    model: NLIWithChunkingAndPooling,
) -> np.ndarray:
    """Run NLI for every (hypothesis, premise) pair.

    Returns an (n_hypotheses, n_premises) matrix of entailment scores.
    """
    scores = []
    for h in hypotheses:
        results = model(premises, h)
        scores.append([r.entailment_score for r in results])
    return np.array(scores)


def pairwise_functional_similarity(
    scores: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Pairwise NMI between hypotheses based on their binary answer patterns.

    Two hypotheses are functionally similar (NMI close to 1) when knowing one
    question's yes/no answer across the collection perfectly predicts the
    other's — whether correlated or anti-correlated.

    Args:
        scores: (n_hypotheses, n_premises) entailment score matrix.
        threshold: binarisation cutoff applied before computing NMI.

    Returns:
        Symmetric (n_hypotheses, n_hypotheses) NMI matrix with zeros on the
        diagonal.
    """
    n = scores.shape[0]
    similarities = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            similarities[i, j] = nmi_binary_similarity(
                scores[i], scores[j], threshold
            )
            similarities[j, i] = similarities[i, j]
    return similarities
