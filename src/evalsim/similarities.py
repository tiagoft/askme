"""SimilarityCalculator: lexical, semantic, logical, and functional similarity."""

import numpy as np
from pydantic import BaseModel

from .commons import nmi_binary_similarity
from .lexical_similarity import pairwise_jaccard_ngram_similarity
from .semantic_similarity import pairwise_cosine_similarity
from .logical_similarity import pairwise_logical_similarity


class PooledResults(BaseModel):
    mean: float
    std: float


class Similarity(BaseModel):
    lexical: PooledResults | None = None
    semantic: PooledResults | None = None
    logical: PooledResults | None = None
    functional: PooledResults | None = None


def pairwise_functional_similarity(
    scores: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Pairwise NMI between hypotheses based on their binary answer patterns.

    Args:
        scores: (n_hypotheses, n_premises) entailment score matrix.
        threshold: binarisation cutoff applied before computing NMI.

    Returns:
        Symmetric (n_hypotheses, n_hypotheses) NMI matrix with zeros on diagonal.
    """
    n = scores.shape[0]
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i, j] = nmi_binary_similarity(scores[i], scores[j], threshold)
            matrix[j, i] = matrix[i, j]
    return matrix


class SimilarityCalculator:
    def __init__(
        self,
        max_ngram: int = 3,
        use_lexical: bool = True,
        use_semantic: bool = True,
        use_logical: bool = True,
        use_functional: bool = False,
    ):
        self.use_lexical = use_lexical
        self.use_semantic = use_semantic
        self.use_logical = use_logical
        self.use_functional = use_functional

        if use_semantic:
            from sentence_transformers import SentenceTransformer
            self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

        if use_logical:
            from askme.rtp.nli import NLIWithChunkingAndPooling
            self.logical_model = NLIWithChunkingAndPooling()

        self.max_ngram = max_ngram

    def _pool(self, matrix: np.ndarray) -> PooledResults:
        n = matrix.shape[0]
        values = matrix[np.triu_indices(n, k=1)]
        return PooledResults(mean=float(np.mean(values)), std=float(np.std(values)))

    def calculate_lexical_similarity(self, texts: list[str]) -> PooledResults:
        return self._pool(pairwise_jaccard_ngram_similarity(texts, self.max_ngram))

    def calculate_semantic_similarity(self, texts: list[str]) -> PooledResults:
        return self._pool(pairwise_cosine_similarity(texts, self.semantic_model))

    def calculate_logical_similarity(self, texts: list[str]) -> PooledResults:
        return self._pool(pairwise_logical_similarity(texts, self.logical_model))

    def calculate_functional_similarity(self, scores: np.ndarray) -> PooledResults:
        return self._pool(pairwise_functional_similarity(scores))

    def calculate_similarity(
        self,
        texts: list[str],
        functional_scores: np.ndarray | None = None,
    ) -> Similarity:
        lexical = self.calculate_lexical_similarity(texts) if self.use_lexical else None
        semantic = self.calculate_semantic_similarity(texts) if self.use_semantic else None
        logical = self.calculate_logical_similarity(texts) if self.use_logical else None
        functional = None
        if self.use_functional and functional_scores is not None:
            functional = self.calculate_functional_similarity(functional_scores)
        return Similarity(lexical=lexical, semantic=semantic, logical=logical, functional=functional)

    def __call__(
        self,
        texts: list[str],
        functional_scores: np.ndarray | None = None,
    ) -> Similarity:
        return self.calculate_similarity(texts, functional_scores)
