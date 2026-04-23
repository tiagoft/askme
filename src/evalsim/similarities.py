"""SimilarityCalculator: lexical, semantic, logical, and functional similarity."""

import numpy as np
from pydantic import BaseModel

from .lexical_similarity import pairwise_jaccard_ngram_similarity
from .semantic_similarity import pairwise_cosine_similarity
from .logical_similarity import pairwise_logical_similarity


class Similarity(BaseModel):
    lexical: float
    semantic: float
    logical: float
    functional: float | None = None


class SimilarityCalculator:
    def __init__(
        self,
        max_ngram: int = 3,
        pooling_fn: np.ufunc = np.mean,
        use_lexical: bool = True,
        use_semantic: bool = True,
        use_logical: bool = True,
        use_functional: bool = False,
    ):
        self.use_lexical = use_lexical
        self.use_semantic = use_semantic
        self.use_logical = use_logical
        self.use_functional = use_functional
        self.max_ngram = max_ngram
        self.pooling_fn = pooling_fn

        if use_semantic:
            from sentence_transformers import SentenceTransformer
            self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

        if use_logical or use_functional:
            from askme.rtp.nli import NLIWithChunkingAndPooling
            self.nli_model = NLIWithChunkingAndPooling()
            self.logical_model = self.nli_model  # backward-compat alias

    def calculate_lexical_similarity(self, texts: list[str]) -> float:
        return self.pooling_fn(pairwise_jaccard_ngram_similarity(texts, self.max_ngram))

    def calculate_semantic_similarity(self, texts: list[str]) -> float:
        return self.pooling_fn(pairwise_cosine_similarity(texts, self.semantic_model))

    def calculate_logical_similarity(self, texts: list[str]) -> float:
        return self.pooling_fn(pairwise_logical_similarity(texts, self.nli_model))

    def calculate_functional_similarity(
        self, questions: list[str], collection: list[str]
    ) -> float:
        """NMI-based functional similarity of `questions` w.r.t. `collection`.

        Runs NLI to get binary answer patterns for each question across the
        collection, then computes pairwise NMI and pools the result.

        Args:
            questions: The yes/no hypotheses to compare.
            collection: The document collection used as NLI premises.

        Returns:
            Pooled NMI score in [0, 1].
        """
        from .functional_similarity import all_entailment_scores, pairwise_functional_similarity
        scores = all_entailment_scores(questions, collection, self.nli_model)
        matrix = pairwise_functional_similarity(scores)
        return float(self.pooling_fn(matrix))

    def calculate_similarity(self, texts: list[str]) -> Similarity:
        lexical = self.calculate_lexical_similarity(texts) if self.use_lexical else 0.0
        semantic = self.calculate_semantic_similarity(texts) if self.use_semantic else 0.0
        logical = self.calculate_logical_similarity(texts) if self.use_logical else 0.0
        return Similarity(lexical=lexical, semantic=semantic, logical=logical)

    def __call__(self, texts: list[str]) -> Similarity:
        return self.calculate_similarity(texts)
