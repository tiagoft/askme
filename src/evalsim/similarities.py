
from .lexical_similarity import pairwise_jaccard_ngram_similarity
from .semantic_similarity import pairwise_cosine_similarity
from .logical_similarity import pairwise_logical_similarity

import numpy as np

from pydantic import BaseModel

class PooledResults(BaseModel):
    mean: float
    std: float

class Similarity(BaseModel):
    lexical: PooledResults
    semantic: PooledResults
    logical: PooledResults

class SimilarityCalculator:
    def __init__(self, 
                 max_ngram: int = 3,
                 pooling_fn: np.ufunc = np.mean,
                 use_lexical = True,
                 use_semantic = True,
                 use_logical = True,
                 ):
        
        self.use_lexical = use_lexical
        self.use_semantic = use_semantic
        self.use_logical = use_logical
        
        if use_semantic:

            from sentence_transformers import SentenceTransformer
            self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

        if use_logical:
            from askme.rtp.nli import NLIWithChunkingAndPooling
            self.logical_model = NLIWithChunkingAndPooling(disable_tqdm=True)
    
        self.max_ngram = max_ngram
        self.pooling_fn = pooling_fn

    
    def calculate_lexical_similarity(self, texts: list[str]) -> PooledResults:
        sim = pairwise_jaccard_ngram_similarity(texts, self.max_ngram)
        mean = self.pooling_fn(sim[np.triu_indices_from(sim, k=1)])
        std = np.std(sim[np.triu_indices_from(sim, k=1)])
        return PooledResults(
            mean=mean,
            std=std
        )
    
    def calculate_semantic_similarity(self, texts: list[str]) -> PooledResults:
        sim = pairwise_cosine_similarity(texts, self.semantic_model)
        mean = self.pooling_fn(sim[np.triu_indices_from(sim, k=1)])
        std = np.std(sim[np.triu_indices_from(sim, k=1)])
        return PooledResults(
            mean=mean,
            std=std
        )
    
    def calculate_logical_similarity(self, texts: list[str]) -> PooledResults:
        sim = pairwise_logical_similarity(texts, self.logical_model)
        mean = self.pooling_fn(sim[np.triu_indices_from(sim, k=1)])
        std = np.std(sim[np.triu_indices_from(sim, k=1)])
        return PooledResults(
            mean=mean,
            std=std
        )

    def calculate_similarity(self, texts: list[str]) -> Similarity:
        lexical = self.calculate_lexical_similarity(texts) if self.use_lexical else 0
        semantic = self.calculate_semantic_similarity(texts) if self.use_semantic else 0
        logical = self.calculate_logical_similarity(texts) if self.use_logical else 0
        return Similarity(lexical=lexical, semantic=semantic, logical=logical)
    
    def __call__(self, texts: list[str]) -> Similarity:
        return self.calculate_similarity(texts)