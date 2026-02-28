
from .lexical_similarity import pairwise_jaccard_ngram_similarity
from .semantic_similarity import pairwise_cosine_similarity
from .logical_similarity import pairwise_logical_similarity

import numpy as np

from pydantic import BaseModel

class Similarity(BaseModel):
    lexical: float
    semantic: float
    logical: float

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
            self.logical_model = NLIWithChunkingAndPooling()
    
        self.max_ngram = max_ngram
        self.pooling_fn = pooling_fn

    
    def calculate_lexical_similarity(self, texts: list[str]) -> float:
        return self.pooling_fn(pairwise_jaccard_ngram_similarity(texts, self.max_ngram))
    
    def calculate_semantic_similarity(self, texts: list[str]) -> float:
        return self.pooling_fn(pairwise_cosine_similarity(texts, self.semantic_model))
    
    def calculate_logical_similarity(self, texts: list[str]) -> float:
        return self.pooling_fn(pairwise_logical_similarity(texts, self.logical_model))
    
    def calculate_similarity(self, texts: list[str]) -> Similarity:
        lexical = self.calculate_lexical_similarity(texts) if self.use_lexical else 0
        semantic = self.calculate_semantic_similarity(texts) if self.use_semantic else 0
        logical = self.calculate_logical_similarity(texts) if self.use_logical else 0
        return Similarity(lexical=lexical, semantic=semantic, logical=logical)
    
    def __call__(self, texts: list[str]) -> Similarity:
        return self.calculate_similarity(texts)