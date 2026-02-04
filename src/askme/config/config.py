"""Configuration models for the AskMe applications."""

from pydantic import BaseModel
from typing import Optional
import toml
from askme.assets import config_data as default_config_data


def config_factory(
    config_class: type[BaseModel],
    config_data: dict | None = None,
    override_data: str | None = None,
) -> BaseModel:
    if config_data is None:
        config_data = default_config_data
    config_dict = config_data[config_class.__name__]
    
    if override_data is not None:
        override_dict = config_data[override_data]
        config_dict.update(override_dict)
    
    return config_class(**config_dict)

class LabelPropagationConfig(BaseModel):
    n_neighbors: int = 10
    sigma: float = 1.0
    alpha: float = 0.99
    max_iter: int = 100
    tol: float = 1e-3
    
class MakeQuestionsConfig(BaseModel):
    """Configuration for making questions."""

    model_name: str = "gpt-oss:20b"
    temperature: float = 0.7
    max_tokens: int = 150
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    max_words_per_text: int = 350
    retries: int = 10
    blacklist: Optional[list[str]] = []
    cache: str = "./make_questions_cache.cache"

class TextEmbeddingConfig(BaseModel):
    """Configuration for text embedding model."""
    model_name: str = "sentence-transformers/all-mpnet-base-v2"
    device: str = "auto"  # 'auto', 'cpu', or 'cuda'
    chunk_size: int = 100
    overlap: int = 25
    cache: str = "./embedding_cache.cache"

class NLIBatchingChukingConfig(BaseModel):
    """Configuration for NLI batching and chunking."""

    model_name: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    chunk_size: int = 512
    overlap: int = 50
    device: str = "auto"  # 'auto', 'cpu', or 'cuda'
    label_names: list[str] = ["entailment", "neutral", "contradiction"]
    batch_size: int = 16
    max_chunks_per_minibatch: int = 64
    max_characters_per_chunk: int = 10000
    
class SamplingConfig(BaseModel):
    """Configuration for sampling strategies."""
    selection_strategy: str = "vote_k"  # e.g., 'random', 'vote_k', 'kmeans'
    n_select: int | float = 10
    n_samples: int = 5
    k_neighbors: int = 15 # Number of neighbors for VoteKSampler
    seed: int = 42
    total_size : Optional[int] = None  # Used for samplers that need total size
    use_gpu: bool = True # Use GPU foi FAISS kmeans
    niter: int = 50
    spherical: bool = True