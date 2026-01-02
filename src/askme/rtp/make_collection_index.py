import faiss
import numpy as np
from ..utils import TextEmbeddingWithChunker
from collections.abc import Iterable


def make_faiss_index(
    text_collection: Iterable[str],
    embedding_model: TextEmbeddingWithChunker,
    dimension: int,
    use_gpu: bool = True,
    return_embeddings: bool = False,
) -> faiss.Index | tuple[faiss.Index, np.ndarray]:
    
    index = faiss.IndexFlatL2(dimension)
    
    if use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
        
    embeddings = []
    
    for text in text_collection:
        embedding = embedding_model(text)
        embedding = embedding.astype('float32').reshape(1, -1)
        if return_embeddings:
            embeddings.append(embedding)
        index.add(embedding)
    
    if return_embeddings:
        embeddings = np.vstack(embeddings)
        return index, embeddings
    return index



